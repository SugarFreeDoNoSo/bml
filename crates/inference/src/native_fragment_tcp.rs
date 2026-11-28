//! Opt-in loopback TCP experiment for the native ELF row-fragment protocol.
//!
//! The wire format and numerical kernel are exactly the stdio v1 protocol.
//! Descriptor equality binds model, weights and row ownership, but is **not**
//! cryptographic peer authentication or TLS. Any local process able to reach
//! the listener can connect; this module is not a multi-host deployment path.
//! Only literal loopback endpoints are admitted, without DNS or retries.
//!
//! Client connection/handshakes and each full projection share a 20-second
//! deadline. Workers accept one connection within 20 seconds, cap session I/O
//! at 120 seconds, and renew a 20-second exchange deadline only after a complete
//! Ready/Output flush. Partial reads/writes never renew a deadline. The stdio
//! protocol's 256 requests, 256 KiB frames and numerical quotas remain intact.
//! These I/O checks do not preempt CPU computation; a process supervisor is
//! required for a hard elapsed-time limit on the complete worker process.

use crate::native_fragment_protocol::{self as wire, Descriptor};
use crate::native_fragments::{layout, FragmentExecutable};
use crate::{Error, Result};
use std::cell::Cell;
use std::collections::HashSet;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

const PHASE_TIMEOUT: Duration = Duration::from_secs(20);
const SESSION_TIMEOUT: Duration = Duration::from_secs(120);

fn endpoint(address: SocketAddr, allow_zero_port: bool) -> Result<()> {
    let unusual_v6 = match address {
        SocketAddr::V6(value) => value.scope_id() != 0 || value.flowinfo() != 0,
        SocketAddr::V4(_) => false,
    };
    if !address.ip().is_loopback() || unusual_v6 || (!allow_zero_port && address.port() == 0) {
        return Err(Error(
            "native fragment TCP requires a literal loopback endpoint and a nonzero port for clients"
                .into(),
        ));
    }
    Ok(())
}

/// Parse a literal loopback endpoint without resolving names.
/// Port zero is allowed only when selecting a local listener's ephemeral port.
pub fn parse_loopback_endpoint(text: &str, allow_zero_port: bool) -> Result<SocketAddr> {
    let address = text
        .parse::<SocketAddr>()
        .map_err(|_| Error("native fragment TCP requires a literal loopback endpoint".into()))?;
    endpoint(address, allow_zero_port)?;
    Ok(address)
}

fn timeout() -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        "native fragment TCP deadline exceeded",
    )
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(timeout)
}

fn io_error(error: io::Error) -> io::Error {
    if matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ) {
        timeout()
    } else {
        error
    }
}

fn read_before(stream: &TcpStream, deadline: Instant, bytes: &mut [u8]) -> io::Result<usize> {
    stream.set_read_timeout(Some(remaining(deadline)?))?;
    let mut channel = stream;
    let count = channel.read(bytes).map_err(io_error)?;
    remaining(deadline)?;
    Ok(count)
}

fn write_before(stream: &TcpStream, deadline: Instant, bytes: &[u8]) -> io::Result<usize> {
    stream.set_write_timeout(Some(remaining(deadline)?))?;
    let mut channel = stream;
    let count = channel.write(bytes).map_err(io_error)?;
    remaining(deadline)?;
    Ok(count)
}

struct DeadlineIo<'a> {
    stream: &'a TcpStream,
    deadline: Instant,
}

impl Read for DeadlineIo<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        read_before(self.stream, self.deadline, bytes)
    }
}

impl Write for DeadlineIo<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        write_before(self.stream, self.deadline, bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        remaining(self.deadline)?;
        let mut channel = self.stream;
        channel.flush().map_err(io_error)?;
        remaining(self.deadline).map(|_| ())
    }
}

/// A local TCP group with exact row ownership and no retry after failure.
///
/// This client checks the compiled descriptors; it cannot authenticate the
/// remote executable hash over TCP. Use only the explicitly opt-in local test.
pub struct TcpProjectionGroup {
    streams: Vec<TcpStream>,
    descriptors: Vec<Descriptor>,
    rows: usize,
    cols: usize,
    sequence: u32,
    failed: bool,
    phase_timeout: Duration,
}

impl TcpProjectionGroup {
    /// Validate all endpoints and complete row coverage before connecting.
    /// All Ready handshakes must pass before any activation can be transmitted.
    pub fn connect(addresses: &[SocketAddr], fragments: &[FragmentExecutable]) -> Result<Self> {
        Self::connect_with_timeout(addresses, fragments, PHASE_TIMEOUT)
    }

    fn connect_with_timeout(
        addresses: &[SocketAddr],
        fragments: &[FragmentExecutable],
        phase_timeout: Duration,
    ) -> Result<Self> {
        let (rows, cols) = layout(fragments)?;
        if addresses.len() != fragments.len()
            || addresses.iter().collect::<HashSet<_>>().len() != addresses.len()
        {
            return Err(Error(
                "native fragment TCP endpoint count or duplicates rejected".into(),
            ));
        }
        for &address in addresses {
            endpoint(address, false)?;
        }
        if phase_timeout.is_zero() || phase_timeout > PHASE_TIMEOUT {
            return Err(Error("invalid native fragment TCP deadline".into()));
        }
        let deadline = Instant::now() + phase_timeout;
        let mut group = Self {
            streams: Vec::new(),
            descriptors: fragments
                .iter()
                .map(|item| item.descriptor.clone())
                .collect(),
            rows,
            cols,
            sequence: 0,
            failed: false,
            phase_timeout,
        };
        for (&address, fragment) in addresses.iter().zip(fragments) {
            let stream =
                TcpStream::connect_timeout(&address, remaining(deadline)?).map_err(|error| {
                    Error(format!("native fragment TCP connection failed: {error}"))
                })?;
            stream.set_nodelay(true)?;
            group.streams.push(stream);
            let stream = group
                .streams
                .last()
                .ok_or_else(|| Error("missing TCP stream".into()))?;
            wire::handshake(&mut DeadlineIo { stream, deadline }, &fragment.descriptor)?;
        }
        remaining(deadline)?;
        Ok(group)
    }

    fn invalidate(&mut self) {
        self.failed = true;
        for stream in &self.streams {
            let _ = stream.shutdown(Shutdown::Both);
        }
        self.streams.clear();
    }

    /// Send all activations, then concatenate exact row responses in plan order.
    /// Any failure permanently invalidates this group and closes its sockets.
    pub fn project(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.failed {
            return Err(Error("native fragment TCP session is invalid".into()));
        }
        if self.sequence >= wire::MAX_REQUESTS
            || input.len() != self.cols
            || input.iter().any(|value| !value.is_finite())
        {
            self.invalidate();
            return Err(Error(
                "native fragment TCP activation or request budget rejected".into(),
            ));
        }
        let result = (|| {
            let deadline = Instant::now() + self.phase_timeout;
            for stream in &self.streams {
                wire::write_request(&mut DeadlineIo { stream, deadline }, self.sequence, input)?;
            }
            let mut output = Vec::new();
            output
                .try_reserve_exact(self.rows)
                .map_err(|_| Error("native fragment TCP result allocation failed".into()))?;
            for (stream, descriptor) in self.streams.iter().zip(&self.descriptors) {
                output.extend(wire::read_response(
                    &mut DeadlineIo { stream, deadline },
                    descriptor,
                    self.sequence,
                )?);
            }
            remaining(deadline)?;
            if output.len() != self.rows || output.iter().any(|value| !value.is_finite()) {
                return Err(Error("native fragment TCP aggregate rejected".into()));
            }
            Ok(output)
        })();
        if result.is_err() {
            self.invalidate();
        } else {
            self.sequence += 1;
        }
        result
    }
}

impl Drop for TcpProjectionGroup {
    fn drop(&mut self) {
        self.invalidate();
    }
}

struct ServerBudget {
    phase_end: Cell<Instant>,
    session_end: Instant,
    phase_timeout: Duration,
}

impl ServerBudget {
    fn new(phase_timeout: Duration, session_timeout: Duration) -> Self {
        let now = Instant::now();
        let session_end = now + session_timeout;
        Self {
            phase_end: Cell::new((now + phase_timeout).min(session_end)),
            session_end,
            phase_timeout,
        }
    }
    fn next_exchange(&self) -> io::Result<()> {
        remaining(self.phase_end.get())?;
        remaining(self.session_end)?;
        self.phase_end
            .set((Instant::now() + self.phase_timeout).min(self.session_end));
        Ok(())
    }
}

struct ServerReader<'a> {
    stream: &'a TcpStream,
    budget: &'a ServerBudget,
}

impl Read for ServerReader<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        read_before(self.stream, self.budget.phase_end.get(), bytes)
    }
}

struct ServerWriter<'a> {
    stream: &'a TcpStream,
    budget: &'a ServerBudget,
}

impl Write for ServerWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        write_before(self.stream, self.budget.phase_end.get(), bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        let mut io = DeadlineIo {
            stream: self.stream,
            deadline: self.budget.phase_end.get(),
        };
        io.flush()?;
        self.budget.next_exchange()
    }
}

fn accept_before(listener: &TcpListener, deadline: Instant) -> Result<TcpStream> {
    listener.set_nonblocking(true)?;
    loop {
        remaining(deadline)?;
        match listener.accept() {
            Ok((stream, address)) => {
                endpoint(address, false)?;
                remaining(deadline)?;
                return Ok(stream);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(remaining(deadline)?.min(Duration::from_millis(2)));
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
}

fn serve_stream(
    descriptor: &Descriptor,
    weights: &[u8],
    stream: TcpStream,
    phase_timeout: Duration,
    session_timeout: Duration,
) -> Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    let budget = ServerBudget::new(phase_timeout, session_timeout);
    let result = wire::serve(
        descriptor,
        weights,
        ServerReader {
            stream: &stream,
            budget: &budget,
        },
        ServerWriter {
            stream: &stream,
            budget: &budget,
        },
    );
    let _ = stream.shutdown(Shutdown::Both);
    result
}

/// Announce a literal loopback listener, serve exactly one connection and exit.
///
/// Weights are authenticated before bind. Stdout receives exactly one JSON line
/// containing format, actual address and descriptor; TCP carries wire v1 frames.
/// The listener has no peer authentication and is for the opt-in local test only.
pub fn serve_listener(descriptor: &Descriptor, weights: &[u8], address: SocketAddr) -> Result<()> {
    descriptor.validate(weights)?;
    endpoint(address, true)?;
    let listener = TcpListener::bind(address)?;
    let accept_deadline = Instant::now() + PHASE_TIMEOUT;
    {
        let mut stdout = std::io::stdout().lock();
        serde_json::to_writer(
            &mut stdout,
            &serde_json::json!({
                "format": "bml-native-fragment-tcp-listener-v1",
                "address": listener.local_addr()?.to_string(),
                "descriptor": descriptor,
            }),
        )?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    let stream = accept_before(&listener, accept_deadline)?;
    drop(listener);
    serve_stream(descriptor, weights, stream, PHASE_TIMEOUT, SESSION_TIMEOUT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::thread::{self, JoinHandle};

    fn fragments() -> Vec<(FragmentExecutable, Vec<u8>)> {
        [
            [1.0f32, -2.0, 3.0, -0.0, -0.0, -0.0],
            [2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(i, weights)| {
            let weights: Vec<u8> = weights.into_iter().flat_map(f32::to_le_bytes).collect();
            (
                FragmentExecutable {
                    file_name: format!("fragment_{i}"),
                    sha256: "ab".repeat(32),
                    descriptor: Descriptor {
                        version: wire::VERSION,
                        source_sha256: "cd".repeat(32),
                        tensor_name: "output.weight".into(),
                        row_start: i * 2,
                        row_end: (i + 1) * 2,
                        total_rows: 4,
                        cols: 3,
                        weights_sha256: format!("{:x}", Sha256::digest(&weights)),
                    },
                },
                weights,
            )
        })
        .collect()
    }

    fn server(
        descriptor: Descriptor,
        weights: Vec<u8>,
        phase: Duration,
        session: Duration,
    ) -> (SocketAddr, JoinHandle<Result<()>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let stream = accept_before(&listener, Instant::now() + PHASE_TIMEOUT)?;
            drop(listener);
            serve_stream(&descriptor, &weights, stream, phase, session)
        });
        (address, join)
    }

    #[test]
    fn endpoints_are_literal_loopback_and_zero_is_bind_only() {
        assert_eq!(
            parse_loopback_endpoint("127.0.0.1:0", true).unwrap().port(),
            0
        );
        assert!(parse_loopback_endpoint("[::1]:12", false).is_ok());
        for text in [
            "localhost:12",
            "0.0.0.0:12",
            "[::]:12",
            "192.168.1.1:12",
            "8.8.8.8:12",
            "http://127.0.0.1:12",
            "127.0.0.1:0",
            " 127.0.0.1:12",
            "[::ffff:127.0.0.1]:12",
            "[::1%3]:12",
        ] {
            assert!(parse_loopback_endpoint(text, false).is_err(), "{text}");
        }
    }

    #[test]
    fn full_preflight_happens_before_any_connection() {
        let expected: Vec<_> = fragments().into_iter().map(|x| x.0).collect();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        assert!(TcpProjectionGroup::connect(&[address, address], &expected).is_err());
        assert!(TcpProjectionGroup::connect(&[address], &expected).is_err());
        assert!(
            TcpProjectionGroup::connect(&[address, "0.0.0.0:9".parse().unwrap()], &expected)
                .is_err()
        );
        let mut incomplete = expected.clone();
        incomplete[1].descriptor.row_start += 1;
        assert!(TcpProjectionGroup::connect(
            &[address, "127.0.0.1:9".parse().unwrap()],
            &incomplete
        )
        .is_err());
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }

    #[test]
    fn two_tcp_workers_preserve_bits_row_order_and_invalidation() {
        let fixtures = fragments();
        let expected: Vec<_> = fixtures.iter().map(|x| x.0.clone()).collect();
        let mut addresses = Vec::new();
        let mut threads = Vec::new();
        for (fragment, weights) in fixtures {
            let (address, join) =
                server(fragment.descriptor, weights, PHASE_TIMEOUT, SESSION_TIMEOUT);
            addresses.push(address);
            threads.push(join);
        }
        let mut group = TcpProjectionGroup::connect(&addresses, &expected).unwrap();
        for input in [[2.0f32, 3.0, 4.0], [1.0, 1.0, 1.0]] {
            let actual = group.project(&input).unwrap();
            let want: Vec<f32> = fragments()
                .into_iter()
                .flat_map(|(_, weights)| {
                    crate::model::project_f32_rows(&weights, 2, 3, &input).unwrap()
                })
                .collect();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(actual[1].to_bits(), (-0.0f32).to_bits());
        }
        assert!(group.project(&[1.0]).is_err());
        assert!(group.streams.is_empty());
        assert!(group
            .project(&[1.0; 3])
            .unwrap_err()
            .0
            .contains("session is invalid"));
        drop(group);
        for join in threads {
            join.join().unwrap().unwrap();
        }
    }

    #[test]
    fn wrong_ready_identity_is_rejected_before_activations() {
        let (mut expected, weights) = fragments().remove(0);
        expected.descriptor.total_rows = 2;
        let mut wrong = expected.descriptor.clone();
        wrong.source_sha256 = "ef".repeat(32);
        let (address, join) = server(wrong, weights, PHASE_TIMEOUT, SESSION_TIMEOUT);
        assert!(TcpProjectionGroup::connect(&[address], &[expected])
            .err()
            .unwrap()
            .0
            .contains("identity mismatch"));
        join.join().unwrap().unwrap();
    }

    #[test]
    fn server_rejects_bad_frame_length_and_authenticates_before_bind() {
        let (fragment, weights) = fragments().remove(0);
        let (address, join) = server(
            fragment.descriptor.clone(),
            weights.clone(),
            PHASE_TIMEOUT,
            SESSION_TIMEOUT,
        );
        let stream = TcpStream::connect(address).unwrap();
        let mut io = DeadlineIo {
            stream: &stream,
            deadline: Instant::now() + PHASE_TIMEOUT,
        };
        wire::handshake(&mut io, &fragment.descriptor).unwrap();
        io.write_all(&((wire::MAX_FRAME_BYTES + 1) as u32).to_le_bytes())
            .unwrap();
        assert!(wire::read_response(&mut io, &fragment.descriptor, 0).is_err());
        assert!(join.join().unwrap().unwrap_err().0.contains("frame length"));
        let occupied = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut wrong = fragment.descriptor;
        wrong.weights_sha256 = "00".repeat(32);
        assert!(
            serve_listener(&wrong, &weights, occupied.local_addr().unwrap())
                .unwrap_err()
                .0
                .contains("SHA-256")
        );
    }

    #[test]
    fn disconnected_peer_invalidates_the_group_without_reconnecting() {
        let (mut fragment, weights) = fragments().remove(0);
        fragment.descriptor.total_rows = 2;
        let mut ready = Vec::new();
        wire::serve(&fragment.descriptor, &weights, &[][..], &mut ready).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut peer, _) = listener.accept().unwrap();
            peer.write_all(&ready).unwrap();
            peer.shutdown(Shutdown::Both).unwrap();
        });
        let mut group = TcpProjectionGroup::connect(&[address], &[fragment]).unwrap();
        join.join().unwrap();
        assert!(group.project(&[1.0; 3]).is_err());
        assert!(group.streams.is_empty());
        assert!(group
            .project(&[1.0; 3])
            .unwrap_err()
            .0
            .contains("session is invalid"));
    }

    #[test]
    fn partial_request_does_not_renew_server_deadline() {
        let (fragment, weights) = fragments().remove(0);
        let (address, join) = server(
            fragment.descriptor.clone(),
            weights,
            Duration::from_millis(100),
            Duration::from_secs(2),
        );
        let stream = TcpStream::connect(address).unwrap();
        let mut io = DeadlineIo {
            stream: &stream,
            deadline: Instant::now() + PHASE_TIMEOUT,
        };
        wire::handshake(&mut io, &fragment.descriptor).unwrap();
        // A fresh 100 ms idle timeout per read would accept this complete valid
        // request: every gap is 40 ms, but the exchange takes over 200 ms.
        let mut request = Vec::new();
        wire::write_request(&mut request, 0, &[1.0; 3]).unwrap();
        for byte in &request[..5] {
            if io.write_all(std::slice::from_ref(byte)).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(40));
        }
        let _ = io.write_all(&request[5..]);
        let _ = stream.shutdown(Shutdown::Write);
        assert!(join.join().unwrap().unwrap_err().0.contains("deadline"));
    }

    #[test]
    fn accept_and_session_deadlines_are_bounded_independently() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        assert!(
            accept_before(&listener, Instant::now() + Duration::from_millis(20))
                .unwrap_err()
                .0
                .contains("deadline")
        );
        let (fragment, weights) = fragments().remove(0);
        let (address, join) = server(
            fragment.descriptor.clone(),
            weights,
            Duration::from_secs(2),
            Duration::from_millis(100),
        );
        let stream = TcpStream::connect(address).unwrap();
        wire::handshake(
            &mut DeadlineIo {
                stream: &stream,
                deadline: Instant::now() + PHASE_TIMEOUT,
            },
            &fragment.descriptor,
        )
        .unwrap();
        assert!(join.join().unwrap().unwrap_err().0.contains("deadline"));
    }

    #[test]
    fn partial_response_does_not_renew_client_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut peer, _) = listener.accept().unwrap();
            for byte in 0..6u8 {
                if peer.write_all(&[byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(40));
            }
        });
        let stream = TcpStream::connect(address).unwrap();
        let mut io = DeadlineIo {
            stream: &stream,
            deadline: Instant::now() + Duration::from_millis(100),
        };
        // Six bytes arrive within each 100 ms idle window but beyond the one
        // shared absolute deadline. A timeout renewed per read would succeed.
        let mut bytes = [0; 6];
        assert_eq!(
            io.read_exact(&mut bytes).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        join.join().unwrap();
    }
}
