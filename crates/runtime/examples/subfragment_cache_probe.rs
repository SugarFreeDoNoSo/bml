//! Controlled borrowed-subfragment workload for externally operated PMU counters.
//! Run `--verify` for a short correctness check without the stdin protocol.
use bml_compiler::{distributed::SubFragment, RpnOp};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer, Runtime};
use std::hint::black_box;
use std::io::{self, Read, Write};

const ROUTE: &str = "Runtime::execute_sub_fragments";
const FRAGMENTS: usize = 4;
const STACK_CAPACITY: usize = 8; // This workload reaches at most four entries.
const INPUTS: [[f64; 2]; 4] = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
const CONSTANTS: [f64; 1] = [0.5];
const EXPECTED_RESULTS: [f64; 4] = [13.0, 18.0, 24.0, 32.0];
const EXPECTED_ACCUMULATORS: [f64; 4] = [5.0, 10.0, 16.0, 24.0];
// At most 32 is added per call, so every intermediate integer sum is exact.
const MAX_ITERATIONS: u64 = (1_u64 << 53) / 32;

struct Config {
    warmup: u64,
    iterations: u64,
    verify: bool,
}

fn positive_count(value: Option<String>, flag: &str) -> Result<u64, String> {
    let count = value
        .ok_or_else(|| format!("{flag} requires a positive integer"))?
        .parse::<u64>()
        .map_err(|_| format!("{flag} requires a positive integer"))?;
    if count == 0 || count > MAX_ITERATIONS {
        return Err(format!("{flag} must be in 1..={MAX_ITERATIONS}"));
    }
    Ok(count)
}

fn config() -> Result<Option<Config>, String> {
    let mut args = std::env::args().skip(1);
    let (mut warmup, mut iterations, mut verify) = (None, None, false);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" if warmup.is_none() => {
                warmup = Some(positive_count(args.next(), "--warmup")?);
            }
            "--iterations" if iterations.is_none() => {
                iterations = Some(positive_count(args.next(), "--iterations")?);
            }
            "--verify" if !verify => verify = true,
            "--help" | "-h" => {
                println!(
                    "Usage: subfragment_cache_probe [--warmup N] [--iterations N]\n\
                     Defaults: --warmup 10000 --iterations 1000000; counts must be positive.\n\
                     Or: subfragment_cache_probe --verify\n\
                     --verify checks four literal cases and 12 calls, without stdin or PMU.\n\
                     Normal mode: wait for READY, enable counters, send go;\n\
                     after FINISHED, disable counters and send stop."
                );
                return Ok(None);
            }
            _ => return Err(format!("unknown or repeated argument: {arg}")),
        }
    }
    if verify && (warmup.is_some() || iterations.is_some()) {
        return Err("--verify is a fixed short check; use it without count options".into());
    }
    Ok(Some(Config {
        warmup: warmup.unwrap_or(10_000),
        iterations: iterations.unwrap_or(1_000_000),
        verify,
    }))
}

fn subfragment(sub_id: u32, ops: Vec<RpnOp>) -> SubFragment {
    SubFragment {
        fragment_id: 0,
        sub_id,
        layer_start: 0,
        layer_end: 1,
        ops,
        weight_refs: vec![],
        depends_on: if sub_id == 0 {
            vec![]
        } else {
            vec![sub_id - 1]
        },
    }
}

struct Workload {
    runtime: Runtime,
    fragments: [SubFragment; FRAGMENTS],
    buffer: ResultBuffer,
}

impl Workload {
    fn new() -> Self {
        let fragments = [
            subfragment(0, vec![RpnOp::Var(0), RpnOp::One, RpnOp::Bml]),
            subfragment(
                1,
                vec![
                    RpnOp::Loop {
                        count: 2,
                        body_len: 5,
                    },
                    RpnOp::Drop,
                    RpnOp::Loop {
                        count: 2,
                        body_len: 3,
                    },
                    RpnOp::Drop,
                    RpnOp::Var(1),
                    RpnOp::FAdd,
                ],
            ),
            subfragment(
                2,
                vec![
                    RpnOp::Dup,
                    RpnOp::Zero,
                    RpnOp::StoreResult { slot: 0 },
                    RpnOp::Loop {
                        count: 3,
                        body_len: 6,
                    },
                    RpnOp::Dup,
                    RpnOp::VarIndexed { base: 8 },
                    RpnOp::Const(0),
                    RpnOp::FMul,
                    RpnOp::Swap,
                    RpnOp::StoreResult { slot: 1 },
                ],
            ),
            subfragment(
                3,
                vec![
                    RpnOp::Zero,
                    RpnOp::VarIndexed { base: 4 },
                    RpnOp::FAdd,
                    RpnOp::One,
                    RpnOp::VarIndexed { base: 4 },
                    RpnOp::FAdd,
                ],
            ),
        ];
        let mut buffer = ResultBuffer::new(3, 4);
        for slot in 0..3 {
            for offset in 0..4 {
                buffer.write(slot, offset, -999.0);
            }
        }
        for (offset, value) in [6.0, 10.0, 14.0].into_iter().enumerate() {
            buffer.write(2, offset as u32, value);
        }
        Self {
            runtime: Runtime::new(STACK_CAPACITY, 1),
            fragments,
            buffer,
        }
    }

    fn execute_case(&mut self, case: usize) -> f64 {
        let context = EvalContext::new(black_box(&INPUTS[case]), black_box(&CONSTANTS));
        black_box(self.runtime.execute_sub_fragments(
            black_box(&self.fragments),
            black_box(&context),
            black_box(&mut self.buffer),
        ))
    }

    // Only borrowed slices, scalar arithmetic, and the reserved stack/buffer are
    // used here. Construction, the oracle, command parsing, and I/O stay outside.
    fn run(&mut self, iterations: u64) -> f64 {
        let mut checksum = 0.0;
        for iteration in 0..iterations {
            checksum += self.execute_case((iteration & 3) as usize);
        }
        black_box(checksum)
    }

    fn check_state(&self, case: usize) {
        assert_eq!(self.runtime.stack_depth(), 0);
        assert_eq!(self.runtime.stack_capacity(), STACK_CAPACITY);
        assert_eq!(
            self.buffer.slot(0),
            &[EXPECTED_ACCUMULATORS[case], -999.0, -999.0, -999.0]
        );
        assert_eq!(self.buffer.slot(1), &[3.0, 5.0, 7.0, -999.0]);
        assert_eq!(self.buffer.slot(2), &[6.0, 10.0, 14.0, -999.0]);
    }

    fn verify_cases(&mut self) {
        // Independent oracle: B(x, 1) = 2^x. Four additions of y produce
        // [5,10,16,24]; buffer scaling gives [3,5,7], so add 3+5 to return.
        // These literals do not call the runtime or compiler to derive results.
        for (case, expected) in EXPECTED_RESULTS.into_iter().enumerate() {
            assert_eq!(self.execute_case(case), expected);
            self.check_state(case);
        }
    }
}

fn expected_checksum(iterations: u64) -> f64 {
    // Each complete four-case cycle sums to 87, with these literal prefixes.
    const PREFIX: [u64; 4] = [0, 13, 31, 55];
    ((iterations / 4) * 87 + PREFIX[(iterations & 3) as usize]) as f64
}

fn wait_for(input: &mut impl Read, expected: &[u8]) -> io::Result<()> {
    // Bounded command storage also avoids allocation on the synchronization path.
    let mut command = [0_u8; 8];
    let mut len = 0;
    loop {
        let mut byte = [0_u8; 1];
        input.read_exact(&mut byte)?;
        if byte[0] == b'\n' {
            break;
        }
        if len == command.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "command too long",
            ));
        }
        command[len] = byte[0];
        len += 1;
    }
    if len > 0 && command[len - 1] == b'\r' {
        len -= 1;
    }
    if &command[..len] != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unexpected command",
        ));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(config) = config()? else {
        return Ok(());
    };
    let mut workload = Workload::new();
    workload.verify_cases();
    if config.verify {
        let checksum = workload.run(12);
        assert_eq!(checksum, 261.0);
        workload.check_state(3);
        println!("VERIFIED route={ROUTE} cases=4 iterations=12 checksum={checksum}");
        return Ok(());
    }

    // Initialize standard I/O and all oracle/metadata values before warmup.
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let pid = std::process::id();
    let expected = expected_checksum(config.iterations);
    // Count every adjacent pair of executed arrays, including between calls.
    let total_fragment_transitions = config.iterations * FRAGMENTS as u64 - 1;
    let warmup_checksum = workload.run(config.warmup);
    assert_eq!(warmup_checksum, expected_checksum(config.warmup));
    workload.check_state(((config.warmup - 1) & 3) as usize);

    writeln!(
        output,
        "READY pid={pid} route={ROUTE} warmup={} iterations={} totalfragmenttransitions={total_fragment_transitions}",
        config.warmup, config.iterations,
    )?;
    output.flush()?;
    wait_for(&mut input, b"go")?;

    let checksum = workload.run(config.iterations);
    assert_eq!(checksum, expected);
    workload.check_state(((config.iterations - 1) & 3) as usize);

    writeln!(output, "FINISHED")?;
    output.flush()?;
    wait_for(&mut input, b"stop")?;
    writeln!(
        output,
        "RESULT route={ROUTE} iterations={} subfragments={FRAGMENTS} totalfragmenttransitions={total_fragment_transitions} checksum={checksum} expected={expected} stack_capacity={STACK_CAPACITY}",
        config.iterations,
    )?;
    output.flush()?;
    Ok(())
}
