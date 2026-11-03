#!/usr/bin/env python3
"""Instrument the pinned SegQueue used by runtime WorkQueue, without a mock queue."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_SHA256 = "f4d2e9500ef31a31883075c3b151b8538fa664422e15e1c2fa9045b329aec270"


def remove_method(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    # Remove the doc block belonging to this unused exclusive API too.
    lines = source[:start].splitlines(keepends=True)
    while lines and (lines[-1].lstrip().startswith("///") or not lines[-1].strip()):
        lines.pop()
    return "".join(lines) + "\n" + source[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lap", type=int, choices=[4, 32], default=32)
    parser.add_argument("--weaken-publication", action="store_true")
    parser.add_argument("--weaken-acquire", action="store_true")
    parser.add_argument("--omit-retired-free", action="store_true")
    parser.add_argument("--omit-final-free", action="store_true")
    parser.add_argument("--premature-retirement", action="store_true",
                        help="Negative control: bypass deferral to an unfinished consumer")
    parser.add_argument("--omit-payload-drop", action="store_true",
                        help="Negative control: leak queued payloads during queue destruction")
    args = parser.parse_args()
    original = (args.source / "src/seg_queue.rs").read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == SOURCE_SHA256
    source = original[:original.index("impl<T> IntoIterator for SegQueue<T>")]
    for signature in ["    pub fn push_mut(", "    pub fn pop_mut("]:
        source = remove_method(source, signature)
    start = source.index("    const LAYOUT: Layout")
    end = source.index("    /// Waits until the next pointer", start)
    source = source[:start] + '''    fn new() -> Box<Self> {
        Box::new(Self {
            allocation_id: crate::allocations::allocate(),
            next: AtomicPtr::new(ptr::null_mut()),
            slots: std::array::from_fn(|_| Slot {
                value: UnsafeCell::new(MaybeUninit::uninit()),
                state: AtomicUsize::new(0),
            }),
        })
    }

''' + source[end:]
    replacements = [
        ("use alloc::alloc::{alloc_zeroed, handle_alloc_error, Layout};\n", ""),
        ("use alloc::boxed::Box;", "use std::boxed::Box;"),
        ("use core::cell::UnsafeCell;", "use loom::cell::UnsafeCell;"),
        ("use core::sync::atomic::{self, AtomicPtr, AtomicUsize, Ordering};",
         "use loom::sync::atomic::{self, AtomicPtr, AtomicUsize, Ordering};"),
        ("use crossbeam_utils::{Backoff, CachePadded};",
         "use crossbeam_utils::CachePadded;\nuse crate::Backoff;"),
        ("pub const fn new()", "pub fn new()"),
        ("const LAP: usize = 32;", f"const LAP: usize = {args.lap};"),
        ("slot.value.get().write(MaybeUninit::new(value));",
         "slot.value.with_mut(|p| p.write(MaybeUninit::new(value)));"),
        ("slot.value.get().read().assume_init()",
         "slot.value.with(|p| p.read()).assume_init()"),
        ("(*slot.value.get()).assume_init_drop();",
         "slot.value.with_mut(|p| (*p).assume_init_drop());"),
    ]
    for before, after in replacements:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    for field in ["self.head.index", "self.tail.index", "self.head.block", "(*block).next"]:
        before = f"*{field}.get_mut()"
        assert source.count(before) == 1, before
        source = source.replace(before, f"{field}.with_mut(|p| *p)")
    if args.weaken_publication:
        before = "slot.state.fetch_or(WRITE, Ordering::Release);"
        assert source.count(before) == 1
        source = source.replace(before, "slot.state.fetch_or(WRITE, Ordering::Relaxed);")
    if args.weaken_acquire:
        before = "self.state.load(Ordering::Acquire) & WRITE"
        assert source.count(before) == 1
        source = source.replace(before, "self.state.load(Ordering::Relaxed) & WRITE")
    if args.omit_payload_drop:
        before = "slot.value.with_mut(|p| (*p).assume_init_drop());"
        assert source.count(before) == 1
        source = source.replace(before, "// Negative control: omit queued payload destruction")
    before = "struct Block<T> {"
    assert source.count(before) == 1
    source = source.replace(before, before + "\n    allocation_id: usize,")
    source += '''
impl<T> Drop for Block<T> {
    fn drop(&mut self) {
        crate::allocations::free(self.allocation_id);
    }
}
'''
    # Observe the reclamation precondition before actually freeing raw memory.
    # The last slot owns retirement; every earlier slot must have published READ.
    before = "drop(Box::from_raw(this));"
    assert source.count(before) == 1
    source = source.replace(before, '''for i in 0..BLOCK_CAP - 1 {
            assert_ne!((*this).slots[i].state.load(Ordering::Relaxed) & READ, 0,
                       "premature segment release: consumer has not published READ");
        }
        drop(Box::from_raw(this));''')
    if args.premature_retirement:
        start = source.index("    unsafe fn destroy(this:")
        end = source.index("    unsafe fn destroy_mut(", start)
        region = source[start:end]
        assert region.count("return;") == 1
        source = source[:start] + region.replace("return;", "break; // bypass retirement deferral") + source[end:]
    if args.omit_retired_free:
        before = "drop(Box::from_raw(this));"
        assert source.count(before) == 1
        source = source.replace(before, "// Negative control: leak retired segment")
    if args.omit_final_free:
        start = source.index("impl<T> Drop for SegQueue<T>")
        region = source[start:]
        before = "drop(Box::from_raw(block));"
        assert region.count(before) == 2
        source = source[:start] + region.replace(before, "// Negative control: leak final segment")
    args.output.mkdir(parents=True, exist_ok=False)
    src = args.output / "src"
    src.mkdir()
    (src / "seg_queue.rs").write_text(source)
    (src / "lib.rs").write_text(f'''mod seg_queue;
mod allocations;
pub use seg_queue::SegQueue;
#[cfg(test)] const BLOCK_CAP: usize = {args.lap - 1};
struct Backoff;
impl Backoff {{
    fn new() -> Self {{ Self }}
    fn spin(&self) {{ loom::thread::yield_now(); }}
    fn snooze(&self) {{ loom::thread::yield_now(); }}
}}
#[cfg(test)] mod slots;
''')
    tests = Path(__file__).parent / "loom/segmented_work_slots.rs"
    shutil.copyfile(tests, src / "slots.rs")
    allocations = Path(__file__).parent / "loom/segmented_work_allocations.rs"
    shutil.copyfile(allocations, src / "allocations.rs")
    for name in ["LICENSE-APACHE", "LICENSE-MIT"]:
        shutil.copyfile(args.source / name, args.output / name)
    (args.output / "Cargo.toml").write_text('''[package]
name = "bml-segmented-work-loom-probe"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
crossbeam-utils = "=0.8.22"
loom = "=0.7.2"
''')
    (args.output / "adaptation.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True),
        fromfile="crossbeam-queue-0.3.13/src/seg_queue.rs", tofile="src/seg_queue.rs")))
    (args.output / "provenance.json").write_text(json.dumps({
        "source_sha256": SOURCE_SHA256,
        "adapted_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "tests_sha256": hashlib.sha256(tests.read_bytes()).hexdigest(),
        "lap": args.lap, "weaken_publication": args.weaken_publication,
        "weaken_acquire": args.weaken_acquire,
        "omit_payload_drop": args.omit_payload_drop,
        "omit_retired_free": args.omit_retired_free,
        "omit_final_free": args.omit_final_free,
        "premature_retirement": args.premature_retirement,
        "retirement_guard": "Before physical free, assert READ for each non-final slot using Relaxed loads; fail before an unsafe free in the negative control",
        "allocations_sha256": hashlib.sha256(allocations.read_bytes()).hexdigest(),
        "adaptations": "Loom atomics/cells/backoff; explicit initialized allocation; exclusive destructor access; omit unused push_mut/pop_mut/IntoIterator APIs",
        "production_operations": "push, pop, len, is_empty, block retirement and Drop retained",
        "scope": "bounded scenarios only; not a general liveness proof",
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
