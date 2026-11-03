#!/usr/bin/env python3
"""Prepare offline Loom probes from pinned actual deque source.

Slots are optionally tracked; epoch reclamation remains native in every probe.
The generated diff exposes every adaptation; production sources are untouched.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_SHA256 = "ef56bda48e2e534a315b32b67f4e40d603e3e80e2bd96dfe802c1559dbc9706b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Cached crossbeam-deque-0.8.7 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-control", action="store_true",
                        help="Break successful thief CAS advancement; must fail ownership tests")
    parser.add_argument("--slots", action="store_true",
                        help="Track Buffer payload accesses with Loom UnsafeCell")
    parser.add_argument("--weaken-publication", action="store_true",
                        help="Remove push's release fence; requires --slots")
    parser.add_argument("--weaken-acquire", action="store_true",
                        help="Relax thief back-index acquire load; requires --slots")
    parser.add_argument("--slot-capacity", type=int, default=64,
                        help="Minimum buffer capacity for bounded reuse models (default upstream 64)")
    args = parser.parse_args()
    if (args.weaken_publication or args.weaken_acquire) and not args.slots:
        parser.error("ordering controls require --slots")
    if args.slot_capacity < 2 or args.slot_capacity & (args.slot_capacity - 1):
        parser.error("--slot-capacity must be a power of two >= 2")
    if args.slot_capacity != 64 and not args.slots:
        parser.error("reduced capacity requires --slots")
    original = (args.source / "src/deque.rs").read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == SOURCE_SHA256
    adapted = original.replace(
        "use std::sync::atomic::{self, AtomicIsize, AtomicPtr, AtomicUsize, Ordering};",
        "use loom::sync::atomic::{self, AtomicIsize, AtomicPtr, AtomicUsize, Ordering};",
    )
    # Loom exposes exclusive atomic access as with_mut rather than get_mut.
    for field in ["self.back", "self.front", "self.head.index", "self.tail.index",
                  "self.head.block", "(*block).next"]:
        before = f"*{field}.get_mut()"
        assert adapted.count(before) == 1, before
        adapted = adapted.replace(before, f"{field}.with_mut(|value| *value)")
    if args.slots:
        replacements = [
            ("ptr: *mut T,", "ptr: *mut loom::cell::UnsafeCell<MaybeUninit<T>>,"),
            (".map(|_| MaybeUninit::<T>::uninit())",
             ".map(|_| loom::cell::UnsafeCell::new(MaybeUninit::<T>::uninit()))"),
            (".cast::<T>();", ".cast::<loom::cell::UnsafeCell<MaybeUninit<T>>>();"),
            ("self.ptr.cast::<MaybeUninit<T>>()", "self.ptr"),
            ("unsafe fn at(&self, index: isize) -> *mut T",
             "unsafe fn at(&self, index: isize) -> *mut loom::cell::UnsafeCell<MaybeUninit<T>>"),
            ("ptr::write_volatile(self.at(index).cast::<MaybeUninit<T>>(), task)",
             "(*self.at(index)).with_mut(|slot| ptr::write_volatile(slot, task))"),
            ("ptr::read_volatile(self.at(index).cast::<MaybeUninit<T>>())",
             "(*self.at(index)).with(|slot| ptr::read_volatile(slot))"),
            ("buffer.deref().at(i).drop_in_place();",
             "(*buffer.deref().at(i)).with_mut(|slot| (*slot).assume_init_drop());"),
            ("ptr::copy_nonoverlapping(buffer.at(i), new.at(i), 1);",
             "new.write(i, buffer.read(i));"),
            ("const MIN_CAP: usize = 64;", f"const MIN_CAP: usize = {args.slot_capacity};"),
        ]
        for before, after in replacements:
            assert adapted.count(before) == 1, before
            adapted = adapted.replace(before, after)
        # In this non-resizing probe the native epoch still owns reclamation.
        # Mirror the SeqCst fence supplied by pin() (or the reentrant branch)
        # in the single-item thief path, so a missing modeled epoch fence
        # cannot manufacture a publication/reuse failure.
        start = adapted.index("pub fn steal(&self)")
        end = adapted.index("pub fn steal_batch", start)
        region = adapted[start:end]
        needle = "if epoch::is_pinned() {\n            atomic::fence(Ordering::SeqCst);\n        }"
        assert region.count(needle) == 1
        region = region.replace(needle, "atomic::fence(Ordering::SeqCst); // model pin/reentrant fence")
        if args.weaken_acquire:
            needle = "self.inner.back.load(Ordering::Acquire)"
            assert region.count(needle) == 1
            region = region.replace(needle, "self.inner.back.load(Ordering::Relaxed)")
        adapted = adapted[:start] + region + adapted[end:]
    if args.weaken_publication:
        start = adapted.index("pub fn push(&self, task: T)")
        end = adapted.index("pub fn pop(&self)", start)
        region = adapted[start:end]
        needle = "atomic::fence(Ordering::Release);"
        assert region.count(needle) == 1
        adapted = adapted[:start] + region.replace(needle, "{} // publication negative control") + adapted[end:]
    if args.negative_control:
        # The single-item steal CAS is distinguished from batch/LIFO operations.
        start = adapted.index("pub fn steal(&self)")
        end = adapted.index("pub fn steal_batch", start)
        region = adapted[start:end]
        needle = "f.wrapping_add(1)"
        assert region.count(needle) == 1
        adapted = adapted[:start] + region.replace(needle, "f") + adapted[end:]
    args.output.mkdir(parents=True, exist_ok=False)
    src = args.output / "src"
    src.mkdir()
    (src / "deque.rs").write_text(adapted)
    (src / "lib.rs").write_text(
        "mod deque;\npub use deque::{Injector, Steal, Stealer, Worker};\n"
        "#[cfg(test)] mod indices;\n" +
        (f"#[cfg(test)] const SLOT_CAPACITY: usize = {args.slot_capacity};\n"
         "#[cfg(test)] mod slots;\n" if args.slots else ""))
    shutil.copyfile(Path(__file__).parent / "loom/work_stealing_indices.rs", src / "indices.rs")
    if args.slots:
        shutil.copyfile(Path(__file__).parent / "loom/work_stealing_slots.rs", src / "slots.rs")
    for name in ["LICENSE-APACHE", "LICENSE-MIT"]:
        shutil.copyfile(args.source / name, args.output / name)
    (args.output / "Cargo.toml").write_text('''[package]
name = "bml-work-stealing-loom-probe"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
crossbeam-epoch = "=0.9.20"
crossbeam-utils = "=0.8.22"
loom = "=0.7.2"
''')
    (args.output / "adaptation.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), adapted.splitlines(True),
        fromfile="crossbeam-deque-0.8.7/src/deque.rs", tofile="src/deque.rs")))
    (args.output / "provenance.json").write_text(json.dumps({
        "upstream": "crossbeam-deque 0.8.7", "source_sha256": SOURCE_SHA256,
        "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "negative_control": args.negative_control,
        "slots_instrumented": args.slots,
        "publication_negative_control": args.weaken_publication,
        "acquire_negative_control": args.weaken_acquire,
        "single_steal_pin_fence_modeled": args.slots,
        "minimum_capacity": args.slot_capacity,
        "instrumented": "deque indices and fences, including owner fetch_add and thief CAS",
        "native_uninstrumented": ("" if args.slots else "slot memory, ") + "Arc refcounts, epoch reclamation and backoff",
        "scope": "bounded FIFO models; preemption bound 2; see test source for publication/reuse scenarios",
        "closes_task_9_2": False,
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
