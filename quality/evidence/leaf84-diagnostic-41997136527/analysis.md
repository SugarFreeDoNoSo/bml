# Leaf84 diagnostic: memory growth before runner shutdown

Run [41997136527](https://github.com/SugarFreeDoNoSo/bml/actions/runs/41997136527) failed in leaf84 after prepare succeeded. The envelope was skipped as required by diagnostic mode. There is no leaf artifact, complete raw report or seal. This evidence does not approve mutation quality.

Controller `cca182543987dfc8cf39e3966ca527c28011166e`, source `9a6bc9680d5915576ca92528f60956efe9796cef`, repository and attempt1 match the API. The only artifact is plan `13400523795`; ZIP SHA256 `2a17d2cd83d81540208b940d9db60da5312635f2ce2de8a54453a0742f032f6f` matches its immutable API digest. Frozen controller tools validate the plan, all256 independent assignments, diagnostic-only84 schedule, original pinned manifest and real successful workspace Build/Test preflight. Plan SHA256 is `e873b1558df2044653069863f0f3556adb4c343885268dae77b8ea65c7db2c9c`; the original manifest file remains `719be79f7dfba0746291522fa312b395a9caf857abf16e71f61af70b8a5e7c63`.

The17 untruncated stdout samples span `2026-11-05T08:51:13.994553Z` to `2026-11-05T08:59:21.368800Z`. At `2026-11-05T08:58:44.177422Z`, cgroup memory.current was 2992349184 bytes and subtree RSS sum was 296644 kB. The final sample records memory.current 16025559040 bytes and peak 16025993216 bytes. PID39844 has RSS15,447,988 kB (about14.73GiB). At least 89656631296 bytes of disk remained free in every sample.

All sampled cgroup `oom` and `oom_kill` counters are0; `memory.max` is `max`. At `2026-11-05T08:59:44.6106395Z` the log reports exit143, followed by an explicit runner shutdown message. **Memory growth is proved; the cause of shutdown remains unknown.** These samples cannot prove or exclude a later or host-level memory event.

The last `active_mutant` is null. The newest log filename is `crates__domain__src__symbolic__synthesis.rs_line_187_col_34.log`; it is a filesystem observation, not active-candidate identification. Neither this path nor the preceding MISSED warning identifies the process responsible. Final partial counters are11 mutants (9caught/1missed/1unviable),12 outcome rows including baseline, and no raw end time. They remain diagnostic and unattested.

`analysis.json` retains all numeric snapshots, API chronology, annotations, uncertainty and raw capture hashes. The exact downloaded `leaf.log` has SHA256 `d864bdfcc7bc64b2f60927c7cd6252831a0d8e754ca2d2e3f609e5f3148142ab`. Per-process reads are non-atomic; RSS sums may count shared pages multiple times. No Cargo, retry, dispatch, source edits or classification changes were performed.
