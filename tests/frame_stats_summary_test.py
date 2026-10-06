#!/usr/bin/env python3
# Copyright 2026 Mocktail Project Authors
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts/frame_stats_summary.py"
SPEC = importlib.util.spec_from_file_location("frame_stats_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)

LOG = """\
[launcher] starting session
  [vulkan] ETC2/EAC emulation enabled: compressed uploads decode to RGBA8/R16 host images
  [vulkan] ETC2 queues: family 0 (G|C|T|S) x1 of 16, family 2 (C|T) x1 of 8
  [perf] frame_stats frames=200 fps=100.0 median=9.9ms p99=12.0ms max=15.0ms over33=0 over50=0 over100=0
  [vulkan] ETC2 decode stall 150 ms: 1264 uploads, 0.8 MiB compressed, 48.0 MiB written (prepare 0, copy 4, decode 4, write 141 ms)
  [vulkan] ETC2 decode stall 40 ms: 55 uploads, 3.3 MiB compressed, 26.7 MiB written (prepare 0, copy 20, decode 11, write 6 ms) [threads 8, path fast, src t3 3.0 t4 0.3 MiB]
  [vulkan] ETC2 decode totals so far: images=353 (upscaled 344) uploads=2171 in 4 submits compressed=4.6 MiB host=71.1 MiB (15.6x) decode=206 ms longest=145 ms
  [perf] frame_stats frames=50 fps=25.0 median=9.0ms p99=300.0ms max=400.0ms over33=3 over50=2 over100=2
  [perf] etc2_window submits=2 cpu_ms=190.5 (max 150.2) gpu_ms=0.0 record_calls=18 record_ms=9.1 staging_allocs=18 release_ms=7.0 uploads=1319 gpu=0 fallback=1319 rescued=0 arena_mib=64.0(+8.0) images=+5/-1 copyimg=0 aliasviews=0 regions=1319 fam=0:1300/3.9,2:19/0.2 nocompute=0/0.0 unknownfam=0 begin=ot:40,sim:0,other:9 simcbs=0 sec=0/0 simsec=0 submit=60/61 submit2=0/0 upload_batches=2 multibatch=0 src=t3:3.8,t4:0.3 nonhost=0 blits=0 rawcopies=0 aliasbinds=0 formatlists=0 afterdispatch=0 copies=0/1/0/0/0 new_threads="RenderJob"
cookie: .ROBLOSECURITY=do-not-print-this
  [vulkan] ETC2 GPU submit: 2000 uploads, 48.0 MiB compressed, 384.0 MiB written, record 0.4 ms, gpu 7.1 ms (copy-in 4.1, decode 3.0; lag 1)
  [perf] frame_stats frames=240 fps=120.0 median=8.1ms p99=10.0ms max=12.0ms over33=0 over50=0 over100=0
  [perf] etc2_window submits=1 cpu_ms=0.0 (max 0.0) gpu_ms=7.1 record_calls=3 record_ms=0.5 staging_allocs=0 release_ms=0.2 uploads=2000 gpu=2000 fallback=0 rescued=0 arena_mib=96.0(+0.0) images=+1/-0 copyimg=1 aliasviews=0 regions=2000 fam=0:2000/48.0 nocompute=0/0.0 unknownfam=0 begin=ot:12,sim:1,other:0 simcbs=0 sec=1/1 simsec=0 submit=58/58 submit2=2/3 upload_batches=1 multibatch=0 src=- nonhost=0 blits=0 rawcopies=1 aliasbinds=0 formatlists=0 afterdispatch=0 copies=1/0/2/0/0 new_threads="RenderJob","Worker 2"
  [vulkan] ETC2 decode totals: images=354 (upscaled 344) uploads=4171 in 6 submits compressed=52.6 MiB host=455.1 MiB (8.7x) decode=396 ms longest=150 ms | gpu uploads=2000 submits=1
"""


def run(*arguments: str) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = summary.main(list(arguments))
    assert status == 0
    return output.getvalue()


class FrameStatsSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log = Path(directory.name) / "session.log"
        self.log.write_text(LOG)

    def test_matches_the_summ_line_the_live_runs_used(self) -> None:
        text = run(str(self.log))
        self.assertIn(
            "all: stalls>=20ms n=2 total=190 ms max=150 ms | windows=3 avg "
            "fps=81.7 worst-window fps=25.0 max frame=400 ms over50=2 "
            "over100=2\n", text)
        self.assertNotIn("ROBLOSECURITY", text)
        self.assertNotIn("do-not-print-this", text)

    def test_reports_every_kind_of_line(self) -> None:
        phase = json.loads(run(str(self.log), "--json"))["phases"][0]
        self.assertEqual(phase["name"], "all")
        frames = phase["frames"]
        self.assertEqual(frames["windows"], 3)
        self.assertEqual(frames["frames"], 490)
        self.assertEqual(frames["over33"], 3)
        self.assertEqual(frames["p99_of_window_p99_ms"], 300.0)
        # 400 ms long frame, 190 ms of it named by the stall lines before it.
        self.assertEqual(frames["unexplained_long_frame_ms"], 210.0)

        stalls = phase["stalls"]
        self.assertEqual(stalls["count"], 2)
        self.assertEqual(stalls["at_least_50ms"], 1)
        self.assertEqual(stalls["at_least_100ms"], 1)
        self.assertEqual(stalls["uploads"], 1319)
        self.assertEqual(stalls["split_ms"],
                         {"prepare": 0, "copy": 24, "decode": 15,
                          "write": 147})
        self.assertEqual(stalls["paths"], {"fast": 1})
        self.assertEqual(stalls["source_mib"], {"t3": 3.0, "t4": 0.3})

        gpu = phase["gpu_submits"]
        self.assertEqual(gpu["count"], 1)
        self.assertEqual(gpu["uploads"], 2000)
        self.assertEqual(gpu["gpu_ms"], 7.1)

        window = phase["etc2_window"]
        self.assertEqual(window["windows"], 2)
        self.assertEqual(window["submits"], 3)
        self.assertEqual(window["cpu_ms"], 190.5)
        self.assertEqual(window["cpu_ms_max"], 150.2)
        self.assertEqual(window["uploads"], 3319)
        self.assertEqual(window["arena_mib_peak"], 96.0)
        self.assertEqual(window["arena_over_cap_mib_peak"], 8.0)
        self.assertEqual(window["images_created"], 6)
        self.assertEqual(window["images_destroyed"], 1)
        self.assertEqual(window["fam"]["0"], {"uploads": 3300, "mib": 51.9})
        self.assertEqual(window["fam"]["2"], {"uploads": 19, "mib": 0.2})
        self.assertEqual(window["begin"], {"ot": 52, "sim": 1, "other": 9})
        self.assertEqual(window["submit"], [118, 119])
        self.assertEqual(window["submit2"], [2, 3])
        self.assertEqual(window["sec"], [1, 1])
        self.assertEqual(window["copies"], [1, 1, 2, 0, 0])
        self.assertEqual(window["src"], {"t3": 3.8, "t4": 0.3})
        self.assertEqual(window["rawcopies"], 1)
        self.assertEqual(window["new_threads"], ["RenderJob", "Worker 2"])

        totals = phase["totals"]
        self.assertEqual(totals["images"], 354)
        self.assertEqual(totals["uploads"], 4171)
        self.assertEqual(totals["submits"], 6)
        self.assertEqual(totals["longest_ms"], 150)
        self.assertTrue(totals["line"].endswith("| gpu uploads=2000 submits=1"))

    def test_phases_cover_the_lines_after_a_count(self) -> None:
        result = json.loads(run(str(self.log), "--phase", "load=:7",
                                "--phase", "move=7:", "--json"))
        load, move = result["phases"]
        self.assertEqual(load["lines"], [0, 7])
        self.assertEqual(load["stalls"]["count"], 2)
        self.assertEqual(load["frames"]["windows"], 1)
        self.assertEqual(load["totals"]["images"], 353)
        self.assertEqual(move["lines"], [7, None])
        self.assertEqual(move["stalls"]["count"], 0)
        self.assertEqual(move["frames"]["windows"], 2)
        self.assertEqual(move["totals"]["images"], 354)

        lines = run(str(self.log), "--lines", "3:4")
        self.assertIn("== lines (lines 4-4)", lines)
        self.assertIn("lines: stalls>=20ms n=0 total=0 ms max=0 ms | "
                      "windows=1 avg fps=100.0", lines)

    def test_reads_the_stall_suffix_with_and_without_a_path(self) -> None:
        # plus.2 development builds named the CPU path; the release does
        # not, since only one is left.
        self.log.write_text(
            "  [vulkan] ETC2 decode stall 30 ms: 9 uploads, 1.0 MiB "
            "compressed, 8.0 MiB written (prepare 0, copy 10, decode 10, "
            "write 10 ms) [threads 4, src t4 1.0 MiB]\n"
            "  [vulkan] ETC2 decode stall 25 ms: 7 uploads, 0.5 MiB "
            "compressed, 4.0 MiB written (prepare 0, copy 5, decode 10, "
            "write 10 ms) [threads 8, path legacy, src t3 0.5 MiB]\n")
        stalls = json.loads(run(str(self.log), "--json"))["phases"][0][
            "stalls"]
        self.assertEqual(stalls["count"], 2)
        self.assertEqual(stalls["paths"], {"legacy": 1})
        self.assertEqual(stalls["source_mib"], {"t3": 0.5, "t4": 1.0})

    def test_old_logs_without_these_lines_summarise_to_zero(self) -> None:
        self.log.write_text("[launcher] nothing measured\n")
        phase = json.loads(run(str(self.log), "--json"))["phases"][0]
        self.assertEqual(phase["frames"]["windows"], 0)
        self.assertEqual(phase["frames"]["avg_fps"], 0.0)
        self.assertEqual(phase["stalls"]["count"], 0)
        self.assertIsNone(phase["totals"])
        self.assertIn("windows=0 avg fps=0.0", run(str(self.log)))


if __name__ == "__main__":
    unittest.main()
