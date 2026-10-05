#!/usr/bin/env python3
# Copyright 2026 Mocktail Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Summarise the MOCKTAIL_FRAME_STATS lines of a Mocktail session log.

Run a game with MOCKTAIL_FRAME_STATS=1, note the session log's line count
(wc -l) before and after each part of the run, then:

    frame_stats_summary.py <log> --phase idle=410:530 --phase move=530:1700

A phase A:B covers the lines after the first A up to line B, so the two
numbers are line counts taken while the game ran; either may be left out.
Without --phase or --lines the whole log is one phase, "all".

Per phase it reports:

* frame pacing from the "[perf] frame_stats" lines: windows, average and
  worst-window fps, the longest frame, frames over 33/50/100 ms, the p99 of
  the windows' p99s, and long-frame time that no ETC2 stall line explains;
* ETC2 CPU stalls ("[vulkan] ETC2 decode stall"): count, sum, max, p99, how
  many reach 33/50/100 ms, and the prepare/copy/decode/write split;
* GPU decode submits ("[vulkan] ETC2 GPU submit"): count, CPU recording and
  GPU time;
* the "[perf] etc2_window" lines, summed field by field;
* the last "[vulkan] ETC2 decode totals" line.

Older logs carry only some of these lines; the rest are reported as zero.
Only numbers and Mocktail's own lines are printed, never other log text.
--json prints the same as one JSON object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Iterable

FRAME_STATS = re.compile(
    r"frames=(\d+) fps=([\d.]+) median=([\d.]+)ms p99=([\d.]+)ms "
    r"max=([\d.]+)ms over33=(\d+) over50=(\d+) over100=(\d+)")
STALL = re.compile(
    r"ETC2 decode stall (\d+) ms: (\d+) uploads, ([\d.]+) MiB compressed, "
    r"([\d.]+) MiB written \(prepare (\d+), copy (\d+), decode (\d+), "
    r"write (\d+) ms\)(?: \[threads (\d+), path (\w+), src ([^\]]*) MiB\])?")
STALL_LOOSE = re.compile(r"decode stall (\d+) ms")
GPU_SUBMIT = re.compile(
    r"ETC2 GPU submit: (\d+) uploads, ([\d.]+) MiB compressed, ([\d.]+) MiB "
    r"written, record ([\d.]+) ms, gpu ([\d.]+) ms")
WINDOW = re.compile(r"\[perf\] etc2_window (.*)$")
TOTALS = re.compile(r"\[vulkan\] ETC2 decode totals( so far)?: (images=.*)$")
TOTALS_PREFIX = re.compile(
    r"images=(\d+) \(upscaled (\d+)\) uploads=(\d+) in (\d+) submits "
    r"compressed=([\d.]+) MiB host=([\d.]+) MiB(?: \(([\d.]+)x\))? "
    r"decode=(\d+) ms longest=(\d+) ms")
LONG_FRAME_MS = 100.0


def percentile(values: list[float], fraction: float) -> float:
    """The nearest-rank percentile, as frame_stats computes its p99."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def parse_range(text: str) -> tuple[int, int | None]:
    start, separator, end = text.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected A:B, got {text!r}")
    try:
        return (int(start) if start else 0, int(end) if end else None)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected line counts in {text!r}") from error


def parse_phase(text: str) -> tuple[str, tuple[int, int | None]]:
    name, separator, span = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError(f"expected name=A:B, got {text!r}")
    return name, parse_range(span)


def number(text: str) -> float:
    value = float(text)
    return int(value) if value.is_integer() and "." not in text else value


def add_window_field(sums: dict, key: str, value: str) -> None:
    """Adds one key=value of an etc2_window line to the phase's sums."""
    if key == "new_threads":
        threads = sums.setdefault("new_threads", [])
        for name in re.findall(r'"([^"]*)"', value):
            if name not in threads:
                threads.append(name)
        return
    if key == "arena_mib":
        match = re.fullmatch(r"([\d.]+)\(\+([\d.]+)\)", value)
        if match:
            sums["arena_mib_peak"] = max(sums.get("arena_mib_peak", 0.0),
                                         float(match[1]))
            sums["arena_over_cap_mib_peak"] = max(
                sums.get("arena_over_cap_mib_peak", 0.0), float(match[2]))
        return
    if key == "images":
        match = re.fullmatch(r"\+(\d+)/-(\d+)", value)
        if match:
            sums["images_created"] = sums.get("images_created", 0) + int(match[1])
            sums["images_destroyed"] = (sums.get("images_destroyed", 0) +
                                        int(match[2]))
        return
    if key in ("fam", "begin", "src"):
        # "0:28/2.0,2:4/0.5", "ot:40,sim:0,other:9", "t3:2.5,unknown:1.0"
        group = sums.setdefault(key, {})
        if value == "-":
            return
        for item in value.split(","):
            name, _, counts = item.partition(":")
            parts = counts.split("/")
            if key == "fam":
                entry = group.setdefault(name, {"uploads": 0, "mib": 0.0})
                entry["uploads"] += int(parts[0])
                entry["mib"] = round(entry["mib"] + float(parts[1]), 3)
            else:
                group[name] = round(group.get(name, 0) + number(parts[0]), 3)
        return
    if "/" in value:
        # "sec=1/1", "submit=20/21", "nocompute=3/0.5", "copies=0/2/1/0/0"
        parts = value.split("/")
        try:
            values = [number(part) for part in parts]
        except ValueError:
            return
        current = sums.setdefault(key, [0] * len(values))
        if len(current) == len(values):
            sums[key] = [round(a + b, 3) for a, b in zip(current, values)]
        return
    try:
        sums[key] = round(sums.get(key, 0) + number(value), 3)
    except ValueError:
        pass


def parse_window(text: str, sums: dict) -> None:
    # "cpu_ms=26.5 (max 25.3)" carries the window's longest submit.
    longest = re.search(r"cpu_ms=[\d.]+ \(max ([\d.]+)\)", text)
    if longest:
        sums["cpu_ms_max"] = max(sums.get("cpu_ms_max", 0.0),
                                 float(longest[1]))
        text = text.replace(longest[0], longest[0].split(" ")[0])
    for key, value in re.findall(r'(\w+)=("[^"]*"(?:,"[^"]*")*|\S+)', text):
        add_window_field(sums, key, value)


def summarise(lines: Iterable[str]) -> dict:
    frames = []          # (frames, fps, median, p99, max, o33, o50, o100)
    stalls = []          # milliseconds
    split = {"prepare": 0, "copy": 0, "decode": 0, "write": 0}
    stall_uploads = 0
    stall_compressed = 0.0
    stall_written = 0.0
    paths: dict[str, int] = {}
    stall_sources: dict[str, float] = {}
    gpu = {"count": 0, "uploads": 0, "compressed_mib": 0.0,
           "written_mib": 0.0, "record_ms": 0.0, "gpu_ms": 0.0}
    window_sums: dict = {}
    windows = 0
    totals = None
    stalls_since_window = 0
    unexplained = 0.0
    for line in lines:
        stall = STALL.search(line)
        if stall or STALL_LOOSE.search(line):
            if stall:
                milliseconds = int(stall[1])
                stall_uploads += int(stall[2])
                stall_compressed += float(stall[3])
                stall_written += float(stall[4])
                for index, name in enumerate(split, start=5):
                    split[name] += int(stall[index])
                if stall[10]:
                    paths[stall[10]] = paths.get(stall[10], 0) + 1
                    for item in re.findall(r"(t\d+|unknown) ([\d.]+)",
                                           stall[11] or ""):
                        stall_sources[item[0]] = round(
                            stall_sources.get(item[0], 0.0) + float(item[1]),
                            3)
            else:
                milliseconds = int(STALL_LOOSE.search(line)[1])
            stalls.append(milliseconds)
            stalls_since_window += milliseconds
            continue
        frame = FRAME_STATS.search(line)
        if frame:
            values = (int(frame[1]), float(frame[2]), float(frame[3]),
                      float(frame[4]), float(frame[5]), int(frame[6]),
                      int(frame[7]), int(frame[8]))
            frames.append(values)
            if values[4] >= LONG_FRAME_MS:
                unexplained += max(0.0, values[4] - stalls_since_window)
            stalls_since_window = 0
            continue
        submit = GPU_SUBMIT.search(line)
        if submit:
            gpu["count"] += 1
            gpu["uploads"] += int(submit[1])
            gpu["compressed_mib"] += float(submit[2])
            gpu["written_mib"] += float(submit[3])
            gpu["record_ms"] += float(submit[4])
            gpu["gpu_ms"] += float(submit[5])
            continue
        window = WINDOW.search(line)
        if window:
            windows += 1
            parse_window(window[1], window_sums)
            continue
        total = TOTALS.search(line)
        if total:
            totals = total[2].rstrip()

    frame_count = sum(value[0] for value in frames)
    seconds = sum(value[0] / value[1] for value in frames if value[1] > 0)
    result = {
        "frames": {
            "windows": len(frames),
            "frames": frame_count,
            "seconds": round(seconds, 3),
            "avg_fps": round(frame_count / seconds, 1) if seconds else 0.0,
            "worst_window_fps": min((value[1] for value in frames),
                                    default=0.0),
            "max_frame_ms": max((value[4] for value in frames), default=0.0),
            "p99_of_window_p99_ms": percentile([value[3] for value in frames],
                                               0.99),
            "over33": sum(value[5] for value in frames),
            "over50": sum(value[6] for value in frames),
            "over100": sum(value[7] for value in frames),
            "unexplained_long_frame_ms": round(unexplained, 1),
        },
        "stalls": {
            "count": len(stalls),
            "sum_ms": sum(stalls),
            "max_ms": max(stalls, default=0),
            "p99_ms": percentile(stalls, 0.99),
            "at_least_33ms": sum(1 for value in stalls if value >= 33),
            "at_least_50ms": sum(1 for value in stalls if value >= 50),
            "at_least_100ms": sum(1 for value in stalls if value >= 100),
            "uploads": stall_uploads,
            "compressed_mib": round(stall_compressed, 1),
            "written_mib": round(stall_written, 1),
            "split_ms": split,
            "paths": paths,
            "source_mib": stall_sources,
        },
        "gpu_submits": {key: (round(value, 2) if isinstance(value, float)
                              else value) for key, value in gpu.items()},
        "etc2_window": dict(window_sums, windows=windows),
        "totals": None,
    }
    if totals is not None:
        prefix = TOTALS_PREFIX.match(totals)
        result["totals"] = {"line": totals}
        if prefix:
            result["totals"].update({
                "images": int(prefix[1]),
                "upscaled_images": int(prefix[2]),
                "uploads": int(prefix[3]),
                "submits": int(prefix[4]),
                "compressed_mib": float(prefix[5]),
                "host_mib": float(prefix[6]),
                "decode_ms": int(prefix[8]),
                "longest_ms": int(prefix[9]),
            })
    return result


def format_phase(name: str, span: tuple[int, int | None], result: dict) -> str:
    frames = result["frames"]
    stalls = result["stalls"]
    gpu = result["gpu_submits"]
    window = result["etc2_window"]
    end = "end" if span[1] is None else str(span[1])
    out = [f"== {name} (lines {span[0] + 1}-{end})"]
    # The summ.py line the live protocol compared runs with.
    out.append(
        f"{name}: stalls>=20ms n={stalls['count']} total={stalls['sum_ms']} ms "
        f"max={stalls['max_ms']} ms | windows={frames['windows']} avg "
        f"fps={frames['avg_fps']:.1f} worst-window "
        f"fps={frames['worst_window_fps']:.1f} max "
        f"frame={frames['max_frame_ms']:.0f} ms over50={frames['over50']} "
        f"over100={frames['over100']}")
    out.append(
        f"  frames: over33={frames['over33']} p99 of window "
        f"p99s={frames['p99_of_window_p99_ms']:.1f} ms unexplained long "
        f"frames={frames['unexplained_long_frame_ms']:.0f} ms over "
        f"{frames['seconds']:.1f} s")
    split = stalls["split_ms"]
    out.append(
        f"  stalls: p99={stalls['p99_ms']} ms >=33={stalls['at_least_33ms']} "
        f">=50={stalls['at_least_50ms']} >=100={stalls['at_least_100ms']} "
        f"uploads={stalls['uploads']} compressed={stalls['compressed_mib']:.1f}"
        f" MiB written={stalls['written_mib']:.1f} MiB (prepare "
        f"{split['prepare']}, copy {split['copy']}, decode {split['decode']}, "
        f"write {split['write']} ms)")
    out.append(
        f"  gpu submits: n={gpu['count']} uploads={gpu['uploads']} "
        f"record={gpu['record_ms']:.1f} ms gpu={gpu['gpu_ms']:.1f} ms")
    if window["windows"]:
        fields = " ".join(f"{key}={json.dumps(value, separators=(',', ':'))}"
                          for key, value in window.items() if key != "windows")
        out.append(f"  etc2_window: windows={window['windows']} {fields}")
    if result["totals"] is not None:
        out.append(f"  totals: {result['totals']['line']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="a Mocktail session log")
    parser.add_argument("--lines", type=parse_range, metavar="A:B",
                        help="summarise only lines A+1..B")
    parser.add_argument("--phase", type=parse_phase, action="append",
                        default=[], metavar="NAME=A:B",
                        help="a named range; may be repeated")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON object")
    arguments = parser.parse_args(argv)
    try:
        lines = arguments.log.read_text(errors="replace").splitlines()
    except OSError as error:
        print(f"frame_stats_summary: {error}", file=sys.stderr)
        return 1
    phases = list(arguments.phase)
    if arguments.lines is not None:
        phases.insert(0, ("lines", arguments.lines))
    if not phases:
        phases = [("all", (0, None))]
    results = []
    for name, (start, end) in phases:
        results.append((name, (start, end),
                        summarise(lines[start:end])))
    if arguments.json:
        print(json.dumps({
            "log": str(arguments.log),
            "phases": [{"name": name, "lines": [start, end], **result}
                       for name, (start, end), result in results],
        }, indent=2))
    else:
        print("\n".join(format_phase(name, span, result)
                        for name, span, result in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
