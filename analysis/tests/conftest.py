"""Helpers of the analysis tests: running a script on another results folder, copies of runs, fake runs.

The scripts never run on results/ itself: they get a results folder of their own (ANALYSIS_RESULTS, see common.py),
either with fake runs (job-info.txt and empty files, for the checks) or with links to the inputs of real runs (the
regression tests), so the outputs in results/ stay untouched.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parent.parent
REPO = ANALYSIS.parent
RESULTS = REPO / "results"
REFERENCE = Path(__file__).resolve().parent / "reference"

sys.path.insert(0, str(ANALYSIS))  # the tests import the scripts as modules (analyze_tuning, common, ...)

# inputs of a run as written by experiment.sh; everything else in a run folder is written by the analysis
INPUT_NAMES = {"job-info.txt", "profiler-out.tar.gz", "variant-env.txt"}


def pytest_addoption(parser):
    parser.addoption("--update-reference", action="store_true",
                     help="write the outputs of the regression tests to analysis/tests/reference/ instead of "
                          "comparing them (after a deliberate change of the outputs; check the diff)")


def run_script(script, *args, results):
    """runs analysis/<script> with ANALYSIS_RESULTS=results; deterministic PDFs (no creation date)"""
    env = {**os.environ, "ANALYSIS_RESULTS": str(results), "SOURCE_DATE_EPOCH": "0"}
    return subprocess.run([sys.executable, str(ANALYSIS / script), *map(str, args)], env=env, cwd=REPO,
                          capture_output=True, text=True)


def link_run(run, results):
    """
    results/<experiment>/<job> -> <results>/<experiment>/<job> with symlinks to its inputs (and to the extracted
    profiler outputs, which the analysis only reads once they exist); returns the copy
    """
    run = Path(run)
    target = Path(results) / run.relative_to(RESULTS)
    for path in sorted(run.rglob("*")):
        if "profiler-out-extracted" in path.relative_to(run).parts[:-1]:
            continue
        if path.name == "profiler-out-extracted" or (path.is_file() and
                                                     (path.name in INPUT_NAMES or path.suffix == ".out")):
            dest = target / path.relative_to(run)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(path)
    return target


def outputs(run):
    """
    files the analysis wrote into a copy made by link_run (not the symlinked inputs, not profiler outputs it
    extracted itself), as relative paths
    """
    return sorted(p.relative_to(run) for p in Path(run).rglob("*")
                  if p.is_file() and not p.is_symlink() and "profiler-out-extracted" not in p.relative_to(run).parts)


def write_job_info(path, header, variants=None, gpus=None, ranks=None, env=None):
    """
    a job-info.txt as experiment.sh writes it, with only the parts the analysis reads: the "key: value" header,
    the variant list (tuning), "GPUs per node" ({host: [GPU names]}), "GPUs visible per rank" ({host: number of
    ranks}) and the job environment ({variable: value})
    """
    lines = [f"{k}: {v}" for k, v in header.items()]
    if variants is not None:
        lines += ["variants:", *(f"  {name}: {env}" for name, env in variants.items())]
    if gpus is not None:
        lines += ["", "GPUs per node:"]
        lines += [f"{host}: " + " ".join(f"GPU {i}: {name} (UUID: GPU-{host}-{i})" for i, name in enumerate(names))
                  for host, names in gpus.items()]
    if ranks is not None:
        lines += ["", "GPUs visible per rank (CUDA_VISIBLE_DEVICES, nvidia-smi):"]
        lines += [f"{host} local rank {r}: 0, 1 GPUs" for host, n in ranks.items() for r in range(n)]
    lines += ["", "environment:", "HOME=/nonexistent", *(f"{k}={v}" for k, v in (env or {}).items())]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def results(tmp_path):
    """an empty results folder of the test's own"""
    path = tmp_path / "results"
    path.mkdir()
    return path
