"""Run checks of analyze_tuning.py (check_run_info, check_communicator): a fake tuning run that matches every assumption
(job-info.txt, layout, empty outputs; no data), and one mismatch per test that has to abort the analysis."""

import shutil

import pandas as pd
import pytest

import analyze_tuning as tuning
from conftest import run_script, write_job_info

SIZES = [8, 2**16, 2**19, 2**20, 2**21, 2**26, 2**30]  # all sizes of model_violins.pdf
NCCLBENCH = ("broadcast_bench all_reduce_bench, args: --sizes " + " ".join(map(str, SIZES)) +
             " --data-type float --blocking --csv --warmups 5 --iterations 50")
MUSICA_GPUS = {"n1": ["NVIDIA H100"] * 4, "n2": ["NVIDIA H100"] * 4}


def header(experiment, changes):
    """job-info.txt header of a MUSICA run that matches analyze_tuning.py's assumptions, with `changes`"""
    fields = {"experiment": experiment, "job": "1000 (array job -, task -)", "date": "2026-10-30T23:45:03+01:00",
              "nodes": "n1 n2", "tasks": "8 (per node: 4(x2)), CPUs per task: -", "profilers": "ncclsee",
              "ncclbench": NCCLBENCH, "NCCL logging": "true" if experiment == "logging" else "false",
              "NCCL": "2.27.7 (/path/to/module)"}
    if experiment == "tuning":
        fields["rounds"] = str(tuning.EXPECTED_ROUNDS)
    fields.update(changes)
    return {k: v for k, v in fields.items() if v is not None}  # None: leave the line out


def make_tuning_run(results, header_changes=None, variants=None, gpus=None, logging_changes=None):
    """
    results/tuning/1000 with every round and variant (empty profiler outputs and ncclbench outputs), and the
    logging run at results/logging/1889653/01 (where LOGGING_RUN is relative to ANALYSIS_RESULTS)
    """
    run = results / "tuning" / "1000"
    variants = dict(tuning.EXPECTED_VARIANTS) if variants is None else variants
    write_job_info(run / "job-info.txt", header("tuning", header_changes or {}), variants=variants,
                   gpus=MUSICA_GPUS if gpus is None else gpus)
    for r in range(1, tuning.EXPECTED_ROUNDS + 1):
        for v in tuning.EXPECTED_VARIANTS:
            d = run / f"{r:02d}" / v
            d.mkdir(parents=True)
            for name in ["profiler-out.tar.gz", *(f"{test}_ncclsee.out" for test in tuning.TESTS)]:
                (d / name).touch()
    logging_run = results / "logging" / "1889653" / "01"
    write_job_info(logging_run / "job-info.txt", header("logging", logging_changes or {}), gpus=MUSICA_GPUS)
    return run, logging_run


def problems(run, logging_run):
    """the mismatches check_run_info reports (its SystemExit message), "" if it passes"""
    try:
        tuning.check_run_info(run, logging_run=logging_run)
    except SystemExit as e:
        return str(e)
    return ""


def test_matching_run_passes(results):
    assert problems(*make_tuning_run(results)) == ""


@pytest.mark.parametrize("header_changes, expected", [
    ({"machine": "spark"}, "tuning run: machine spark, expected musica"),
    ({"job": None}, "tuning run: machine unknown, expected musica"),  # no job line: not from MUSICA's Slurm
    ({"NCCL": "2.32.3 (built)"}, "tuning run: NCCL 2.32.3, expected 2.27.7"),
    ({"NCCL": None}, "tuning run: NCCL (not in job-info.txt), expected 2.27.7"),
    ({"tasks": "6 (per node: 3(x2)), CPUs per task: -"}, "tuning run: 6 tasks, expected 8"),
    ({"experiment": "full"}, "tuning run: experiment full, expected tuning"),
    ({"rounds": "2"}, "tuning run: rounds 2, expected 3"),
    ({"rounds": None}, "tuning run: rounds (not in job-info.txt), expected 3"),
    ({"ncclbench": NCCLBENCH.replace(" --blocking", "")}, "tuning run: ncclbench without --blocking"),
    ({"ncclbench": NCCLBENCH.replace("broadcast_bench ", "")}, "tuning run: ncclbench test broadcast_bench missing"),
    ({"ncclbench": NCCLBENCH.replace(" 2097152", "")}, "tuning run: sizes [2097152] of model_violins.pdf (all_reduce)"),
])
def test_job_info_mismatch(results, header_changes, expected):
    assert expected in problems(*make_tuning_run(results, header_changes=header_changes))


@pytest.mark.parametrize("gpus, expected", [
    ({"n1": ["NVIDIA H100"] * 4}, "tuning run: 1 node(s) (n1), expected 2"),
    ({**MUSICA_GPUS, "n3": ["NVIDIA H100"] * 4}, "tuning run: 3 node(s) (n1 n2 n3), expected 2"),
    ({"n1": ["NVIDIA H100"] * 3, "n2": ["NVIDIA H100"] * 4}, "tuning run: n1 has 3 GPU(s)"),
    ({"n1": ["NVIDIA GB10"] * 4, "n2": ["NVIDIA H100"] * 4}, "tuning run: n1 has 4 GPU(s) ['NVIDIA GB10'], expected "
                                                             "4 x NVIDIA H100"),
])
def test_gpu_mismatch(results, gpus, expected):
    assert expected in problems(*make_tuning_run(results, gpus=gpus))


def test_variant_missing(results):
    variants = {k: v for k, v in tuning.EXPECTED_VARIANTS.items() if k != "ring_ll"}
    assert "tuning run: variants ['ring_ll'] missing" in problems(*make_tuning_run(results, variants=variants))


def test_variant_unknown(results):
    variants = {**tuning.EXPECTED_VARIANTS, "ring_ll_4ch": "NCCL_ALGO=ring NCCL_MAX_NCHANNELS=4"}
    assert "tuning run: unknown variants ['ring_ll_4ch']" in problems(*make_tuning_run(results, variants=variants))


def test_variant_environment_differs(results):
    variants = {**tuning.EXPECTED_VARIANTS, "tree_ll": "NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL128"}
    assert "tuning run: variant tree_ll sets 'NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL128'" in \
        problems(*make_tuning_run(results, variants=variants))


def test_variant_without_profiler_output(results):
    run, logging_run = make_tuning_run(results)
    (run / "02" / "tree_ll" / "profiler-out.tar.gz").unlink()
    assert "tuning run: (round, variant) without profiler output: [('02', 'tree_ll')]" in problems(run, logging_run)


def test_unexpected_round(results):
    run, logging_run = make_tuning_run(results)
    shutil.copytree(run / "03", run / "04")
    assert "tuning run: unexpected (round, variant): [('04', 'default')" in problems(run, logging_run)


def test_benchmark_output_missing(results):
    run, logging_run = make_tuning_run(results)
    (run / "01" / "default" / "broadcast_bench_ncclsee.out").unlink()
    assert "tuning run: 01/default/broadcast_bench_ncclsee.out missing" in problems(run, logging_run)


def test_no_job_info(results):
    run, logging_run = make_tuning_run(results)
    (run / "job-info.txt").unlink()
    assert "no job-info.txt in" in problems(run, logging_run)


def test_logging_run_missing(results):
    run, logging_run = make_tuning_run(results)
    shutil.rmtree(logging_run)
    assert f"logging run {logging_run} not found" in problems(run, logging_run)


@pytest.mark.parametrize("logging_changes, expected", [
    ({"NCCL logging": "false"}, "logging run 1889653: without NCCL logging (no TUNING table)"),
    ({"machine": "spark"}, "logging run 1889653: machine spark, expected musica"),
    ({"NCCL": "2.32.3 (built)"}, "logging run 1889653: NCCL 2.32.3, expected 2.27.7"),
])
def test_logging_run_mismatch(results, logging_changes, expected):
    assert expected in problems(*make_tuning_run(results, logging_changes=logging_changes))


def test_all_mismatches_reported(results):
    text = problems(*make_tuning_run(results, header_changes={"machine": "spark", "rounds": "2"},
                                     logging_changes={"NCCL logging": "false"}))
    assert "machine spark" in text and "rounds 2" in text and "without NCCL logging" in text


def test_script_aborts_before_writing(results):
    # the whole script: aborts with the list, writes nothing, and finds the logging run relative to its results/
    run, _ = make_tuning_run(results, header_changes={"rounds": "2"})
    before = sorted(run.rglob("*"))
    proc = run_script("analyze_tuning.py", run, results=results)
    assert proc.returncode != 0
    assert "does not match what analyze_tuning.py assumes" in proc.stderr
    assert "rounds 2, expected 3" in proc.stderr
    assert "logging run" not in proc.stderr
    assert sorted(run.rglob("*")) == before


# check_communicator: what ncclsee recorded and the TUNING table of the logging run

def choices(ranks=range(tuning.COMM_RANKS), default_channels=tuning.COMM_CHANNELS):
    rows = [{"rank": r, "variant": "default", "channels": default_channels} for r in ranks]
    rows += [{"rank": r, "variant": "ring_ll", "channels": 2} for r in ranks]
    return pd.DataFrame(rows)


def table(drop=()):
    return {(op, a, p): (1.0, 1.0) for op, combos in tuning.COMBINATIONS.items() for a, p in combos
            if (op, a, p) not in drop}


def communicator_problems(choices, table):
    try:
        tuning.check_communicator("run", choices, table)
    except SystemExit as e:
        return str(e)
    return ""


def test_communicator_matches():
    assert communicator_problems(choices(), table()) == ""


def test_communicator_rank_missing():
    assert "ncclsee recorded ranks [0, 1, 2, 3, 4, 5, 6], expected 0 - 7" in \
        communicator_problems(choices(ranks=range(7)), table())


def test_communicator_channels():
    assert "at most 4 channels in the default variant, expected 8" in \
        communicator_problems(choices(default_channels=4), table())


def test_communicator_combinations():
    assert "TUNING table of the logging run offers" in \
        communicator_problems(choices(), table(drop=[("all_reduce", "tree", "ll128")]))
