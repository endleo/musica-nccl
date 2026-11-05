"""Run checks of analyze_tuning.py: a fake tuning run and logging run that fit together (job-info.txt, layout, empty
outputs; no data), one problem per test that has to abort the analysis, and other setups that have to be accepted
as long as both runs share them. Also the parts of the cost model that do not need data (communicator, model time)."""

import shutil

import pandas as pd
import pytest

import analyze_tuning as tuning
from conftest import run_script, write_job_info

SIZES = [8, 2**16, 2**19, 2**20, 2**21, 2**26, 2**30]  # all sizes of model_violins.pdf
NCCLBENCH = ("broadcast_bench all_reduce_bench, args: --sizes " + " ".join(map(str, SIZES)) +
             " --data-type float --blocking --csv --warmups 5 --iterations 50")
MUSICA_GPUS = {"n1": ["NVIDIA H100"] * 4, "n2": ["NVIDIA H100"] * 4}
ENV = {"NCCL_ALLGATHERV_ENABLE": "0"}
SAME = object()  # logging run: the same as the tuning run


def header(experiment, changes):
    """job-info.txt header of a MUSICA run (no machine line: from Slurm, so MUSICA), with `changes` (None: no line)"""
    fields = {"experiment": experiment, "job": "1000 (array job -, task -)", "date": "2026-10-30T23:45:03+01:00",
              "profilers": "ncclsee", "ncclbench": NCCLBENCH,
              "NCCL logging": "true" if experiment == "logging" else "false", "NCCL": "2.27.7 (/path/to/module)"}
    if experiment == "tuning":
        fields["rounds"] = "3"
    fields.update(changes)
    return {k: v for k, v in fields.items() if v is not None}


def ranks_of(gpus):
    return {host: len(names) for host, names in gpus.items()}


def make_logging_run(results, job="1889653", changes=None, gpus=MUSICA_GPUS, ranks=None, env=ENV):
    """results/logging/<job>/01 with job-info.txt and an empty profiler-out.tar.gz; returns the run folder"""
    run = results / "logging" / job
    write_job_info(run / "01" / "job-info.txt", header("logging", changes or {}), gpus=gpus,
                   ranks=ranks_of(gpus) if ranks is None else ranks, env=env)
    (run / "01" / "profiler-out.tar.gz").touch()
    return run


def make_tuning_run(results, header_changes=None, variants=None, gpus=MUSICA_GPUS, ranks=None, env=ENV, rounds=3,
                    logging_changes=None, logging_gpus=SAME, logging_ranks=SAME, logging_env=SAME):
    """
    results/tuning/1000 with `rounds` rounds of every variant (empty profiler outputs and ncclbench outputs), and the
    logging run results/logging/1889653 (by default with the same GPUs, ranks and environment); returns both folders
    """
    run = results / "tuning" / "1000"
    variants = dict(tuning.EXPECTED_VARIANTS) if variants is None else variants
    ranks = ranks_of(gpus) if ranks is None else ranks
    write_job_info(run / "job-info.txt", header("tuning", {"rounds": str(rounds), **(header_changes or {})}),
                   variants=variants, gpus=gpus, ranks=ranks, env=env)
    for r in range(1, rounds + 1):
        for v in tuning.EXPECTED_VARIANTS:
            d = run / f"{r:02d}" / v
            d.mkdir(parents=True)
            for name in ["profiler-out.tar.gz", *(f"{test}_ncclsee.out" for test in tuning.TESTS)]:
                (d / name).touch()
    logging_run = make_logging_run(results, changes=logging_changes,
                                   gpus=gpus if logging_gpus is SAME else logging_gpus,
                                   ranks=ranks if logging_ranks is SAME else logging_ranks,
                                   env=env if logging_env is SAME else logging_env)
    return run, logging_run


def problems(run, logging_run):
    """the problems check_run_info reports (its SystemExit message), "" if it passes"""
    try:
        tuning.check_run_info(run, logging_run)
    except SystemExit as e:
        return str(e)
    return ""


# ---------------------------------------------------------------------------------------------
# accepted

def test_matching_run_passes(results):
    assert problems(*make_tuning_run(results)) == ""


@pytest.mark.parametrize("gpus, header_changes", [
    ({f"n{i}": ["NVIDIA H100"] * 4 for i in range(8)}, {}),  # MUSICA, 8 nodes (tuning run 1911532)
    ({"n1": ["NVIDIA H100"] * 4}, {}),  # one node
    ({"n1": ["NVIDIA H100"] * 2, "n2": ["NVIDIA H100"] * 2}, {}),  # fewer GPUs per node
    ({"dgx1": ["NVIDIA GB10"], "dgx2": ["NVIDIA GB10"], "dgx3": ["NVIDIA GB10"]},  # another machine
     {"machine": "spark", "job": None}),
])
def test_other_setup_accepted(results, gpus, header_changes):
    # any setup, as long as the logging run has the same (and the NCCL version has a cost model)
    run, logging_run = make_tuning_run(results, header_changes=header_changes, gpus=gpus,
                                       logging_changes=header_changes)
    assert problems(run, logging_run) == ""


@pytest.mark.parametrize("rounds", [2, 5])
def test_other_round_count_accepted(results, rounds):
    assert problems(*make_tuning_run(results, rounds=rounds)) == ""


def test_other_nodes_accepted(results):
    # the same setup on other nodes (the hosts differ between allocations)
    other = {"n7": MUSICA_GPUS["n1"], "n8": MUSICA_GPUS["n2"]}
    assert problems(*make_tuning_run(results, logging_gpus=other)) == ""


def test_ignored_nccl_environment(results):
    # NCCL's logging and dump variables, and the install path, do not change the tuning
    env = {**ENV, "NCCL_DEBUG": "INFO", "NCCL_TOPO_DUMP_FILE": "/tmp/topo.xml", "NCCL_HOME": "/other"}
    assert problems(*make_tuning_run(results, logging_env=env)) == ""


def test_added_nccl_version_accepted(results, monkeypatch):
    # a new NCCL version is accepted once COST_MODELS has a model for it
    monkeypatch.setitem(tuning.COST_MODELS, "2.32.3", tuning.model_time_2_27_7)
    nccl = {"NCCL": "2.32.3 (built)"}
    assert problems(*make_tuning_run(results, header_changes=nccl, logging_changes=nccl)) == ""


# ---------------------------------------------------------------------------------------------
# problems of the tuning run itself

@pytest.mark.parametrize("header_changes, expected", [
    ({"experiment": "full"}, "tuning run: experiment full, expected tuning"),
    ({"rounds": "1"}, "tuning run: rounds 1, expected at least 2 (MIN_ROUNDS)"),
    ({"rounds": None}, "tuning run: rounds (not in job-info.txt), expected at least 2 (MIN_ROUNDS)"),
    ({"ncclbench": NCCLBENCH.replace(" --blocking", "")}, "tuning run: ncclbench without --blocking"),
    ({"ncclbench": NCCLBENCH.replace("broadcast_bench ", "")}, "tuning run: ncclbench test broadcast_bench missing"),
    ({"ncclbench": NCCLBENCH.replace(" 2097152", "")}, "tuning run: sizes [2097152] of model_violins.pdf (all_reduce)"),
])
def test_job_info_mismatch(results, header_changes, expected):
    assert expected in problems(*make_tuning_run(results, header_changes=header_changes))


@pytest.mark.parametrize("nccl, expected", [
    ("2.32.3 (built)", "tuning run: NCCL 2.32.3, the cost model is only implemented for NCCL 2.27.7 (COST_MODELS)"),
    (None, "tuning run: NCCL (not in job-info.txt), the cost model is only implemented for NCCL 2.27.7"),
])
def test_unknown_nccl_version(results, nccl, expected):
    # refused even if the logging run has the same version: there is no cost model for it
    run, logging_run = make_tuning_run(results, header_changes={"NCCL": nccl}, logging_changes={"NCCL": nccl})
    text = problems(run, logging_run)
    assert expected in text
    assert "logging run 1889653: NCCL" in text  # the logging run as well


def test_mixed_gpu_types(results):
    gpus = {"n1": ["NVIDIA H100"] * 4, "n2": ["NVIDIA A100"] * 4}
    assert "tuning run: GPUs of different types ['NVIDIA A100', 'NVIDIA H100']" in \
        problems(*make_tuning_run(results, gpus=gpus))


def test_no_ranks(results):
    assert "tuning run: no ranks in job-info.txt" in problems(*make_tuning_run(results, ranks={}, logging_ranks={}))


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


# ---------------------------------------------------------------------------------------------
# the logging run: its own problems, and differences to the tuning run

def test_logging_run_missing(results):
    run, logging_run = make_tuning_run(results)
    shutil.rmtree(logging_run)
    assert f"logging run {logging_run} not found" in problems(run, logging_run)


@pytest.mark.parametrize("logging_changes, expected", [
    ({"NCCL logging": "false"}, "logging run 1889653: without NCCL logging (no TUNING table)"),
    ({"experiment": "full"}, "logging run 1889653: experiment full, expected logging"),
    ({"machine": "spark"}, "logging run 1889653: machine spark, tuning run musica"),
    ({"NCCL": "2.32.3 (built)"}, "logging run 1889653: NCCL 2.32.3, tuning run 2.27.7"),
])
def test_logging_run_mismatch(results, logging_changes, expected):
    assert expected in problems(*make_tuning_run(results, logging_changes=logging_changes))


@pytest.mark.parametrize("header_changes, expected", [
    ({"machine": "spark"}, "logging run 1889653: machine musica, tuning run spark"),
    ({"job": None}, "logging run 1889653: machine musica, tuning run unknown"),  # no job line: not MUSICA's Slurm
])
def test_machine_mismatch(results, header_changes, expected):
    assert expected in problems(*make_tuning_run(results, header_changes=header_changes))


@pytest.mark.parametrize("logging_gpus, expected", [
    ({"n1": ["NVIDIA H100"] * 4}, "logging run 1889653: GPUs 1 node(s) with 4 x NVIDIA H100, tuning run 2 node(s) "
                                  "with 4 x NVIDIA H100"),
    ({**MUSICA_GPUS, "n3": ["NVIDIA H100"] * 4}, "logging run 1889653: GPUs 3 node(s) with 4 x NVIDIA H100"),
    ({"n1": ["NVIDIA H100"] * 3, "n2": ["NVIDIA H100"] * 4}, "logging run 1889653: GPUs 1 node(s) with 3 x NVIDIA "
                                                             "H100, 1 node(s) with 4 x NVIDIA H100"),
    ({"n1": ["NVIDIA GB10"] * 4, "n2": ["NVIDIA GB10"] * 4}, "logging run 1889653: GPUs 2 node(s) with 4 x NVIDIA "
                                                             "GB10, tuning run 2 node(s) with 4 x NVIDIA H100"),
])
def test_gpu_mismatch(results, logging_gpus, expected):
    assert expected in problems(*make_tuning_run(results, logging_gpus=logging_gpus, logging_ranks=ranks_of(MUSICA_GPUS)))


def test_rank_mismatch(results):
    assert "logging run 1889653: ranks 2 node(s) with 2 rank(s), tuning run 2 node(s) with 4 rank(s)" in \
        problems(*make_tuning_run(results, logging_ranks={"n1": 2, "n2": 2}))


def test_nccl_environment_mismatch(results):
    assert "logging run 1889653: NCCL environment {'NCCL_ALLGATHERV_ENABLE': '0', 'NCCL_IB_HCA': 'mlx5_0'}" in \
        problems(*make_tuning_run(results, logging_env={**ENV, "NCCL_IB_HCA": "mlx5_0"}))


def test_logging_run_without_profiler_output(results):
    run, logging_run = make_tuning_run(results)
    (logging_run / "01" / "profiler-out.tar.gz").unlink()
    assert "logging run 1889653: no profiler-out.tar.gz in 01/" in problems(run, logging_run)


def test_all_mismatches_reported(results):
    text = problems(*make_tuning_run(results, header_changes={"rounds": "1"}, logging_changes={"machine": "spark",
                                                                                              "NCCL logging": "false"}))
    assert "rounds 1" in text and "machine spark" in text and "without NCCL logging" in text


# ---------------------------------------------------------------------------------------------
# choice of the logging run, command line

def test_newest_matching_logging_run(results):
    # the newest logging run that fits, not the newest logging run
    run, _ = make_tuning_run(results, logging_changes={"date": "2026-10-29T12:00:00+01:00"})
    make_logging_run(results, "older", changes={"date": "2026-10-01T12:00:00+02:00"})
    make_logging_run(results, "newer-8-nodes", changes={"date": "2026-11-05T12:00:00+01:00"},
                     gpus={f"n{i}": ["NVIDIA H100"] * 4 for i in range(8)})
    assert tuning.matching_logging_run(run, results) == results / "logging" / "1889653"


def test_no_matching_logging_run(results):
    run, _ = make_tuning_run(results, logging_changes={"machine": "spark"})
    with pytest.raises(SystemExit) as e:
        tuning.matching_logging_run(run, results)
    assert "no logging run in" in str(e.value)
    assert "1889653: logging run 1889653: machine spark, tuning run musica" in str(e.value)


def test_script_chooses_matching_logging_run(results):
    run, _ = make_tuning_run(results)
    make_logging_run(results, "newer-8-nodes", changes={"date": "2026-11-05T12:00:00+01:00"},
                     gpus={f"n{i}": ["NVIDIA H100"] * 4 for i in range(8)})
    proc = run_script("analyze_tuning.py", run, results=results)
    # the fake run has no data: the script fails after the checks, having named the logging run
    assert f"logging run {results / 'logging' / '1889653'}" in proc.stdout


def test_script_takes_logging_option(results):
    run, _ = make_tuning_run(results)
    other = make_logging_run(results, "other", gpus={"n1": ["NVIDIA H100"] * 4})
    proc = run_script("analyze_tuning.py", run, "--logging", other, results=results)
    assert proc.returncode != 0
    assert "logging run other: GPUs 1 node(s) with 4 x NVIDIA H100" in proc.stderr


def test_script_refuses_other_experiment_as_logging_run(results):
    run, _ = make_tuning_run(results)
    proc = run_script("analyze_tuning.py", run, "--logging", run, results=results)
    assert proc.returncode != 0
    assert "a tuning run, --logging takes logging runs" in proc.stderr


def test_script_aborts_before_writing(results):
    # the whole script: aborts with the list and writes nothing
    run, _ = make_tuning_run(results, header_changes={"rounds": "1"})
    before = sorted(run.rglob("*"))
    proc = run_script("analyze_tuning.py", run, results=results)
    assert proc.returncode != 0
    assert "does not fit what analyze_tuning.py supports" in proc.stderr
    assert "rounds 1, expected at least 2" in proc.stderr
    assert sorted(run.rglob("*")) == before


# ---------------------------------------------------------------------------------------------
# communicator of the logging run, and check_communicator: what ncclsee recorded and the TUNING table

LOG = """\
n1:1:1 [0] NCCL INFO comm 0x1 rank 0 nRanks 8 nNodes 2 localRanks 4 localRank 0 MNNVL 0
n1:1:1 [0] NCCL INFO Channel 00/08 : 0 3 1 2 4 7 5 6
n1:1:1 [0] NCCL INFO Channel 01/08 : 0 2 1 7 4 6 5 3
"""
TOPO = '<system><cpu><gpu dev="0" sm="90" rank="0" gdr="1"/><gpu dev="1" sm="90" rank="1" gdr="1"/></cpu></system>'


def fake_logging_dir(tmp_path, log=LOG, topo=TOPO):
    """an iteration folder of a logging run with already extracted profiler outputs (rank 0's log and topology)"""
    d = tmp_path / "01"
    node = d / "profiler-out-extracted" / "profiler-out" / "ncclsee" / "all_reduce_bench" / "n1"
    node.mkdir(parents=True)
    if log is not None:
        (node / "nccl-rank0.log").write_text(log)
    if topo is not None:
        (node / "topo-rank0.xml").write_text(topo)
    return d


def test_logging_communicator(tmp_path):
    assert tuning.logging_communicator(fake_logging_dir(tmp_path)) == \
        tuning.Communicator(ranks=8, nodes=2, channels=8, min_comp_cap=90)


@pytest.mark.parametrize("log, topo, expected", [
    (None, TOPO, "no nccl-rank0.log"),
    (LOG.replace("nRanks", "ranks"), TOPO, "no 'comm ... nRanks <n> nNodes <n>' line"),
    (LOG.replace("Channel", "Chan"), TOPO, "no 'Channel <i>/<n>' line"),
    (LOG, None, "no topo-rank0.xml"),
])
def test_logging_communicator_incomplete(tmp_path, log, topo, expected):
    with pytest.raises(SystemExit) as e:
        tuning.logging_communicator(fake_logging_dir(tmp_path, log, topo))
    assert expected in str(e.value)


COMM = tuning.Communicator(ranks=8, nodes=2, channels=8, min_comp_cap=90)


def choices(ranks=range(COMM.ranks), default_channels=COMM.channels):
    rows = [{"rank": r, "variant": "default", "channels": default_channels} for r in ranks]
    rows += [{"rank": r, "variant": "ring_ll", "channels": 2} for r in ranks]
    return pd.DataFrame(rows)


def table(drop=()):
    return {(op, a, p): (1.0, 1.0) for op, combos in tuning.COMBINATIONS.items() for a, p in combos
            if (op, a, p) not in drop}


def communicator_problems(choices, table, comm=COMM, nodes=2):
    try:
        tuning.check_communicator("run", choices, table, comm, nodes)
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


def test_communicator_nodes():
    assert "NCCL's communicator has 2 node(s), the tuning run 3" in communicator_problems(choices(), table(), nodes=3)


def test_communicator_combinations():
    assert "TUNING table of the logging run offers" in \
        communicator_problems(choices(), table(drop=[("all_reduce", "tree", "ll128")]))


# ---------------------------------------------------------------------------------------------
# NCCL 2.27.7's model time: the factors that depend on the communicator

def model_time(comm, algo="ring", proto="simple", nbytes=2**20, lat=10.0, bw=10.0):
    return tuning.model_time_2_27_7({("all_reduce", algo, proto): (lat, bw)}, comm, "all_reduce", algo, proto, nbytes)


def test_model_time_ring_plateau():
    transfer = 2**20 / (1000 * 10.0)
    assert model_time(COMM) == pytest.approx(10.0 * 1.4 + transfer)
    # below compute capability 8.0: 1.9
    assert model_time(tuning.Communicator(8, 2, 8, 70)) == pytest.approx(10.0 * 1.9 + transfer)
    # one node, or fewer than 64 bytes per channel and rank: no plateau factor
    assert model_time(tuning.Communicator(4, 1, 8, 90)) == pytest.approx(10.0 + transfer)
    assert model_time(COMM, nbytes=2048) == pytest.approx(10.0 + 2048 / (1000 * 10.0))


def test_model_time_tree_correction():
    # 1 MiB: log2(2^20 / 64) = 14, LL128 factor 0.6
    assert model_time(COMM, algo="tree", proto="ll128") == pytest.approx(10.0 + 2**20 / (1000 * 10.0 * 0.6))


def test_model_time_unavailable():
    assert pd.isna(tuning.model_time_2_27_7({}, COMM, "all_reduce", "ring", "ll", 8))
