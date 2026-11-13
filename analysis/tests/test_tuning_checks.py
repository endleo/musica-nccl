"""Run checks of analyze_tuning.py: a fake tuning run and logging run that fit together (job-info.txt, layout, empty
outputs; no data), one problem per test that has to abort the analysis, and other setups that have to be accepted
as long as both runs share them; the variants (what they pin, read from job-info.txt) and incomplete runs
(--allow-incomplete). Also the parts of the cost model that do not need data (communicator, model time, the
combinations NCCL chooses from), and the noise floor with combinations missing."""

import re
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
# the variants of experiments/tuning.sbatch
VARIANTS = {
    "default": "(job environment)",
    "tree_ll": "NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL",
    "tree_ll128": "NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL128",
    "tree_simple": "NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:Simple",
    "ring_ll": "NCCL_ALGO=allreduce:ring;broadcast:ring NCCL_PROTO=allreduce:LL;broadcast:LL",
    "ring_ll128": "NCCL_ALGO=allreduce:ring;broadcast:ring NCCL_PROTO=allreduce:LL128;broadcast:LL128",
    "ring_simple": "NCCL_ALGO=allreduce:ring;broadcast:ring NCCL_PROTO=allreduce:Simple;broadcast:Simple",
}
# what experiment.sh runs on Leonardo (UNSUPPORTED_PROTOCOLS=(LL128))
NO_LL128 = {k: v for k, v in VARIANTS.items() if "ll128" not in k}


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
    results/tuning/1000 with `rounds` rounds of every variant (default: VARIANTS; empty profiler outputs and
    ncclbench outputs), and the logging run results/logging/1889653 (by default with the same GPUs, ranks and
    environment); returns both folders
    """
    run = results / "tuning" / "1000"
    variants = dict(VARIANTS) if variants is None else variants
    ranks = ranks_of(gpus) if ranks is None else ranks
    write_job_info(run / "job-info.txt", header("tuning", {"rounds": str(rounds), **(header_changes or {})}),
                   variants=variants, gpus=gpus, ranks=ranks, env=env)
    for r in range(1, rounds + 1):
        for v in variants:
            d = run / f"{r:02d}" / v
            d.mkdir(parents=True)
            for name in ["profiler-out.tar.gz", *(f"{test}_ncclsee.out" for test in tuning.TESTS)]:
                (d / name).touch()
    logging_run = make_logging_run(results, changes=logging_changes,
                                   gpus=gpus if logging_gpus is SAME else logging_gpus,
                                   ranks=ranks if logging_ranks is SAME else logging_ranks,
                                   env=env if logging_env is SAME else logging_env)
    return run, logging_run


def problems(run, logging_run, allow_incomplete=False):
    """the problems check_run_info reports (its SystemExit message), "" if it passes"""
    try:
        tuning.check_run_info(run, logging_run, allow_incomplete)
    except SystemExit as e:
        return str(e)
    return ""


def design(run, logging_run, allow_incomplete=False):
    """the Design check_run_info returns (fails the test if it aborts)"""
    return tuning.check_run_info(run, logging_run, allow_incomplete)


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


@pytest.mark.parametrize("nccl", ["2.27.7", "2.32.3"])
def test_nccl_versions_with_cost_model_accepted(results, nccl):
    version = {"NCCL": f"{nccl} (built)"}
    assert problems(*make_tuning_run(results, header_changes=version, logging_changes=version)) == ""


def test_added_nccl_version_accepted(results, monkeypatch):
    # a new NCCL version is accepted once COST_MODELS has a model for it
    monkeypatch.setitem(tuning.COST_MODELS, "2.99.0", tuning.COST_MODELS["2.27.7"])
    nccl = {"NCCL": "2.99.0 (built)"}
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
    ("2.99.0 (built)", "tuning run: NCCL 2.99.0, the cost model is only implemented for NCCL 2.27.7, 2.32.3 "
                       "(COST_MODELS)"),
    (None, "tuning run: NCCL (not in job-info.txt), the cost model is only implemented for NCCL 2.27.7, 2.32.3"),
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


def test_variant_with_other_variables(results):
    variants = {**VARIANTS, "ring_ll_4ch": "NCCL_ALGO=ring NCCL_PROTO=LL NCCL_MAX_NCHANNELS=4"}
    assert "tuning run: variant ring_ll_4ch sets NCCL_MAX_NCHANNELS=4: only NCCL_ALGO and NCCL_PROTO are supported" \
        in problems(*make_tuning_run(results, variants=variants))


def test_combination_pinned_twice(results):
    # what a variant pins comes from its environment, not its name
    variants = {**VARIANTS, "tree_ll": "NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL128"}
    assert "tuning run: AllReduce Tree/LL128 pinned by several variants ['tree_ll', 'tree_ll128']" in \
        problems(*make_tuning_run(results, variants=variants))


def test_no_default_variant(results):
    variants = {k: v for k, v in VARIANTS.items() if k != "default"}
    assert "tuning run: no variant 'default' with the job environment" in \
        problems(*make_tuning_run(results, variants=variants))


def test_default_variant_with_pins(results):
    variants = {**VARIANTS, "default": "NCCL_PROTO=^LL128"}
    assert "tuning run: no variant 'default' with the job environment" in \
        problems(*make_tuning_run(results, variants=variants))


def test_nothing_pinned(results):
    variants = {"default": "(job environment)", "default2": "(job environment)"}
    assert "tuning run: no variant pins a combination" in problems(*make_tuning_run(results, variants=variants))


def test_variant_without_profiler_output(results):
    run, logging_run = make_tuning_run(results)
    (run / "02" / "tree_ll" / "profiler-out.tar.gz").unlink()
    assert "tuning run: incomplete, --allow-incomplete analyses the complete part: 02/tree_ll: no profiler output " \
           "(did not finish)" in problems(run, logging_run)


def test_unexpected_round(results):
    run, logging_run = make_tuning_run(results)
    shutil.copytree(run / "03", run / "04")
    assert "tuning run: unexpected (round, variant): [('04', 'default')" in problems(run, logging_run)


def test_benchmark_output_missing(results):
    run, logging_run = make_tuning_run(results)
    (run / "01" / "default" / "broadcast_bench_ncclsee.out").unlink()
    assert "01/default: broadcast_bench_ncclsee.out missing" in problems(run, logging_run)


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
n1:1:1 [0] NCCL INFO Pattern 4, crossNic 0, nChannels 4, bw 24.000000/24.000000, type NVL/PXN, sameChannels 0
n1:1:1 [0] NCCL INFO Pattern 3, crossNic 0, nChannels 4, bw 48.000000/24.000000, type NVL/PXB, sameChannels 0
n1:1:1 [0] NCCL INFO Pattern 1, crossNic 0, nChannels 4, bw 48.000000/24.000000, type NVL/SYS, sameChannels 0
n1:1:1 [0] NCCL INFO comm 0x1 rank 0 nRanks 8 nNodes 2 localRanks 4 localRank 0 MNNVL 0
n1:1:1 [0] NCCL INFO Channel 00/08 : 0 3 1 2 4 7 5 6
n1:1:1 [0] NCCL INFO Channel 01/08 : 0 2 1 7 4 6 5 3
"""
# graph types of LOG: the first graph per algorithm (patterns 4: Ring, 3: Tree; the later tree graph is ignored)
GRAPHS = (("tree", "NVL", "PXB"), ("ring", "NVL", "PXN"))
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
        tuning.Communicator(ranks=8, nodes=2, channels=8, min_comp_cap=90, graph_types=GRAPHS)


@pytest.mark.parametrize("log, topo, expected", [
    (None, TOPO, "no nccl-rank0.log"),
    (LOG.replace("nRanks", "ranks"), TOPO, "no 'comm ... nRanks <n> nNodes <n>' line"),
    (LOG.replace("Channel", "Chan"), TOPO, "no 'Channel <i>/<n>' line"),
    (LOG.replace("Pattern 4", "Pattern 5"), TOPO, "lines of the graphs ['tree', 'ring'] in nccl-rank0.log (found "
                                                  "['tree'])"),
    (LOG.replace("NVL/PXN", "NVL/XYZ"), TOPO, "unknown path types"),
    (LOG, None, "no topo-rank0.xml"),
])
def test_logging_communicator_incomplete(tmp_path, log, topo, expected):
    with pytest.raises(SystemExit) as e:
        tuning.logging_communicator(fake_logging_dir(tmp_path, log, topo))
    assert expected in str(e.value)


COMM = tuning.Communicator(ranks=8, nodes=2, channels=8, min_comp_cap=90, graph_types=GRAPHS)


def comm(**changes):
    """COMM with changes, e.g. comm(nodes=1)"""
    return tuning.Communicator(**{**COMM.__dict__, **changes})


def choices(ranks=range(COMM.ranks), default_channels=COMM.channels):
    rows = [{"rank": r, "variant": "default", "channels": default_channels} for r in ranks]
    rows += [{"rank": r, "variant": "ring_ll", "channels": 2} for r in ranks]
    return pd.DataFrame(rows)


def table(drop=()):
    return {(op, a, p): (1.0, 1.0) for op, combos in tuning.COMBINATIONS.items() for a, p in combos
            if (op, a, p) not in drop}


def communicator_problems(choices, table, comm=COMM, nodes=2, measured=tuning.COMBINATIONS):
    try:
        tuning.check_communicator("run", choices, table, comm, nodes, measured)
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
    assert "TUNING table of the logging run lacks measured combinations ['AllReduce Tree/LL128']" in \
        communicator_problems(choices(), table(drop=[("all_reduce", "tree", "ll128")]))


def test_communicator_combination_not_measured():
    # a combination the run did not measure (e.g. LL128 skipped on Leonardo) need not be in the table
    measured = {op: [c for c in combos if c[1] != "ll128"] for op, combos in tuning.COMBINATIONS.items()}
    assert communicator_problems(choices(), table(drop=[("all_reduce", "tree", "ll128")]), measured=measured) == ""


# ---------------------------------------------------------------------------------------------
# NCCL 2.27.7's model time: the factors that depend on the communicator

def model_time(comm, algo="ring", proto="simple", nbytes=2**20, lat=10.0, bw=10.0):
    return tuning.model_time_2_27_7({("all_reduce", algo, proto): (lat, bw)}, comm, "all_reduce", algo, proto, nbytes)


def test_model_time_ring_plateau():
    transfer = 2**20 / (1000 * 10.0)
    assert model_time(COMM) == pytest.approx(10.0 * 1.4 + transfer)
    # below compute capability 8.0: 1.9
    assert model_time(comm(min_comp_cap=70)) == pytest.approx(10.0 * 1.9 + transfer)
    # one node, or fewer than 64 bytes per channel and rank: no plateau factor
    assert model_time(comm(ranks=4, nodes=1)) == pytest.approx(10.0 + transfer)
    assert model_time(COMM, nbytes=2048) == pytest.approx(10.0 + 2048 / (1000 * 10.0))


def test_model_time_tree_correction():
    # 1 MiB: log2(2^20 / 64) = 14, LL128 factor 0.6
    assert model_time(COMM, algo="tree", proto="ll128") == pytest.approx(10.0 + 2**20 / (1000 * 10.0 * 0.6))


def test_model_time_unavailable():
    assert pd.isna(tuning.model_time_2_27_7({}, COMM, "all_reduce", "ring", "ll", 8))


# ---------------------------------------------------------------------------------------------
# the variants: what they pin (read from their environment in job-info.txt)

TREE, RING = "tree", "ring"
OLD_RING_LL = "NCCL_ALGO=allreduce:ring NCCL_PROTO=allreduce:LL;broadcast:LL"  # tuning.sbatch before NCCL 2.32.3


@pytest.mark.parametrize("env, expected", [
    ("(job environment)", {"all_reduce": None, "broadcast": None}),
    (VARIANTS["tree_ll"], {"all_reduce": (TREE, "ll"), "broadcast": None}),  # Broadcast: NCCL's choice
    (VARIANTS["ring_simple"], {"all_reduce": (RING, "simple"), "broadcast": (RING, "simple")}),
    (OLD_RING_LL, {"all_reduce": (RING, "ll"), "broadcast": (RING, "ll")}),  # 2.27.7: Broadcast keeps Ring
    ("NCCL_ALGO=allreduce:Tree NCCL_PROTO=allreduce:ll128", {"all_reduce": (TREE, "ll128"), "broadcast": None}),
    # a bare list applies to all operations; Broadcast has no Tree in the analysis, so Tree pins nothing there
    ("NCCL_ALGO=ring NCCL_PROTO=LL", {"all_reduce": (RING, "ll"), "broadcast": (RING, "ll")}),
    # exclusions: everything but LL and LL128 leaves Simple
    ("NCCL_ALGO=allreduce:ring NCCL_PROTO=allreduce:^LL,LL128", {"all_reduce": (RING, "simple"), "broadcast": None}),
    # functions the analysis does not measure are ignored
    ("NCCL_ALGO=allreduce:tree;allgather:ring NCCL_PROTO=allreduce:LL", {"all_reduce": (TREE, "ll"),
                                                                         "broadcast": None}),
])
def test_variant_pins(env, expected):
    assert tuning.variant_pins(env) == expected


@pytest.mark.parametrize("env, resets, expected", [
    ("NCCL_ALGO=ring NCCL_PROTO=LL NCCL_NTHREADS=64", False, "sets NCCL_NTHREADS=64: only NCCL_ALGO and NCCL_PROTO"),
    ("NCCL_PROTO=allreduce:LL", False, "leaves AllReduce 2 combinations ['Tree/LL', 'Ring/LL'], expected exactly one"),
    ("NCCL_ALGO=allreduce:nvls NCCL_PROTO=allreduce:Simple", False, "enables ['nvls'] for AllReduce"),
    # a bare list applies to Broadcast as well, which has no Tree (NCCL would fail)
    ("NCCL_ALGO=Tree NCCL_PROTO=allreduce:LL", False, "leaves Broadcast 0 combinations [], expected exactly one"),
    ("NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL;ll128", False, "every entry but the first needs a prefix"),
    ("NCCL_ALGO=allreduce:tree NCCL_PROTO=allreduce:LL256", False, "unknown ['ll256']"),
    ("NCCL_ALGO=gather:tree", False, "unknown function gather"),
    # NCCL 2.32.3: NCCL_ALGO without broadcast leaves the Broadcast that NCCL_PROTO forces no algorithm
    (OLD_RING_LL, True, "leaves Broadcast no algorithm"),
])
def test_variant_pins_refused(env, resets, expected):
    with pytest.raises(ValueError, match=re.escape(expected)):
        tuning.variant_pins(env, algo_resets_all=resets)


def test_variant_pins_2_32_3_current_variants():
    # the variants of tuning.sbatch are the same under both versions' rules
    for env in VARIANTS.values():
        assert tuning.variant_pins(env, algo_resets_all=True) == tuning.variant_pins(env)


def test_variants_without_ll128(results):
    # Leonardo: experiment.sh skipped the LL128 variants; everything else is measured
    d = design(*make_tuning_run(results, variants=NO_LL128))
    assert d.left_out == ()
    assert d.variants == ["default", "tree_ll", "tree_simple", "ring_ll", "ring_simple"]
    assert d.measured == {"all_reduce": [(TREE, "ll"), (TREE, "simple"), (RING, "ll"), (RING, "simple")],
                          "broadcast": [(RING, "ll"), (RING, "simple")]}
    assert len(d.cells) == 3 * 5


def test_variants_other_names_and_extra(results):
    # names do not matter, and variants that pin nothing (more samples of NCCL's choice) are allowed
    variants = {"default": "(job environment)", "a": VARIANTS["ring_ll"], "b": VARIANTS["tree_simple"],
                "default_again": "(job environment)"}
    d = design(*make_tuning_run(results, variants=variants))
    assert d.variants == ["default", "b", "a", "default_again"]  # by what they pin: AllReduce Tree before Ring
    assert d.pinned("b", "broadcast") is None and d.pinned("a", "broadcast") == (RING, "ll")


def test_old_ring_variants_refused_for_2_32_3(results):
    nccl = {"NCCL": "2.32.3 (built)"}
    variants = {**VARIANTS, "ring_ll": OLD_RING_LL}
    text = problems(*make_tuning_run(results, variants=variants, header_changes=nccl, logging_changes=nccl))
    assert "tuning run: variant ring_ll leaves Broadcast no algorithm" in text
    assert "(NCCL 2.32.3)" in text


def test_old_ring_variants_accepted_for_2_27_7(results):
    assert problems(*make_tuning_run(results, variants={**VARIANTS, "ring_ll": OLD_RING_LL})) == ""


# ---------------------------------------------------------------------------------------------
# incomplete runs: refused, or with --allow-incomplete the largest complete part

def leonardo_like(results):
    """
    as tuning run leonardo-64323905: round 01 has default and tree_ll, tree_ll128 started (no profiler output) and
    the job ended; rounds 02 and 03 never ran
    """
    run, logging_run = make_tuning_run(results)
    for r in ["02", "03"]:
        shutil.rmtree(run / r)
    for v in ["tree_simple", "ring_ll", "ring_ll128", "ring_simple"]:
        shutil.rmtree(run / "01" / v)
    (run / "01" / "tree_ll128" / "profiler-out.tar.gz").unlink()
    return run, logging_run


def test_incomplete_refused(results):
    text = problems(*leonardo_like(results))
    assert "tuning run: incomplete, --allow-incomplete analyses the complete part: 01/tree_ll128: no profiler " \
           "output (did not finish); 01/tree_simple: missing" in text
    assert "02/(all variants): missing; 03/(all variants): missing" in text


def test_incomplete_allowed(results):
    d = design(*leonardo_like(results), allow_incomplete=True)
    assert d.rounds == ["01"] and d.variants == ["default", "tree_ll"]
    assert d.measured == {"all_reduce": [(TREE, "ll")], "broadcast": []}
    assert "01/tree_ll128: no profiler output (did not finish)" in d.left_out
    assert d.left_out[-1] == ("not analysed: round 02, round 03, variant tree_ll128, variant tree_simple, "
                              "variant ring_ll, variant ring_ll128, variant ring_simple")
    assert d.note()[0].startswith("INCOMPLETE RUN (--allow-incomplete): analysed rounds ['01'] x variants")


def test_failed_variant(results):
    # experiment.sh's failed.txt (here: STEP_TIMEOUT) makes a (round, variant) incomplete, even with all outputs
    run, logging_run = make_tuning_run(results)
    (run / "02" / "ring_ll" / "failed.txt").write_text("broadcast_bench ncclsee: timed out after 10m (STEP_TIMEOUT)\n")
    assert "02/ring_ll: failed: broadcast_bench ncclsee: timed out after 10m (STEP_TIMEOUT)" in \
        problems(run, logging_run)
    # leaving out ring_ll (3 rounds x 6 variants) keeps more than leaving out round 02 (2 x 7)
    d = design(run, logging_run, allow_incomplete=True)
    assert d.rounds == ["01", "02", "03"] and "ring_ll" not in d.variants and len(d.variants) == 6


def test_missing_round(results):
    # a whole round missing: the other rounds with all variants (2 x 7) rather than fewer variants
    run, logging_run = make_tuning_run(results)
    shutil.rmtree(run / "03" / "default")
    d = design(run, logging_run, allow_incomplete=True)
    assert d.rounds == ["01", "02"] and len(d.variants) == 7
    assert "not analysed: round 03" in d.left_out


def test_incomplete_without_default(results):
    run, logging_run = make_tuning_run(results)
    for r in ["01", "02", "03"]:
        (run / r / "default" / "profiler-out.tar.gz").unlink()
    assert "tuning run: incomplete, and no round has both the default variant and one that pins a combination" in \
        problems(run, logging_run, allow_incomplete=True)


def test_incomplete_only_default(results):
    run, logging_run = make_tuning_run(results, variants={k: VARIANTS[k] for k in ["default", "tree_ll"]})
    for r in ["01", "02", "03"]:
        shutil.rmtree(run / r / "tree_ll")
    assert "no round has both the default variant and one that pins a combination" in \
        problems(run, logging_run, allow_incomplete=True)


def test_invalid_variant_left_out_of_incomplete_run(results):
    # an invalid variant aborts a complete analysis, but is only left out with --allow-incomplete
    nccl = {"NCCL": "2.32.3 (built)"}
    run, logging_run = make_tuning_run(results, variants={**VARIANTS, "ring_ll": OLD_RING_LL}, header_changes=nccl,
                                       logging_changes=nccl)
    d = design(run, logging_run, allow_incomplete=True)
    assert "ring_ll" not in d.variants and len(d.variants) == 6
    assert any(x.startswith("variant ring_ll: leaves Broadcast no algorithm") for x in d.left_out)


def test_complete_run_with_allow_incomplete(results):
    d = design(*make_tuning_run(results), allow_incomplete=True)
    assert d.left_out == () and d.note() == [] and len(d.cells) == 3 * 7


@pytest.mark.parametrize("ok, expected", [
    # ties in the number of runs: more variants first, then earlier rounds
    ({("1", "default"), ("1", "a"), ("2", "default"), ("2", "b")}, (["1"], ["default", "a"])),
    ({("1", "default"), ("2", "default"), ("2", "a"), ("2", "b")}, (["2"], ["default", "a", "b"])),
    # 1 x 2 = 2 x 1: the round with a pinned variant rather than both rounds of NCCL's choice only
    ({("1", "default"), ("2", "default"), ("1", "a")}, (["1"], ["default", "a"])),
    # NCCL's choice alone is nothing to compare: the round with a pinned variant, though 3 x 1 > 1 x 2
    ({("1", "default"), ("2", "default"), ("3", "default"), ("1", "a")}, (["1"], ["default", "a"])),
    ({("1", "a"), ("2", "b")}, ([], [])),  # no default anywhere
    ({("1", "default"), ("2", "default")}, ([], [])),  # nothing pinned anywhere
])
def test_complete_selection(ok, expected):
    rounds = sorted({r for r, _ in ok} | {"1", "2"})
    assert tuning.complete_selection(ok, rounds, ["default", "a", "b"], pinning=["a", "b"]) == expected


def test_script_allow_incomplete(results):
    run, _ = leonardo_like(results)
    proc = run_script("analyze_tuning.py", run, results=results)
    assert proc.returncode != 0 and "--allow-incomplete analyses the complete part" in proc.stderr
    proc = run_script("analyze_tuning.py", run, "--allow-incomplete", results=results)
    # past the checks (the fake run has no data, so it fails when loading it)
    assert "1 round(s) x 2 variant(s)" in proc.stdout
    assert "INCOMPLETE RUN (--allow-incomplete)" in proc.stdout


# ---------------------------------------------------------------------------------------------
# NCCL 2.32.3: the combinations NCCL chooses from by itself (LL128), and the model time

A100_LEONARDO = comm(min_comp_cap=80, nodes=8, ranks=32, graph_types=(("tree", "NVL", "PHB"), ("ring", "NVL", "PXN")))


@pytest.mark.parametrize("communicator, ll128", [
    (A100_LEONARDO, set()),  # Leonardo: Ring over PXN, Tree over PHB, LL128 needs <= PXB on A100
    (comm(min_comp_cap=80, graph_types=(("tree", "NVL", "PXB"), ("ring", "NVL", "PIX"))), {TREE, RING}),
    (COMM, {TREE, RING}),  # H100 (MUSICA): <= PXN with NCCL_LL128_C2C (default)
    (comm(graph_types=(("tree", "NVL", "PHB"), ("ring", "NVL", "PXN"))), {RING}),
    (comm(graph_types=(("tree", "PIX", "PXB"), ("ring", "NVL", "PXN"))), {RING}),  # intra beyond NVB
    (comm(min_comp_cap=60), set()),
])
def test_enabled_2_32_3(communicator, ll128):
    enabled = tuning.enabled_2_32_3(table(), communicator)
    assert {a for op, a, p in enabled if p == "ll128" and op == "all_reduce"} == ll128
    # everything else of the table stays
    assert {k for k in table() if k[2] != "ll128"} <= enabled


def test_enabled_2_27_7_is_the_table():
    assert tuning.enabled_2_27_7(table(drop=[("all_reduce", TREE, "ll128")]), COMM) == \
        set(table(drop=[("all_reduce", TREE, "ll128")]))


@pytest.mark.parametrize("communicator", [COMM, A100_LEONARDO, comm(min_comp_cap=70), comm(ranks=4, nodes=1)])
@pytest.mark.parametrize("algo, proto", [(RING, "simple"), (TREE, "ll128"), (RING, "ll")])
@pytest.mark.parametrize("nbytes", [8, 2048, 2**20, 2**30])
def test_model_time_2_32_3_same_as_2_27_7(communicator, algo, proto, nbytes):
    # before Blackwell, 2.32.3's formula is 2.27.7's
    t = {("all_reduce", algo, proto): (10.0, 10.0)}
    assert tuning.model_time_2_32_3(t, communicator, "all_reduce", algo, proto, nbytes) == \
        tuning.model_time_2_27_7(t, communicator, "all_reduce", algo, proto, nbytes)


@pytest.mark.parametrize("nbytes, factor", [
    (2**20, 1 + (1 - 0.25**5) * 0.4),  # 32 KiB per rank and channel: the full delta, scaled by the 6 ring steps
    (2**24, 1 + (1 - 0.25**5) * 0.3),  # 512 KiB per rank and channel: delta reduced by 0.1
])
def test_model_time_2_32_3_blackwell_single_node(nbytes, factor):
    blackwell = comm(min_comp_cap=100, ranks=4, nodes=1)
    t = {("all_reduce", RING, "simple"): (10.0, 10.0)}
    assert tuning.model_time_2_32_3(t, blackwell, "all_reduce", RING, "simple", nbytes) == \
        pytest.approx(10.0 * factor + nbytes / (1000 * 10.0))
    # 2.27.7 has no plateau on one node; without NVLink in the ring there is none in 2.32.3 either
    assert tuning.model_time_2_27_7(t, blackwell, "all_reduce", RING, "simple", nbytes) == \
        pytest.approx(10.0 + nbytes / (1000 * 10.0))
    pcie = comm(min_comp_cap=100, ranks=4, nodes=1, graph_types=(("tree", "PIX", "PIX"), ("ring", "PIX", "PIX")))
    assert tuning.model_time_2_32_3(t, pcie, "all_reduce", RING, "simple", nbytes) == \
        pytest.approx(10.0 + nbytes / (1000 * 10.0))


def test_model_choice_only_enabled():
    # LL128 has the lowest estimate, but NCCL 2.32.3 disables it on Leonardo: the choice is the best of the rest
    t = {k: (1.0, 100.0 if k[2] == "ll128" else 1.0) for k in table()}
    t[("all_reduce", TREE, "ll")] = (1.0, 2.0)
    model = tuning.CostModel(nccl="2.32.3", table=t, comm=A100_LEONARDO)
    est = tuning.model_estimates(model, [2**20])
    chosen = est[est["model_choice"]].set_index("operation")
    assert tuple(chosen.loc["all_reduce", ["algo", "proto"]]) == (TREE, "ll")
    assert not est.loc[est["proto"] == "ll128", "enabled"].any()
    assert est["estimate_us"].notna().all()  # LL128 still has an estimate (for the pinned variants)


@pytest.mark.parametrize("line, expected, n", [
    ("LL128", "unsupported according to the machine file: LL128", 0),
    ("-", "unsupported according to the machine file: -  <-- DIFFERS", 1),
    (None, "does not record the machine's UNSUPPORTED_PROTOCOLS", 0),
])
def test_protocol_check(line, expected, n):
    info = {} if line is None else {"unsupported protocols": line}
    text, problems_found = tuning.protocol_check(info, tuning.CostModel(nccl="2.32.3", table=table(),
                                                                        comm=A100_LEONARDO))
    assert "Protocols NCCL disables by itself (cost model, logging run): LL128" in text
    assert expected in text and problems_found == n


def test_unsupported_env_2_32_3(results):
    # NCCL_PROTO in the job environment: 2.32.3's table does not show it, so the run is refused
    nccl = {"NCCL": "2.32.3 (built)"}
    env = {**ENV, "NCCL_PROTO": "^LL128"}
    assert "tuning run: ['NCCL_PROTO'] set in the job environment, not modelled for NCCL 2.32.3" in \
        problems(*make_tuning_run(results, header_changes=nccl, logging_changes=nccl, env=env))


# ---------------------------------------------------------------------------------------------
# noise floor without NCCL's own combination measured

def test_noise_floor_unknown_without_own_combination(tmp_path):
    pins = {"default": {"all_reduce": None, "broadcast": None}, "tree_ll": {"all_reduce": (TREE, "ll"),
                                                                            "broadcast": None}}
    d = tuning.Design(pins=pins, cells=tuple((r, v, tmp_path) for r in ["01", "02"] for v in pins))
    sizes = [8, 2**20]
    # per round medians: default 100 everywhere; tree_ll 98 / 102 at 8 B (NCCL's own choice there), 80 at 1 MiB
    # (NCCL chose something else); tree_ll's Broadcast (NCCL's choice) 105 / 95
    rows = []
    for r, f in [("01", 0.98), ("02", 1.02)]:
        for s in sizes:
            rows += [{"round": r, "variant": "default", "operation": op, "size_bytes": s, "time_us": 100.0}
                     for op in TESTS_OPS]
            rows.append({"round": r, "variant": "tree_ll", "operation": "all_reduce", "size_bytes": s,
                         "time_us": 100 * f if s == 8 else 80.0})
            rows.append({"round": r, "variant": "tree_ll", "operation": "broadcast", "size_bytes": s,
                         "time_us": 105.0 if r == "01" else 95.0})
    bench = pd.DataFrame(rows)
    pvd = pd.DataFrame([
        {"operation": "all_reduce", "size_bytes": s, "algo": a, "proto": p, "ratio": ratio, "ratio_min": ratio * f0,
         "ratio_max": ratio * f1, "same": same}
        for s, ratio, f0, f1, same in [(8, 1.0, 0.98, 1.02, True), (2**20, 0.8, 1, 1, False)]
        for a, p in [(TREE, "ll")]] + [
        {"operation": op, "size_bytes": s, "algo": "default", "proto": "default", "ratio": 1.0, "ratio_min": 1.0,
         "ratio_max": 1.0, "same": False} for op in TESTS_OPS for s in sizes])
    noise = tuning.noise_floor(bench, pvd, d)
    assert noise.loc[("all_reduce", 8)] == pytest.approx(0.02)
    assert pd.isna(noise.loc[("all_reduce", 2**20)])  # unknown: NCCL's own combination not measured
    assert noise.loc[("broadcast", 2**20)] == pytest.approx(0.05)  # tree_ll's Broadcast vs. default
    # 20 % faster in every round, but the noise is unknown: not counted as clearly faster
    assert not tuning.clearly_faster(pvd, noise).any()


TESTS_OPS = list(tuning.TESTS.values())
