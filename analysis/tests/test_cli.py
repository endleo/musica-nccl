"""Command line of the analyze_*.py scripts: which run they take, which runs they refuse (fake runs, no data)."""

import pytest

from conftest import run_script, write_job_info

SCRIPTS = {"full": "analyze_full.py", "quick": "analyze_full.py", "tuning": "analyze_tuning.py",
           "logging": "analyze_logging.py"}


def fake_run(results, experiment, job, date="2026-11-01T12:00:00+01:00"):
    """results/<experiment>/<job> with a job-info.txt only: in the run folder (tuning), else in iteration 01"""
    run = results / experiment / job
    info = run / "job-info.txt" if experiment == "tuning" else run / "01" / "job-info.txt"
    write_job_info(info, {"experiment": experiment, "job": f"{job} (array job -, task -)", "date": date})
    return run


@pytest.mark.parametrize("script, experiment", [
    ("analyze_full.py", "tuning"), ("analyze_full.py", "logging"),
    ("analyze_tuning.py", "full"), ("analyze_tuning.py", "quick"), ("analyze_tuning.py", "logging"),
    ("analyze_logging.py", "full"), ("analyze_logging.py", "quick"), ("analyze_logging.py", "tuning"),
])
def test_refuses_run_of_other_experiment(results, script, experiment):
    run = fake_run(results, experiment, "1000")
    proc = run_script(script, run, results=results)
    assert proc.returncode != 0
    assert f"a {experiment} run" in proc.stderr
    assert f"use analysis/{SCRIPTS[experiment]}" in proc.stderr
    assert not any(p.is_file() and p.name != "job-info.txt" for p in run.rglob("*"))  # nothing written


@pytest.mark.parametrize("script", sorted(set(SCRIPTS.values())))
def test_refuses_missing_folder(results, script):
    proc = run_script(script, results / "full" / "nope", results=results)
    assert proc.returncode != 0
    assert "no such run folder" in proc.stderr


@pytest.mark.parametrize("script", sorted(set(SCRIPTS.values())))
def test_refuses_folder_without_job_info(results, script):
    (results / "full" / "1000" / "01").mkdir(parents=True)
    proc = run_script(script, results / "full" / "1000", results=results)
    assert proc.returncode != 0
    assert "no job-info.txt" in proc.stderr


@pytest.mark.parametrize("experiment", ["full", "tuning", "logging"])
def test_default_is_newest_run_of_the_experiment(results, experiment):
    # newest by the date in job-info.txt, not by name (Slurm job ids and spark-<date> ids do not sort by time),
    # and only among the runs of the script's experiment
    fake_run(results, experiment, "2000", date="2026-10-01T12:00:00+02:00")
    fake_run(results, experiment, "1000", date="2026-11-02T12:00:00+01:00")
    fake_run(results, experiment, "spark-20261101-120000", date="2026-11-01T12:00:00+01:00")
    other = "logging" if experiment != "logging" else "full"
    fake_run(results, other, "3000", date="2026-11-05T12:00:00+01:00")
    proc = run_script(SCRIPTS[experiment], results=results)
    # the fake runs have no data: the scripts fail after choosing the run, naming it
    output = proc.stdout + proc.stderr
    assert str(results / experiment / "1000") in output
    for other_run in [results / experiment / "2000", results / experiment / "spark-20261101-120000",
                      results / other / "3000"]:
        assert str(other_run) not in output


@pytest.mark.parametrize("script, experiment", [("analyze_full.py", "full"), ("analyze_tuning.py", "tuning"),
                                                ("analyze_logging.py", "logging")])
def test_no_runs(results, script, experiment):
    (results / experiment).mkdir()
    proc = run_script(script, results=results)
    assert proc.returncode != 0
    assert "no runs with a job-info.txt" in proc.stderr


def test_full_takes_quick_runs(results):
    run = fake_run(results, "quick", "1000")
    proc = run_script("analyze_full.py", run, results=results)
    # past the experiment check; fails at the missing profiler output of the fake run
    assert "a quick run" not in proc.stderr
    assert "no iterations with profiler output" in proc.stderr


def test_full_refuses_run_without_profiler_output(results):
    run = fake_run(results, "full", "1000")
    proc = run_script("analyze_full.py", run, results=results)
    assert proc.returncode != 0
    assert "no iterations with profiler output" in proc.stderr


def test_logging_stub(results):
    run = fake_run(results, "logging", "1000")
    (run / "02").mkdir()
    proc = run_script("analyze_logging.py", run, results=results)
    assert proc.returncode == 0, proc.stderr
    assert "iterations 01 02" in proc.stdout
    assert "no analysis for the logging experiment yet" in proc.stdout
