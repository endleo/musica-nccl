"""Shared by the analysis scripts (analyze_<experiment>.py): results folder, command line, which run to analyze.

Every script analyzes one run of its experiment: the run folder given on the command line, by default the newest run
in results/<experiment>/ (by the date in its job-info.txt, see profiler_results.newest_run). A run of another
experiment is rejected with the name of the script for it.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # only files, no windows

import profiler_results as pr

RESULTS = (Path(__file__).resolve().parent.parent / "results").resolve()

# analysis script per experiment (EXPERIMENT in experiments/<experiment>.sbatch); quick runs have the layout of full
# runs, so analyze_full.py takes them as well
SCRIPTS = {"full": "analyze_full.py", "quick": "analyze_full.py", "logging": "analyze_logging.py",
           "tuning": "analyze_tuning.py"}


def label(path):
    """short name of a run or iteration folder for figure titles: results/full/1863257/01 -> full/1863257/01"""
    path = Path(path).resolve()
    return str(path.relative_to(RESULTS)) if path.is_relative_to(RESULTS) else str(path)


def run_experiment(run):
    """
    experiment of a run (`experiment:` in its first job-info.txt: <run>/job-info.txt with variants, else
    <run>/<NN>/job-info.txt, as in profiler_results.run_date), None if it has none
    """
    for info in [*sorted(Path(run).glob("job-info.txt")), *sorted(Path(run).glob("*/job-info.txt"))]:
        if m := re.search(r"^experiment: (\S+)", info.read_text(), re.M):
            return m.group(1)
    return None


def run_from_args(doc, experiment, accepted=None):
    """
    the run folder to analyze: the command line argument, default the newest run in results/<experiment>/. Exits if
    the run's experiment is not one of `accepted` (default: only `experiment`)
    """
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help=f"run folder (default: newest in results/{experiment}/)")
    args = parser.parse_args()
    run = args.run or pr.newest_run((RESULTS / experiment).glob("*"))
    if not run.is_dir():
        raise SystemExit(f"{run}: no such run folder")
    found, accepted = run_experiment(run), accepted or [experiment]
    if found is None:
        raise SystemExit(f"{run}: no job-info.txt with the experiment, not a run folder?")
    if found not in accepted:
        script = SCRIPTS.get(found)
        hint = f", use analysis/{script}" if script else ""
        raise SystemExit(f"{run}: a {found} run, this script analyzes {' and '.join(accepted)} runs{hint}")
    return run
