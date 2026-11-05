"""Logging experiment (experiments/logging.sbatch): NCCL's debug log, topology and graph dumps. Stub, no analysis yet.

    uv run analysis/analyze_logging.py [results/logging/<job>]     default: newest run in results/logging/

Lists the iterations of the run. The numbers of report/logging/summary.typ are taken from the logs by hand so far.
TODO: candidates for this script: NCCL's environment and layout (rings/trees, channels), its per-call choices and the
TUNING table (cost model), which analyze_tuning.py reads from the logging run for now.
"""

from common import run_from_args


def main():
    run = run_from_args(__doc__, "logging")
    iterations = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name.isdigit())
    print(f"{run}: logging run, iterations {' '.join(iterations) or '(none)'}")
    print("no analysis for the logging experiment yet")


if __name__ == "__main__":
    main()
