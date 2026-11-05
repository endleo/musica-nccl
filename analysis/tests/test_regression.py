"""Regression tests: every analysis script on copies of existing runs, outputs compared with analysis/tests/reference/.

The runs come from results/ (not in git: fetch.sh); a test is skipped if its run is missing. Each test links the
run's inputs into a results folder of its own (conftest.link_run), so results/ is never written to, and runs the
script on it. Its outputs must be the same files as in reference/<experiment>/<job>/: text files identical; PDFs
identical, or, if the bytes differ (e.g. after a matplotlib update), identical when rendered (pdftoppm, poppler).

After a deliberate change of the outputs: uv run pytest -m slow --update-reference, then check the diff of
reference/ (text files) and the rendered PDFs before committing them.
"""

import difflib
import shutil
import subprocess

import pytest

from conftest import REFERENCE, RESULTS, link_run, outputs, run_script

# run -> (script, other runs it reads)
RUNS = {
    "full/1863257": ("analyze_full.py", []),
    "full/1874191": ("analyze_full.py", []),
    "full/spark-20261104-173019": ("analyze_full.py", []),
    "full/spark-20261104-192246": ("analyze_full.py", []),
    "quick/1863239": ("analyze_full.py", []),
    "tuning/1893400": ("analyze_tuning.py", ["logging/1889653"]),  # LOGGING_RUN: NCCL's TUNING table
}


def rendered(pdf, out_dir):
    """the pages of a PDF as images (PPM, 100 dpi)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-r", "100", str(pdf), str(out_dir / "page")], check=True)
    return [p.read_bytes() for p in sorted(out_dir.glob("page*"))]


def difference(path, reference, tmp_path):
    """None if the output equals its reference, else what differs"""
    new, old = path.read_bytes(), reference.read_bytes()
    if new == old:
        return None
    if path.suffix != ".pdf":
        diff = difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(), "reference", "new",
                                    lineterm="", n=1)
        return "\n".join(list(diff)[:40])
    if shutil.which("pdftoppm") is None:
        return "PDF bytes differ (install poppler's pdftoppm to compare the rendered pages)"
    key = str(path.relative_to(tmp_path)).replace("/", "_")
    new_pages, old_pages = rendered(path, tmp_path / "png" / key / "new"), rendered(reference, tmp_path / "png" / key / "old")
    if len(new_pages) != len(old_pages):
        return f"{len(new_pages)} pages, reference {len(old_pages)}"
    if differ := [i + 1 for i, (a, b) in enumerate(zip(new_pages, old_pages)) if a != b]:
        return f"rendered pages {differ} differ"
    return None  # other bytes, same rendering


@pytest.mark.slow
@pytest.mark.parametrize("run", RUNS)
def test_outputs(run, tmp_path, request):
    script, needs = RUNS[run]
    if missing := [r for r in [run, *needs] if not (RESULTS / r).is_dir()]:
        pytest.skip(f"not in results/: {', '.join(missing)}")
    results = tmp_path / "results"
    for r in [run, *needs]:
        link_run(RESULTS / r, results)
    copy = results / run

    proc = run_script(script, copy, results=results)
    assert proc.returncode == 0, proc.stderr[-5000:]
    produced = outputs(copy)
    reference = REFERENCE / run

    if request.config.getoption("--update-reference"):
        shutil.rmtree(reference, ignore_errors=True)
        for f in produced:
            (reference / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(copy / f, reference / f)
        return

    assert reference.is_dir(), f"no reference outputs in {reference} (--update-reference)"
    expected = sorted(p.relative_to(reference) for p in reference.rglob("*") if p.is_file())
    assert [str(f) for f in produced] == [str(f) for f in expected], "other output files than the reference"
    differences = {str(f): d for f in produced if (d := difference(copy / f, reference / f, tmp_path))}
    assert not differences, "\n\n".join(f"{f}:\n{d}" for f, d in differences.items())
