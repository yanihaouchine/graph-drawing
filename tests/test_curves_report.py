"""benchmark/curves_experiment.py report: tables, paired tests with effect sizes, verdict count, selection."""
import csv
import importlib.util
import json
import random
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("curves_experiment", ROOT / "benchmark" / "curves_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_settings_cover_the_sweep():
    script = load_script()
    assert len(script.SETTINGS["all"]) == 16 and set(script.HARD) <= set(script.SETTINGS["all"])
    assert script.SETTINGS["hard"] == script.HARD


def test_report_end_to_end(tmp_path):
    script = load_script()
    script.OUT = tmp_path
    rng = random.Random(0)
    rows = []
    # one of the three first settings, where curves halve the ambiguous edges, and one where they change nothing
    for (size, radius), gain in (((200, 4.0), 0.5), ((500, 2.0), 1.0)):
        for graph in range(12):
            base = {"ambiguous_edges": rng.uniform(0.1, 1), "edge_overlap": rng.uniform(0.1, 1),
                    "crossings": float(rng.randint(5, 40))}
            for edges in ("straight", "curved"):
                factor = gain if edges == "curved" else 1.0
                rows.append({"size": size, "radius": radius, "width": 1.0, "edges": edges, "graph": graph,
                             "ambiguous_edges": factor * base["ambiguous_edges"], "edge_overlap": base["edge_overlap"],
                             "crossings": base["crossings"], "hidden_nodes": 0.0, "ink_peak": 0.5,
                             "bend": 0.0 if edges == "straight" else 0.1})
    with open(tmp_path / "rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grid = [{"fraction": f, "lr": lr} for f in (0.1, 1.0, 3.0) for lr in (0.5, 1.0, 2.0)]
    (tmp_path / "selection.json").write_text(json.dumps({
        "200px_r4_w1": {"fraction": 3.0, "lr": 1.0, "candidates": grid},
        "500px_r2_w1": {"fraction": 1.0, "lr": 1.0, "candidates": grid}}))

    script.report(types.SimpleNamespace(quick=False))

    md = (tmp_path / "results_curves.md").read_text()
    assert md.count("## s = ") == 2
    count = {tuple(c.strip() for c in line.strip("|").split("|")[:2]): [c.strip() for c in line.strip("|").split("|")[2:]]
             for line in md.split("## Décompte des verdicts")[1].split("## ")[0].splitlines() if line.startswith("| ")}
    assert count[("tous", "Arêtes ambiguës / arête")] == ["1", "0", "1", "2"]  # better in one setting out of two
    assert count[("les trois premiers", "Arêtes ambiguës / arête")] == ["1", "0", "0", "1"]
    assert count[("tous", "Croisements")] == ["0", "0", "2", "2"]  # identical: nothing to tell
    assert "1 choix sur 2 tombent sur une borne" in md and "| 200px_r4_w1 | 3 | 1 | fraction |" in md
