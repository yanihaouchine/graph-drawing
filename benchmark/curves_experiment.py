"""Curved vs straight edges on every setting of the resolution sweep: from the drawing of our model refined in screen space at the
setting (the refinement chosen by benchmark/resolution_sweep.py), the nodes stay where they are and only the
edges change, straight or bent in screen space (src/curves). Measures on the rendered image.

    python benchmark/curves_experiment.py run runs/model.pt [--workers 48] [--settings all|hard] [--quick]
    python benchmark/curves_experiment.py report

Needs benchmark/out/positions.pt (compare.py) and benchmark/out/resolution/{selection.json, val_starts.pt}
(resolution_sweep.py run). Writes benchmark/out/curves/: rows.csv, selection.json, results_curves.md (means with
95 % bootstrap intervals, paired Wilcoxon tests curved vs straight with Holm correction) and sheet.png
(three test graphs, straight and curved, defects of the image in red).

--settings hard runs only the three settings of the first experiment, which were picked because straight edges
hit nodes most there; the default runs the 16 settings of the sweep, so that the result does not rest on that
choice.
"""
import argparse
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

OUT = HERE / "out" / "curves"
SWEEP = HERE / "out" / "resolution"
DATA = HERE.parent / "data" / "rome"
HARD = [(200, 4.0, 1.0), (200, 4.0, 3.0), (300, 8.0, 1.0)]  # where straight edges hit nodes most: the first experiment
SETTINGS = {"hard": HARD,  # "all": the settings of benchmark/resolution_sweep.py
            "all": sorted([(s, r, 1.0) for s in (100, 150, 200, 300, 500) for r in (2.0, 4.0, 8.0)] + [(200, 4.0, 3.0)])}
COMPARED = ("ambiguous_edges", "edge_overlap", "crossings")  # measures of the paired tests, all to minimise
MIN_EFFECT = 0.1  # smallest |rank-biserial| counted as a difference, as in the sweep report
MEASURES = {"ambiguous_edges": "Arêtes ambiguës / arête", "edge_overlap": "Arêtes confondues",
            "crossings": "Croisements", "hidden_nodes": "Nœuds cachés", "ink_peak": "Encre (pic)", "bend": "Courbure"}


def refined_drawings(starts, graphs, size, radius, width, chosen, device):
    """Our model's drawings, in the frame, refined with the screen loss chosen by the sweep for this setting."""
    from resolution import anchor, make_loss, refine_all, rescale_to_frame, with_style
    framed = [rescale_to_frame(p, size, radius) for p in starts]
    styled = [anchor(with_style(g, radius, width), p) for g, p in zip(graphs, framed)]
    P = refine_all(framed, styled, size, make_loss("screen", chosen["weights"]), 25, chosen["lr"], radius, device)
    return list(zip(P, styled))


def run(args):
    from curves.experiment import curves_setting
    from resolution import make_pool
    from data.input import load_dataset, split_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = load_dataset(DATA)
    _, val_files, test_files = split_dataset(DATA)
    val_saved = torch.load(SWEEP / "val_starts.pt")
    val_graphs = [graphs[name] for name in val_saved["names"]]
    test_graphs = [graphs[p.stem] for p in test_files]
    positions = torch.load(HERE / "out" / "positions.pt")["Notre modèle"]
    test_starts = [positions[g.name] for g in test_graphs]
    val_starts = val_saved["starts"]["ours"]
    selection_sweep = json.loads((SWEEP / "selection.json").read_text())
    settings = SETTINGS[args.settings]
    if args.quick:
        settings, val_graphs, val_starts = HARD[:1], val_graphs[:20], val_starts[:20]
        test_graphs, test_starts = test_graphs[:20], test_starts[:20]

    rows_path = OUT / ("rows_quick.csv" if args.quick else "rows.csv")
    selection, writer = {}, None
    with open(rows_path, "w", newline="") as f, make_pool(args.workers) as pool:
        for size, radius, width in settings:
            t = time.perf_counter()
            key = f"{size}px_r{radius:g}_w{width:g}"
            chosen = selection_sweep[key]["screen"]
            val = refined_drawings(val_starts, val_graphs, size, radius, width, chosen, device)
            test = refined_drawings(test_starts, test_graphs, size, radius, width, chosen, device)
            kwargs = dict(fractions=(1.0,), learning_rates=(1.0,), steps=20) if args.quick else {}
            if args.fractions:
                kwargs["fractions"] = tuple(float(x) for x in args.fractions.split(","))
            if args.rates:
                kwargs["learning_rates"] = tuple(float(x) for x in args.rates.split(","))
            rows, chosen_curves, offsets = curves_setting(size, val, test, device, pool, **kwargs)
            selection[key] = chosen_curves
            torch.save({"positions": [p for p, _ in test], "offsets": offsets, "names": [g.name for g in test_graphs]},
                       OUT / f"drawings_{key}{'_quick' if args.quick else ''}.pt")
            for row in rows:
                row = {"size": size, "radius": radius, "width": width, **row}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            f.flush()
            print(f"{key}: {time.perf_counter() - t:.0f} s | fraction {chosen_curves['fraction']}, "
                  f"lr {chosen_curves['lr']}", flush=True)
    (OUT / ("selection_quick.json" if args.quick else "selection.json")).write_text(json.dumps(selection, indent=1))


def report(args):
    from resolution import on_bound
    from stats import bootstrap_ci, holm, paired_difference_ci, paired_test, rank_biserial
    suffix = "_quick" if args.quick else ""
    with open(OUT / f"rows{suffix}.csv", newline="") as f:
        rows = [{k: (v if k == "edges" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    table = {}
    for r in rows:
        table.setdefault((r["size"], r["radius"], r["width"]), {}).setdefault(r["edges"], []).append(r)
    lines = [f"# Arêtes courbes contre arêtes droites ({OUT.name}/rows{suffix}.csv)", "",
             "Mêmes positions des nœuds (notre modèle affiné à l'écran au réglage) ; seules les arêtes changent. "
             "Moyenne sur les graphes de test [IC à 95 %].", ""]
    tests = []
    for setting, by_edges in sorted(table.items()):
        for kind in by_edges.values():
            kind.sort(key=lambda r: r["graph"])
        lines += [f"## s = {setting[0]:g} px, r = {setting[1]:g} px, w = {setting[2]:g} px", "",
                  "| arêtes | " + " | ".join(MEASURES.values()) + " |", "|---|" + "---|" * len(MEASURES)]
        for kind in ("straight", "curved"):
            cells = []
            for m in MEASURES:
                mean, lo, hi = bootstrap_ci([r[m] for r in by_edges[kind]])
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append(f"| {'droites' if kind == 'straight' else 'courbes'} | " + " | ".join(cells) + " |")
        lines.append("")
        for m in COMPARED:
            curved, straight = [r[m] for r in by_edges["curved"]], [r[m] for r in by_edges["straight"]]
            diff, p = paired_test(curved, straight)
            _, lo, hi = paired_difference_ci(curved, straight)
            tests.append({"setting": setting, "measure": m, "diff": diff, "low": lo, "high": hi,
                          "effect": rank_biserial(curved, straight), "p": p})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p"] = p
        different = p < 0.05 and abs(t["effect"]) >= MIN_EFFECT
        t["verdict"] = ("mieux" if t["effect"] < 0 else "pire") if different else "indécis"
    lines += ["## Courbes − droites, apparié par graphe", "",
              f"Différence moyenne [IC à 95 %] (négatif : les courbes font mieux). r : corrélation bisériale de "
              f"rangs appariée. Wilcoxon signé, p corrigé par Holm sur les {len(tests)} tests. Verdict : « mieux » "
              f"ou « pire » si p < 0,05 et |r| ≥ {MIN_EFFECT:g}.", "",
              "| s | r | w | mesure | courbes − droites | r | p (Holm) | verdict |", "|---|---|---|---|---|---|---|---|"]
    for t in tests:
        s_, r_, w_ = t["setting"]
        lines.append(f"| {s_:g} | {r_:g} | {w_:g} | {MEASURES[t['measure']]} | {t['diff']:+.4f} "
                     f"[{t['low']:+.4f}, {t['high']:+.4f}] | {t['effect']:+.2f} | {t['p']:.2g}{' *' if t['p'] < 0.05 else ''} "
                     f"| {t['verdict']} |")
    hard = {tuple(map(float, h)) for h in HARD}
    lines += ["", "## Décompte des verdicts", "",
              "Nombre de réglages par verdict, sur tous les réglages mesurés puis sur les trois de la première "
              "expérience (choisis parce que les arêtes droites y touchent le plus de nœuds).", "",
              "| réglages | mesure | courbes mieux | courbes pire | indécis | total |", "|---|---|---|---|---|---|"]
    for title, kept in (("tous", tests), ("les trois premiers", [t for t in tests if t["setting"] in hard])):
        for m in COMPARED:
            group = [t["verdict"] for t in kept if t["measure"] == m]
            if group:
                lines.append(f"| {title} | {MEASURES[m]} | {group.count('mieux')} | {group.count('pire')} | "
                             f"{group.count('indécis')} | {len(group)} |")
    selection_path = OUT / f"selection{suffix}.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        bounds = {key: on_bound(chosen) for key, chosen in selection.items()}
        lines += ["", "## Courbes choisies sur la validation", "",
                  f"{sum(map(bool, bounds.values()))} choix sur {len(bounds)} tombent sur une borne de la grille essayée.", "",
                  "| réglage | fraction des poids | pas | sur une borne |", "|---|---|---|---|"]
        lines += [f"| {key} | {chosen['fraction']:g} | {chosen['lr']:g} | {', '.join(bounds[key]) or '–'} |"
                  for key, chosen in selection.items()]
    (OUT / f"results_curves{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sheet(suffix)


def defect_map(positions, offsets, graph, size):
    """(size, size) map of the image defects (metrics/image_quality.defect_mask), grown by 2 px to be seen."""
    from scipy import ndimage
    from curves.render import curve_elements
    from metrics.image_quality import defect_mask
    mask = defect_mask(positions, graph, size, size, edges=curve_elements(positions, offsets, graph, size, size))
    return ndimage.binary_dilation(mask > 0, iterations=2).astype(float)


def sheet(suffix=""):
    """Three test graphs of the first setting: straight edges | curved edges, defects in red."""
    from PIL import Image, ImageDraw, ImageFont
    from curves.render import render_curved
    from data.input import load_graphml
    from data.raster import raster
    from resolution import with_style

    size, radius, width = HARD[0]
    key = f"{size}px_r{radius:g}_w{width:g}"
    path = OUT / f"drawings_{key}{suffix}.pt"
    if not path.exists():
        return
    saved = torch.load(path)
    by_size = sorted(range(len(saved["names"])), key=lambda i: len(saved["positions"][i]))
    picks = [by_size[len(by_size) // 4], by_size[len(by_size) // 2], by_size[-1]]
    scale, pad, top = 2, 12, 30
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for i in picks:
        g = with_style(load_graphml(DATA / f"{saved['names'][i]}.graphml"), radius, width)
        P, off = saved["positions"][i], saved["offsets"][i]
        row = []
        for offsets in (torch.zeros_like(off), off):
            img = render_curved(P, offsets, g, size, size, highlight=defect_map(P, offsets, g, size))
            row.append(Image.fromarray(raster.to_uint8(img)).resize((size * scale, size * scale), Image.NEAREST))
        tiles.append(row)
    W = 2 * size * scale + 3 * pad
    H = len(tiles) * (size * scale + pad) + pad + top
    canvas = Image.new("RGB", (W, H), "#fcfcfb")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 7), "arêtes droites", fill="#0b0b0b", font=font)
    draw.text((2 * pad + size * scale, 7), "arêtes courbes (mêmes nœuds)", fill="#0b0b0b", font=font)
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            canvas.paste(tile, (pad + c * (size * scale + pad), top + r * (size * scale + pad)))
    canvas.save(OUT / f"sheet{suffix}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("checkpoint")
    p_run.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 4))
    p_run.add_argument("--settings", default="all", choices=sorted(SETTINGS),
                       help="all: the 16 settings of the resolution sweep; hard: the three of the first experiment")
    p_run.add_argument("--fractions", default="", help="weight fractions tried, comma separated (curves.experiment.FRACTIONS)")
    p_run.add_argument("--rates", default="", help="learning rates tried, comma separated (curves.experiment.LEARNING_RATES)")
    p_run.add_argument("--quick", action="store_true")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--quick", action="store_true")
    a = parser.parse_args()
    run(a) if a.command == "run" else report(a)
