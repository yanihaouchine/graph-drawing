"""Energy per drawing of each method, for the environmental impact analysis (docs/impact.md).

Each method runs under the carbon tracker (src/tools/carbon.py) on the same test graphs, following the
protocol of src/tools/energy_protocol.py: a block repeats the test graphs until --min-seconds have passed, the
power of the idle process is subtracted, and every method is measured --repeats times, the methods taking
turns so that a drift of the machine is spread over all of them.

Methods. The first three start from the .graphml file, so that what each needs before drawing is counted:
  kk               read the file, Kamada-Kawai (networkx, CPU; its shortest paths included)
  ours             read the file, shortest paths and Laplacian eigenvectors, our model, 12 rounds
  ours+refine      the same, then the team's refinement at 500 px (refine.draw: one graph at a time on the CPU)
  ours (cached)    our model alone, from inputs prepared beforehand: the share of the preparation, by difference
  refine@S         the screen-space refinement alone at image side S (100, 200, 500 px), from Kamada-Kawai
                   drawings made beforehand, by batches of 64: to add to the cost of the starting drawing.
                   Weights and learning rate: those chosen by the sweep for this size if --selection has
                   them, else fraction 0.3 and learning rate 1

    python benchmark/energy.py runs/model.pt [--graphs 200] [--repeats 5] [--min-seconds 30]
                                             [--training-kwh 0.049 --training-kwh 0.062] [--out DIR]
Writes energy.csv (codecarbon rows, one per block) and energy.md in benchmark/out/energy/ (or --out).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

OUT = HERE / "out" / "energy"
DATA = HERE.parent / "data" / "rome"
SELECTION = HERE / "out" / "resolution" / "selection.json"
REFINE_SIZES = (100, 200, 500)
RADIUS, WIDTH = 4.0, 1.0  # style of the refine@S drawings, the team's
LABELS = {"kk": "Kamada-Kawai (fichier → dessin, CPU)", "ours": "Notre modèle (fichier → dessin)",
          "ours+refine": "Notre modèle + affinage de l'équipe (fichier → dessin)",
          "ours-cached": "Notre modèle seul (entrées déjà préparées)",
          **{f"refine@{s}": f"Affinage écran seul à {s} px (paquets de 64)" for s in REFINE_SIZES}}


def refine_parameters(size, selection):
    """(fraction, learning rate) of the screen refinement at this size: the sweep's choice if it has one."""
    chosen = selection.get(f"{size}px_r{RADIUS:g}_w{WIDTH:g}", {}).get("screen")
    return (chosen["fraction"], chosen["lr"]) if chosen else (0.3, 1.0)


def report_lines(summaries, idle, n, device, machine, min_seconds, training_kwh=()):
    """Markdown report. summaries: {method: energy_protocol.summarize(...)}; idle: watts per component."""
    from tools.energy_protocol import break_even

    samples = next(iter(summaries.values()))["samples"]
    lines = [f"# Énergie par dessin ({n} graphes de test de Rome, {device}, {machine})", "",
             f"Chaque méthode est mesurée {samples} fois, par blocs d'au moins {min_seconds:g} s. Énergie nette : "
             f"au-dessus de la puissance du processus au repos (CPU {idle['cpu']:.1f} W, GPU {idle['gpu']:.1f} W, "
             f"RAM {idle['ram']:.1f} W), soustraite sur la durée du bloc. Énergie brute : tout ce que le traceur "
             "attribue au processus, matériel au repos compris. Moyenne ± écart-type entre les mesures.", "",
             "| méthode | temps / dessin | énergie nette / dessin (mWh) | énergie brute / dessin (mWh) |",
             "|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {LABELS.get(name, name)} | {s['ms']:.1f} ms | {s['net_mwh']:.3f} ± {s['net_std']:.3f} | "
                     f"{s['gross_mwh']:.3f} ± {s['gross_std']:.3f} |")
    if training_kwh and {"kk", "ours"} <= set(summaries):
        lines += ["", "## Seuil de rentabilité de l'entraînement face à Kamada-Kawai", "",
                  "Énergie de l'entraînement divisée par ce que le modèle économise par dessin (énergies nettes, "
                  "du fichier au dessin)."]
        saving = summaries["kk"]["net_mwh"] - summaries["ours"]["net_mwh"]
        for kwh in training_kwh:
            drawings, certain = break_even(kwh, summaries["kk"], summaries["ours"])
            result = (f"environ {drawings:,.0f} dessins".replace(",", " ") if math.isfinite(drawings)
                      else "jamais : le modèle ne consomme pas moins par dessin")
            doubt = "" if certain else " (écart inférieur à la dispersion des mesures : signe non établi)"
            lines.append(f"- entraînement de {kwh:g} kWh, économie de {saving:+.3f} mWh par dessin : {result}{doubt}")
    return lines


def main():
    import networkx as nx
    from data.input import load_graphml, split_dataset
    from infer import load_model
    from loss import Loss
    from refine import ROUNDS, STEPS, draw
    from resolution import (anchor, balanced_weights, make_loss, refine_all, rescale_to_frame, term_means,
                            with_style)
    from tools.carbon import read_runs, track
    from tools.energy_protocol import Sample, idle_power, repeat_for, summarize
    from train import REFINE_WEIGHTS

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--graphs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5, help="measures of each method")
    parser.add_argument("--min-seconds", type=float, default=30, help="shortest duration of a measured block")
    parser.add_argument("--idle-seconds", type=float, default=30, help="duration of each idle measure")
    parser.add_argument("--training-kwh", type=float, action="append", default=[],
                        help="energy of a training, for the break-even against Kamada-Kawai (repeatable)")
    parser.add_argument("--selection", type=Path, default=SELECTION, help="selection.json of the resolution sweep")
    parser.add_argument("--out", type=Path, default=OUT, help="folder of energy.csv and energy.md")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "energy.csv"
    csv_path.unlink(missing_ok=True)
    device = args.device
    paths = split_dataset(DATA)[2][:args.graphs]
    model = load_model(Path(args.checkpoint)).to(device)
    refine_loss = Loss(REFINE_WEIGHTS)
    selection = json.loads(args.selection.read_text()) if args.selection.exists() else {}

    def kamada_kawai(path):
        G = nx.read_graphml(path)
        p = nx.kamada_kawai_layout(G)
        return torch.tensor([p[node] for node in G.nodes()], dtype=torch.float32)

    def ours(path, steps):
        return draw(model, load_graphml(path).to(device), refine_loss, ROUNDS, steps)

    # prepared beforehand, outside the measures: inputs of "ours-cached", starting drawings of "refine@S"
    inputs = [load_graphml(path) for path in paths]
    kk_drawings = [kamada_kawai(path) for path in paths]
    workloads = {
        "kk": lambda: [kamada_kawai(path) for path in paths],
        "ours": lambda: [ours(path, 0) for path in paths],
        "ours+refine": lambda: [ours(path, STEPS) for path in paths],
        "ours-cached": lambda: [draw(model, g.clone().to(device), refine_loss, ROUNDS, 0) for g in inputs],
    }
    for size in REFINE_SIZES:
        fraction, lr = refine_parameters(size, selection)
        framed = [rescale_to_frame(p, size, RADIUS) for p in kk_drawings]
        styled = [anchor(with_style(g, RADIUS, WIDTH), p) for g, p in zip(inputs, framed)]
        loss = make_loss("screen", balanced_weights(term_means(list(zip(framed, styled)), "screen", size), fraction))
        workloads[f"refine@{size}"] = (lambda framed=framed, styled=styled, size=size, loss=loss, lr=lr:
                                       refine_all(framed, styled, size, loss, 25, lr, RADIUS, device))

    drawings = {}  # block name: drawings made

    def measured(block, workload=None):
        """One tracked block: the workload repeated for min-seconds, or a sleep (idle) if there is none."""
        with track(f"energy-{block}", csv_path, measure_power_secs=1):
            if workload is None:
                time.sleep(args.idle_seconds)
                drawings[block] = 0
            else:
                drawings[block] = len(paths) * repeat_for(workload, args.min_seconds)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()

    for workload in workloads.values():  # warm-up, not measured: lazy imports, CUDA kernels, caches
        workload()
    measured("idle-before")
    for repeat in range(args.repeats):
        for name, workload in workloads.items():
            measured(f"{name}#{repeat}", workload)
            print(f"{name} #{repeat}: {drawings[f'{name}#{repeat}']} dessins", flush=True)
    measured("idle-after")

    runs = {r["name"].removeprefix("energy-"): r for r in read_runs(csv_path)}
    sample = lambda block: Sample(runs[block]["duration_s"], runs[block]["cpu_kwh"], runs[block]["gpu_kwh"],  # noqa: E731
                                  runs[block]["ram_kwh"], drawings[block])
    before, after = idle_power(sample("idle-before")), idle_power(sample("idle-after"))
    idle = {c: (before[c] + after[c]) / 2 for c in before}
    summaries = {name: summarize([sample(f"{name}#{r}") for r in range(args.repeats)], idle) for name in workloads}
    machine = runs["idle-before"]["gpu"] if device.startswith("cuda") else runs["idle-before"]["cpu"]
    lines = report_lines(summaries, idle, len(paths), device, machine, args.min_seconds, args.training_kwh)
    lines += ["", f"Puissance au repos mesurée avant et après les méthodes : GPU {before['gpu']:.1f} W puis "
                  f"{after['gpu']:.1f} W, CPU {before['cpu']:.1f} W puis {after['cpu']:.1f} W."]
    (args.out / "energy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
