"""benchmark/energy.py: the report and the choice of the refinement parameters (the measures need a tracker,
a model and the data; they are run on the server)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("energy", ROOT / "benchmark" / "energy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(net, std=0.01, gross=1.0):
    return {"net_mwh": net, "net_std": std, "gross_mwh": gross, "gross_std": std, "ms": 20.0, "samples": 5}


IDLE = {"cpu": 0.5, "gpu": 52.0, "ram": 10.0}


def report(summaries, training_kwh=(0.05,)):
    return load_script().report_lines(summaries, IDLE, 200, "cuda", "RTX", 30, training_kwh)


def test_report_table_and_break_even():
    lines = report({"kk": summary(0.50), "ours": summary(0.30), "refine@100": summary(0.20)})
    text = "\n".join(lines)
    assert "mesurée 5 fois" in text and "GPU 52.0 W" in text
    assert "| Kamada-Kawai (fichier → dessin, CPU) | 20.0 ms | 0.500 ± 0.010 | 1.000 ± 0.010 |" in lines
    assert "| Affinage écran seul à 100 px (paquets de 64) | 20.0 ms | 0.200 ± 0.010 | 1.000 ± 0.010 |" in lines
    assert lines[-1] == "- entraînement de 0.05 kWh, économie de +0.200 mWh par dessin : environ 250 000 dessins"


def test_report_says_never_when_the_model_saves_nothing():
    lines = report({"kk": summary(0.24), "ours": summary(0.38)})
    assert lines[-1].endswith("jamais : le modèle ne consomme pas moins par dessin")
    assert "-0.140 mWh" in lines[-1]


def test_report_flags_a_saving_within_the_spread():
    lines = report({"kk": summary(0.31, std=0.02), "ours": summary(0.30, std=0.02)})
    assert lines[-1].endswith("(écart inférieur à la dispersion des mesures : signe non établi)")


def test_report_without_training_energy_has_no_break_even():
    assert not any("rentabilité" in line for line in report({"kk": summary(0.5), "ours": summary(0.3)}, ()))


def test_refine_parameters_follow_the_sweep():
    script = load_script()
    selection = {"200px_r4_w1": {"screen": {"fraction": 3.0, "lr": 0.5}}}
    assert script.refine_parameters(200, selection) == (3.0, 0.5)
    assert script.refine_parameters(100, selection) == (0.3, 1.0)  # not in the sweep: the previous values
