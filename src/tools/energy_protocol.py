"""Protocol of the energy-per-drawing measures (benchmark/energy.py), apart from the tracker so that it can be
tested without measuring anything.

A measure is a Sample: the duration and the energy of each component (CPU, GPU, RAM) of a block that made a
known number of drawings. Three rules, each answering a flaw of a single short measure:

- long enough: a block repeats its workload until `min_seconds` have passed (repeat_for), so that the tracker,
  which samples the power about once a second, takes many samples;
- net of idle: the power drawn by the process when it does nothing (idle_power, from a Sample of a sleep) is
  subtracted over the duration of the block (net_kwh). A GPU that is allocated but unused draws tens of watts;
  charged to a method that runs on the CPU, they exceed the energy of the method itself;
- repeated: a method is measured several times and reported as a mean and a spread (summarize).
"""
import math
import statistics
import time
from dataclasses import dataclass

COMPONENTS = ("cpu", "gpu", "ram")
WH_PER_KWH = 1000.0


@dataclass
class Sample:
    """One measured block: its duration, the energy of each component (kWh), the drawings it made."""
    duration_s: float
    cpu_kwh: float
    gpu_kwh: float
    ram_kwh: float
    drawings: int = 0

    @property
    def gross_kwh(self):
        return self.cpu_kwh + self.gpu_kwh + self.ram_kwh


def repeat_for(workload, min_seconds, clock=time.perf_counter):
    """Run workload() until min_seconds have passed, at least once. Returns the number of runs."""
    start, runs = clock(), 0
    while True:
        workload()
        runs += 1
        if clock() - start >= min_seconds:
            return runs


def idle_power(sample):
    """Power of each component (W) during a block where the process did nothing: {component: watts}."""
    if sample.duration_s <= 0:
        raise ValueError("the idle sample has no duration")
    hours = sample.duration_s / 3600
    return {c: WH_PER_KWH * getattr(sample, f"{c}_kwh") / hours for c in COMPONENTS}


def net_kwh(sample, idle):
    """Energy of the block above the idle power (kWh): for each component, its energy minus the idle power
    over the duration of the block, not below zero (a component measured under its idle level did no work)."""
    hours = sample.duration_s / 3600
    return sum(max(0.0, getattr(sample, f"{c}_kwh") - idle[c] * hours / WH_PER_KWH) for c in COMPONENTS)


def summarize(samples, idle):
    """Per drawing, over the repeated samples of one method: mean and standard deviation of the net and gross
    energy (mWh), mean time (ms), number of samples. The standard deviation is 0 for a single sample."""
    if not samples or any(s.drawings <= 0 for s in samples):
        raise ValueError("every sample needs its number of drawings")
    net = [1e6 * net_kwh(s, idle) / s.drawings for s in samples]
    gross = [1e6 * s.gross_kwh / s.drawings for s in samples]
    spread = lambda values: statistics.stdev(values) if len(values) > 1 else 0.0  # noqa: E731
    return {"net_mwh": statistics.fmean(net), "net_std": spread(net),
            "gross_mwh": statistics.fmean(gross), "gross_std": spread(gross),
            "ms": statistics.fmean(1000 * s.duration_s / s.drawings for s in samples), "samples": len(samples)}


def break_even(training_kwh, baseline, model):
    """Number of drawings after which training the model is repaid by what it saves per drawing against the
    baseline. baseline, model: summaries (summarize). Returns (drawings, certain):
    drawings = training energy / (baseline - model) on the net means, math.inf if the model saves nothing;
    certain = False when the saving is smaller than the spread of the two measures (one standard deviation
    each), in which case the sign of the saving itself is not established."""
    saving = baseline["net_mwh"] - model["net_mwh"]
    certain = abs(saving) > baseline["net_std"] + model["net_std"]
    if saving <= 0:
        return math.inf, certain
    return 1e6 * training_kwh / saving, certain
