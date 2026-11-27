"""Energy protocol: long enough, net of idle, repeated. No tracker involved."""
import math

import pytest

from tools.energy_protocol import Sample, break_even, idle_power, net_kwh, repeat_for, summarize


def sample(seconds, cpu_w, gpu_w, ram_w=10.0, drawings=0):
    """A block that drew constant powers (W) for `seconds`."""
    kwh = lambda watts: watts * seconds / 3600 / 1000  # noqa: E731
    return Sample(seconds, kwh(cpu_w), kwh(gpu_w), kwh(ram_w), drawings)


def test_repeat_for_runs_until_the_time_is_reached():
    now = [0.0]
    calls = []

    def workload():
        calls.append(1)
        now[0] += 4.0  # each run takes 4 s of the fake clock

    assert repeat_for(workload, 10, clock=lambda: now[0]) == 3  # 12 s >= 10 s
    assert repeat_for(workload, 0, clock=lambda: now[0]) == 1  # at least once
    assert len(calls) == 4


def test_idle_power_recovers_the_watts():
    idle = idle_power(sample(20, cpu_w=0.5, gpu_w=52.0))
    assert idle == pytest.approx({"cpu": 0.5, "gpu": 52.0, "ram": 10.0})
    with pytest.raises(ValueError):
        idle_power(Sample(0, 0, 0, 0))


def test_net_energy_removes_the_idle_gpu():
    """The case of the first measure: Kamada-Kawai on the CPU while an idle GPU draws 52 W."""
    idle = idle_power(sample(30, cpu_w=0.0, gpu_w=52.0))
    kk = sample(3.55, cpu_w=39.3, gpu_w=52.0, drawings=200)  # the GPU does nothing more than idle
    assert net_kwh(kk, idle) == pytest.approx(39.3 * 3.55 / 3600 / 1000)  # the CPU alone
    assert net_kwh(kk, idle) < 0.4 * kk.gross_kwh  # most of the gross energy was the idle GPU and the RAM
    # a component below its idle level counts for zero, not for a negative energy
    assert net_kwh(sample(10, cpu_w=5.0, gpu_w=40.0), idle) == pytest.approx(5.0 * 10 / 3600 / 1000)


def test_summarize_per_drawing():
    idle = idle_power(sample(30, cpu_w=0.0, gpu_w=50.0, ram_w=10.0))
    runs = [sample(60, cpu_w=36.0, gpu_w=50.0, drawings=1000), sample(60, cpu_w=72.0, gpu_w=50.0, drawings=1000)]
    s = summarize(runs, idle)
    # net: 36 W and 72 W of CPU for 60 s over 1000 drawings = 0.6 and 1.2 mWh
    assert s["net_mwh"] == pytest.approx(0.9) and s["net_std"] == pytest.approx(0.6 / math.sqrt(2))
    assert s["gross_mwh"] == pytest.approx(0.9 + 1.0)  # + 60 W of idle GPU and RAM for 60 s = 1 mWh
    assert s["ms"] == pytest.approx(60.0) and s["samples"] == 2
    assert summarize(runs[:1], idle)["net_std"] == 0.0
    with pytest.raises(ValueError):
        summarize([sample(60, 36, 50)], idle)  # no drawings


def test_break_even():
    kk = {"net_mwh": 0.50, "net_std": 0.01}
    ours = {"net_mwh": 0.30, "net_std": 0.01}
    drawings, certain = break_even(0.05, kk, ours)  # 50 Wh / 0.2 mWh
    assert drawings == pytest.approx(250_000) and certain
    # the model costs more per drawing: never repaid
    assert break_even(0.05, ours, kk) == (math.inf, True)
    # a saving within the spread of the measures: a number, flagged as not established
    drawings, certain = break_even(0.05, {"net_mwh": 0.31, "net_std": 0.02}, ours)
    assert drawings == pytest.approx(5_000_000) and not certain
