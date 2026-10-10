# Fleet Memory

When one car slows down on a road cell, the next cars could be warned. **Fleet Memory** is a shared, decaying memory of what recent
vehicles experienced on each road cell, tested on **real data from 384 cars** (the Vehicle Energy Dataset: OBD-II logs around Ann Arbor,
Michigan, November 2017 to November 2018, 1.1 million passes through 2,429 road cells). No simulation is involved.

**Task.** A vehicle is about to enter a road cell (a 165 m square seen from one of four headings). Will its pass be *slow*, meaning mean
speed below 60% of the cell's free-flow speed (85th percentile of its training-period passes; cells under 40 km/h free-flow are dropped)?
Time-ordered replay: a pass becomes visible only when it ends, and a vehicle queries before its own pass is folded in (tested).

| predictor | idea |
|---|---|
| static | P(slow) per cell, weekday/weekend and 3-hour block, from the first 150 days |
| last pass | was the most recent visible pass of the cell slow (within 30 min) |
| fleet memory | decayed count of slow passes (half-life 10 min), shrunk toward the static prior |
| fused | logistic regression on the above plus confidence, fitted on days 150 to 240 |

## Results (test: days 240 to 365; intervals resample whole days)

| | average precision | AUROC | Brier |
|---|---|---|---|
| static | 0.485 [0.478, 0.491] | 0.816 | 0.122 |
| last pass | 0.505 [0.498, 0.512] | 0.802 | 0.129 |
| fleet memory | 0.533 [0.525, 0.541] | 0.834 | 0.118 |
| **fused** | **0.541 [0.533, 0.550]** | **0.837** | **0.115** |

![PR curves and fleet size](results/fleet_memory.png)

- **The gain is real but modest**: fused beats static by **+0.057 AP** [+0.051, +0.061]. Most of what a road cell does is already
  predictable from its own history (traffic lights, rush hours).
- **It is concentrated where history cannot help.** On "unexpected" passes (static prior below 10%, 45% of queries, 4.2% of them slow) AP
  rises from 0.071 to 0.228. With fresh evidence it is 0.728 against 0.680; without it, fleet memory is no better than history.
- **It needs a dense fleet.** Fresh evidence exists for only **8%** of queries even with all 384 vehicles; feeding the memory from
  10% of them gives AP 0.493 against static 0.485, and 50% gives 0.515.
- **Warnings**: at the loosest threshold with 50% precision on validation, history warns on 202 per 1000 passes and catches 52% of slow ones;
  the fused model warns on 249 and catches 65%.

Limits: slow passes include recurring stop-and-go, there is no ground truth for incidents, GPS in VED is noisy, and one year of one city
say nothing about others.

## Run

```bash
pip install -e .[dev]
fleetmem download      # VED, 176 MB (needs 7z, or Windows' tar)
fleetmem passes        # about a minute: 22 million records -> 1.5 million passes
fleetmem run           # replay, evaluate, write results/
pytest                 # 11 tests: memory decay, no look-ahead in the replay, labels, buckets
```

Settings are in `configs/default.yaml`. Data: Oh, Leblanc, Peng, "Vehicle Energy Dataset (VED)", IEEE T-ITS 2022 (Apache-2.0,
<https://github.com/gsoh/VED>). The raw data is not redistributed here. Code: MIT.
