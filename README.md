# Fleet Memory

A shared road-condition map built from many vehicles' observations. When one vehicle sees something that changes a stretch of road (less grip, an obstruction), the vehicles behind it get a warning with the evidence cited.

## What I did
- Wrote a simulator that replays multi-vehicle drives with injected road-condition changes and known ground truth.
- Built a memory that keeps a condition score for each road segment. Old evidence fades with a half-life, and a vehicle is only warned by vehicles that passed before it.
- Compared four ways of detecting a change:

| method | what it uses |
|---|---|
| `timestamp` | one CAN threshold, flag held for a fixed time |
| `can_only` | learned model on CAN signals (slip ratio, deceleration) |
| `camera_only` | learned model on camera embeddings compared to the segment's normal look |
| `fusion` | both, falling back to one or none if a sensor is missing |

- Added a recommender that turns a warning into actions, cites the vehicles, signals and camera frames behind it, and says nothing when the evidence is weak.

## Results (5 held-out seeds, simulated fleet)

| method | precision | recall | F1 | false-alarm rate | event recall | detection delay (s) | lead time (s) |
|---|---|---|---|---|---|---|---|
| timestamp | 0.196 | 0.568 | 0.285 | 0.221 | 0.743 | 71 | 825 |
| can_only | 0.626 | 0.284 | 0.386 | 0.015 | 0.429 | 555 | 612 |
| camera_only | 0.507 | 0.481 | 0.493 | 0.041 | 0.600 | 270 | 698 |
| fusion | 0.654 | 0.490 | 0.555 | 0.022 | 0.871 | 788 | 523 |

- Fusion has the best F1 and event recall. It also keeps working when a sensor is dropped: without the camera it matches CAN-only (0.386), and without CAN it matches camera-only (0.493).
- Fusion is slower to detect (788 s vs 71 s for the timestamp method) but has about 10 times fewer false alarms (0.022 vs 0.221).

![F1 under sensor dropout](results/dropout_f1.png)

Full tables: [results/results.md](results/results.md).

## How it was built

```
Observation (vehicle, time, segment, CAN, camera embedding)
   -> Detector: log-likelihood ratio that the segment changed
   -> DecayMemory: per-segment log-odds that fade over time
   -> warn if probability and evidence are high enough
   -> Recommender: cites the evidence, or abstains
```

All settings are in `configs/default.yaml`. The full spec is in `docs/BLUEPRINT.md`.

## Tech stack
Python, NumPy, PyYAML, matplotlib, pytest (53 tests).
