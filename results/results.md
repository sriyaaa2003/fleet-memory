1,142,925 passes through 2,429 road cells by 384 vehicles. Slow-pass rate: train 20.6%, val 19.2%, test 19.1%.

| test days, all passes | AP [95% CI] | AUROC [95% CI] | Brier |
|---|---|---|---|
| static (history only) | 0.485 [0.478, 0.491] | 0.816 [0.812, 0.821] | 0.122 |
| last pass | 0.505 [0.498, 0.512] | 0.802 [0.799, 0.806] | 0.129 |
| fleet memory | 0.533 [0.525, 0.541] | 0.834 [0.831, 0.837] | 0.118 |
| fused | 0.541 [0.533, 0.550] | 0.837 [0.834, 0.841] | 0.115 |

Fused minus static: AP +0.057 [+0.051, +0.061], AUROC +0.022 [+0.019, +0.023].

Fresh evidence (decayed pass count at least the configured level) exists for 8% of queries. 'Unexpected' passes (static prior below 0.10) are 45% of queries and 4.2% of them are slow.

| subset | static AP | last AP | memory AP | fused AP |
|---|---|---|---|---|
| fresh evidence | 0.680 | 0.651 | 0.720 | 0.728 |
| no fresh evidence | 0.441 | 0.423 | 0.441 | 0.440 |
| unexpected (prior < 0.10) | 0.071 | 0.201 | 0.225 | 0.228 |

| memory fed by this share of vehicles | static AP | memory AP | fused AP | share with fresh evidence |
|---|---|---|---|---|
| 10% | 0.485 | 0.492 | 0.493 | 1% |
| 25% | 0.485 | 0.494 | 0.496 | 2% |
| 50% | 0.485 | 0.510 | 0.515 | 3% |
| 100% | 0.485 | 0.533 | 0.541 | 8% |

| warning rule (threshold from validation days) | threshold | precision | recall | warnings per 1000 passes |
|---|---|---|---|---|
| static (history only) | 0.400 | 0.493 | 0.524 | 202.4 |
| fleet memory | 0.371 | 0.493 | 0.630 | 243.7 |
| fused | 0.309 | 0.495 | 0.647 | 249.3 |
