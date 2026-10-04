# Results (synthetic fleet simulator)

> All numbers come from the synthetic fleet simulator, not from real vehicle data. Mean ± std over 5 held-out seeds; `n/a` means undefined (e.g. precision with zero warnings).

## Clean sensors

| method | precision | recall | F1 | false-alarm rate | event recall | detection delay (s) | lead time (s) |
|---|---|---|---|---|---|---|---|
| timestamp | 0.196 ± 0.048 | 0.568 ± 0.073 | 0.285 ± 0.049 | 0.2207 ± 0.0910 | 0.743 ± 0.125 | 71 ± 28 | 825 ± 91 |
| can_only | 0.626 ± 0.042 | 0.284 ± 0.055 | 0.386 ± 0.050 | 0.0148 ± 0.0040 | 0.429 ± 0.045 | 555 ± 107 | 612 ± 58 |
| camera_only | 0.507 ± 0.027 | 0.481 ± 0.030 | 0.493 ± 0.018 | 0.0405 ± 0.0055 | 0.600 ± 0.035 | 270 ± 45 | 698 ± 71 |
| fusion | 0.654 ± 0.032 | 0.490 ± 0.103 | 0.555 ± 0.076 | 0.0220 ± 0.0035 | 0.871 ± 0.114 | 788 ± 100 | 523 ± 57 |

## F1 when the camera signal is dropped (fraction of passes)

| method | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| timestamp | 0.285 ± 0.049 | 0.285 ± 0.049 | 0.285 ± 0.049 | 0.285 ± 0.049 | 0.285 ± 0.049 |
| can_only | 0.386 ± 0.050 | 0.386 ± 0.050 | 0.386 ± 0.050 | 0.386 ± 0.050 | 0.386 ± 0.050 |
| camera_only | 0.493 ± 0.018 | 0.493 ± 0.019 | 0.467 ± 0.026 | 0.423 ± 0.037 | 0.000 ± 0.000 |
| fusion | 0.555 ± 0.076 | 0.536 ± 0.082 | 0.475 ± 0.090 | 0.406 ± 0.069 | 0.386 ± 0.050 |

## F1 when the CAN signal is dropped (fraction of passes)

| method | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| timestamp | 0.285 ± 0.049 | 0.301 ± 0.043 | 0.326 ± 0.027 | 0.368 ± 0.062 | 0.000 ± 0.000 |
| can_only | 0.386 ± 0.050 | 0.367 ± 0.054 | 0.344 ± 0.055 | 0.272 ± 0.076 | 0.000 ± 0.000 |
| camera_only | 0.493 ± 0.018 | 0.493 ± 0.018 | 0.493 ± 0.018 | 0.493 ± 0.018 | 0.493 ± 0.018 |
| fusion | 0.555 ± 0.076 | 0.554 ± 0.064 | 0.562 ± 0.046 | 0.531 ± 0.042 | 0.493 ± 0.018 |

## Evidence-grounded recommender (fusion, clean sensors)

- Warned passes that produced a recommendation (rest abstained): 0.981 ± 0.006
- Precision of emitted recommendations: 0.647 ± 0.032 (vs. warning precision 0.654 ± 0.032)
- Emitted recommendations passing citation verification: 1.000 ± 0.000

Citation verification is 1.0 by construction for the template renderer; it becomes a real test once an LLM phrases the claims.
