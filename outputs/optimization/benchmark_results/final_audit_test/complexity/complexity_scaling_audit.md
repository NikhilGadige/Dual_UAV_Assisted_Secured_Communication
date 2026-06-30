# Complexity Scaling Audit

## Power-Law Fits

| Parameter | c | p | R^2 | T = c * N^p |
|-----------|---|----|-----|-------------|
| N_RIS | 2.2915e+00 | 0.1737 | 0.5708 | T = 2.2915e+00 * N^0.1737 |
| N_time | 2.0195e-01 | 1.7928 | 0.9955 | T = 2.0195e-01 * N^1.7928 |
| N_eve | 3.0072e+00 | 0.1814 | 0.9364 | T = 3.0072e+00 * N^0.1814 |
| N_veh | 2.9683e+00 | 0.2154 | 0.9091 | T = 2.9683e+00 * N^0.2154 |

## FLAGS

  FLAG: N_RIS has R^2=0.5708 < 0.9

## Interpretation

- p close to 1: linear scaling
- p close to 2: quadratic scaling
- p > 2: super-quadratic (may need optimization)
