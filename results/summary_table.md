
### Fashion-MNIST, clean

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random | 86.28 ± 0.41 | 82.08 ± 0.22 |
| Entropy **(best)** | 87.11 ± 0.48 | 82.80 ± 0.33 |
| BALD | 88.03 ± 0.70 | 82.57 ± 0.68 |
| Coreset | 83.86 ± 0.48 | 80.95 ± 0.35 |
| IRIS (ours) | 87.61 ± 0.65 | 82.65 ± 0.52 |
| IRIS w/o diversity | 86.31 ± 0.74 | 81.32 ± 0.88 |

### Fashion-MNIST, noisy γ=0.2

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random | 83.71 ± 0.54 | 79.61 ± 0.39 |
| Entropy | 84.30 ± 0.47 | 79.20 ± 0.91 |
| BALD | 81.92 ± 1.30 | 77.81 ± 0.46 |
| Coreset | 81.89 ± 0.46 | 78.39 ± 0.46 |
| IRIS (ours) **(best)** | 85.68 ± 0.61 | 80.40 ± 0.38 |
| IRIS w/o reliability | 84.55 ± 0.25 | 79.28 ± 0.48 |

### BloodMNIST, clean

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random | 89.47 ± 0.68 | 83.63 ± 0.98 |
| Entropy | 91.25 ± 0.97 | 82.10 ± 1.33 |
| BALD **(best)** | 91.50 ± 0.73 | 83.94 ± 0.84 |
| Coreset | 85.83 ± 1.27 | 82.10 ± 0.80 |
| IRIS (ours) | 90.62 ± 0.91 | 81.14 ± 0.53 |
| IRIS w/o diversity | 87.67 ± 1.73 | 79.27 ± 0.95 |

### BloodMNIST, noisy γ=0.2

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random **(best)** | 86.94 ± 0.75 | 80.00 ± 0.98 |
| Entropy | 86.52 ± 1.69 | 76.68 ± 1.34 |
| BALD | 83.32 ± 1.08 | 75.25 ± 1.47 |
| Coreset | 87.09 ± 0.45 | 78.97 ± 1.08 |
| IRIS (ours) | 87.82 ± 0.75 | 78.52 ± 0.77 |
| IRIS w/o reliability | 87.35 ± 1.15 | 77.33 ± 0.90 |

### CIFAR-10, clean

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random **(best)** | 74.50 ± 0.46 | 61.99 ± 0.58 |
| Entropy | 73.42 ± 2.41 | 58.92 ± 1.43 |
| BALD | 73.71 ± 1.14 | 59.91 ± 0.53 |
| Coreset | 71.43 ± 0.57 | 59.27 ± 0.51 |
| IRIS (ours) | 70.91 ± 1.71 | 59.26 ± 0.78 |
| IRIS w/o diversity | 68.44 ± 3.15 | 56.67 ± 0.78 |

### CIFAR-10, noisy γ=0.2

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random **(best)** | 63.24 ± 0.55 | 54.00 ± 0.51 |
| Entropy | 62.16 ± 1.29 | 51.47 ± 0.83 |
| BALD | 60.55 ± 1.42 | 49.00 ± 1.28 |
| Coreset | 60.36 ± 0.29 | 49.89 ± 0.39 |
| IRIS (ours) | 62.20 ± 0.30 | 51.69 ± 1.07 |
| IRIS w/o reliability | 62.09 ± 0.40 | 50.97 ± 0.53 |
