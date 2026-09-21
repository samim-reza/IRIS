
### Fashion-MNIST, clean

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random | 86.25 ± 0.36 | 82.00 ± 0.26 |
| Entropy | 86.73 ± 0.54 | 82.42 ± 0.33 |
| BALD | 87.85 ± 0.62 | 82.52 ± 0.70 |
| Coreset | 83.63 ± 0.37 | 80.74 ± 0.56 |
| IRIS (ours) **(best)** | 87.71 ± 0.47 | 82.66 ± 0.67 |
| IRIS w/o diversity | 86.36 ± 0.72 | 81.09 ± 0.79 |

### Fashion-MNIST, noisy γ=0.2

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random | 83.85 ± 0.55 | 79.56 ± 0.29 |
| Entropy | 84.22 ± 0.60 | 79.20 ± 0.70 |
| BALD | 81.89 ± 0.93 | 77.77 ± 0.38 |
| Coreset | 81.76 ± 0.92 | 78.33 ± 0.35 |
| IRIS (ours) **(best)** | 84.82 ± 0.50 | 79.93 ± 0.37 |
| IRIS w/o reliability | 84.12 ± 0.69 | 79.46 ± 0.49 |

### BloodMNIST, clean

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random **(best)** | 89.45 ± 1.03 | 83.70 ± 0.41 |
| Entropy | 91.00 ± 1.16 | 82.55 ± 1.18 |
| BALD | 91.05 ± 0.91 | 83.32 ± 1.30 |
| Coreset | 86.26 ± 1.80 | 81.71 ± 0.77 |
| IRIS (ours) | 91.16 ± 0.77 | 80.95 ± 0.82 |
| IRIS w/o diversity | 86.52 ± 2.30 | 79.01 ± 1.45 |

### BloodMNIST, noisy γ=0.2

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random **(best)** | 86.69 ± 0.91 | 80.10 ± 0.96 |
| Entropy | 86.50 ± 0.91 | 77.22 ± 1.02 |
| BALD | 83.31 ± 0.69 | 75.34 ± 0.92 |
| Coreset | 86.68 ± 0.46 | 78.63 ± 0.61 |
| IRIS (ours) | 88.08 ± 0.56 | 77.39 ± 1.02 |
| IRIS w/o reliability | 86.74 ± 0.96 | 77.07 ± 1.30 |

### CIFAR-10, clean

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random **(best)** | 74.16 ± 0.57 | 62.08 ± 0.88 |
| Entropy | 73.24 ± 1.79 | 60.03 ± 0.95 |
| BALD | 74.07 ± 0.11 | 61.14 ± 0.19 |
| Coreset | 71.39 ± 0.47 | 60.10 ± 0.38 |
| IRIS (ours) | 73.25 ± 1.22 | 59.24 ± 0.42 |

### CIFAR-10, noisy γ=0.2

| Method | Final acc (%) | AUBC (%) |
|---|---|---|
| Random **(best)** | 63.69 ± 0.96 | 53.74 ± 0.38 |
| Entropy | 62.12 ± 3.47 | 51.84 ± 0.37 |
| BALD | 61.98 ± 1.88 | 50.31 ± 1.10 |
| Coreset | 60.15 ± 0.89 | 50.31 ± 0.39 |
| IRIS (ours) | 58.28 ± 1.24 | 51.45 ± 0.74 |
