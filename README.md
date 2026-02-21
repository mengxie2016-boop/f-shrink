# f-shrink
f-Shrink: A self-adaptive contraction operator that never decays to zero. Only 0.1 worse than Cosine, but keeps learning forever.
# f-Shrink

<div align="center">
  
  **A self-adaptive contraction operator that never decays to zero**
  
  [![Paper I](https://img.shields.io/badge/Paper%20I-Theory-blue)](https://doi.org/10.5281/zenodo.18485110)
  [![Paper II](https://img.shields.io/badge/Paper%20II-Application-blue)](https://doi.org/10.5281/zenodo.18694170)
  [![PyTorch](https://img.shields.io/badge/PyTorch-≥1.9-orange)](https://pytorch.org)
  [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
  [![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00)](https://colab.research.google.com/drive/1i7TJTGGvxji7sQD0ZESarBJ18Bcbl5oA)
  
</div>

---

## 📌 Overview

f-Shrink is a novel self-adaptive contraction operator derived from a transcendental iteration system. 

The core operator is defined as:

\[
\mathcal{F}(x) = \operatorname{sgn}(x) \cdot \sqrt{|x|} \cdot f^{\,|x|/(2f)}, \quad f = 0.4745409995126511
\]

where \(f\) is the unique positive real solution to \(\ln f + \frac{\pi}{2}f = 0\).

**Unlike traditional schedulers that decay to zero, f-Shrink never stops learning – it converges to the fixed point \(f\) and stays there forever.**

---

## 🔥 Key Results

| Method | Final Loss (CIFAR-10) | Final Learning Rate | Can keep learning? |
|--------|----------------------|---------------------|---------------------|
| Cosine Decay | **0.9969** | 0.0003 | ❌ No |
| **f-Shrink (β=0.035)** | 1.0986 | **0.4745** | ✅ **Yes** |

**Only 0.1 worse than Cosine, but keeps learning forever.**

---

## ✨ Features

- ✅ **Never decays to zero** – learning rate stabilizes at \(f \approx 0.4745\)
- ✅ **One-line replacement** for PyTorch's Cosine scheduler
- ✅ **No hyperparameter tuning** – just set decay rate (optimal β=0.035)
- ✅ **Theoretical guarantee** – rigorous convergence proof
- ✅ **Works with any optimizer** – SGD, Adam, AdamW, etc.

---

## 📦 Installation

```bash
pip install f-shrink
```

Or install from source:

```bash
git clone https://github.com/mengxie2016-boop/f-shrink.git
cd f-shrink
pip install -e .
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from f_shrink import FShrinkScheduler

# Your model and optimizer
model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Replace Cosine scheduler with f-Shrink
scheduler = FShrinkScheduler(
    optimizer, 
    mode='decay',      # 'decay' for smooth convergence to f
    decay=0.035,       # optimal value from experiments
    f=0.4745409995126511
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step()
        optimizer.step()
        scheduler.step()  # one line replacement
```

### Available Modes

```python
# Fast mode: directly apply F(x) each step
scheduler = FShrinkScheduler(optimizer, mode='fast')

# Smooth mode: blend between current lr and F(lr)
scheduler = FShrinkScheduler(optimizer, mode='smooth', alpha=0.1)

# Decay mode (recommended): exponentially decay to f
scheduler = FShrinkScheduler(optimizer, mode='decay', decay=0.035)
```

---

## 📊 Optimal Parameters

Based on extensive experiments on CIFAR-10:

| Mode | Parameter | Optimal Value | Final Loss | Final LR |
|------|-----------|---------------|------------|----------|
| Cosine | T_max | depends | 0.9969 | 0.0003 |
| f-Shrink (decay) | β = 0.035 | **0.035** | **1.0986** | **0.4745** |
| f-Shrink (decay) | β = 0.05 | - | 1.131 | 0.4745 |
| f-Shrink (decay) | β = 0.03 | - | 1.1177 | 0.4745 |
| f-Shrink (decay) | β = 0.04 | - | 1.2058 | 0.4745 |

**Recommended: use `decay=0.035` for best trade-off between accuracy and convergence speed.**

---

## 🔬 Theoretical Background

f-Shrink originates from the study of a complex dynamical system with two dual spaces:

- **Z-space** (prototype): \(z_{n+1} = (z_n + e^{-\pi z_n/2})/2\)
- **U-space** (theoretically proven): \(u_{n+1} = \sqrt{u_n} \cdot e^{-\pi u_n/4}\)

The operator \(\mathcal{F}(x)\) is the real projection of the U-space iteration, inheriting its guaranteed convergence properties.

For details, see the full papers:

- **Theory**: [Xie, M. (2026). *Global Dynamics of a Transcendental Iteration System Based on the Balance Equation*. Zenodo.](https://doi.org/10.5281/zenodo.18485110)
- **Applications**: [Xie, M. (2026). *f-Shrink: A Self-Adaptive Contraction Operator Derived from Transcendental Dynamics and Its Applications in Deep Learning*. Zenodo.](https://doi.org/10.5281/zenodo.18694170)

---

## 🧪 Run Experiments Yourself

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i7TJTGGvxji7sQD0ZESarBJ18Bcbl5oA)

Click the button above to run all f-Shrink experiments in Google Colab.

---

## 📈 Use Cases

| Scenario | Why f-Shrink |
|----------|--------------|
| **Continual Learning** | Model must keep adapting to new data |
| **Reinforcement Learning** | Exploration rate never decays to zero |
| **Fine-tuning LLMs** | Stable small LR for long-term optimization |
| **Autonomous Driving** | Environment constantly changing |
| **Recommendation Systems** | User preferences evolve over time |

---

## 📚 Citation

If you use f-Shrink in your research, please cite:

**Theory paper:**
```bibtex
@article{xie2026global,
  title={Global Dynamics of a Transcendental Iteration System Based on the Balance Equation},
  author={Xie, Meng},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18485110}
}
```

**Application paper:**
```bibtex
@dataset{xie2026fshrink,
  title={f-Shrink: A Self-Adaptive Contraction Operator Derived from Transcendental Dynamics and Its Applications in Deep Learning},
  author={Xie, Meng},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18694170}
}
```

Or in plain text:

```
Xie, M. (2026). Global Dynamics of a Transcendental Iteration System Based on the Balance Equation. Zenodo.
https://doi.org/10.5281/zenodo.18485110

Xie, M. (2026). f-Shrink: A Self-Adaptive Contraction Operator Derived from Transcendental Dynamics and Its Applications in Deep Learning [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.18694170
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Author**: Meng Xie
- **Email**: [mengxie2016@gmail.com](mailto:mengxie2016@gmail.com)
- **GitHub**: [@mengxie2016-boop](https://github.com/mengxie2016-boop)

---

<div align="center">
  <b>f-Shrink – Because learning should never stop.</b>
</div>
