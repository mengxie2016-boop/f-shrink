# f-shrink
f-Shrink: A self-adaptive contraction operator that never decays to zero. Only 0.1 worse than Cosine, but keeps learning forever.
# f-Shrink

<div align="center">
  
  **A self-adaptive contraction operator that never decays to zero**
  
  [![Paper I](https://img.shields.io/badge/Theory%20Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.18485110)
  [![Paper II](https://img.shields.io/badge/Application%20Paper-Zenodo-blue)](https://doi.org/10.5281/zenodo.18694170)
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
| Cosine Decay | 0.9969 | 0.0003 | ❌ No |
| **f-Shrink (β=0.035)** | **1.0986** | **0.4745** | ✅ **Yes** |

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
    mode='decay',
    decay=0.035,  # optimal value from experiments
    f=0.4745409995126511
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step()
        optimizer.step()
        scheduler.step()  # one line replacement
```

---

## 🧪 Run Experiments Yourself

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i7TJTGGvxji7sQD0ZESarBJ18Bcbl5oA)

Click the button above to run all f-Shrink experiments in Google Colab.

---

## 📚 Citation

If you use f-Shrink in your research, please cite:

**Theory paper:**

```
Xie, M. (2026). Global Dynamics of a Transcendental Iteration System Based on the Balance Equation. Zenodo.
https://doi.org/10.5281/zenodo.18485110
```

**Application paper:**

```
Xie, M. (2026). f-Shrink: A Self-Adaptive Contraction Operator Derived from Transcendental Dynamics and Its Applications in Deep Learning [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.18694170
```

**BibTeX:**

```bibtex
@article{xie2026global,
  title={Global Dynamics of a Transcendental Iteration System Based on the Balance Equation},
  author={Xie, Meng},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18485110}
}

@dataset{xie2026fshrink,
  title={f-Shrink: A Self-Adaptive Contraction Operator Derived from Transcendental Dynamics and Its Applications in Deep Learning},
  author={Xie, Meng},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18694170}
}
```

---

## 📄 License

MIT License

Copyright (c) 2026 Meng Xie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📬 Contact

- **Author**: Meng Xie
- **Email**: [mengxie2016@gmail.com](mailto:mengxie2016@gmail.com)
- **GitHub**: [@mengxie2016-boop](https://github.com/mengxie2016-boop)

---

<div align="center">
  <b>f-Shrink – Because learning should never stop.</b>
</div>
