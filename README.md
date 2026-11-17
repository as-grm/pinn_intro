# From Deep Learning to Physics-Informed Neural Networks

A hands-on introduction to **Physics-Informed Neural Networks (PINNs)** for engineers.
The repository starts with the fundamentals of deep learning (a multi-layer perceptron on MNIST,
written once from scratch in NumPy and once in PyTorch) and then applies the same ideas to three
engineering problems solved with PINNs in PyTorch.

Every notebook is self-contained: theory, code and validation against a reference or analytical
solution are in one place.

**Author:** Aleksander GRM — ULFPP, University of Ljubljana, Slovenia
([aleksander.grm@fpp.uni-lj.si](mailto:aleksander.grm@fpp.uni-lj.si))

---

## Repository structure

```
pinn_intro/
├── 01_intro_NN/                          Part I  – deep-learning foundations (MNIST)
│   ├── MNIST_MLP_numpy.ipynb             MLP from scratch: forward pass, back-propagation, SGD/Adam, dropout
│   ├── MNIST_MLP_pytorch.ipynb           twin notebook: the same experiments in PyTorch (nn.Module, Autograd)
│   ├── mnist_report.pdf                  written report accompanying the two notebooks
│   └── mnist_slides.pdf                  slide summary of the MNIST study
├── 02_cases/                             Part II – PINN case studies
│   ├── 01_2DOF_double_pendulum.ipynb     hybrid PINN: coupled nonlinear ODEs + noisy data
│   ├── 02_Duffing_inverse_PINN.ipynb     inverse PINN: identify an unknown physical parameter
│   └── 03_1D_heat_transfer.ipynb         forward PINN: steady conduction, physics only, LR schedulers
├── LICENSE                               Apache License 2.0
└── README.md
```

---

## Part I — Deep-learning foundations (`01_intro_NN/`)

Two **twin notebooks** classify hand-written digits from MNIST (60 000 training / 10 000 test images,
28×28 pixels) with a fully-connected MLP. The section numbers, hyper-parameters and loss definitions are
identical, so you can read the two side by side and see exactly what a framework does for you.

| Notebook | What you learn |
|---|---|
| `MNIST_MLP_numpy.ipynb` | Every operation written by hand in **NumPy**: neuron and layer, activations, MSE / cross-entropy loss, back-propagation, gradient check, SGD / mini-batch / momentum / Adam, weight initialisation, dropout, train/validation/test split, evaluation metrics |
| `MNIST_MLP_pytorch.ipynb` | The same model and experiments in **PyTorch**: tensors on CPU / CUDA / ROCm / Apple MPS, `nn.Module`, Autograd, `torch.optim`, comparison of the Autograd gradient with the manual one |

Experiments in both notebooks: (A) stochastic gradient descent, (B) mini-batch gradient descent,
(C) dropout on vs. off over 5 seeds with checkpoint selection, (D) a modern MLP on the full training set,
followed by an analysis of the final model (per-class accuracy, confusions, misclassified images).

Result of the final model (MLP `[784, 256, 128, 10]`, 235 146 parameters, 15 epochs), as stored in the notebook outputs:

| Implementation | Test accuracy |
|---|---|
| NumPy  | 98.16 % (9 816 / 10 000) |
| PyTorch | 98.29 % (9 829 / 10 000) |

The MNIST files are downloaded automatically on the first run into `./data/MNIST/raw/`
(the NumPy notebook uses the same folder layout as `torchvision.datasets.MNIST`, so the data are shared).

---

## Part II — PINN case studies (`02_cases/`)

A PINN is an ordinary neural network whose loss also contains the **residual of the governing equation**,
evaluated at collocation points. The derivatives in the residual are computed exactly with
**automatic differentiation**, so no mesh is needed. The three cases show the three main ways a PINN is used:

| # | Notebook | Physics | PINN mode | Validation |
|---|---|---|---|---|
| 1 | `01_2DOF_double_pendulum.ipynb` | Double pendulum — two coupled nonlinear ODEs (Lagrange), $t\in[0,5]$ s | **Hybrid**: initial conditions + physics + noisy angle measurements | `scipy` `solve_ivp` (DOP853) reference |
| 2 | `02_Duffing_inverse_PINN.ipynb` | Forced Duffing oscillator $m\ddot x + c\dot x + kx + \alpha x^3 = F\cos\Omega t$, $t\in[0,30]$ s | **Inverse**: cubic stiffness $\alpha$ (optionally also damping $c$) is a trainable parameter | `solve_ivp` (DOP853) reference + true parameter value |
| 3 | `03_1D_heat_transfer.ipynb` | 1-D steady conduction with heat source $kT'' + q = 0$, Dirichlet + Neumann BCs | **Forward**: physics only, no temperature data | Analytical solution $T(x)=T_0+\frac{q}{k}\left(Lx-\frac{x^2}{2}\right)$ |

### Case 1 — Double pendulum (hybrid PINN)
One network maps normalised time to both angles $(\theta_1,\theta_2)$. The loss combines initial conditions,
the ODE residuals at $N_f = 1000$ collocation points and $N_{data} = 250$ noisy measurements
(noise σ = 0.005 rad). Default: 20 000 Adam epochs, learning rate halved every quarter of training, seed 123.
Shows input/output normalisation and how to balance the weights of the loss terms.

### Case 2 — Duffing oscillator (inverse PINN)
The unknown $\alpha$ is a `torch.nn.Parameter` that is trained together with the network weights.
The parameters get their own, larger learning rate (`lr_params = 1e-2` vs. `lr = 1e-3`), so they
converge much faster. Result stored in the notebook (α only, initial guess 0, 10 000 epochs):
**α̂ = 1.0121** (true 1.0, error 1.21 %), relative L2 error of $x(t)$ = 4.8·10⁻³.
Suggested experiments: a far initial guess (`alpha_init = 5.0`), two unknowns (`identify_c = True`),
fewer data points, more noise, or `lambda_data = 0` (then α is no longer identifiable).

### Case 3 — 1-D heat conduction (forward PINN)
Trained from physics and boundary conditions alone ($N_f = 100$ points, 8000 epochs, seed 42).
Three output strategies are available (`bc_type`): raw output, shifted output, or a hard Dirichlet BC
$\hat T = T_0 + x\,N(x)$. The same problem is trained with **five learning-rate schedulers** from identical
initial weights. Results stored in the notebook:

| Scheduler | max \|error\| [K] | rel. L2 |
|---|---|---|
| none (constant) | 7.3·10⁻² | 7.3·10⁻⁵ |
| step | 4.2·10⁻² | 9.5·10⁻⁵ |
| cosine | 1.0·10⁻¹ | 2.3·10⁻⁴ |
| plateau | 7.3·10⁻² | 7.3·10⁻⁵ |
| exponential | 4.8 | 1.1·10⁻² |

Lesson: decaying the learning rate too early freezes training. Here, exponential decay ends about two orders of magnitude worse than the other schedules.

> The numbers above come from the outputs saved in the notebooks. Your results will differ slightly with hardware (CPU vs. GPU) and library versions.

---

## Getting started

### Requirements
- Python 3.10+ (the notebooks were run with Python 3.12)
- `numpy`, `matplotlib`, `tqdm` (all notebooks)
- `torch`, `torchvision` (PyTorch MNIST notebook)
- `torch`, `scipy` (PINN case studies)
- Jupyter (`jupyterlab` or `notebook`)

### Installation
```bash
git clone https://github.com/as-grm/pinn_intro.git
cd pinn_intro
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib tqdm jupyterlab
pip install torch torchvision        # see https://pytorch.org for the CUDA / ROCm build
jupyter lab
```

A GPU is optional. The PyTorch notebooks select CUDA / ROCm or Apple MPS when available and fall back to the CPU otherwise; the NumPy notebook runs on the CPU.
The PINN cases run in seconds to a few minutes on a laptop CPU.

### Suggested order
1. `01_intro_NN/MNIST_MLP_numpy.ipynb` — understand every gradient by hand
2. `01_intro_NN/MNIST_MLP_pytorch.ipynb` — see how PyTorch automates it
3. `02_cases/03_1D_heat_transfer.ipynb` — the simplest PINN (forward, physics only)
4. `02_cases/01_2DOF_double_pendulum.ipynb` — a system of ODEs with data (hybrid)
5. `02_cases/02_Duffing_inverse_PINN.ipynb` — identifying unknown physics (inverse)

---

## References
- M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, J. Comput. Phys. **378** (2019) 686–707.
- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
- A. Trask, *Grokking Deep Learning*, Manning, 2019.
- S. Cuomo *et al.*, *Scientific machine learning through physics-informed neural networks: where we are and what's next*, J. Sci. Comput. **92** (2022) 88.
- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-based learning applied to document recognition*, Proc. IEEE **86** (1998) 2278–2324.

## License
Released under the [Apache License 2.0](LICENSE).

## Citation
If you use this material in teaching or research, please cite the repository:

```bibtex
@misc{grm_pinn_intro_2026,
  author       = {Grm, Aleksander},
  title        = {From Deep Learning to Physics-Informed Neural Networks},
  year         = {2026},
  howpublished = {\url{https://github.com/as-grm/pinn_intro}}
}
```

