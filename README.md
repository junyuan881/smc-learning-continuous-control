# SMC-Learning for Continuous-Action Reinforcement Learning

**Student:** 謝濬遠  
**Student ID:** 114024511

A clean, runnable reproduction/demo project for:

> Alessandro Lazaric, Marcello Restelli, Andrea Bonarini, **“Reinforcement Learning in Continuous Action Spaces through Sequential Monte Carlo Methods,”** NIPS 2007.

The project reconstructs the paper's boat-control experiment and implements the core **SMC-learning** actor-critic algorithm, together with the comparison baselines used in the paper.

## What is included

- Paper boat environment with the published nonlinear dynamics and reward zones.
- SMC-learning with:
  - continuous action particles,
  - importance-weight update,
  - effective sample size (ESS),
  - systematic resampling,
  - local uniform-kernel particle movement,
  - SARSA critic.
- Discrete SARSA with 5, 10, 20, and 40 actions.
- Two-tiling action-CMAC baseline with 2.25° effective resolution (80 action candidates).
- Continuous-action Q-learning interpolation baseline with 40 anchors.
- Fast particle-mechanism demo, a compact boat demo, and a 100,000-episode paper-scale runner.
- Plotting of learning curves, boat trajectories, and SMC particle distributions.
- Unit tests and a project verification script.
- Explicit reproduction notes for details that the paper does not specify.

## Repository structure

```text
.
├── CITATION.cff
├── configs/
│   └── paper_boat.json
├── docs/
│   ├── PAPER_SUMMARY.md
│   └── REPRODUCTION_NOTES.md
├── notebooks/
│   └── demo.ipynb
├── results/
│   └── demo/                 # verified example outputs included
├── scripts/
│   ├── run_particle_demo.py
│   ├── run_demo.py
│   ├── run_paper_scale.py
│   └── verify_project.py
├── src/
│   ├── agents/
│   │   ├── continuous_q.py
│   │   ├── sarsa.py
│   │   ├── smc_learning.py
│   │   └── tile_coding.py
│   ├── environment.py
│   ├── plotting.py
│   ├── training.py
│   └── utils.py
├── tests/
│   └── test_core.py
├── references.bib
├── requirements.txt
└── README.md
```

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the fast SMC particle demo

This is a small continuous-action example that makes the paper's importance-weighting, ESS, resampling, and particle-moving mechanism easy to see:

```bash
python scripts/run_particle_demo.py
```

It finishes quickly and writes `particle_convergence.png`, `particle_evolution.png`, and a JSON summary.

## Run the compact boat demo

```bash
python scripts/run_demo.py --episodes 500
```

This runs SMC-learning with 5/10 particles and SARSA with 5/40 actions, then writes:

- `results/runs/demo/learning_curves.png`
- `results/runs/demo/smc10_trajectories.png`
- `results/runs/demo/smc_particles.png`
- `results/runs/demo/rewards.csv`
- `results/runs/demo/summary.json`

To include the two slower comparison baselines:

```bash
python scripts/run_demo.py --episodes 500 --all-baselines
```

## Run the paper-scale comparison

The published Figure 2 reaches **100,000 episodes**. The configuration in `configs/paper_boat.json` reproduces the published hyperparameters and uses 10 bins for each of the two explicitly presented boat coordinates (`x`,`y`).

```bash
python scripts/run_paper_scale.py --episodes 100000 --seeds 3
```

For a quicker paper-parameter smoke run:

```bash
python scripts/run_paper_scale.py --episodes 5000 --seeds 1 --skip-slow-baselines
```

## Validate the repository

```bash
pytest -q
python scripts/verify_project.py
```

## Paper parameters reproduced

| Item | Value |
|---|---:|
| Current force `fc` | 1.25 |
| Inertia `I` | 0.1 |
| Max speed `sMAX` | 2.5 |
| Desired speed `sD` | 1.75 |
| Proportional coefficient `p` | 0.9 |
| Quay | `(200, 110)` |
| Success-zone width | 0.2 |
| Viability-zone width | 20 |
| State representation | x/y, 10 bins each |
| Discount `gamma` | 0.99 |
| Initial learning rate / decay | 0.5 / 0.01 |
| SARSA temperature / decay | 3.0 / 0.0001 |
| SMC ESS ratio `sigma` | 0.95 |
| SMC temperature / decay | 25.0 / 0.0005 |
| Continuous-Q epsilon / decay | 0.4 / 0.005 |

The paper's decay rule is implemented as:

```text
x(N) = x(0) / (1 + delta_x * N)
```

## Reproduction scope

This is intentionally **not presented as a bit-for-bit recovery of the authors' original implementation**. Some numerical implementation choices are absent from the 8-page paper, including exact starting y coordinates, full state-variable discretization bounds, CMAC details, and the exact boat adaptation of Continuous Q-learning. All such assumptions are listed in [`docs/REPRODUCTION_NOTES.md`](docs/REPRODUCTION_NOTES.md).

The uploaded paper also refers to mini-golf and swing-up pendulum experiments in an appendix, but that appendix is not present in the supplied 8-page file. This repository therefore does not invent unsupported appendix experiments.

## Reference

See `references.bib`. Official paper: https://proceedings.neurips.cc/paper/2007/hash/0f840be9b8db4d3fbd5ba2ce59211f55-Abstract.html

