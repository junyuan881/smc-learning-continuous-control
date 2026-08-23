# Reproduction notes

This repository is a **transparent reconstruction**, not a claim that the exact private 2007 experiment code was recovered.

## What is directly specified by the paper

The implementation follows the paper for:

- Continuous action range: `[-90°, 90°]`.
- Boat dynamics parameters: `fc=1.25`, `I=0.1`, `sMAX=2.5`, `sD=1.75`, `p=0.9`.
- Quay location `(200, 110)`, success-zone width `0.2`, viability-zone width `20`.
- State discretization: 10 intervals per state variable in paper-scale mode.
- Learning parameters: `alpha0/delta_alpha = 0.5/0.01`, `gamma=0.99`, SARSA `tau0/delta_tau=3/0.0001`, SMC `sigma=0.95`, SMC `tau0/delta_tau=25/0.0005`, Continuous-Q `epsilon0/delta_epsilon=0.4/0.005`.
- Parameter decay: `x(N)=x(0)/(1+delta_x*N)`.
- SMC actor: finite action samples and normalized importance weights.
- Weight update from Equation (2), ESS from Equation (3), systematic resampling, and local uniform-kernel movement.
- Comparison sets: SARSA with 5/10/20/40 actions; SMC with 5/10 samples; tile coding with effective 80-action resolution; Continuous Q-learning with 40 anchors.

## Details not fully specified in the 8-page paper

The following are therefore explicit reconstruction choices:

1. **Initial left-bank points.** The text says the boat starts at one of the points shown in the figure, but exact numeric values are not listed. We use `y={20,40,...,180}`.
2. **Initial dynamic variables.** We initialize `delta=0`, `Omega=0`, and `speed=sD`.
3. **Which variables form the tabular state.** Section 4.1 explicitly introduces the boat coordinates `x,y` and later says each state variable is discretized into 10 intervals, but it does not spell out a state vector. The default reproduction therefore discretizes `x,y` only. A `full` diagnostic mode including delta/Omega/speed is also implemented.
4. **The y-dynamics equation contains `s_(t-1)`.** The default `dynamics_variant="paper"` implements it literally. `dynamics_variant="s_next"` is available as a diagnostic alternative rather than silently correcting the equation.
5. **Meaning of zone "width".** We interpret width as total width centered at y=110; thus the success half-width is 0.1 and viability half-width is 10.
6. **D(x,y).** The paper says it decreases linearly from +10 to -10 relative to distance from the success zone. We implement exactly that description within the viability zone.
7. **Tile-coding details.** The paper gives two tilings and 2.25° resolution but not the full CMAC feature map. The provided baseline tile-codes the action dimension while retaining tabular discrete states.
8. **Continuous Q-learning details in the boat implementation.** We reproduce the three-anchor continuous interpolation rule from Millán et al. (2002), but use TD(0) and the boat paper's discrete states because its ITPM/eligibility-trace setup is not specified in the SMC paper.
9. **Number of independent runs / smoothing.** Figure 2 does not fully specify these plotting details. The long script lets you choose the number of seeds and saves raw reward arrays.
10. **Appendix experiments.** The uploaded 8-page paper says mini-golf and swing-up pendulum results are in an appendix, but that appendix is not included. They are intentionally not invented here.

Because of these missing details, the repository aims to reproduce the **algorithmic behavior and qualitative comparison**, not pixel-match the published curves.
