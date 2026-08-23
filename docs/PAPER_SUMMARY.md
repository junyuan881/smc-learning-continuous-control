# Paper summary

**Paper:** *Reinforcement Learning in Continuous Action Spaces through Sequential Monte Carlo Methods*  
**Authors:** Alessandro Lazaric, Marcello Restelli, Andrea Bonarini  
**Venue:** NIPS 2007 / Advances in Neural Information Processing Systems 20

## Core idea

The actor does not parameterize a Gaussian or neural policy. Instead, for every state it stores a small weighted set of continuous action samples (particles). The critic estimates the Q-value of those samples. When a sample's Q-value improves, its importance weight increases. If the weights become too concentrated, the actor resamples and locally moves particles so that resolution automatically increases near promising actions.

## Algorithm in this project

1. Discretize the continuous state into a sparse tabular state key.
2. Initialize `N` action particles uniformly over `[-90°, 90°]`, all with weight `1/N`.
3. Sample the action according to particle weights.
4. Update its critic Q-value with SARSA.
5. Update the action weight with `exp(Delta Q / tau)` and normalize.
6. Compute effective sample size `1/sum(w_i^2)`.
7. If `ESS/N < sigma`, systematic-resample and locally move particles using uniform kernels.

This implements the distinctive contribution of the paper: continuous action search without a global parametric policy shape assumption.
