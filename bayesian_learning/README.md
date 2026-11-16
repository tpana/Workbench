# Bayesian learning

**Bayesian learning**: a Bayesian linear regression learner, built up over three units: its posterior over weights, then where to sample next by upper confidence bound, then forgetting

| # | Unit | Folder | Follows |
|---|---|---|---|
| 10 | Bayesian linear regression | [`part1_linear_regression/`](part1_linear_regression/) | Krasser's notebook, then worksheet steps 1–8 |
| 11 | Where to look, UCB | [`part2_ucb/`](part2_ucb/) | Distill's UCB section and the shape of Krasser's loop, then steps 9–12 |
| 12 | Forgetting | [`part3_forgetting/`](part3_forgetting/) | Worksheet steps 13–21, with Shin and Lee (2020) for background |

All three share one toy world: a single bump on [0, 1], seen through 20 Gaussian features with noisy looks. Each unit types along with a tutorial notebook as far as one fits, then continues with my Bench worksheet, whose 21 steps run across all three units. Unit 12 has no tutorial, so the worksheet is its tutorial. The units share one environment: NumPy, SciPy, matplotlib and scikit-learn.

Credit: units 10 and 11 follow Martin Krasser's [bayesian-machine-learning](https://github.com/krasserm/bayesian-machine-learning) notebooks (Apache-2.0) and Agnihotri and Batra's [*Exploring Bayesian Optimization*](https://distill.pub/2020/bayesian-optimization/) (Distill, 2020, CC BY 4.0). Unit 12's background reading is Shin and Lee's [survey of forgetting in recursive least squares](https://arxiv.org/abs/2004.03910) (2020).
