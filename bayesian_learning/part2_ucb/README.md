# Where to look, UCB

**Where to look, UCB**: the unit 10 map choosing its next look where μ + κσ is largest, scored by how many of 30 looks land on the object, against random, raster, greedy and uncertainty-only search and across a sweep of κ

**Tutorials:** the Upper Confidence Bound section of [*Exploring Bayesian Optimization*](https://distill.pub/2020/bayesian-optimization/) (Distill), which writes the rule as μ + λσ, and the loop in [Krasser's Bayesian optimization notebook](https://github.com/krasserm/bayesian-machine-learning/blob/dev/bayesian-optimization/bayesian_optimization.ipynb) for its shape. His surrogate is a Gaussian process and his rule is expected improvement; here they are the unit 10 map and UCB. Then steps 9–12 of my Bench worksheet.

Credit: this unit follows Apoorv Agnihotri and Nipun Batra's [*Exploring Bayesian Optimization*](https://distill.pub/2020/bayesian-optimization/) (Distill, 2020, CC BY 4.0) and Martin Krasser's notebook above, from his [bayesian-machine-learning](https://github.com/krasserm/bayesian-machine-learning) repo (Apache-2.0).
