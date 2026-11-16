# Bayesian linear regression

**Bayesian linear regression**: a posterior over the weights of 20 Gaussian basis features, built in one batch or one look at a time, as a map of a toy one-dimensional world: its mean, its uncertainty band, and a normalised surprise ε² that reads true noise over assumed noise

**Tutorial notebook:** [Bayesian regression with linear basis function models](https://github.com/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression.ipynb), typed along through the straight-line and Gaussian-basis examples, skipping the evidence sections. Then steps 1–8 of my Bench worksheet, on a toy world: one bump at x = 0.7, seen through 20 Gaussian features with noise of standard deviation 0.1.

Credit: this unit follows Martin Krasser's notebook above, from his [bayesian-machine-learning](https://github.com/krasserm/bayesian-machine-learning) repo (Apache-2.0), which follows chapter 3 of Bishop's [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006).
