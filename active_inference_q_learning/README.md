# Gridworld, two agents

**Gridworld, two agents**: a discrete active inference agent in plain numpy (A, B, C, D arrays, belief updates, expected free energy as risk plus ambiguity), then tabular Q-learning on the same world, compared

Requires `inferactively-pymdp==0.0.7.1`. The tutorial uses pymdp's original numpy API, which the 1.x JAX rewrite removed.

Credit: this unit follows the "Active inference from scratch" notebook from [pymdp](https://github.com/infer-actively/pymdp). The Q-learning agent is written from the pseudocode in Section 6.5 of Sutton and Barto's [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.).
