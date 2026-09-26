# workbench

Hand-built implementations of the ML and cognitive-modeling ideas I work with, one unit at a time.

I research cognitive architectures and AI as a Vector Instutite scholar and a master's student in Cognitive Science at Carleton University (ANIMUS Lab). My research combines predictive processing, active inference and vector symbolic architectures, and I co-founded TLOL. In this repo I build the underlying tools and machinery myself, from an empty file, all hand-written.

## How this repo works

Every line of code here was typed by hand. Some units follow a tutorial closely, and those are credited below and in the unit's own README. The goal is to build understanding and familiarity with tools in ML and cognitive modelling.

Each unit folder has a short README covering what I built and the sources I followed.

## Tracks

| Track | Covers |
|---|---|
| A. ML core | Autograd, PyTorch, language modeling, transformers, finetuning, evals, agents |
| B. RL and planning | Tabular RL, deep RL (DQN, PPO), RL for LLMs, model-based planning |
| C. Probabilistic and cognitive modeling | Bayesian inference, probabilistic programs, predictive coding, discrete active inference, VSA/HRR, cognitive architectures |
| D. Neuro-symbolic | Program synthesis, proposer–verifier design |

## Units

| # | Unit | Tracks | Status |
|---|---|---|---|
| 01 | **micrograd**: a scalar autograd engine (`Value`, topological-sort `backward()`), then `Neuron`, `Layer` and `MLP` trained on a toy dataset | A | Done |
| 02 | **Gridworld, two agents**: a discrete active inference agent in plain numpy (A, B, C, D arrays, belief updates, expected free energy as risk plus ambiguity), then tabular Q-learning on the same world, compared | B, C | In progress |

What's next: a GPT from scratch that loads real GPT-2 weights, PPO on CartPole, a predictive coding network beside a backprop twin, an HRR capacity curve, an enumerative program synthesizer for a grid DSL, and an LLM agent loop with no framework.

## Credits

- **01 micrograd** follows Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and his [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) lecture. The design is his, and the original is MIT-licensed.
- **02 Gridworld** follows the "Active inference from scratch" notebook from [pymdp](https://github.com/infer-actively/pymdp). The Q-learning agent is written from the pseudocode in Section 6.5 of Sutton and Barto's [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.).

## Contact

Theo Pana · [LinkedIn](https://www.linkedin.com/in/theo-pana/)
