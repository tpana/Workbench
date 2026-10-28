# workbench

Hand-built implementations of the ML and cognitive-modeling ideas I work with, one unit at a time.

I research cognitive architectures and AI as a Vector Institute scholar and a master's student in Cognitive Science at Carleton University (ANIMUS Lab). My research combines predictive processing, active inference and vector symbolic architectures, and I co-founded TLOL. In this repo I build the underlying tools and machinery myself, from an empty file, all hand-written.

## How this repo works

Every line of code here was typed by hand. Some units follow a tutorial closely, and those are credited below and in the unit's own README. The goal is to build understanding and familiarity with tools in ML and cognitive modelling.

Each unit folder has a short README covering what I built and the sources I followed. Units that follow one lecture series share a parent folder, like `makemore/`, with a README for the series.

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
| 02 | **Gridworld, two agents**: a discrete active inference agent in plain numpy (A, B, C, D arrays, belief updates, expected free energy as risk plus ambiguity), then tabular Q-learning on the same world, compared | B, C | Done |
| 03 | **makemore 1, bigrams**: a character-level bigram language model over 32,033 names, first as a table of counts, then as a one-layer neural net trained by gradient descent to the same negative log-likelihood | A | Done |
| 04 | **makemore 2, MLP**: the MLP language model of Bengio et al. (2003), with learned character embeddings, a tanh hidden layer, minibatch training, a learning-rate sweep and train/dev/test splits | A | Done |
| 05 | **makemore 3, activations and BatchNorm**: diagnosing a deeper MLP's activations and gradients (initialization scale, tanh saturation, Kaiming init), adding batch normalization, and rewriting the network as PyTorch-style layers | A | Planned |
| 06 | **makemore 4, backprop by hand**: manual backpropagation through the whole MLP (cross-entropy, linear layers, BatchNorm, tanh, embeddings), each gradient checked against PyTorch autograd, then training with no `loss.backward()` | A | Planned |

What's next: a GPT from scratch that loads real GPT-2 weights, PPO on CartPole, a predictive coding network beside a backprop twin, an HRR capacity curve, an enumerative program synthesizer for a grid DSL, and an LLM agent loop with no framework.

## Credits

- **01 micrograd** follows Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and his [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) lecture. The design is his, and the original is MIT-licensed.
- **02 Gridworld** follows the ["Active inference from scratch" notebook](https://github.com/infer-actively/pymdp/blob/v0.0.7.1/docs/notebooks/active_inference_from_scratch.ipynb) from [pymdp](https://github.com/infer-actively/pymdp). The Q-learning agent is written from the pseudocode in Section 6.5 of Sutton and Barto's [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.).
- **03–06 makemore** follows parts 1–4 of the makemore lectures in Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), and `names.txt` comes from his [makemore](https://github.com/karpathy/makemore) repo. The design is his, and the original is MIT-licensed.

## Contact

Theo Pana · [LinkedIn](https://www.linkedin.com/in/theo-pana/)
