# makemore

**makemore**: character-level language models that learn from 32,033 names and generate new ones, built up over four units from a bigram count table to an MLP with BatchNorm, backpropagated by hand

| # | Unit | Folder |
|---|---|---|
| 03 | makemore 1, bigrams | [`part1_bigrams/`](part1_bigrams/) |
| 04 | makemore 2, MLP | [`part2_mlp/`](part2_mlp/) |
| 05 | makemore 3, activations and BatchNorm | [`part3_batchnorm/`](part3_batchnorm/) |
| 06 | makemore 4, backprop by hand | [`part4_backprop/`](part4_backprop/) |

All four units train on [`names.txt`](names.txt), one name per line, from Karpathy's [makemore](https://github.com/karpathy/makemore) repo. Each notebook reads it from its unit folder as `../names.txt`. The units share one environment: PyTorch and matplotlib.

Credit: these units follow parts 1–4 of the makemore lectures in Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). The design is his, and the original is MIT-licensed.
