# makemore

**makemore**: character-level language models that learn from 32,033 names and generate new ones, built up over five units from a bigram count table, through an MLP with BatchNorm, to a WaveNet-style hierarchical model

| # | Unit | Folder |
|---|---|---|
| 03 | makemore 1, bigrams | [`part1_bigrams/`](part1_bigrams/) |
| 04 | makemore 2, MLP | [`part2_mlp/`](part2_mlp/) |
| 05 | makemore 3, activations and BatchNorm | [`part3_batchnorm/`](part3_batchnorm/) |
| 06 | makemore 4, backprop by hand | [`part4_backprop/`](part4_backprop/) |
| 07 | makemore 5, WaveNet | [`part5_wavenet/`](part5_wavenet/) |

All five units train on [`names.txt`](names.txt), one name per line, from Karpathy's [makemore](https://github.com/karpathy/makemore) repo. Each notebook reads it from its unit folder as `../names.txt`. The units share one environment: PyTorch and matplotlib.

Credit: these units follow parts 1–5 of the makemore lectures in Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). The design is his, and the original is MIT-licensed.
