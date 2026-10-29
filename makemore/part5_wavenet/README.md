# makemore 5, WaveNet

**makemore 5, WaveNet**: the MLP rebuilt from `torch.nn`-style modules (`Embedding`, `FlattenConsecutive`, `Sequential`), its context widened from 3 to 8 characters and fused two at a time in a WaveNet-style tree, with BatchNorm fixed for (B, T, C) tensors

**Lecture:** [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) · [lecture notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)

Credit: this unit follows the lecture above from Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), which takes its hierarchical architecture from DeepMind's [*WaveNet: A Generative Model for Raw Audio*](https://arxiv.org/abs/1609.03499) (van den Oord et al., 2016). The design is his, and the original is MIT-licensed.
