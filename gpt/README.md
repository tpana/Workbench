# GPT from scratch

**GPT from scratch**: a character-level, decoder-only Transformer trained on Tiny Shakespeare, built up from a bigram baseline through masked self-attention, multi-head attention, feed-forward layers, residual connections, LayerNorm and dropout

**Lecture:** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) · [lecture notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-) (Colab) · [lecture code](https://github.com/karpathy/ng-video-lecture) (`bigram.py`, `gpt.py`)

Trains on [`input.txt`](input.txt), Tiny Shakespeare: 1.1 million characters of Shakespeare's plays, from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repo.

Credit: this unit follows the lecture above from Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), which implements the decoder from Vaswani et al.'s [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017). The design is his.
