# GPT tokenizer

**GPT tokenizer**: byte-level byte-pair encoding from scratch, trained on UTF-8 bytes with encode and decode, then GPT-2's regex pre-split and special tokens, checked against tiktoken's GPT-2 and GPT-4 tokenizers and compared with SentencePiece

**Lecture:** [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) · [lecture notebook](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L) (Colab) · [lecture notes](https://github.com/karpathy/minbpe/blob/master/lecture.md) · [exercise](https://github.com/karpathy/minbpe/blob/master/exercise.md)

Credit: this unit follows the lecture above from Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and his [minbpe](https://github.com/karpathy/minbpe) repo. Byte-pair encoding for subwords comes from Sennrich et al.'s [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) (2016), and the byte-level variant from Radford et al.'s [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019). The design is his, and minbpe is MIT-licensed.
