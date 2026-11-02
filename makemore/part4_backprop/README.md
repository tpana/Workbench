# makemore 4, backprop by hand

**makemore 4, backprop by hand**: manual backpropagation through the MLP's cross-entropy loss and output layer, one intermediate tensor at a time from the loss back to the hidden activations, each gradient matching PyTorch autograd exactly

This unit covers the first stretch of the lecture's exercise 1, from the loss back to the hidden layer `h`.

**Lecture:** [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) · [lecture notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb)

Credit: this unit follows the lecture above from Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). The design is his, and the original is MIT-licensed.
