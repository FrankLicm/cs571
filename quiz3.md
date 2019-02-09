- What is a [transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) in neural networks and how does [BERT](https://arxiv.org/abs/1810.04805) use transformers for the optimization of word sequences?

  A block of transformer is made of multihead attention mechnisim layer and feed forward layer which actually is a kind of fully connected dense neural networks with so called multihead attention mechnism. 

  Bert uses the transfomer encoder to build a bi-directional neural network to train word representations. It has two tasks, the first task is masked lm and the second task is next sentence prediction. By these two tasks, it can generate word representations by using transformer.

  

- What are the key challenges in sentiment analysis (opinion mining) and how do neural network-based approaches handle those better than traditional bag-of-words approaches?

  key challenges:

  1.It is very context sensitive. It is hard to express the meaning of longer phrases

  2.It is quite domain dependent, the same expression in different domain may have different sentiment 

  3.It has order dependences. Different order with the same sentences may generate different sentiment

  Advantage of neural network-based approaches.

  1. The neural network can consider the context of the word which can solve the context problem while bag of words is just collection of words without any context
  2. The neural network can consider the order of the sentences and tokens, while the bag of words has no order information.

  

  

  