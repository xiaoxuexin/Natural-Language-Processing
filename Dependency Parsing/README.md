# Dependency Parsing

Despite its seeming chaos, natural language has lots of structure. We’ve already seen some of this structure in part of speech tags and how the order of parts of speech are predictive of what kinds of words might come next (via their parts of speech). In Homework 4, you’ll get a deeper view of this structure by implementing a dependency parser. We covered this topic in Week 10 of the course and it’s covered extensively in Speech & Language Processing chapter 13, if you want to brushup.1 Briefly,dependencyparsingidentifiesthesyntaticrelationshipbetweenwordpairsto create a parse tree, like the one seen in Figure 1.

In Homework 4, you’ll implement the shift-reduce neural dependency parser of Chen and Man- ning [2014],2 which was one of the first neural network-based parser and is quite famous. Thank- fully, its neural network is also fairly straight-forward to implement. We’ve provided the parser’s skeleton code in Python 3 that you can use to finish the implementation, with comments that out- line the steps you’ll need to finish. And, importantly, we’ve provided a lot of boilerplate code that handles loading in the training, evaluation, and test dataset, and converting that data into a representation suitable for the network. Your part essentially boils down to two steps: (1) fill in the implementation of the neural network and (2) fill in the main training loop that processes each batch of instances and does backprop. Thankfully, unlike in Homeworks 1 and 2, you’ll be leveraging the miracles of modern deep learning libraries to accomplish both of these!

1. Gain a working knowledge of the PyTorch library, including constructing a basic network, using layers, dropout, and loss functions.

2. Learn how to train a network with PyTorch

3. Learn how to use pre-trained embeddings in downstream applications

4. Gain a basic familiarity with dependency parsing and how a shift-reduce parser works.

You’ll notice that most of the learning goals are based on deep learning topics, which is the primary focus of this homework. The skills you learn with this homework will hopefully help you with your projects and (ideally) with any real-world situation where you’d need to build a new network. However, you’re welcome—encouraged, even!—to wade into the parsing setup and evaluation code to understand more of how this kind of model works.

#A book with PyTorch and NLP

https://learning.oreilly.com/library/view/natural-language-processing/9781491978221/
