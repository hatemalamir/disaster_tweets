# Disaster Tweets

Disaster Tweets is a [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started). "In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t".

Sentament classification is an important and common task in NLP. Instead of using a ready-to-use off-the-shelf pretrained model, I took the time to build things up from scratch, with a detail-oriented language, C++, and experiment with different architectures and data preparation techniques. The goal is better understading of the inner working of those architectures and techniques, and also greater familiarity with PyTorch and its C++ API.

I started with an implementation of a similar task in the wonderful book [Hands-On Machine Learning with C++](https://www.packtpub.com/product/hands-on-machine-learning-with-c/9781789955330), Ch 11. It was based on LSTMs. I needed to fit the code to my use case and also fix different parts to work with the current version of PyTorch.
