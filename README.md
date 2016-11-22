# matlab-bagging-ensemble-neural-network
A simple bagging ensemble neural network wrote in MATLAB

This is a CSC578 school project. And the main goal of this project is adding Nesterov Momentum, dropout and ensemble learning with bagging to the Neural Network from project 2.

1. Nesterov momentum can be considered as an upgrade version of standard momentum. Use ball rolling down a hill example, nesterov momentum is makeing the ball much smarter that it can have a notion of where it is going so that it knows to slow down before the hill slopes up again.

2. Dropout is one of most common ways for reducing overfitting in neural networks as well as a regularization technique to prevent complex co-adaptations on train dataset.

3. Bootstrap Aggregation (or Bagging) is training each model independently and then we combine the results from each model in the ensemble vote with equal weight. In order to promote model variance, each basic model is using a randomly drawn subset of the training data, different initial weights and biases as well as dropout.
