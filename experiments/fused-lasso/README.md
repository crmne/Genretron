# Experiments for the thesis

## Goals
Understand the effect of Fused Lasso on:

* MNIST
* GTZAN
* Softmax Regression
* Convolutional Neural Networks
* L1

Therefore, we need the following experiments:

1) MNIST + l1, MNIST + l1+FL (logistic regression)
2) GTZAN + L1, GTZAN L1+FL (logistic regression)
3) GTZAN + L1, GTZAN L1+FL (convnet)

within these I would do something like

a) find best L1 over some range of values
b) find best lambda FL over range (with 0 <= L1 <= best L1) (i.e. it may prefer a smaller l1 since the FL is also a type of regulariser).

## Experiments

1. Standard MNIST model + L1, softmax regression. Find best L1 over some range of values.
2. MNIST with horizontal FL + L1, softmax. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.
3. MNIST with vertical FL + L1, softmax. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.
4. MNIST with horizontal FL + L1, softmax. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.

1. Standard GTZAN model + L1, softmax regression. Find best L1 over some range of values.
2. GTZAN with horizontal FL + L1, softmax. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.
3. GTZAN with vertical FL + L1, softmax. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.
4. GTZAN with horizontal FL + L1, softmax. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.

1. Standard GTZAN model + L1, convnets. Find best L1 over some range of values.
2. GTZAN with horizontal FL + L1, convnets. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.
3. GTZAN with vertical FL + L1, convnets. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.
4. GTZAN with horizontal FL + L1, convnets. See what is the effect of FL on the weights. Multiple experiments with different fl lambdas, where FL lambda >= L1 lambda.
