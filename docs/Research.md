# SOTA for MNIST

- https://beta.hyper.ai/en/sota/tasks/graph-classification/benchmark/graph-classification-on-mnist

# SOTA Research Summary

Model Type                    |Accuracy(%) | Error(%)  | Notes
------------------------------|------------| ----------| ---------------------------
SOTA Hybrid (CNN + ViT)       | 99.97%     | 0.03%     | A hybrid model approach.
SOTA Ensemble                 | 99.87%     | 0.13%     | Ensemble (multiple models).
Capsule Networks              |~99.84%     | 0.16%     | Single model from Hinton
SOTA Single CNN Model         | 99.83%     | 0.17%     | Single CNN highly tuned
Standard CNN                  |~99.2%      | 0.8%      | Single CNN basic
Multi-layer Perceptron (MLP)  |~98.1%      | 1.9%      | Classical NN method.
Logistic Regression           |~92.5%      | 7.5%      | Classical Logit method.

# Decision

Run with Hintons Capsule network (https://arxiv.org/abs/1710.09829). Its good enough and from a trusted source so its likely stable and generalizes well..
