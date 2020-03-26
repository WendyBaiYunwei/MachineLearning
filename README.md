# MachineLearning
Machine Learning Projects

SplitNN implementation using Pytorch and MNIST dataset.
SplitNN link: https://arxiv.org/abs/1812.00564

The implementation is tweaked to a multi-processing version with multiple pairs of server and client.

Assuming that the number of GPUs is equal to the number of server-client pairs, each pair of server and client is assigned to one GPU for simultaneous training.
