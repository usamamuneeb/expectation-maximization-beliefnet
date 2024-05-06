## Expectation-Maximization in Bayesian Belief Networks

This repository accompanies the blog post [Expectation-Maximization (EM) Algorithm and application to Bayesian Belief Networks](https://usamamuneeb.github.io/articles/expectation-maximization-algorithm).

The code requires Python 3.9+ and can be executed as

```bash
python em_belief_network.py
```

In `em_belief_network.py`, we define the topology of the belief network using two dictionaries: one that represents the number of possible states of each random variable, and another that maps each variable to a list containing the names of random variables it is conditioned upon. Using this information, a `BeliefNetwork` object is constructed with randomly initialized PMF tensors for each node.

These PMF tensors will then be used to sample data from belief network, which we will then use to estimate the original PMFs via EM.

**Hidden Variables**: When sampling data, the user can also specify a probability with which certain variables are masked (hidden). When doing EM, we will maintain a belief over the hidden variables (which will be combined with the states of the known variables).

This implementation performs individual node updates for each random variable (i.e. corresponds to Algorithm 2 in the blog post) and the user may choose to update a subset of the nodes. With minor changes, it can be modified to implement Algorithm 1 (which maintains a joint PMF to represent the belief network).
