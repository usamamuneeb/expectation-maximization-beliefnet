"""
Implementation of Expectation Maximization algorithm for Maximum-Likelihood
estimation of a Bayesian Belief Network.

Copyright 2021-2022 Usama Muneeb

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
import numpy as np

from belief_network import BeliefNetwork, add_delta
from pmf_tensor import PMFTensor

var_nstates = {
    "a": 3,
    "b": 4,
    "c": 5,
    "d": 4,
    "e": 3,
    "f": 4,
    "g": 5,
    "h": 4
}

# make sure there are no cycles!!
var_parents = {
    "a": [],
    "b": [],
    "c": ["a"],
    "d": ["a", "b"],
    "e": ["a", "c"],
    "f": ["b", "d"],
    "g": ["e"],
    "h": ["f"]
}

# user may choose to update only a subset of nodes
vars_to_update = ["a", "b", "c", "d", "e", "f", "g", "h"]

np.random.seed(0)
net = BeliefNetwork(var_nstates, var_parents, vars_to_update)
dataset_size = 1000
sample_vars, samples = net.sample_data(dataset_size, 0.3)
print(samples[:10]) # print first 10 as a sanity check


max_iterations = 10
logLikelihoodAll = []


# the variational distribution Q is not a persistent property of the belief network
# and therefore `q` is not a component of the `BeliefNetwork` class
q = {}
for node in net.vars:
    q[node] = PMFTensor.zero_init(net.p[node].var_nstates, net.p[node].parent_nstates)

for iter in range(max_iterations):
    for node in net.vars:
        q[node].pmf.fill(0)

    logLikelihood = 0

    # E-step
    # use current \hat P and data to accumulatively build distribution Q
    for n in range(dataset_size):
        sample_hidden_vars = [sample_vars[x] for x in np.where(samples[n] == -1)[0]]
        sample_known_vars = [sample_vars[x] for x in np.where(samples[n] >= 0)[0]]

        hidden_vars = {}
        known_vars = {}
        for node in net.p:
            hidden_vars[node] = list(set(sample_hidden_vars) & set(net.p[node].get_vars()))
            known_vars[node] = list(set(sample_known_vars) & set(net.p[node].get_vars()))

        # using the states of known (unmasked) variables, get the
        # belief function over the hidden variables
        # we can either take a slice of P (the joint PMF tensor), or compute the
        # joint of the sliced individual P_i tensors (the former is more efficient)
        hatP = net.get_joint_pmf("estimated")
        hatP_given_y = hatP.get_slice_using_observation(samples[n], sample_vars)

        # get log likelihood for this sample and add to total `logLikelihood`
        logLikelihood = logLikelihood + np.log(np.sum(hatP_given_y.pmf))

        hatP_given_y.normalize()

        # accumulatively build Q for the nodes we want to update
        for node in net.vars_to_update:

            if len(hidden_vars[node])==0:
                # NOTE: Q is configured to be a conditional PMF tensor
                # while accumulatively building Q here, we ignore this fact over here
                # (it will become a valid conditional PMF tensor in the M-step
                # when we call `normalize()` on it)
                add_delta(q[node], samples[n], sample_vars) # Q_i = Q_i + OneHot(sample)
            else:
                known_var_nstates = {
                    var: net.var_nstates[var] for var in known_vars[node]
                }

                # this is ingredient 1 of 2 for constructing the variational PMF Q_i
                # a deterministic (one hot) PMF over the known variables
                delta_y = PMFTensor.zero_init(known_var_nstates)
                add_delta(delta_y, samples[n], sample_vars) # see NOTE preceding `add_delta` above

                # this is ingredient 2 of 2 for constructing the variational PMF 
                # a belief over the missing variables
                marginalTensor = hatP_given_y.get_marginal(hidden_vars[node])

                # we take the outer product (tensor product) of both ingredients
                q[node].pmf = q[node].pmf + delta_y.multiply(
                    marginalTensor,
                    q[node].get_vars() # custom order of nodes, to match q[node] for easy addition
                ).pmf

    # M-step
    # use Q to update \hat P
    for node in net.vars_to_update:
        q[node].normalize()

        net.phat[node].pmf = q[node].pmf.copy() # set \hat P to normalized Q

    # store this epoch's log-likelihood for plotting
    print("Epoch: {}, Log-Likelihood: {}".format(iter, logLikelihood))
    logLikelihoodAll.append(logLikelihood)

    if (iter > 1):
        if abs(logLikelihoodAll[-1] - logLikelihoodAll[-2]) < 0.1:
            break

plt.plot(list(range(iter+1)), logLikelihoodAll)
plt.xlabel('Epoch')
plt.ylabel('Log Likelihood')
plt.show()

print("Complete")

np.set_printoptions(precision=2, suppress=True)

for var in net.vars:
    print(f"p[{var}]\n" + str(net.p[var].pmf))
    print(f"phat[{var}]\n" + str(net.phat[var].pmf))
    # print(f"diff[{var}]\n" + str(net.phat[var].pmf - net.p[var].pmf))
    print("\n")
