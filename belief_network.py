"""
A class for representing a Bayesian Belief Network.

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

import numpy as np

from pmf_tensor import PMFTensor


class BeliefNetwork():
    def __init__(self, var_nstates, var_parents, vars_to_update=None):
        self.vars = list(var_nstates.keys())
        self.var_nstates = var_nstates
        self.var_parents = var_parents

        # generate the PMFs
        self.p = {}
        self.phat = {}
        for node in self.vars:
            conditional_vars = {
                node: var_nstates[node]
            }
            conditioning_vars = {
                parent: var_nstates[parent] for parent in var_parents[node]
            }

            # we will generate data from these
            self.p[node] = PMFTensor.rand_init(conditional_vars, conditioning_vars)

            # we will estimate these from the data
            self.phat[node] = PMFTensor.rand_init(conditional_vars, conditioning_vars)

        # if `vars_to_update` not specified, update all
        self.vars_to_update = vars_to_update or self.vars

        # numerical equivalents of `self.vars`
        self.var_idx = {}
        for idx, node in enumerate(self.vars):
            self.var_idx[node] = idx

    def get_joint_pmf(self, which="true"):
        tensor_dict = self.p if which=="true" else self.phat

        product = tensor_dict[self.vars[0]]
        for node in self.vars[1:]:
            product = product.multiply(tensor_dict[node])
        return product

    def sample_data(self, datasetSize = 100, missingProbability = 0.3):
        product = self.get_joint_pmf()

        # flatten the PMF tensor to get a univariate distribution
        pmf_flat = product.pmf.flatten()

        # sample `datasetSize` indices from the flattened PMF
        sample_idx = np.random.choice(len(pmf_flat), datasetSize, p=pmf_flat)

        # unravel the indices to get the corresponding coordinates
        # for the original (i.e. unflattened) PMF tensor
        samples = np.vstack(list(np.unravel_index(sample_idx, product.pmf.shape))).T

        # randomly mask each node with probability `missingProbability`
        mask = np.random.rand(*samples.shape) < missingProbability
        samples[mask] = -1

        return list(product.get_vars()), samples


def add_delta(tensor, sample, sample_vars):
    known_vars = {}
    for var, val in zip(sample_vars, sample):
        if val != -1:
            known_vars[var] = val

    tensor_one_hot_index = []
    for var in tensor.get_vars():
        tensor_one_hot_index.append(known_vars[var])

    tensor.pmf[*tensor_one_hot_index] += 1.0
