"""
A class for representing a Probability Mass Function with arbitrary number of
variables.

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


class PMFTensor():
    def __init__(self, pmf, var_axes, var_nstates, parent_nstates = {}, normalize=False):
        self.pmf = pmf
        self.var_axes = var_axes
        self.var_nstates = var_nstates
        self.parent_nstates = parent_nstates

        if normalize:
            self.normalize()

    def normalize(self):
        normalization_axes = tuple([self.var_axes[var] for var in self.var_nstates.keys()])
        self.pmf /= np.sum(self.pmf, normalization_axes, keepdims=True)

    @classmethod
    def rand_init(cls, var_nstates, parent_nstates = {}):
        # combined dictionary with lengths of both conditional and conditioning variable axes
        all_nstates = var_nstates | parent_nstates # requires Python 3.9+

        # a dictionary mapping variable (key) to its axis (value)
        var_axes = {}
        for idx, var in enumerate(all_nstates):
            var_axes[var] = idx

        # generate a random tensor
        shape = list(all_nstates.values())
        pmf = np.random.rand(*shape)
        return cls(pmf, var_axes, var_nstates, parent_nstates, normalize=True)

    @classmethod
    def zero_init(cls, var_nstates, parent_nstates = {}):
        # combined dictionary with lengths of both conditional and conditioning variable axes
        all_nstates = var_nstates | parent_nstates # requires Python 3.9+

        # a dictionary mapping variable (key) to its axis (value)
        var_axes = {}
        for idx, var in enumerate(all_nstates):
            var_axes[var] = idx

        # generate a zero tensor
        shape = list(all_nstates.values())
        pmf = np.zeros(shape)
        return cls(pmf, var_axes, var_nstates, parent_nstates) # default: normalize=False


    def multiply(self, other, product_axes_order = None):
        if product_axes_order is None:
            # get the UNION of the variables contained in both PMFs
            combined_vars = list(set(self.get_vars()) | set(other.get_vars()))

            # In order for `sample_data` output to be deterministic for a specific NumPy
            # seed, the order of joint PMF axes must be consistent.
            # The order of joint PMF axes relies on this `multiply` routine.
            # Since set UNION (done above) does not have deterministic ordering (unless
            # Python hash seed is set), we will sort it.
            combined_vars.sort()
        else:
            combined_vars = list(product_axes_order)

        # besides serving as the axes of this variable in the product PMF, the values of
        # this dictionary will also be used the unique axes identifiers during `np.einsum`
        axes_combined = {}
        for idx, var in enumerate(combined_vars):
            axes_combined[var] = idx

        # get outer product using `np.einsum`
        axes_left = [axes_combined[node] for node in self.var_axes]
        axes_right = [axes_combined[node] for node in other.var_axes]
        axes_product = [axes_combined[node] for node in combined_vars]

        prod = np.einsum(self.pmf, axes_left, other.pmf, axes_right, axes_product)

        # also get `var_nstates` and `parent_nstates` for the newly created tensor
        prod_var_nstates = self.var_nstates | other.var_nstates

        # P(A, B | C, D) x P(C | D)
        # should give
        # P(A, B, C | D)
        # we take a UNION of the variables on the left of | in each tensor
        # we do likewise for the variables on the right of |
        # except that any variables that appear in the left side of | in product tensor
        # should be excluded from right side (now that we do have a PMF available over them)
        prod_parent_nstates = self.parent_nstates | other.parent_nstates
        prod_parent_nstates = dict(
            filter(lambda item: item[0] not in prod_var_nstates, prod_parent_nstates.items())
        )
        product = PMFTensor(prod, axes_combined, prod_var_nstates, prod_parent_nstates)
        return product

    def get_vars(self):
        return list(self.var_axes.keys())

    def print_description(self):
        def dict_to_str(mydict, printvals=False): 
            if printvals:
                # the values (from `self.*_nstates` dictionaries) can be compared with
                # the shape of the `self.pmf` object for debugging purposes
                mydict = mydict.items()
                mydict = list(map(lambda pair : f"{pair[0]}:{pair[1]}", mydict))
            else:
                # otherwise, printing without values prints in a more "mathematical" manner
                mydict = mydict.keys()
            return ', '.join(mydict)

        if self.parent_nstates == {}:
            print(f"P({dict_to_str(self.var_nstates)}) of shape {self.pmf.shape}")
        else:
            print(f"P({dict_to_str(self.var_nstates)}|{dict_to_str(self.parent_nstates)}) \
            of shape {self.pmf.shape}")

    def get_marginal(self, subset_vars):
        assert set(subset_vars).issubset(set(self.get_vars())) # sanity check

        sum_over_vars = set(self.get_vars()) - set(subset_vars)

        sum_over_axes = tuple([self.var_axes[x] for x in sum_over_vars])
        marginal_pmf = np.sum(self.pmf, sum_over_axes)

        # This will ensure that `subset_vars` gets ordered according to their element's
        # positions in `self.get_vars()`
        # `self.get_vars()` is the sequence of the unreduced tensor and the reduced
        # tensor's axes will still be sorted according to it
        subset_vars = list(filter(
            lambda x : x in subset_vars,
            self.get_vars()
        ))

        marginal_axes = {}
        for idx, var in enumerate(subset_vars):
            marginal_axes[var] = idx

        marginal_var_nstates = dict(
            filter(lambda item: item[0] not in sum_over_vars, self.var_nstates.items())
        )
        marginal_parent_nstates = dict(
            filter(lambda item: item[0] not in sum_over_vars, self.parent_nstates.items())
        )
        return PMFTensor(marginal_pmf, marginal_axes, marginal_var_nstates, marginal_parent_nstates)

    def get_slice_using_observation(self, sample, sample_vars):
        # the present (unmasked) variables in the sample
        sample_y = [sample_vars[x] for x in np.where(sample >= 0)[0]]

        # the missing (masked) variables in the sample
        sample_z = [sample_vars[x] for x in np.where(sample == -1)[0]]

        # move present variables to the front
        axes_src = []
        axes_dst = []
        for (idx, var) in enumerate(sample_y + sample_z):
            axes_src.append(self.var_axes[var]) # move this var from this axis
            axes_dst.append(idx) # to this axis

        pmf_y_z = np.moveaxis(self.pmf, axes_src, axes_dst)

        # slice the tensor
        sample_y_vals = sample[np.where(sample >= 0)[0]]
        sliced_pmf = pmf_y_z[*sample_y_vals]
        slice_axes = {}
        for (idx, var) in enumerate(sample_z):
            slice_axes[var] = idx

        # also get `var_nstates` and `parent_nstates` for the newly created tensor
        slice_var_nstates = {
            var: self.var_nstates[var] for var in sample_z if var in self.var_nstates
        }

        # this will be empty for now because currently this function is only called on the
        # joint PMF (which has no conditioning variables)
        slice_parent_nstates = {
            var: self.parent_nstates[var] for var in sample_z if var in self.parent_nstates
        }
        return PMFTensor(sliced_pmf, slice_axes, slice_var_nstates, slice_parent_nstates)


if __name__ == '__main__':
    # a quick test to verify consistency of `PMFTensor` class
    np.random.seed(0)
    np.set_printoptions(precision=4, suppress=True)

    # P(A,B|C) = P(A|C) x P(B|C)
    # P(A,B,C) = P(A,B|C) x P(C)
    # overall = ( P(A|C) x P(B|C) ) x P(C)
    pagc = PMFTensor.rand_init({"A": 3}, {"C": 5})
    pbgc = PMFTensor.rand_init({"B": 4}, {"C": 5})
    prod1a = pagc.multiply(pbgc)
    pc = PMFTensor.rand_init({"C": 5}, {})
    prod1b = prod1a.multiply(pc)
    print("prod1b")
    print(prod1b.pmf)

    # P(A,C) = P(A|C) x P(C)
    # P(A,B,C) = P(A,C) x P(B|C)
    # overall = ( P(A|C) x P(C) ) x P(B|C)
    pac = pagc.multiply(pc)
    pbc = pbgc.multiply(pc)
    prod2 = pac.multiply(pbgc)
    print("prod2")
    print(prod2.pmf) # consistent with prod1b
