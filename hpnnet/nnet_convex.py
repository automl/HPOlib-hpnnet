##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from skdata.larochelle_etal_2007.view import ConvexVectorXV as Protocol
import skdata
# If anybody knows a better solutions, please tell me;)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bergstra_2011_helper
import hpnnet
import hpnnet.nips2011
import benchmark_util
__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def main(params, **kwargs):
    """
    Function to be included by run_instance.py.
    """
    dataset = skdata.larochelle_etal_2007.dataset.Convex()
    protocol_class = Protocol
    train, valid, test, train_targets, valid_targets, test_targets = \
        bergstra_2011_helper.fetch_data(dataset, "convex", **kwargs)
    kwargs['train'] = train
    kwargs['train_targets'] = train_targets
    kwargs['valid'] = valid
    kwargs['valid_targets'] = valid_targets
    kwargs['test'] = test
    kwargs['test_targets'] = test_targets

    """
    convex_cv = np.load("../../convex_cv.npy")
    convex_labels_cv = np.load("../../convex_labels_cv.npy")
    convex_test = np.load("../../convex_test.npy")
    convex_labels_test = np.load("../../convex_labels_test.npy")
    print np.all(train == convex_cv[:6500])
    print np.all(train_targets == convex_labels_cv[:6500])
    print np.all(valid == convex_cv[6500:])
    print np.all(valid_targets == convex_labels_cv[6500:])
    print np.all(test == convex_test)
    print np.all(test_targets == convex_labels_test)
    """

    print 'Params: ', params, '\n'
    space = hpnnet.nips2011.nnet1_preproc_space()
    y = bergstra_2011_helper.wrapping_nnet_split_decorator(
        params, space, protocol_class, **kwargs)
    print 'Result: ', y
    return y


def run_test(params, **kwargs):
    """
    Test function to be included by run_instance.py.
    """
    dataset = skdata.larochelle_etal_2007.dataset.Convex()
    protocol_class = Protocol
    train, valid, test, train_targets, valid_targets, test_targets = \
        bergstra_2011_helper.fetch_data(dataset, "convex", **kwargs)
    kwargs['train'] = train
    kwargs['train_targets'] = train_targets
    kwargs['valid'] = valid
    kwargs['valid_targets'] = valid_targets
    kwargs['test'] = test
    kwargs['test_targets'] = test_targets
    print 'Params: ', params, '\n'
    space = hpnnet.nips2011.nnet1_preproc_space()
    y = bergstra_2011_helper.wrapping_nnet_split_decorator(
        params, space, protocol_class, 'test', **kwargs)
    print 'Result: ', y
    return y


if __name__ == "__main__":
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    print "Result", result
