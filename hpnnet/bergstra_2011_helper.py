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

import ast
from functools import partial
import types
import numpy as np
import os

from hpnnet.skdata_learning_algo import eval_fn, PyllLearningAlgo
import hpnnet.nnet  # -- load scope with nnet symbols
import hyperopt
from skdata.base import Task

import HPOlib.data_util as data_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def fetch_data(dataset, data_name, **kwargs):
    dataset.fetch(True)
    print os.path.join(dataset.home(), "convex_inputs.npy")
    convex_labels = data_util.load_file(os.path.join(dataset.home(),
                                       data_name + "_labels.npy"), "numpy", 100)
    convex_inputs = data_util.load_file(os.path.join(dataset.home(),
                                       data_name + "_inputs.npy"), "numpy", 100)
    descr = dataset.descr
    n_train = descr['n_train']
    n_valid = descr['n_valid']
    n_test = descr['n_test']

    fold = int(kwargs['fold'])
    folds = int(kwargs['folds'])

    if folds == 1:
        train = convex_inputs[:n_train]
        valid = convex_inputs[n_train:n_train+n_valid]
        test = convex_inputs[n_train+n_valid:]
        train_targets = convex_labels[:n_train]
        valid_targets = convex_labels[n_train:n_train+n_valid]
        test_targets = convex_labels[n_train+n_valid:]
    elif folds > 1:
        cv_data = np.copy(convex_inputs[:n_train+n_valid])
        train, valid = data_util.prepare_cv_for_fold(cv_data, fold, folds)
        cv_labels = np.copy(convex_labels[:n_train+n_valid])
        train_targets, valid_targets = data_util.prepare_cv_for_fold(cv_labels,
                                                                    fold, folds)
        test = convex_inputs[n_train+n_valid:]
        test_targets = convex_labels[n_train+n_valid:]
    else:
        raise ValueError("Folds cannot be less than 1")

    return train, valid, test, train_targets, valid_targets, test_targets


def custom_split_protocol(self, algo, train, train_targets, valid,
                          valid_targets, test, test_targets):
    """
    This is modification from skdata/larochelle_et_al_2007/view.py and will be
    injected in there instead of the original protocol
    """
    ds = self.dataset
    ds.fetch(True)
    ds.build_meta()
    n_cv = ds.descr['n_train'] + ds.descr['n_valid']
    n_test = ds.descr['n_test']

    print ds.descr['n_train'], ds.descr['n_valid'], ds.descr['n_test']
    print "Split assertion 1", len(train), len(valid), n_cv
    assert(len(train) + len(valid) == n_cv)
    print "Split assertion 2", len(train_targets), len(valid_targets), n_cv
    assert(len(train_targets) + len(valid_targets) == n_cv)
    print "Split assertion 3", len(test), n_test
    assert(len(test) == n_test)
    print "Split assertion 4", len(test_targets), n_test
    assert(len(test_targets) == n_test)

    train_task = Task('vector_classification',
                      name='train',
                      x=train.reshape(train.shape[0], -1),
                      y=train_targets,
                      n_classes=ds.descr['n_classes'])

    valid_task = Task('vector_classification',
                      name='valid',
                      x=valid.reshape(valid.shape[0], -1),
                      y=valid_targets,
                      n_classes=ds.descr['n_classes'])

    test_task = Task('vector_classification',
                     name='test',
                     x=test.reshape(test.shape[0], -1),
                     y=test_targets,
                     n_classes=ds.descr['n_classes'])

    model = algo.best_model(train=train_task, valid=valid_task)
    algo.loss(model, train_task)
    algo.loss(model, valid_task)
    algo.loss(model, test_task)


def wrapping_nnet_split_decorator(params, space, protocol_class, mode='train',
                                  **kwargs):
    """
    This function changes the protocol_class and injects the cv_protocol.
    """
    # Prepare the custom_split_protocol with the data for the data split
    protocol = partial(custom_split_protocol, train=kwargs['train'],
                       train_targets=kwargs['train_targets'],
                       valid=kwargs['valid'], valid_targets=
                       kwargs['valid_targets'], test=kwargs['test'],
                       test_targets=kwargs['test_targets'])

    # by only using functools.partial, the method is unbound, meaning it does
    # not belong to the respective Protocol class. The next line adds the method
    # to the protocol class
    protocol_class.protocol = types.MethodType(protocol, None, protocol_class)
    return wrapping_nnet(params, space, protocol_class, mode, **kwargs)


def wrapping_nnet(params, space, protocol_class, mode='train', **kwargs):
    """
    This function takes a protocol class and uses the skdata_learning_algo to
    train the neural network. Protocol_class can either be the one specified at
    the top of this script or the original one from skdata
    """
    dataset_eval_fn = partial(eval_fn, protocol_cls=protocol_class)

    # Get the unevaluated search space from arguments,, extract all hyper-
    # parameters and create a memo object. This contains the random expression
    # as a key and the new pyll Literal as a value. The random expression will
    # then be replaced with the Literal by the dataset_eval_fn

    # This is only for testing purposes
    #space = hpnnet.nips2011.nnet1_preproc_space(sup_min_epochs = 10,
    #                                            sup_max_epochs = 100)
    hps = {}
    if type(space) == tuple:
        hpspace = space[0]
    else:
        hpspace = space

    hyperopt.pyll_utils.expr_to_config(hpspace, (), hps)
    memo = {}
    for param in params:
        node = hps[param]['node']
        # print "###"
        # print "Label:", hps[param]['label']
        # print "Node:", node
        # We have to convert the parameter back to a string so literal eval can
        # actually work with it
        try:
            value = ast.literal_eval(str(params[param]))
        except ValueError as e:
            print "Malformed String:", params[param]
            raise e
        memo[node] = hyperopt.pyll.Literal(value)
        # print "Memo[node]:", memo[node]
        # print "Value", value
    hpspace = hyperopt.pyll.stochastic.recursive_set_rng_kwarg(hpspace)
    if type(space) == tuple:
        space = (hpspace, space[1])
        space = hyperopt.pyll.as_apply(space)
    else:
        space = hpspace


    print "Evaluation mode", mode
    if mode == 'train':
        rval = dataset_eval_fn(space, memo, None)
        return rval['loss']

    elif mode == 'test':
        """
        Ad-hoc implementation for testing a configuration.
        """
        import hpnnet.nnet as nnet # -- ensure pyll symbols are loaded
        assert 'time' in hyperopt.pyll.scope._impls
        protocol = protocol_class()
        algo = PyllLearningAlgo(space, memo, None)
        protocol.protocol(algo)

        true_loss = None
        print algo.results
        for dct in algo.results['loss']:
            if dct['task_name'] == 'test':
                true_loss = dct['err_rate']
        return true_loss
    else:
        raise Exception("No evaluation mode specified")