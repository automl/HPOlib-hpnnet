import hpnnet
import hpnnet.nips2011
import hpnnet.nips2011_dbn
import skdata
from hpnnet.skdata_learning_algo import eval_fn, PyllLearningAlgo
import hyperopt
from skdata.base import Task
from skdata.larochelle_etal_2007.view import \
    MNIST_RotatedBackgroundImages_VectorXV as Protocol
from skdata.larochelle_etal_2007.view import ConvexVectorXV as Protocol
import theano
