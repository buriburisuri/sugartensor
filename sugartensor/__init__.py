from __future__ import absolute_import
from tensorflow import *

from .sg_util import sg_opt
from .sg_main import *
from .sg_logging import *
from .sg_train import *
from .sg_queue import sg_producer_func
from .sg_layer import sg_emb

from . import sg_optimize
from . import sg_data
from . import sg_initializer

from ._version import __version__


__author__ = 'buriburisuri@gmail.com'


#
# augmenting tensorflow Variable and Tensor type by injecting custom methods
#

modules = ['sg_activation', 'sg_metric', 'sg_layer', 'sg_loss', 'sg_transform', 'sg_net']
for mod in modules:
    sg_inject(__path__[0], mod)
