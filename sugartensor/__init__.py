# -*- coding: utf-8 -*-
# pylint: disable=unused-import
from tensorflow import *

from sg_util import sg_opt
from sg_main import *
from sg_logging import *
import sg_optimize
from sg_train import *

import sg_data

__author__ = 'buriburisuri@gmail.com'


#
# augmenting tensorflow Variable and Tensor type by injecting custom methods
#

modules = ['sg_activation', 'sg_metric', 'sg_layer', 'sg_loss', 'sg_transform', 'sg_net']
for mod in modules:
    sg_inject(__path__[0], mod)

