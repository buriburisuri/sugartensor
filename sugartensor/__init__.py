# -*- coding: utf-8 -*-
# pylint: disable=unused-import
from tensorflow import *

from sg_util import sg_opt
from sg_main import *
from sg_logging import *
from sg_train import sg_train
import sg_data

__author__ = 'njkim@jamonglab.com'


#
# augmenting tensorflow Variable and Tensor type by injecting custom methods
#

modules = ['sg_activation', 'sg_layer', 'sg_loss', 'sg_transform']
for mod in modules:
    sg_inject(__path__, mod)
