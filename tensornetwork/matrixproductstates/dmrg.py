# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensornetwork.network import TensorNetwork
from tensornetwork.matrixproductstates import mpslib as mpslib
from tensornetwork.matrixproductstates.mps import FiniteMPS
from tensornetwork.matrixproductstates.mpo import FiniteMPO
import numpy as np
from typing import Any, List, Optional, Text, Type, Union, Dict
Tensor = Any


class FiniteDMRG:

  def __init__(self, mps: FiniteMPS, mpo: FiniteMPO):
    self.mps = mps
    self.mpo = mpo
    if mps.backend.name != mpo.backend.name:
      raise ValueError('FiniteMPS and FiniteMPO have different backends.'
                       'Please use same backend for both.')
    if mps.dtype is not mpo.dtype:
      raise TypeError('FiniteMPS and FiniteMPO have different dtypes.'
                      'Please use same dtype for both.')

    self.backend = self.mps.backend

  def _add_left_layer(self, left_block: Tensor, mps_tensor: Tensor,
                      mpo_tensor: Tensor):
    return mpslib.add_left_mpo_layer(left_block, mps_tensor, mpo_tensor,
                                     self.backend.conj(mps_tensor),
                                     self.backend.name)

  def _add_right_layer(self, right_block: Tensor, mps_tensor: Tensor,
                       mpo_tensor: Tensor):

    return mpslib.add_right_mpo_layer(right_block, mps_tensor, mpo_tensor,
                                      self.backend.conj(mps_tensor),
                                      self.backend.name)
