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
  """
  DMRG simulation for finite quantumm systems.
  """

  def __init__(self, mps: FiniteMPS, mpo: FiniteMPO):
    """
    Args:
      mps: An initial `FiniteMPS` object
      mpo: A Hammiltonian in mpo form
    Returns:
      None:
    Raises:
      ValueError: If `FiniteMPS` and `FiniteMPO` have
        different backends
      TypeError: If `FiniteMPS` and `FiniteMPO` have
        different dtypes
    """
    self.mps = mps
    self.mpo = mpo
    if mps.backend.name != mpo.backend.name:
      raise ValueError('FiniteMPS and FiniteMPO have different backends.'
                       'Please use same backend for both.')
    if mps.dtype is not mpo.dtype:
      raise TypeError('FiniteMPS and FiniteMPO have different dtypes.'
                      'Please use same dtype for both.')

    self.backend = self.mps.backend
    self.left_blocks = {
        0:
            self.backend.ones(
                (self.mps.nodes[0].shape[0], self.mps.nodes[0].shape[0],
                 self.mpo.nodes[0].shape[0]))
    }
    self.right_blocks = {
        len(self.mps) - 1:
            self.backend.ones((self.mps.nodes[len(self.mps) - 1].shape[2],
                               self.mps.nodes[len(self.mps) - 1].shape[2],
                               self.mpo.nodes[len(self.mps) - 1].shape[1]))
    }
    if not all([
        self.mps.check_orthonormality('left', site) < 1E-10
        for site in range(len(self.mps))
    ]):
      if all([
          self.mps.check_orthonormality('right', site) < 1E-10
          for site in range(len(self.mps))
      ]):
        self.mps.position(0)
      else:
        self.mps.position(len(self.mps) - 1)
        self.mps.position(0)

    self.mps.position(0)

  def add_left_layer(self, site: int):
    self.left_blocks[site + 1] = mpslib.add_left_mpo_layer(
        self.left_blocks[site], self.mps.nodes[site].tensor,
        self.mpo.nodes[site].tensor,
        self.backend.conj(self.mps.nodes[site].tensor), self.backend.name)

  def add_right_layer(self, site: int):
    self.right_blocks[site - 1] = mpslib.add_right_mpo_layer(
        self.right_blocks[site], self.mps.nodes[site].tensor,
        self.mpo.nodes[site].tensor,
        self.backend.conj(self.mps.nodes[site].tensor), self.backend.name)

  def position(self, site):
    #`site` has to be between 0 and len(mps) - 1
    if site >= len(self.mps) or site < 0:
      raise ValueError('site = {} not between values'
                       ' 0 < site < N = {}'.format(site, len(self.mps)))

    old_center_position = self.mps.center_position
    if site > old_center_position:
      self.mps.position(site)
      [self.add_left_layer(n) for n in range(old_center_position, site + 1)]
    elif site < old_center_position:
      self.mps.position(site)
      [
          self.add_right_layer(n)
          for n in reversed(range(site, old_center_position + 1))
      ]
