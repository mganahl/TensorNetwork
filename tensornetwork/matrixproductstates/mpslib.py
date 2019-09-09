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
import numpy as np
from typing import Any, List, Optional, Text, Type, Union, Dict
Tensor = Any


def add_left_mpo_layer(left_block: Tensor, mps_tensor: Tensor,
                       mpo_tensor: Tensor, conj_mps_tensor: Tensor,
                       backend: str):
  net = TensorNetwork(backend=backend)
  L = net.add_node(left_block)
  A = net.add_node(mps_tensor)
  conj_A = net.add_node(conj_mps_tensor)
  M = net.add_node(mpo_tensor)
  L[0] ^ A[0]
  L[1] ^ conj_A[0]
  L[2] ^ M[0]
  A[1] ^ M[3]
  conj_A[1] ^ M[2]

  output_order = [A[2], conj_A[2], M[1]]
  result = L @ A @ M @ conj_A
  result.reorder_edges(output_order)
  return result.tensor


def add_right_mpo_layer(right_block: Tensor, mps_tensor: Tensor,
                        mpo_tensor: Tensor, conj_mps_tensor: Tensor,
                        backend: str):
  net = TensorNetwork(backend=backend)
  R = net.add_node(right_block)
  A = net.add_node(mps_tensor)
  conj_A = net.add_node(conj_mps_tensor)
  M = net.add_node(mpo_tensor)
  R[0] ^ A[2]
  R[1] ^ conj_A[2]
  R[2] ^ M[1]
  A[1] ^ M[3]
  conj_A[1] ^ M[2]

  output_order = [A[0], conj_A[0], M[0]]
  result = R @ A @ M @ conj_A
  result.reorder_edges(output_order)
  return result.tensor
