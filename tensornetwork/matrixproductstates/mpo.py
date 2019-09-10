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


class FiniteMPO(TensorNetwork):
  """
  A base class for mpos of finite systems. Custom mpos should be subclassed from 
  `FiniteMPO`.
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Text] = None,
               dtype: Optional[Type[np.number]] = None) -> None:
    """
    Args:
      tensors: A list of `Tensor` objects.
      backend: The name of the backend that should be used to perform 
        contractions. See documentation of TensorNetwork.__init__ for 
        a list of supported backends.
      dtype: An optional `dtype` for the FiniteMPO. See documentation of 
        TensorNetwork.__init__ for more details.
    Returns:
      None
    """
    super().__init__(backend=backend, dtype=dtype)
    self._nodes = [
        self.add_node(tensors[n], name='node{}'.format(n))
        for n in range(len(tensors))
    ]
    for site in range(len(self._nodes) - 1):
      self.connect(self._nodes[site][1], self._nodes[site + 1][0])

  def save(self, path: str):
    raise NotImplementedError()

  @property
  def nodes(self):
    return self._nodes

  @property
  def bond_dimensions(self) -> List:
    """
    Return a list of bond dimensions of FiniteMPS
    """
    return [self._nodes[0].shape[0]] + [node.shape[1] for node in self._nodes]

  @property
  def physical_input_dimensions(self) -> List:
    """
    Return a list of physical Hilbert-space dimensions of FiniteMPS
    """

    return [node.shape[3] for node in self._nodes]

  @property
  def physical_output_dimensions(self) -> List:
    """
    Return a list of physical Hilbert-space dimensions of FiniteMPS
    """
    return [node.shape[2] for node in self._nodes]


class FiniteXXZ(FiniteMPO):
  """
  The famous Heisenberg Hamiltonian, the one we all know and love 
  (almost as much as the TFI)!
  """

  def __init__(self,
               Jz: np.ndarray,
               Jxy: np.ndarray,
               Bz: np.ndarray,
               backend: Optional[Text] = None,
               dtype: Optional[np.dtype] = None) -> None:
    """
    returns an mpo representation of the XXZ Hamiltonian
    Args:
      Jz:  The Sz*Sz coupling strength between nearest 
        neighbor lattice sites
      Jxy: The (Sx*Sx + Sy*Sy) coupling strength between 
        nearest neighbor lattice sites
      Bz: Magnetic field on each lattice site
      backend: The name of the backend that should be used to perform 
        contractions. See documentation of TensorNetwork.__init__ for 
        a list of supported backends.
      dtype: An optional `dtype` for the FiniteXXZ. See documentation of 
        TensorNetwork.__init__ for more details.
    Returns:
        FiniteXXZ: An mpo representation of the finite XXZ model
    """
    self.Jz = Jz
    self.Jxy = Jxy
    self.Bz = Bz
    N = len(Bz)
    mpo_tensors = []
    temp = np.zeros((1, 5, 2, 2), dtype=dtype)
    #BSz
    temp[0, 0, 0, 0] = -0.5 * Bz[0]
    temp[0, 0, 1, 1] = 0.5 * Bz[0]

    #Sm
    temp[0, 1, 0, 1] = Jxy[0] / 2.0 * 1.0
    #Sp
    temp[0, 2, 1, 0] = Jxy[0] / 2.0 * 1.0
    #Sz
    temp[0, 3, 0, 0] = Jz[0] * (-0.5)
    temp[0, 3, 1, 1] = Jz[0] * 0.5

    #11
    temp[0, 4, 0, 0] = 1.0
    temp[0, 4, 1, 1] = 1.0
    mpo_tensors.append(temp)
    for n in range(1, N - 1):
      temp = np.zeros((5, 5, 2, 2), dtype=dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #Sp
      temp[1, 0, 1, 0] = 1.0
      #Sm
      temp[2, 0, 0, 1] = 1.0
      #Sz
      temp[3, 0, 0, 0] = -0.5
      temp[3, 0, 1, 1] = 0.5
      #BSz
      temp[4, 0, 0, 0] = -0.5 * Bz[n]
      temp[4, 0, 1, 1] = 0.5 * Bz[n]

      #Sm
      temp[4, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
      #Sp
      temp[4, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
      #Sz
      temp[4, 3, 0, 0] = Jz[n] * (-0.5)
      temp[4, 3, 1, 1] = Jz[n] * 0.5
      #11
      temp[4, 4, 0, 0] = 1.0
      temp[4, 4, 1, 1] = 1.0

      mpo_tensors.append(temp)
    temp = np.zeros((5, 1, 2, 2), dtype=dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #Sp
    temp[1, 0, 1, 0] = 1.0
    #Sm
    temp[2, 0, 0, 1] = 1.0
    #Sz
    temp[3, 0, 0, 0] = -0.5
    temp[3, 0, 1, 1] = 0.5
    #BSz
    temp[4, 0, 0, 0] = -0.5 * Bz[-1]
    temp[4, 0, 1, 1] = 0.5 * Bz[-1]

    mpo_tensors.append(temp)
    super().__init__(
        mpo_tensors, backend=backend)  #derives its dtype from mpo_tensors
