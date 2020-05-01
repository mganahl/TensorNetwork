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
"""implementation of different Matrix Product Operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.backends import backend_factory
from tensornetwork.backend_contextmanager import get_default_backend
from typing import List, Union, Text, Optional, Any, Type
Tensor = Any


#TODO: this class is very similar to BaseMPS. The two could probably
#      be merged.
class BaseMPO:
  """
  Base class for MPOs.
  """

  def __init__(self, tensors: List[Tensor],
               backend: Optional[Text] = None) -> None:
    """
    Initialize a BaseMPO.
    Args:
      tensors: A list of `Tensor` objects.
      backend: The name of the backend that should be used to perform 
        contractions. Available backends are currently 'numpy', 'tensorflow',
        'pytorch', 'jax'
    """
    if backend is None:
      backend = get_default_backend()

    self.backend = backend_factory.get_backend(backend)

    # we're no longer connecting MPS nodes because it's barely needed
    self.tensors = [self.backend.convert_to_tensor(t) for t in tensors]

  def __iter__(self):
    return iter(self.tensors)

  def __len__(self):
    return len(self.tensors)

  @property
  def dtype(self):
    if not all(
        [self.tensors[0].dtype == tensor.dtype for tensor in self.tensors]):
      raise ValueError('not all dtype in FiniteMPS.nodes are the same')
    return self.tensors[0].dtype

  @property
  def bond_dimensions(self):
    """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
    return [self.tensors[0].shape[0]
           ] + [tensor.shape[1] for tensor in self.tensors]


class InfiniteMPO(BaseMPO):
  """
  Base class for implementation of infinite MPOs; the user should implement 
  specific infinite MPOs by deriving from InfiniteMPO
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Text] = None,
               name: Optional[Text] = None) -> None:

    super().__init__(tensors=tensors, backend=backend)
    if not (self.bond_dimensions[0] == self.bond_dimensions[-1]):
      raise ValueError('left and right MPO ancillary dimension have to match')
    self.name = name

  def roll(self, num_sites):
    tensors = [self.tensors[n] for n in range(num_sites, len(self.tensors))
              ] + [self.tensors[n] for n in range(num_sites)]
    self.tensors = tensors


class FiniteMPO(BaseMPO):
  """
  Base class for implementation of finite MPOs; the user should implement 
  specific finite MPOs by deriving from FiniteMPO
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Text] = None,
               name: Optional[Text] = None) -> None:
    super().__init__(tensors=tensors, backend=backend)
    if not (self.bond_dimensions[0] == 1) and (self.bond_dimensions[-1] == 1):
      raise ValueError('left and right MPO ancillary dimensions have to be 1')
    self.name = name


class FiniteXXZ(FiniteMPO):
  """
  the famous Heisenberg Hamiltonian, the one we all know and love (almost as much as the TFI)!
  """

  def __init__(self,
               Jz: np.ndarray,
               Jxy: np.ndarray,
               Bz: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Text] = None):
    """
    returns the MPO of the XXZ model
    Args:
      Jz:  the Sz*Sz coupling strength between nearest neighbor lattice sites
      Jxy: the (Sx*Sx + Sy*Sy) coupling strength between nearest neighbor lattice sites
      Bz:  magnetic field on each lattice site
      dtype: the dtype of the MPO
      backend: An optional backend.
    Returns:
        FiniteXXZ:   the mpo of the finite XXZ model
    """
    self.Jz = Jz
    self.Jxy = Jxy
    self.Bz = Bz
    N = len(Bz)
    mpo = []
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
    mpo.append(temp)
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

      mpo.append(temp)
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

    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name='XXZ_MPO')


class FiniteTFI(FiniteMPO):
  """
  The good old transverse field Ising model
  convention: sigma_z=diag([-1,1])
  """

  def __init__(self,
               Jx: np.ndarray,
               Bz: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Text] = None):
    """
    Returns the MPO of the finite TFI model.
    Args:
      Jx:  The Sx*Sx coupling strength between nearest neighbor lattice sites.
      Bz:  Magnetic field on each lattice site.
      dtype: The dtype of the MPO.
      backend: An optional backend.
    Returns:
      FiniteTFI: The mpo of the infinite TFI model
    """
    self.Jx = Jx.astype(dtype)
    self.Bz = Bz.astype(dtype)
    N = len(Bz)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([-1, 1]).astype(dtype)
    mpo = []
    temp = np.zeros(shape=[1, 3, 2, 2], dtype=dtype)
    #Bsigma_z
    temp[0, 0, :, :] = self.Bz[0] * sigma_z
    #sigma_x
    temp[0, 1, :, :] = self.Jx[0] * sigma_x
    #11
    temp[0, 2, 0, 0] = 1.0
    temp[0, 2, 1, 1] = 1.0
    mpo.append(temp)
    for n in range(1, N - 1):
      temp = np.zeros(shape=[3, 3, 2, 2], dtype=dtype)
      #11
      temp[0, 0, 0, 0] = 1.0
      temp[0, 0, 1, 1] = 1.0
      #sigma_x
      temp[1, 0, :, :] = sigma_x
      #Bsigma_z
      temp[2, 0, :, :] = self.Bz[n] * sigma_z
      #sigma_x
      temp[2, 1, :, :] = self.Jx[n] * sigma_x
      #11
      temp[2, 2, 0, 0] = 1.0
      temp[2, 2, 1, 1] = 1.0
      mpo.append(temp)

    temp = np.zeros([3, 1, 2, 2], dtype=dtype)
    #11
    temp[0, 0, 0, 0] = 1.0
    temp[0, 0, 1, 1] = 1.0
    #sigma_x
    temp[1, 0, :, :] = sigma_x
    #Bsigma_z
    temp[2, 0, :, :] = self.Bz[-1] * sigma_z
    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name='TFI_MPO')