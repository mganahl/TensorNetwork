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
"""implementations of finite and infinite Density Matrix Renormalization Group algorithms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import pickle
import time
import tensornetwork as tn
import numpy as np
import tensorflow as tf
import experiments.MPS_classifier.batchtensornetwork as btn
import experiments.MPS.misc_mps as misc_mps
import experiments.MPS.matrixproductoperators as MPO
from sys import stdout
import experiments.MPS.matrixproductstates as MPS
import functools as fct
from experiments.MPS.matrixproductstates import InfiniteMPSCentralGauge, FiniteMPSCentralGauge


def block_MPO(mpo, block_length):
  if block_length == 1:
    return mpo
  assert len(mpo) % block_length == 0
  tensors = []
  for n in range(0, len(mpo), block_length):
    net = tn.TensorNetwork()
    nodes = [net.add_node(mpo[n + b]) for b in range(block_length)]
    for n in range(len(nodes) - 1):
      nodes[n][1] ^ nodes[n + 1][0]
    bottom_edges = [n[2] for n in nodes]
    top_edges = [n[3] for n in nodes]
    out_edges = [nodes[0][0]] + [nodes[-1][1]] + bottom_edges + top_edges
    node = nodes[0]
    for n in range(1, len(nodes)):
      node = node @ nodes[n]
    node.reorder_edges(out_edges)
    M1, M2 = node.shape[0], node.shape[1]

    tensors.append(
        tf.reshape(node.tensor, (M1, M2, 2**block_length, 2**block_length)))
  return MPO.FiniteMPO(tensors)


def generate_probability_mps(mps):
  ds = mps.d
  Ds = mps.D
  tensors = []
  for site in range(len(mps)):
    copy_tensor = np.zeros((ds[site], ds[site], ds[site]),
                           dtype=mps.dtype.as_numpy_dtype)
    for n in range(ds[site]):
      copy_tensor[n, n, n] = 1.0
      tensor = misc_mps.ncon(
          [mps.get_tensor(site),
           tf.conj(mps.get_tensor(site)), copy_tensor],
          [[-1, 1, -4], [-2, 2, -5], [1, 2, -3]])
      tmp = tf.reshape(tensor, (Ds[site]**2, ds[site], Ds[site + 1]**2))
    tensors.append(tmp)
  return MPS.FiniteMPSCentralGauge(tensors)


def absorb_two_body_gates(mps, two_body_gates, Dmax=None):
  mps_out = copy.deepcopy(mps)
  for site in range(0, len(mps_out) - 1, 2):
    mps_out.apply_2site(two_body_gates[(site, site + 1)], site)
  for site in range(1, len(mps) - 2, 2):
    mps_out.apply_2site(two_body_gates[(site, site + 1)], site)
  mps_out.position(0)
  mps_out.position(len(mps_out), normalize=True)
  tw = mps_out.position(0, normalize=True, D=Dmax)
  return mps_out, tw


def absorb_one_body_gates(mps, one_body_gates):
  tensors = [
      misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]],
                    [[-1, 1, -3], [-2, 1]]) for site in range(len(mps))
  ]
  mps_out = MPS.FiniteMPSCentralGauge(tensors)
  return mps_out


def generate_basis(N):
  l = [[0, 1] for _ in range(N)]
  l = list(itertools.product(*l))
  return np.stack([list(k) for k in l])


def apply_random_positive_gates(mps):
  for site in range(0, len(mps) - 1, 2):
    mps.apply_2site(
        tf.random_uniform(
            shape=[mps.d[site], mps.d[site + 1], mps.d[site], mps.d[site + 1]],
            dtype=mps.dtype), site)
  for site in range(1, len(mps) - 2, 2):
    mps.apply_2site(
        tf.random_uniform(
            shape=[mps.d[site], mps.d[site + 1], mps.d[site], mps.d[site + 1]],
            dtype=mps.dtype), site)
  mps.position(0)
  mps.position(len(mps))
  mps.position(0)
  return mps


def get_gate_from_generator(g, shape):
  """
    Args:
        g (tf.Tensor of shape (d,d)):   a generator matrix
        shape (tuple): the desired output shape of the gate
    
    """
  if g.dtype in (tf.complex128, tf.complex64):
    return tf.reshape(tf.linalg.expm(g - tf.conj(tf.transpose(g))), shape)
  elif g.dtype in (tf.float32, tf.float64):
    return tf.reshape(tf.linalg.expm(g - tf.conj(tf.transpose(g))), shape)


def get_generator_from_gate(g):
  """
    only works properly if g has a complex dtype
    Args:
        g (tf.Tensor):   a gate
    
    """
  shape = g.shape
  din, dout = 1, 1
  for n in range(len(shape) // 2):
    dout *= shape[n]
  for n in range(len(shape) // 2, len(shape)):
    din *= shape[n]
  if g.dtype in (tf.complex128, tf.complex64):
    return tf.linalg.logm(tf.reshape(g, (dout, din))) / 2
  elif g.dtype in (tf.float32, tf.float64):
    matrix = tf.linalg.logm(
        tf.complex(
            tf.reshape(g, (dout, din)),
            tf.zeros(shape=[dout, din], dtype=g.dtype)))
    return (matrix - tf.transpose(tf.conj(matrix))) / 2


def get_generators_from_gates(gates):
  return {s: get_generator_from_gate(g) for s, g in gates.items()}


def randomize_even_two_body_gates(gates, noise):
  assert (noise >= 0.0)
  for s in range(0, len(gates), 2):
    if gates[(s, s + 1)].dtype in (tf.complex128, tf.complex64):
      gates[(s, s + 1)] += tf.complex(
          tf.random_uniform(
              shape=gates[(s, s + 1)].shape,
              dtype=gates[(s, s + 1)].dtype.real_dtype,
              minval=-noise / 2,
              maxval=noise / 2),
          tf.random_uniform(
              shape=gates[(s, s + 1)].shape,
              dtype=gates[(s, s + 1)].dtype.real_dtype,
              minval=-noise / 2,
              maxval=noise / 2))
    elif gates[(s, s + 1)].dtype in (tf.float32, tf.float64):
      gates[(s, s + 1)] += tf.random_uniform(
          shape=gates[(s, s + 1)].shape,
          dtype=gates[(s, s + 1)].dtype.real_dtype,
          minval=-noise / 2,
          maxval=noise / 2)

  return gates


def randomize_odd_two_body_gates(gates, noise):
  assert (noise >= 0.0)
  for s in range(1, len(gates) - 1, 2):
    if gates[(s, s + 1)].dtype in (tf.complex128, tf.complex64):

      gates[(s, s + 1)] += tf.complex(
          tf.random_uniform(
              shape=gates[(s, s + 1)].shape,
              dtype=gates[(s, s + 1)].dtype.real_dtype,
              minval=-noise / 2,
              maxval=noise / 2),
          tf.random_uniform(
              shape=gates[(s, s + 1)].shape,
              dtype=gates[(s, s + 1)].dtype.real_dtype,
              minval=-noise / 2,
              maxval=noise / 2))
    elif gates[(s, s + 1)].dtype in (tf.float32, tf.float64):
      gates[(s, s + 1)] += tf.random_uniform(
          shape=gates[(s, s + 1)].shape,
          dtype=gates[(s, s + 1)].dtype.real_dtype,
          minval=-noise / 2,
          maxval=noise / 2)
  return gates


def randomize_gates(gates, noise):
  assert (noise >= 0.0)
  for k in gates.keys():
    if gates[k].dtype in (tf.complex128, tf.complex64):
      gates[k] += tf.complex(
          tf.random_uniform(
              shape=gates[k].shape,
              dtype=gates[k].dtype.real_dtype,
              minval=-noise / 2,
              maxval=noise / 2),
          tf.random_uniform(
              shape=gates[k].shape,
              dtype=gates[k].dtype.real_dtype,
              minval=-noise / 2,
              maxval=noise / 2))
    elif gates[k].dtype in (tf.float32, tf.float64):
      gates[k] += tf.random_uniform(
          shape=gates[k].shape,
          dtype=gates[k].dtype.real_dtype,
          minval=-noise / 2,
          maxval=noise / 2)
  return gates


def initialize_even_two_body_gates(ds, dtype, which, noise=0.0):
  """
    initialize two body gates
    Args:
        ds (iterable of int):   physical dimensions at each site
        dtype (tf.Dtype):       data type
        which (str):   the type to which gates should be reset
                       `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                       or in ('h','haar') for Haar random unitaries
        noise (float): nose parameter; if nonzero, add noise to the identities
    Returns:
        dict:          maps (s,s+1) to gate for s even
    Raises:
        ValueError
    """
  two_body_gates = {}
  if which in ('i', 'identities', 'eye', 'e'):
    for site in range(0, len(ds), 2):
      if dtype in (tf.float32, tf.float64):
        two_body_gates[(site, site + 1)] = tf.reshape(
            tf.eye(ds[site] * ds[site + 1], dtype=dtype),
            (ds[site], ds[site + 1], ds[site], ds[site + 1]))
      elif dtype in (tf.complex128, tf.complex64):
        two_body_gates[(site, site + 1)] = tf.reshape(
            tf.complex(
                tf.eye(ds[site] * ds[site + 1], dtype=dtype.real_dtype),
                noise * tf.random_uniform(
                    shape=[ds[site] * ds[site + 1], ds[site] * ds[site + 1]],
                    dtype=dtype.real_dtype,
                    minval=-1,
                    maxval=1)),
            (ds[site], ds[site + 1], ds[site], ds[site + 1]))

  elif which in ('h', 'haar'):
    for site in range(0, len(ds), 2):
      two_body_gates[(site, site + 1)] = tf.reshape(
          misc_mps.haar_random_unitary(
              (ds[site] * ds[site + 1], ds[site] * ds[site + 1]), dtype=dtype),
          (ds[site], ds[site + 1], ds[site], ds[site + 1]))
  else:
    raise ValueError('unrecognized value which = {0}'.format(which))

  return two_body_gates


def initialize_odd_two_body_gates(ds, dtype, which, noise=0.0):
  """
    reset the two body gates
    Args:
        ds (iterable of int):   physical dimensions at each site
        dtype (tf.Dtype):       data type
        which (str):   the type to which gates should be reset
                       `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                       or in ('h','haar') for Haar random unitaries
        noise (float): nose parameter; if nonzero, add noise to the identities
    Returns:
        dict:          maps (s,s+1) to gate for s odd

    Raises:
        ValueError
    """
  two_body_gates = {}
  if which in ('i', 'identities', 'eye', 'e'):
    for site in range(1, len(ds) - 2, 2):
      if dtype in (tf.float32, tf.float64):
        two_body_gates[(site, site+1)] = tf.reshape(tf.eye(ds[site]*ds[site+1], dtype=dtype) + \
                                                         noise * tf.random_uniform(shape = [ds[site]*ds[site+1], ds[site]*ds[site+1]], dtype=dtype),
                                                         (ds[site], ds[site+1], ds[site], ds[site+1]))
      elif dtype in (tf.complex128, tf.complex64):
        two_body_gates[(site, site + 1)] = tf.reshape(
            tf.complex(
                tf.eye(ds[site] * ds[site + 1], dtype=dtype.real_dtype),
                noise * tf.random_uniform(
                    shape=[ds[site] * ds[site + 1], ds[site] * ds[site + 1]],
                    dtype=dtype.real_dtype,
                    minval=-1,
                    maxval=1)),
            (ds[site], ds[site + 1], ds[site], ds[site + 1]))

  elif which in ('h', 'haar'):
    for site in range(1, len(ds) - 2, 2):
      two_body_gates[(site, site + 1)] = tf.reshape(
          misc_mps.haar_random_unitary(
              (ds[site] * ds[site + 1], ds[site] * ds[site + 1]), dtype),
          (ds[site], ds[site + 1], ds[site], ds[site + 1]))
  else:
    raise ValueError('unrecognized value which = {0}'.format(which))

  return two_body_gates


def initialize_one_body_gates(ds, dtype, which, noise=0.0):
  """
    reset the two one gates
    Args:
        ds (iterable of int):   physical dimensions at each site
        dtype (tf.Dtype):       data type
        which (str):   the type to which gates should be reset
                       `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                       or in ('h','haar') for Haar random unitaries
        noise (float): nose parameter; if nonzero, add noise to the identities
    Returns:
        dict:          maps int to tf.Tensor (sites to gates)

    Raises:
        ValueError
    """
  if which in ('i', 'identities', 'eye', 'e'):
    if dtype in (tf.float32, tf.float64):
      one_body_gates = {site: tf.eye(ds[site], dtype=dtype) + \
                             noise * tf.random_uniform(shape=[ds[site], ds[site]], dtype=dtype.real_dtype, minval=-1, maxval=1)
                             for site in range(0,len(ds))}
    elif dtype in (tf.complex128, tf.complex64):
      one_body_gates = {site: tf.complex(tf.eye(ds[site], dtype=dtype.real_dtype) + \
                                              noise * tf.random_uniform(shape=[ds[site], ds[site]], dtype=dtype.real_dtype, minval=-1, maxval=1),
                                              noise * tf.random_uniform(shape=[ds[site], ds[site]], dtype=dtype.real_dtype, minval=-1, maxval=1))
                             for site in range(0,len(ds))}
  elif which in ('h', 'haar'):
    one_body_gates = {
        site: misc_mps.haar_random_unitary((ds[site], ds[site]), dtype)
        for site in range(0, len(ds))
    }
  else:
    raise ValueError('unrecognized value which = {0}'.format(which))
  return one_body_gates


def initialize_gates_from_generators(generators, ds):
  gates = {}
  for s, g in generators.items():
    try:
      if len(s) == 2:
        shape = (ds[s[0]], ds[s[1]], ds[s[0]], ds[s[1]])
    except TypeError:
      shape = (ds[s], ds[s])
    gates[s] = get_gate_from_generator(g, shape)
  return gates


def initialize_two_body_generators(ds, dtype, which='identities'):
  """
    initialize two body generator matrices `g`. Note that the "real" generator
    is given by `g`- herm(`g`) and U = expm(`g` - herm(`g`)))
    Thus, if `g = 11`, then `U = 11`.
    Args:
        ds (list):     list of local dimensions, one for each site 
        dtype (tf.Dtype): the type of the tensors
        which (str):   the type to which gates should be reset
                       `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                       or in ('r','random') for random matrices
    Returns:
        dict:          the generators; maps tuples (site, site+1) -> tf.Tensor
    Raises:
        ValueError
    """

  two_body_generators = initialize_even_two_body_generators(ds, dtype, which)
  two_body_generators.update(
      initialize_odd_two_body_generators(ds, dtype, which))
  return two_body_generators


def initialize_even_two_body_generators(ds, dtype, which='identities'):
  """
    initialize even two body generator matrices `g`. Note that the "real" generator
    is given by `g`- herm(`g`) and U = expm(`g` - herm(`g`)))
    Thus, if `g = 11`, then `U = 11`.
    Args:
        ds (list):     list of local dimensions, one for each site 
        dtype (tf.Dtype): the type of the tensors
        which (str):   the type to which gates should be reset
                       `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                       or in ('r','random') for random matrices
    Returns:
        dict:          the generators; maps tuples (site, site+1) -> tf.Tensor
    Raises:
        ValueError
    """

  two_body_generators = {}
  if which in ('i', 'identities', 'eye', 'e'):
    for site in range(0, len(ds) - 1, 2):
      if dtype in (tf.float32, tf.float64):
        two_body_generators[(site, site + 1)] = tf.eye(
            ds[site] * ds[site + 1], dtype=dtype)
      elif dtype in (tf.complex128, tf.complex64):
        two_body_generators[(site, site + 1)] = tf.complex(
            tf.eye(ds[site] * ds[site + 1], dtype=dtype.real_dtype),
            tf.zeros(
                shape=[ds[site] * ds[site + 1], ds[site] * ds[site + 1]],
                dtype=dtype.real_dtype))
  elif which in ('r', 'random'):
    for site in range(0, len(ds) - 1, 2):
      if dtype in (tf.float32, tf.float64):
        two_body_generators[(site, site + 1)] = tf.random_uniform(
            shape=(ds[site] * ds[site + 1], ds[site] * ds[site + 1]),
            dtype=dtype,
            minval=-0.1,
            maxval=0.1)
      elif dtype in (tf.complex128, tf.complex64):
        two_body_generators[(site, site + 1)] = tf.complex(
            tf.random_uniform(
                shape=(ds[site] * ds[site + 1], ds[site] * ds[site + 1]),
                dtype=dtype.real_dtype,
                minval=-0.1,
                maxval=0.1),
            tf.random_uniform(
                shape=(ds[site] * ds[site + 1], ds[site] * ds[site + 1]),
                dtype=dtype.real_dtype,
                minval=-0.1,
                maxval=0.1))

  else:
    raise ValueError('unrecognized value which = {0}'.format(which))
  return two_body_generators


def initialize_odd_two_body_generators(ds, dtype, which='identities'):
  """
    initialize odd two body generator matrices `g`. Note that the "real" generator
    is given by `g`- herm(`g`) and U = expm(`g` - herm(`g`)))
    Thus, if `g = 11`, then `U = 11`.
    Args:
        ds (list):     list of local dimensions, one for each site 
        dtype (tf.Dtype): the type of the tensors
        which (str):   the type to which gates should be reset
                       `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                       or in ('r','random') for random matrices
    Returns:
        dict:          the generators; maps tuples (site, site+1) -> tf.Tensor
    Raises:
        ValueError
    """
  two_body_generators = {}
  if which in ('i', 'identities', 'eye', 'e'):
    for site in range(1, len(ds) - 2, 2):
      if dtype in (tf.float32, tf.float64):
        two_body_generators[(site, site + 1)] = tf.eye(
            ds[site] * ds[site + 1], dtype=dtype)
      elif dtype in (tf.complex128, tf.complex64):
        two_body_generators[(site, site + 1)] = tf.complex(
            tf.eye(ds[site] * ds[site + 1], dtype=dtype.real_dtype),
            tf.zeros(
                shape=[ds[site] * ds[site + 1], ds[site] * ds[site + 1]],
                dtype=dtype.real_dtype))
  elif which in ('r', 'random'):
    for site in range(1, len(ds) - 2, 2):
      if dtype in (tf.float32, tf.float64):
        two_body_generators[(site, site + 1)] = tf.random_uniform(
            shape=(ds[site] * ds[site + 1], ds[site] * ds[site + 1]),
            dtype=dtype,
            minval=-0.1,
            maxval=0.1)
      elif dtype in (tf.complex128, tf.complex64):
        two_body_generators[(site, site + 1)] = tf.complex(
            tf.random_uniform(
                shape=(ds[site] * ds[site + 1], ds[site] * ds[site + 1]),
                dtype=dtype.real_dtype,
                minval=-0.1,
                maxval=0.1),
            tf.random_uniform(
                shape=(ds[site] * ds[site + 1], ds[site] * ds[site + 1]),
                dtype=dtype.real_dtype,
                minval=-0.1,
                maxval=0.1))
  else:
    raise ValueError('unrecognized value which = {0}'.format(which))
  return two_body_generators


def initialize_one_body_generators(ds, dtype, which='identities'):
  """
    initialize one-body generator matrices `g`. Note that the "real" generator
    is given by `g`- herm(`g`) and U = expm(`g` - herm(`g`)))
    Thus, if `g = 11`, then `U = 11`.
    Args:
        ds (list):     list of local dimensions, one for each site 
        dtype (tf.Dtype): the type of the tensors
        which (str):   the type to which gates should be reset
                       `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                       or in ('r','random') for random matrices
    Returns:
        dict:          the generators; maps integers `site` -> tf.Tensor
    Raises:
        ValueError
    """
  if which in ('i', 'identities', 'eye', 'e'):
    if dtype in (tf.float32, tf.float64):
      one_body_generators = {
          site: tf.eye(ds[site], dtype=dtype) for site in range(0, len(ds))
      }
    elif dtype in (tf.float64, tf.complex128):
      one_body_generators = {
          site: tf.complex(
              tf.eye(ds[site], dtype=dtype.real_dtype),
              tf.zeros(ds[site], dtype=dtype.real_dtype))
          for site in range(0, len(ds))
      }
  elif which in ('r', 'random'):
    if dtype in (tf.float32, tf.float64):
      one_body_generators = {
          site: tf.random_uniform(
              shape=(ds[site], ds[site]),
              dtype=dtype.real_dtype,
              minval=-0.1,
              maxval=0.1) for site in range(0, len(ds))
      }
    elif dtype in (tf.float64, tf.complex128):
      one_body_generators = {
          site: tf.complex(
              tf.random_uniform(
                  shape=(ds[site], ds[site]),
                  dtype=dtype.real_dtype,
                  minval=-0.1,
                  maxval=0.1),
              tf.random_uniform(
                  shape=(ds[site], ds[site]),
                  dtype=dtype.real_dtype,
                  minval=-0.1,
                  maxval=0.1)) for site in range(0, len(ds))
      }
  else:
    raise ValueError('unrecognized value which = {0}'.format(which))
  return one_body_generators


class OverlapMaximizer:
  """
    maximizes the overlap between `mps` and a given reference mps, using a double layer of two-body unitaries.
    For now `mps` has to have even length

    """

  def __init__(self,
               mps,
               one_body_gates=None,
               two_body_gates=None,
               name='overlap_maximizer'):
    """
        initialize an OverlapMaximizer object
        This object maximizes the overlap between `mps` and a reference mps 
        using a unitary circuit with three layers:
        the first layer 1 contains `N` one-body unitaries, as provided in `one_body_gates`
        the second layer 2 contains `N/2` two-body unitaries on sites (site1, site2) with site1 even 
        the third layer 3 contains `N/2 - 1` two-body unitaries on sites (site1, site2) with site1 odd 
        Two body unitaries are provided in `two_body_gates`

        index conventions for one-body unitaries:

               1
               |
              ___
             |   |
             |___|
               |
               0

        index conventions for two-body unitaries:

               2   3
               |   |
              _______
             |       |
             |_______|
               |   |
               0   1
    
        2,3 are the physical outgoing and incoming indices, respectively. The conjugated 
        side of the MPS is on the bottom (at index 2)

        MPS index convention:

              ___
             |   |
         0---     ---2
             |___|
               |
               1

    
        An MPS by this convention is contracted from above:
                    ___
                 --|   |--
                    ---
                     |
                    ___
                   |   |
                   |___|
                     |

                ___     ___    
             --|   | --|   |--
                ---     ---   
                 |       |     
                ___________ 
               |           |
               |___________|
                 |       |  


    
        Args:
            mps (FiniteMPSCentralGauge):       an mps of even length
            one_body_gates (iterable or None): an iterable mapping sites to matrices
                                               `one_body_gates[site]` is the one-body unitary  at site `site
                                               if `None`, one-body gates are initialized with identities
            two_body_gates (dict or None):     dictionary mapping tuples `(site`, site2)` to rank-4 unitary tensors
                                               the convention 
                                               `one_body_gates[site]` is the one-body unitary  at site `site
                                               if `None`, two-body gates are initialized with identities
            name (str):                        an optional name for the object
        """

    self.name = name
    self.mps = mps
    if (one_body_gates == None) or (len(one_body_gates) == 0):
      self.reset_one_body_gates(which='e')
    else:
      self.one_body_gates = one_body_gates

    if (two_body_gates == None) or (len(two_body_gates) == 0):
      self.reset_two_body_gates(which='e')
    else:
      self.two_body_gates = two_body_gates

    self.right_envs = {}
    self.left_envs = {}
    self.right_envs_batched = {}
    self.left_envs_batched = {}

  def save(self, name=None):
    data = {
        'one_body_gates': self.one_body_gates,
        'two_body_gates': self.two_body_gates,
        'mps': self.mps,
        'name': self.name
    }
    if name is None:
      name = self.name
    with open(name + '.mz', 'wb') as f:
      pickle.dump(data, f)

  @classmethod
  def load(cls, name):
    with open(name, 'rb') as f:
      data = pickle.load(f)
    one_body_gates = data['one_body_gates']
    two_body_gates = data['two_body_gates']
    mps = data['mps']
    name = data['name']
    return cls(
        mps=mps,
        one_body_gates=one_body_gates,
        two_body_gates=two_body_gates,
        name=name)

  def reset_two_body_gates(self, which='identities', dtype=None, noise=0.0):
    """
        reset the two body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
            dtype (tf.Dtype):       data type
            noise (float): nose parameter; if nonzero, add noise to the identities
        Returns:
            dict:          maps (s,s+1) to gate for s even
        Raises:
            ValueError
        """
    if not dtype:
      dtype = self.mps.dtype

    self.two_body_gates = initialize_even_two_body_gates(
        self.mps.d, self.mps.dtype, which, noise=noise)
    self.two_body_gates.update(
        initialize_odd_two_body_gates(
            self.mps.d, self.mps.dtype, which, noise=noise))

  def reset_even_two_body_gates(self, which='identities', dtype=None,
                                noise=0.0):
    """
        reset the even two body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
            dtype (tf.Dtype):       data type
            noise (float): nose parameter; if nonzero, add noise to the identities
        Returns:
            dict:          maps (s,s+1) to gate for s even
        Raises:
            ValueError
        """
    self.two_body_gates.update(
        initialize_even_two_body_gates(
            self.mps.d, self.mps.dtype, which, noise=noise))

  def reset_odd_two_body_gates(self, which='identities', dtype=None, noise=0.0):
    """
        reset the even odd body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
            dtype (tf.Dtype):       data type
            noise (float): nose parameter; if nonzero, add noise to the identities
        Returns:
            dict:          maps (s,s+1) to gate for s even
        Raises:
            ValueError
        """
    self.two_body_gates.update(
        initialize_odd_two_body_gates(
            self.mps.d, self.mps.dtype, which, noise=noise))

  def reset_one_body_gates(self, which='eye', dtype=None, noise=0.0):
    """
        reset the one-body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
            dtype (tf.Dtype): data type
            noise (float): nose parameter; if nonzero, add noise to the identities
        Returns:
            dict:          maps (s,s+1) to gate for s even
        Raises:
            ValueError
        """

    if which in ('e', 'eye', 'h', 'haar', 'i', 'identities'):
      ds = [self.mpo.get_tensor(site).shape[2] for site in range(len(self.mpo))]
      self.gates = initialize_one_body_gates(
          self.mps.d, self.mps.dtype, which, noise=noise)
    else:
      raise ValueError('wrong value {} for argument `which`'.format(which))

  @staticmethod
  def add_unitary_batched_right(site, right_envs, mps, samples, one_body_gates,
                                two_body_gates):
    """ 
        samples (tf.Tensor of shape (Nt, len(mps))
        """
    #if site is even, add an odd gate
    #if site is odd, add an even gate
    #fixme: calling tf.one_hot could be slowing things down.
    assert (site > 0)
    assert (len(mps) % 2 == 0)
    if site == (len(mps) - 1):
      right_envs[site - 1] = tf.squeeze(
          misc_mps.ncon([
              mps.get_tensor(site), one_body_gates[site],
              two_body_gates[(site - 1, site)],
              tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype)
          ], [[-2, 1, -5], [2, 1], [-4, 3, -3, 2], [-1, 3]]), 4)
    elif (site < len(mps) - 1) and (site % 2 == 0):
      tmp = misc_mps.ncon([
          right_envs[site],
          mps.get_tensor(site), one_body_gates[site], two_body_gates[(site - 1,
                                                                      site)]
      ], [[-1, 1, 3, 4], [-2, 2, 1], [3, 2], [-4, -5, -3, 4]
         ])  #shape (Nt, D, din, dout, d)
      Nt, D, din, dout, d = tmp.shape
      tmp = tf.reshape(tmp, (Nt, D * din * dout, d))  #(Nt, D * din * dout, d)
      tmp2 = tf.expand_dims(
          tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),
          2)  #(Nt, d, 1)
      right_envs[site - 1] = tf.reshape(
          tf.matmul(tmp, tmp2), (Nt, D, din, dout))

    elif (site < len(mps) - 1) and (site % 2 == 1):
      tmp = misc_mps.ncon([
          right_envs[site],
          mps.get_tensor(site), one_body_gates[site], two_body_gates[(site - 1,
                                                                      site)]
      ], [[-1, 1, 4, -5], [-2, 2, 1], [3, 2], [-4, 4, -3, 3]
         ])  #has shape (Nt, Dl, din1, dout1, d)
      Nt, D, din, dout, d = tmp.shape
      tmp = tf.reshape(tmp, (Nt, D * din * dout, d))
      tmp2 = tf.expand_dims(
          tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),
          2)  #has dim (Nt, d, 1)
      right_envs[site - 1] = tf.reshape(
          tf.matmul(tmp, tmp2), (Nt, D, din, dout))

  @staticmethod
  def add_unitary_batched_left(site, left_envs, mps, samples, one_body_gates,
                               two_body_gates):
    #if site is even, add an odd gate
    #if site is odd, add an even gate
    assert (site < len(mps))
    assert (len(mps) % 2 == 0)

    if site == 0:
      left_envs[site + 1] = tf.squeeze(
          misc_mps.ncon([
              mps.get_tensor(site), one_body_gates[site],
              two_body_gates[(site, site + 1)],
              tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype)
          ], [[-5, 1, -2], [2, 1], [3, -4, 2, -3], [-1, 3]]), 4)
    elif (site > 0) and (site % 2 == 0):
      tmp = misc_mps.ncon([
          left_envs[site],
          mps.get_tensor(site), one_body_gates[site], two_body_gates[(site,
                                                                      site + 1)]
      ], [[-1, 1, 4, -5], [1, 2, -2], [3, 2], [4, -4, 3, -3]
         ])  #has shape (Nt, D, di, do, d)
      Nt, D, di, do, d = tmp.shape
      tmp2 = tf.expand_dims(
          tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),
          2)  #has dim (Nt, d, 1)
      tmp = tf.reshape(tmp, (Nt, D * di * do, d))
      left_envs[site + 1] = tf.reshape(tf.matmul(tmp, tmp2), (Nt, D, di, do))

    elif (site > 0) and (site % 2 == 1):
      tmp = misc_mps.ncon([
          left_envs[site],
          mps.get_tensor(site), one_body_gates[site], two_body_gates[(site,
                                                                      site + 1)]
      ], [[-1, 1, 3, 4], [1, 2, -2], [3, 2], [-5, -4, 4, -3]
         ])  #has shape (Nt, D, di, do, d)
      Nt, D, di, do, d = tmp.shape
      tmp2 = tf.expand_dims(
          tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),
          2)  #has dim (Nt, d, 1)
      tmp = tf.reshape(tmp, (Nt, D * di * do, d))
      left_envs[site + 1] = tf.reshape(tf.matmul(tmp, tmp2), (Nt, D, di, do))

  @staticmethod
  def get_two_body_env_batched(sites, left_envs, right_envs, one_body_gates,
                               mps, samples):
    assert ((len(mps) % 2) == 0)
    assert (sites[0] >= 0)
    assert (sites[1] < len(mps))
    if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(left_envs[sites[0]])
      renv_node = bnet.add_node(right_envs[sites[1]])
      right_sample_node = bnet.add_node(
          tf.one_hot(samples[:, sites[1]], ds[sites[1]], dtype=mps.dtype))
      left_sample_node = bnet.add_node(
          tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype))
      right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
      left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
      right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
      left_unitary_node = bnet.add_node(one_body_gates[sites[0]])

      order = [
          lenv_node[0], lenv_node[2], renv_node[2], left_unitary_node[0],
          right_unitary_node[0]
      ]
      e1 = bnet.connect(lenv_node[3], left_sample_node[1])
      e2 = bnet.connect(lenv_node[1], left_tensor_node[0])
      e3 = bnet.connect(renv_node[3], right_sample_node[1])
      e4 = bnet.connect(renv_node[1], right_tensor_node[2])
      e5 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
      e6 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
      e7 = bnet.connect(right_tensor_node[1], right_unitary_node[1])

      bnet.batched_contract_between(lenv_node, left_sample_node, lenv_node[0],
                                    left_sample_node[0])
      bnet.batched_contract_between(renv_node, right_sample_node, renv_node[0],
                                    right_sample_node[0])
      bnet.contract(e6)
      bnet.contract(e7)
      rtmp = bnet.contract(e4)
      ltmp = bnet.contract(e2)
      out = bnet.batched_contract_between(ltmp, rtmp, lenv_node[0],
                                          renv_node[0])
      out = out.reorder_edges(order)
      return out.tensor

    #odd sites
    elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(tf.expand_dims(
          left_envs[sites[0]], 4))  #add fake legs to the tensors
      renv_node = bnet.add_node(tf.expand_dims(right_envs[sites[1]], 4))
      right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
      left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
      left_sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype),
              2))
      right_sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, sites[1]], ds[sites[0]], dtype=mps.dtype),
              2))
      right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
      right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
      left_unitary_node = bnet.add_node(one_body_gates[sites[0]])

      order = [
          lenv_node[0], left_sample_node[1], right_sample_node[1], lenv_node[3],
          renv_node[3]
      ]

      e1 = bnet.connect(lenv_node[1], left_tensor_node[0])
      e2 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
      e3 = bnet.connect(lenv_node[2], left_unitary_node[0])
      e4 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
      e5 = bnet.connect(right_tensor_node[1], right_unitary_node[1])
      e6 = bnet.connect(renv_node[2], right_unitary_node[0])
      e7 = bnet.connect(renv_node[1], right_tensor_node[2])
      e8 = bnet.connect(lenv_node[4], left_sample_node[2])
      e9 = bnet.connect(renv_node[4], right_sample_node[2])

      ltmp = bnet.contract(e2)  #contract one-body unitaries
      rtmp = bnet.contract(e5)

      ltmp = bnet.contract_between(lenv_node, ltmp)
      rtmp = bnet.contract_between(renv_node, rtmp)

      ctmp = bnet.batched_contract_between(ltmp, rtmp, lenv_node[0],
                                           renv_node[0])
      ctmp = bnet.batched_contract_between(ctmp, left_sample_node, lenv_node[0],
                                           left_sample_node[0])
      out = bnet.batched_contract_between(ctmp, right_sample_node, lenv_node[0],
                                          right_sample_node[0])
      out = out.reorder_edges(order)
      return out.tensor

    elif sites[0] == 0:
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      renv_node = bnet.add_node(tf.expand_dims(right_envs[sites[1]], 4))
      right_sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, sites[1]], ds[sites[1]], dtype=mps.dtype),
              2))
      left_sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype),
              2))
      right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
      left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
      right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
      left_unitary_node = bnet.add_node(one_body_gates[sites[0]])

      order = [
          renv_node[0], left_sample_node[1], renv_node[2], left_unitary_node[0],
          right_unitary_node[0]
      ]

      e1 = bnet.connect(left_tensor_node[0], left_sample_node[2])
      e2 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
      e3 = bnet.connect(renv_node[1], right_tensor_node[2])
      e4 = bnet.connect(renv_node[3], right_sample_node[1])
      e5 = bnet.connect(renv_node[4], right_sample_node[2])
      e6 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
      e7 = bnet.connect(right_tensor_node[1], right_unitary_node[1])

      bnet.contract(e6)
      bnet.contract(e7)

      bnet.batched_contract_between(renv_node, right_sample_node, renv_node[0],
                                    right_sample_node[0])
      bnet.contract(e3)
      tmp = bnet.contract(e2)
      out = bnet.batched_contract_between(tmp, left_sample_node, renv_node[0],
                                          left_sample_node[0])
      out = out.reorder_edges(order)
      return out.tensor

    elif sites[1] == (len(mps) - 1):
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(tf.expand_dims(left_envs[sites[0]], 4))
      right_sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, sites[1]], ds[sites[1]], dtype=mps.dtype),
              2))
      left_sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype),
              2))
      right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
      left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
      right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
      left_unitary_node = bnet.add_node(one_body_gates[sites[0]])

      order = [
          lenv_node[0], lenv_node[2], right_sample_node[1],
          left_unitary_node[0], right_unitary_node[0]
      ]
      e1 = bnet.connect(right_tensor_node[2], right_sample_node[2])
      e2 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
      e3 = bnet.connect(lenv_node[1], left_tensor_node[0])
      e4 = bnet.connect(lenv_node[3], left_sample_node[1])
      e5 = bnet.connect(lenv_node[4], left_sample_node[2])
      e6 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
      e7 = bnet.connect(right_tensor_node[1], right_unitary_node[1])

      bnet.contract(e6)
      bnet.contract(e7)

      bnet.batched_contract_between(lenv_node, left_sample_node, lenv_node[0],
                                    left_sample_node[0])
      bnet.contract(e3)
      tmp = bnet.contract(e2)
      out = bnet.batched_contract_between(tmp, right_sample_node, lenv_node[0],
                                          right_sample_node[0])
      out = out.reorder_edges(order)
      return out.tensor

  @staticmethod
  def get_one_body_env_batched(site, left_envs, right_envs, mps, samples):
    if (site not in (0, len(mps) - 1)) and (site % 2 == 1):
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(tf.expand_dims(left_envs[site], 4))
      renv_node = bnet.add_node(right_envs[site])
      sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype), 2))
      tensor_node = bnet.add_node(mps.get_tensor(site))

      order = [lenv_node[0], lenv_node[2], tensor_node[1]]

      e1 = bnet.connect(lenv_node[1], tensor_node[0])
      e2 = bnet.connect(renv_node[1], tensor_node[2])
      e3 = bnet.connect(lenv_node[3], renv_node[2])
      e4 = bnet.connect(renv_node[3], sample_node[1])
      e5 = bnet.connect(lenv_node[4], sample_node[2])

      tmp = bnet.contract_between(lenv_node, tensor_node)
      tmp = bnet.batched_contract_between(tmp, renv_node, lenv_node[0],
                                          renv_node[0])
      out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0],
                                          sample_node[0])
      out = out.reorder_edges(order)
      return out.tensor

    elif (site not in (0, len(mps) - 1)) and (site % 2) == 0:
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(tf.expand_dims(left_envs[site], 4))
      renv_node = bnet.add_node(right_envs[site])
      sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype), 2))
      tensor_node = bnet.add_node(mps.get_tensor(site))

      order = [lenv_node[0], renv_node[2], tensor_node[1]]

      e1 = bnet.connect(lenv_node[1], tensor_node[0])
      e2 = bnet.connect(renv_node[1], tensor_node[2])
      e3 = bnet.connect(lenv_node[2], renv_node[3])
      e4 = bnet.connect(lenv_node[3], sample_node[1])
      e5 = bnet.connect(lenv_node[4], sample_node[2])

      tmp = bnet.contract_between(lenv_node, tensor_node)
      tmp = bnet.batched_contract_between(tmp, renv_node, lenv_node[0],
                                          renv_node[0])
      out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0],
                                          sample_node[0])
      out = out.reorder_edges(order)
      return out.tensor

    elif site == 0:
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      renv_node = bnet.add_node(right_envs[site])
      sample_node = bnet.add_node(
          tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype))
      tensor_node = bnet.add_node(
          tf.squeeze(mps.get_tensor(site), 0)
      )  #remove trailing 0th dimension of the mps-tensor at the left boundary

      order = [renv_node[0], renv_node[2], tensor_node[0]]

      e1 = bnet.connect(renv_node[1], tensor_node[1])
      e2 = bnet.connect(renv_node[3], sample_node[1])

      tmp = bnet.contract_between(renv_node, tensor_node)
      out = bnet.batched_contract_between(tmp, sample_node, renv_node[0],
                                          sample_node[0])
      out = out.reorder_edges(order)
      return out.tensor

    elif site == (len(mps) - 1):
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(left_envs[site])
      sample_node = bnet.add_node(
          tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype))
      tensor_node = bnet.add_node(tf.squeeze(
          mps.get_tensor(site), 2))  #remove trailing dimension from mps tensor

      order = [lenv_node[0], lenv_node[2], tensor_node[1]]

      e1 = bnet.connect(lenv_node[1], tensor_node[0])
      e2 = bnet.connect(lenv_node[3], sample_node[1])

      tmp = bnet.contract_between(lenv_node, tensor_node)
      out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0],
                                          sample_node[0])
      out = out.reorder_edges(order)
      return out.tensor

  @staticmethod
  def check_overlap_batched(site, left_envs, right_envs, mps, samples):
    """
        a check ; this should return one mps was normalized and samples
        are all basis states
        """
    if site % 2 == 1:
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(tf.expand_dims(left_envs[site], 4))
      renv_node = bnet.add_node(right_envs[site])
      sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype), 2))
      tensor_node = bnet.add_node(mps.get_tensor(site))

      e1 = bnet.connect(lenv_node[1], tensor_node[0])
      e2 = bnet.connect(renv_node[1], tensor_node[2])
      e3 = bnet.connect(lenv_node[2], tensor_node[1])
      e4 = bnet.connect(lenv_node[3], renv_node[2])
      e5 = bnet.connect(renv_node[3], sample_node[1])
      e6 = bnet.connect(lenv_node[4], sample_node[2])
      tmp = bnet.contract_between(lenv_node, tensor_node)
      tmp = bnet.batched_contract_between(tmp, renv_node, lenv_node[0],
                                          renv_node[0])
      out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0],
                                          sample_node[0])
      return tf.math.reduce_sum(tf.pow(out.tensor, 2), axis=0)

    elif site % 2 == 0:
      ds = mps.d
      bnet = btn.BatchTensorNetwork()
      lenv_node = bnet.add_node(tf.expand_dims(left_envs[site], 4))
      renv_node = bnet.add_node(right_envs[site])
      sample_node = bnet.add_node(
          tf.expand_dims(
              tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype), 2))
      tensor_node = bnet.add_node(mps.get_tensor(site))

      e1 = bnet.connect(lenv_node[1], tensor_node[0])
      e2 = bnet.connect(renv_node[1], tensor_node[2])
      e3 = bnet.connect(renv_node[2], tensor_node[1])
      e4 = bnet.connect(lenv_node[2], renv_node[3])
      e5 = bnet.connect(lenv_node[3], sample_node[1])
      e6 = bnet.connect(lenv_node[4], sample_node[2])
      tmp = bnet.contract_between(lenv_node, tensor_node)
      tmp = bnet.batched_contract_between(tmp, renv_node, lenv_node[0],
                                          renv_node[0])
      out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0],
                                          sample_node[0])
      return tf.math.reduce_sum(tf.pow(out.tensor, 2), axis=0)

  @staticmethod
  def overlap_batched(site, left_envs, right_envs, one_body_gates, mps,
                      samples):

    env = tf.math.reduce_mean(
        OverlapMaximizer.get_one_body_env_batched(site, left_envs, right_envs,
                                                  mps, samples),
        axis=0)
    return misc_mps.ncon([env, one_body_gates[site]], [[1, 2], [1, 2]])

    # if site%2 == 1:
    #     ds = mps.d
    #     bnet = btn.BatchTensorNetwork()
    #     lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
    #     renv_node = bnet.add_node(right_envs[site])
    #     unitary_node = bnet.add_node(one_body_gates[site])
    #     sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
    #     tensor_node = bnet.add_node(mps.get_tensor(site))

    #     e1 = bnet.connect(lenv_node[1], tensor_node[0])
    #     e2 = bnet.connect(renv_node[1], tensor_node[2])
    #     e3 = bnet.connect(unitary_node[1], tensor_node[1])
    #     e4 = bnet.connect(lenv_node[2], unitary_node[0])
    #     e5 = bnet.connect(lenv_node[3], renv_node[2])
    #     e6 = bnet.connect(renv_node[3], sample_node[1])
    #     e7 = bnet.connect(lenv_node[4], sample_node[2])

    #     tmp = bnet.contract_between(tensor_node, unitary_node)
    #     tmp = bnet.contract_between(lenv_node, tmp)
    #     tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
    #     out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
    #     return tf.math.reduce_mean(out.tensor, axis=0)#/np.sqrt(samples.shape[0])

    # elif site%2 == 0:
    #     ds = mps.d
    #     bnet = btn.BatchTensorNetwork()
    #     lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
    #     renv_node = bnet.add_node(right_envs[site])
    #     unitary_node = bnet.add_node(one_body_gates[site])
    #     sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
    #     tensor_node = bnet.add_node(mps.get_tensor(site))

    #     e1 = bnet.connect(lenv_node[1], tensor_node[0])
    #     e2 = bnet.connect(renv_node[1], tensor_node[2])
    #     e3 = bnet.connect(tensor_node[1], unitary_node[1])
    #     e4 = bnet.connect(renv_node[2], unitary_node[0])
    #     e5 = bnet.connect(lenv_node[2], renv_node[3])
    #     e6 = bnet.connect(lenv_node[3], sample_node[1])
    #     e7 = bnet.connect(lenv_node[4], sample_node[2])

    #     tmp = bnet.contract_between(tensor_node, unitary_node)
    #     tmp = bnet.contract_between(lenv_node, tmp)
    #     tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
    #     out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
    #     return tf.math.reduce_mean(out.tensor, axis=0)#/np.sqrt(samples.shape[0])

  @staticmethod
  def add_unitary_right(site,
                        right_envs,
                        mps,
                        conj_mps,
                        one_body_gates,
                        two_body_gates,
                        normalize=False):
    #if site is even, add an odd gate
    #if site is odd, add an even gate
    assert (site > 0)
    assert (len(mps) % 2 == 0)
    tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]],
                           [[-1, 1, -3], [-2, 1]])
    if site == (len(mps) - 1):
      right_envs[site - 1] = misc_mps.ncon([
          tensor,
          tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site - 1, site)]
      ], [[-1, 2, 1], [-2, 3, 1], [-4, 3, -3, 2]])
    elif (site < len(mps) - 1) and (site % 2 == 0):
      right_envs[site - 1] = misc_mps.ncon([
          right_envs[site], tensor,
          tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site - 1, site)]
      ], [[1, 3, 2, 5], [-1, 2, 1], [-2, 4, 3], [-4, 4, -3, 5]])
    elif (site < len(mps) - 1) and (site % 2 == 1):
      right_envs[site - 1] = misc_mps.ncon([
          right_envs[site], tensor,
          tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site - 1, site)]
      ], [[1, 4, 3, 5], [-1, 2, 1], [-2, 5, 4], [-4, 3, -3, 2]])
    if normalize:
      right_envs[site - 1] /= tf.linalg.norm(right_envs[site - 1])

  @staticmethod
  def add_unitary_left(site,
                       left_envs,
                       mps,
                       conj_mps,
                       one_body_gates,
                       two_body_gates,
                       normalize=False):
    #if site is even, add an odd gate
    #if site is odd, add an even gate
    assert (site < len(mps) - 1)
    assert (len(mps) % 2 == 0)
    tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]],
                           [[-1, 1, -3], [-2, 1]])
    if site == 0:
      left_envs[site + 1] = misc_mps.ncon([
          tensor,
          tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site, site + 1)]
      ], [[1, 2, -1], [1, 3, -2], [3, -4, 2, -3]])
    elif (site > 0) and (site % 2 == 0):
      left_envs[site + 1] = misc_mps.ncon([
          left_envs[site], tensor,
          tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site, site + 1)]
      ], [[1, 4, 3, 5], [1, 2, -1], [4, 5, -2], [3, -4, 2, -3]])
    elif (site > 0) and (site % 2 == 1):
      left_envs[site + 1] = misc_mps.ncon([
          left_envs[site], tensor,
          tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site, site + 1)]
      ], [[1, 3, 2, 5], [1, 2, -1], [3, 4, -2], [4, -4, 5, -3]])
    if normalize:
      left_envs[site + 1] /= tf.linalg.norm(left_envs[site + 1])

  def compute_right_envs(self, ref_mps):
    [
        self.add_unitary_right(site, self.right_envs, self.mps, ref_mps,
                               self.one_body_gates, self.two_body_gates)
        for site in reversed(range(1, len(self.mps)))
    ]

  def compute_left_envs(self, ref_mps):
    [
        self.add_unitary_left(site, self.left_envs, self.mps, ref_mps,
                              self.one_body_gates, self.two_body_gates)
        for site in range(len(self.mps) - 1)
    ]

  def compute_right_envs_batched(self, samples):
    [
        self.add_unitary_batched_right(site, self.right_envs_batched, self.mps,
                                       samples, self.one_body_gates,
                                       self.two_body_gates)
        for site in reversed(range(1, len(self.mps)))
    ]

  def compute_left_envs_batched(self, samples):
    [
        self.add_unitary_batched_left(site, self.left_envs_batched, self.mps,
                                      samples, self.one_body_gates,
                                      self.two_body_gates)
        for site in range(len(self.mps) - 1)
    ]

  @staticmethod
  def get_two_body_env(sites,
                       left_envs,
                       right_envs,
                       one_body_gates,
                       mps,
                       conj_mps,
                       normalize=False):
    """
        compute the environment of the two-body unitary at sites `sites`
        of the network <conj_mps|U|mps>
        """
    assert ((len(mps) % 2) == 0)
    tensor_left = misc_mps.ncon(
        [mps.get_tensor(sites[0]), one_body_gates[sites[0]]],
        [[-1, 1, -3], [-2, 1]])
    tensor_right = misc_mps.ncon(
        [mps.get_tensor(sites[1]), one_body_gates[sites[1]]],
        [[-1, 1, -3], [-2, 1]])

    if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
      env = misc_mps.ncon([
          left_envs[sites[0]], tensor_left, tensor_right, right_envs[sites[1]],
          tf.conj(conj_mps.get_tensor(sites[0])),
          tf.conj(conj_mps.get_tensor(sites[1]))
      ], [[7, 1, -1, 2], [7, -3, 6], [6, -4, 8], [8, 4, -2, 3], [1, 2, 5],
          [5, 3, 4]])
    elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
      env = misc_mps.ncon([
          left_envs[sites[0]], tensor_left, tensor_right, right_envs[sites[1]],
          tf.conj(conj_mps.get_tensor(sites[0])),
          tf.conj(conj_mps.get_tensor(sites[1]))
      ], [[1, 7, 2, -3], [1, 2, 3], [3, 4, 5], [5, 8, 4, -4], [7, -1, 6],
          [6, -2, 8]])
    elif sites[0] == 0:
      env = misc_mps.ncon([
          tensor_left, tensor_right, right_envs[sites[1]],
          tf.conj(conj_mps.get_tensor(sites[0])),
          tf.conj(conj_mps.get_tensor(sites[1]))
      ], [[1, -3, 2], [2, -4, 3], [3, 4, -2, 5], [1, -1, 6], [6, 5, 4]])
    elif sites[1] == (len(mps) - 1):
      env = misc_mps.ncon([
          left_envs[sites[0]], tensor_left, tensor_right,
          tf.conj(conj_mps.get_tensor(sites[0])),
          tf.conj(conj_mps.get_tensor(sites[1]))
      ], [[5, 4, -1, 3], [5, -3, 1], [1, -4, 6], [4, 3, 2], [2, -2, 6]])
    if normalize:
      env /= tf.linalg.norm(env)
    return env

  @staticmethod
  def get_one_body_env(site,
                       left_envs,
                       right_envs,
                       mps,
                       conj_mps,
                       ref_sym=False,
                       normalize=False):
    """
        compute the environment of the one-body unitary at site `site`
        of the network <conj_mps|U|mps>
        """

    if (site not in (0, len(mps) - 1)) and (site % 2 == 1):
      env = misc_mps.ncon([
          left_envs[site],
          mps.get_tensor(site),
          tf.conj(conj_mps.get_tensor(site)), right_envs[site]
      ], [[1, 5, -1, 4], [1, -2, 3], [5, 6, 7], [3, 7, 4, 6]])

    elif (site not in (0, len(mps) - 1)) and (site % 2 == 0):
      env = misc_mps.ncon([
          left_envs[site],
          mps.get_tensor(site),
          tf.conj(conj_mps.get_tensor(site)), right_envs[site]
      ], [[1, 5, 4, 6], [1, -2, 3], [5, 6, 7], [3, 7, -1, 4]])
    elif site == 0:
      env = misc_mps.ncon([
          mps.get_tensor(site),
          tf.conj(conj_mps.get_tensor(site)), right_envs[site]
      ], [[1, -2, 2], [1, 3, 4], [2, 4, -1, 3]])
    elif site == (len(mps) - 1):
      env = misc_mps.ncon([
          left_envs[site],
          mps.get_tensor(site),
          tf.conj(conj_mps.get_tensor(site))
      ], [[1, 3, -1, 4], [1, -2, 2], [3, 4, 2]])

    if normalize:
      env /= tf.linalg.norm(env)
    if ref_sym:
      env = (env + tf.transpose(tf.conj(env))) / 2.0

    return env

  @staticmethod
  def one_body_gradient(site, left_envs, right_envs, mps, conj_mps,
                        one_body_generators):
    """
        gradient with respect to the one-body generator at site `site` of the network <conj_mps|U|mps>.
        one-body unitaries are parametrized as tf.linalg.expm(g - herm(g)) with g an arbitrary real or complex matrix g.
        The gradient takes this into account
        """
    env = OverlapMaximizer.get_one_body_env(site, left_envs, right_envs, mps,
                                            conj_mps)
    with tf.GradientTape() as tape:
      g = one_body_generators[site]
      tape.watch(g)
      if mps.dtype in (tf.complex128, tf.complex64):
        tmp = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[1, 2], [1, 2]])
        #res = (1.0 + 1j)/2 * tmp + (1.0 - 1j)/2 * tf.conj(tmp)
        #overlap = tf.real(tmp) - tf.abs(tf.imag(tmp))
        overlap = 1 / 2 * (tmp + tf.conj(tmp))
        #the ones below shift the means of the signs around in the complex plane
        #overlap = -tf.real(tmp) + tf.abs(tf.imag(tmp))
        #overlap = tf.real(tmp) - tf.abs(tf.imag(tmp))
      elif mps.dtype in (tf.float64, tf.float32):
        overlap = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[1, 2], [1, 2]])
      else:
        raise TypeError('unsupported dtype {0}'.format(dtype))

    return tape.gradient(overlap, g), overlap

  @staticmethod
  def two_body_gradient(sites, left_envs, right_envs, one_body_gates, mps,
                        conj_mps, two_body_generators):
    """
        gradient with respect to the two-body generator at sites `sites` of the network <conj_mps|U|mps>.
        two-body unitaries are parametrized as tf.linalg.expm(g - herm(g)) with g an arbitrary real or complex matrix g.
        The gradient takes this into account
        """

    env = OverlapMaximizer.get_two_body_env(sites, left_envs, right_envs,
                                            one_body_gates, mps, conj_mps)
    ds = mps.d
    with tf.GradientTape() as tape:
      g = two_body_generators[sites]
      tape.watch(g)
      if mps.dtype in (tf.complex128, tf.complex64):
        tmp = misc_mps.ncon([
            tf.reshape(
                env,
                (ds[sites[0]] * ds[sites[1]], ds[sites[0]] * ds[sites[1]])),
            tf.linalg.expm(g - tf.conj(tf.transpose(g)))
        ], [[1, 2], [1, 2]])
        res = (1.0 + 1j) / 2 * tmp + (1.0 - 1j) / 2 * tf.conj(tmp)
      elif mps.dtype in (tf.float64, tf.float32):
        res = misc_mps.ncon([
            tf.reshape(
                env,
                (ds[sites[0]] * ds[sites[1]], ds[sites[0]] * ds[sites[1]])),
            tf.linalg.expm(g - tf.conj(tf.transpose(g)))
        ], [[1, 2], [1, 2]])
      else:
        raise TypeError('unsupported dtype {0}'.format(dtype))

    return tape.gradient(res, g)

  @staticmethod
  def one_body_gradient_cost_function_batched(site,
                                              left_envs,
                                              right_envs,
                                              mps,
                                              samples,
                                              one_body_generators,
                                              activation=None,
                                              gamma=0.1):
    """
        computes the gradients for the one-body gates given `samples`. 
        `samples` represent a collection of basis states.
        This function computes

                  \sum_{\sigma} \partial C_{\sigma} + 2 * C_{\sigma} * \Re \partial(\log(\psi_{\sigma}))
        with 
                  C_{\sigma} = ((1-\gamma) `activation`(\Re(\psi_{\sigma}))) - \gamma * |\Im(\psi_{\sigma})|

        which is the gradient of the cost function C

                   C = \sum_{\sigma} |\psi_{\sigma}|^2  C_{\sigma} 
        
        Args: 
            site (int):         site of the one-body unitary for which to return the gradient
            left_envs (dict mapping int -> tf.Tensor):   dictionary of left-environments
            right_envs (dict mapping int -> tf.Tensor):   dictionary of right-environments
            mps (FiniteMPSCentralGauge):   an mps
            samples (tf.Tensor of shape (Nt, len(mps)):   the samples
            one_body_generators (dict mapping integer `site` -> tf.Tensor of shape `(mps.d[site],mps.d[site])`):  the generators `g` of the one-body unitaries
                                                                                                                  U = expm(`g`-herm(`g`))
           activation (callable):  activation function, see above
           gamma (float):          see above
        Returns:
            tuple containing (gradient, avsigns, C)
            gradient (tf.Tensor):  the gradient of `one_body_generator[site]'
            avsigns( tf.Tensor):   the average current sign
            C (tf.Tensor (scalar)):the value of the cost function
        """
    env = OverlapMaximizer.get_one_body_env_batched(site, left_envs, right_envs,
                                                    mps, samples)
    ds = mps.d
    with tf.GradientTape(persistent=True) as tape:
      g = one_body_generators[site]
      tape.watch(g)
      if mps.dtype in (tf.complex128, tf.complex64):
        psi_sigma = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[-1, 1, 2], [1, 2]])
        avsigns = np.average(np.sign(np.real(psi_sigma.numpy(
        )))) + 1j * np.average(np.sign(np.imag(psi_sigma.numpy())))
        log_psi = tf.math.reduce_mean(tf.math.log(psi_sigma), axis=0)
        if activation is not None:
          C = tf.complex(
              tf.math.reduce_mean(
                  (1 - gamma) * activation(tf.real(psi_sigma)) -
                  gamma * np.abs(tf.imag(psi_sigma)),
                  axis=0), tf.zeros(shape=[1], dtype=mps.dtype.real_dtype))
        else:
          C = tf.complex(
              tf.math.reduce_mean(
                  (1 - gamma) * tf.real(psi_sigma) -
                  gamma * np.abs(tf.imag(psi_sigma)),
                  axis=0), tf.zeros(shape=[1], dtype=mps.dtype.real_dtype))
          #C =  (tf.math.reduce_sum(psi_sigma) + tf.math.reduce_sum(tf.conj(psi_sigma)))/np.sqrt(samples.shape[0])
      elif mps.dtype in (tf.float64, tf.float32):
        psi_sigma = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[-1, 1, 2], [1, 2]])
        log_psi = tf.math.reduce_mean(tf.math.log(psi_sigma), axis=0)
        avsigns = np.average(np.sign(psi_sigma.numpy()))
        if activation is not None:
          C = tf.math.reduce_mean(activation(psi_sigma), axis=0)
        else:
          C = tf.math.reduce_mean(psi_sigma, axis=0)
      else:
        raise TypeError('unsupported dtype {0}'.format(dtype))
    g1 = tape.gradient(C, g)
    if mps.dtype in (tf.complex128, tf.complex64):
      g2 = tf.complex(
          tf.real(tape.gradient(log_psi, g)),
          tf.zeros(shape=g1.shape, dtype=g1.dtype.real_dtype))
    elif mps.dtype in (tf.float64, tf.float32):
      g2 = tape.gradient(log_psi, g)
    del tape
    return g1 + 2 * C * g2, avsigns, C
    #return  g1, avsigns, C

  @staticmethod
  def one_body_gradient_overlap_batched(site, left_envs, right_envs, mps,
                                        samples, one_body_generators):
    """
        computes the gradients for the one-body gates given `samples`. 
        
        `samples` represent a collection of basis states.
        This function computes the gradient for minimizing the difference
        between `mps` and the equal superposition of `samples`
        
        Args: 
            site (int):         site of the one-body unitary for which to return the gradient
            left_envs (dict mapping int -> tf.Tensor):   dictionary of left-environments
            right_envs (dict mapping int -> tf.Tensor):   dictionary of right-environments
            mps (FiniteMPSCentralGauge):   an mps
            samples (tf.Tensor of shape (Nt, len(mps)):   the samples
            one_body_generators (dict mapping integer `site` -> tf.Tensor of shape `(mps.d[site],mps.d[site])`):  the generators `g` of the one-body unitaries
                                                                                                                  U = expm(`g`-herm(`g`))
        Returns:
            tuple containing (gradient, avsigns, C)
            gradient (tf.Tensor):  the gradient of `one_body_generator[site]'
            avsigns( tf.Tensor):   the average current sign
            C (tf.Tensor (scalar)):the value of the cost function
        """
    env = OverlapMaximizer.get_one_body_env_batched(site, left_envs, right_envs,
                                                    mps, samples)
    ds = mps.d
    with tf.GradientTape(persistent=True) as tape:
      g = one_body_generators[site]
      tape.watch(g)
      if mps.dtype in (tf.complex128, tf.complex64):
        psi_sigma = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[-1, 1, 2], [1, 2]])
        avsigns = np.average(np.sign(np.real(psi_sigma.numpy(
        )))) + 1j * np.average(np.sign(np.imag(psi_sigma.numpy())))
        log_psi = tf.math.reduce_mean(tf.math.log(psi_sigma), axis=0)
        C = tf.math.reduce_mean(
            psi_sigma, axis=0) + tf.math.reduce_mean(
                tf.conj(psi_sigma), axis=0)
      elif mps.dtype in (tf.float64, tf.float32):
        psi_sigma = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[-1, 1, 2], [1, 2]])
        log_psi = tf.math.reduce_mean(tf.math.log(psi_sigma), axis=0)
        avsigns = np.average(np.sign(psi_sigma.numpy()))
        C = tf.math.reduce_mean(psi_sigma, axis=0)
      else:
        raise TypeError('unsupported dtype {0}'.format(dtype))

    return tape.gradient(C, g), avsigns, C

  @staticmethod
  def test_one_body_gradient_overlap_batched(site, left_envs, right_envs, mps,
                                             samples, one_body_generators):
    """
        computes the gradients for the one-body gates given `samples`. 
        
        `samples` represent a collection of basis states.
        This function computes the gradient for minimizing the difference
        between `mps` and the equal superposition of `samples`
        
        Args: 
            site (int):         site of the one-body unitary for which to return the gradient
            left_envs (dict mapping int -> tf.Tensor):   dictionary of left-environments
            right_envs (dict mapping int -> tf.Tensor):   dictionary of right-environments
            mps (FiniteMPSCentralGauge):   an mps
            samples (tf.Tensor of shape (Nt, len(mps)):   the samples
            one_body_generators (dict mapping integer `site` -> tf.Tensor of shape `(mps.d[site],mps.d[site])`):  the generators `g` of the one-body unitaries
                                                                                                                  U = expm(`g`-herm(`g`))
        Returns:
            tuple containing (gradient, avsigns, C)
            gradient (tf.Tensor):  the gradient of `one_body_generator[site]'
            avsigns( tf.Tensor):   the average current sign
            C (tf.Tensor (scalar)):the value of the cost function
        """
    env = OverlapMaximizer.get_one_body_env_batched(site, left_envs, right_envs,
                                                    mps, samples)
    ds = mps.d
    with tf.GradientTape(persistent=True) as tape:
      g = one_body_generators[site]
      tape.watch(g)
      if mps.dtype in (tf.complex128, tf.complex64):
        psi_sigma = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[-1, 1, 2], [1, 2]])
        avsigns = np.average(np.sign(np.real(psi_sigma.numpy(
        )))) + 1j * np.average(np.sign(np.imag(psi_sigma.numpy())))
        C1 = (tf.math.reduce_sum(psi_sigma, axis=0) + tf.math.reduce_sum(
            tf.conj(psi_sigma), axis=0)) / np.sqrt(samples.shape[0])
        C2 = tf.math.reduce_sum(
            psi_sigma * tf.conj(psi_sigma), axis=0)  #the norm
      elif mps.dtype in (tf.float64, tf.float32):
        psi_sigma = misc_mps.ncon(
            [env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],
            [[-1, 1, 2], [1, 2]])
        avsigns = np.average(np.sign(psi_sigma.numpy()))
        C1 = tf.math.reduce_sum(psi_sigma, axis=0) / np.sqrt(samples.shape[0])
        C2 = tf.math.reduce_sum(tf.pow(psi_sigma, 2), axis=0)  #the norm
      else:
        raise TypeError('unsupported dtype {0}'.format(dtype))

    return tape.gradient(C1, g), avsigns, C1, C2

  @staticmethod
  def two_body_gradient_batched(sites,
                                left_envs,
                                right_envs,
                                one_body_gates,
                                mps,
                                samples,
                                two_body_generators,
                                activation=None,
                                gamma=0.1):
    """
        computes the gradients for the two-body gates given `samples`. 
        `samples` represent a collection of basis states.
        This function computes


                  \sum_{\sigma} \partial C_{\sigma} + 2 * C_{\sigma} * \Re \partial(\log(\psi_{\sigma}))
        with 
                  C_{\sigma} = ((1-\gamma) `activation`(\Re(\psi_{\sigma}))) - \gamma * |\Im(\psi_{\sigma})|

        which is the gradient of the cost function C

                   C = \sum_{\sigma} |\psi_{\sigma}|^2  C_{\sigma} 
        Args: 
            sites (tuple of int):         site of the one-body unitary for which to return the gradient
            left_envs (dict mapping int -> tf.Tensor):   dictionary of left-environments
            right_envs (dict mapping int -> tf.Tensor):   dictionary of right-environments
            one_body_gates (iterable or None): an iterable mapping integer `site` to `tf.Tensor`
                                               `one_body_gates[site]` is the one-body unitary  at site `site`
            mps (FiniteMPSCentralGauge):   an mps
            samples (tf.Tensor of shape (Nt, len(mps)):   the samples
            two_body_generators (dict mapping tuple (site, site+1) -> tf.Tensor 
                                 of shape `(mps.d[sites[0]] * mps.d[sites[1]], mps.d[sites[0]] * mps.d[sites[1]])`):  the generators `g` of the two-body unitaries
                                                                                                                      U = expm(`g`-herm(`g`))
            activation (callable):  activation function, see above
            gamma (float):          see above
        Returns:
            tuple containing (gradient, avsigns, C)
            gradient (tf.Tensor):  the gradient of `one_body_generator[site]'
            avsigns( tf.Tensor):   the average current sign
            C (tf.Tensor (scalar)):the value of the cost function
          """

    env = OverlapMaximizer.get_two_body_env_batched(
        sites, left_envs, right_envs, one_body_gates, mps,
        samples)  #shape (Nt, d,d,d,d)
    ds = mps.d
    with tf.GradientTape(persistent=True) as tape:
      g = two_body_generators[sites]
      tape.watch(g)
      if mps.dtype in (tf.complex128, tf.complex64):
        psi_sigma = misc_mps.ncon([
            tf.reshape(env, (samples.shape[0], ds[sites[0]] * ds[sites[1]],
                             ds[sites[0]] * ds[sites[1]])),
            tf.linalg.expm(g - tf.conj(tf.transpose(g)))
        ], [[-1, 1, 2], [1, 2]])
        avsigns = np.average(np.sign(np.real(psi_sigma.numpy(
        )))) + 1j * np.average(np.sign(np.imag(psi_sigma.numpy())))
        log_psi = tf.math.reduce_mean(tf.math.log(psi_sigma), axis=0)
        if activation is not None:
          C = tf.complex(
              tf.math.reduce_mean(
                  (1 - gamma) * activation(tf.real(psi_sigma)) -
                  gamma * np.abs(tf.imag(psi_sigma)),
                  axis=0), tf.zeros(shape=[1], dtype=mps.dtype.real_dtype))
        else:
          C = tf.math.reduce_mean(psi_sigma, axis=0)
      elif mps.dtype in (tf.float64, tf.float32):
        psi_sigma = misc_mps.ncon([
            tf.reshape(env, (samples.shape[0], ds[sites[0]] * ds[sites[1]],
                             ds[sites[0]] * ds[sites[1]])),
            tf.linalg.expm(g - tf.conj(tf.transpose(g)))
        ], [[-1, 1, 2], [1, 2]])
        avsigns = np.average(np.sign(psi_sigma.numpy()))
        log_psi = tf.math.reduce_mean(tf.math.log(psi_sigma), axis=0)
        if activation is not None:
          C = tf.math.reduce_mean(activation(psi_sigma), axis=0)
        else:
          C = tf.math.reduce_mean(psi_sigma, axis=0)
      else:
        raise TypeError('unsupported dtype {0}'.format(dtype))
    g1 = tape.gradient(C, g)
    if mps.dtype in (tf.complex128, tf.complex64):
      g2 = tf.complex(
          tf.real(tape.gradient(log_psi, g)),
          tf.zeros(shape=g1.shape, dtype=g1.dtype.real_dtype))
    elif mps.dtype in (tf.float64, tf.float32):
      g2 = tape.gradient(log_psi, g)
    del tape
    return g1 + 2 * C * g2, avsigns, C

  @staticmethod
  def overlap(site, left_envs, right_envs, one_body_gates, mps, conj_mps):
    """
        compute the overlap of U * `mps` with `conj_mps`
        """
    #assert(site>0)
    #assert(site<len(mps))
    env = OverlapMaximizer.get_one_body_env(site, left_envs, right_envs, mps,
                                            conj_mps)
    return misc_mps.ncon([one_body_gates[site], env], [[1, 2], [1, 2]])
    # if site%2 == 1:
    #     tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]], [[-1,1,-3],[-2, 1]])
    #     return misc_mps.ncon([left_envs[site], tensor, tf.conj(conj_mps.get_tensor(site)),
    #                           right_envs[site]],
    #                          [[1,5,2,4], [1,2,3], [5,6,7], [3,7,4,6]])
    # elif site%2 == 0:
    #     tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]], [[-1,1,-3],[-2, 1]])
    #     return misc_mps.ncon([left_envs[site], tensor, tf.conj(conj_mps.get_tensor(site)),
    #                     right_envs[site]],
    #                    [[1,5,4,6], [1,2,3], [5,6,7], [3,7,2,4]])
  @staticmethod
  def two_body_update_svd_numpy(wIn):
    """
        obtain the update to the disentangler using numpy svd
        Fixme: this currently only works with numpy arrays
        Args:
            wIn (np.ndarray or Tensor):  unitary tensor of rank 4
        Returns:
            The svd update of `wIn`
        """
    shape = tf.shape(wIn)
    ut, st, vt = np.linalg.svd(
        np.reshape(wIn, (shape[0] * shape[1], shape[2] * shape[3])),
        full_matrices=False)
    mat = misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1, -2]])
    return tf.reshape(mat, shape)

  @staticmethod
  def one_body_update_svd_numpy(env):
    """
        obtain the update to the disentangler using numpy svd
        Fixme: this currently only works with numpy arrays
        Args:
            wIn (np.ndarray or Tensor):  unitary tensor of rank 4
        Returns:
            The svd update of `wIn`
        """
    ut, st, vt = np.linalg.svd(env, full_matrices=False)
    return misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1, -2]])

  def absorb_one_body_gates(self):
    """
        absorb the one-body gates into a copy of self.mps 
        Args:
            no args
        Returns:
            FiniteMPSCentralGauge
        
        """
    tensors = [
        misc_mps.ncon([self.mps.get_tensor(site), self.one_body_gates[site]],
                      [[-1, 1, -3], [-2, 1]]) for site in range(len(self.mps))
    ]
    mps = MPS.FiniteMPSCentralGauge(tensors)
    mps.position(0)
    mps.position(len(mps), normalize=True)
    return mps

  def absorb_two_body_gates(self, Dmax=None):
    """
        absorb the two-body gates into a copy of self.mps 
        Args:
            Dmax (int):   the maximal bond dimension to be kept after absorbtion
        Returns:
            tuple containing 
            (FiniteMPSCentralGauge, float)
            an MPS and the truncated weight resulting from optional truncation
        
        """
    mps = copy.deepcopy(self.mps)
    for site in range(0, len(mps) - 1, 2):
      mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
    for site in range(1, len(mps) - 2, 2):
      mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
    mps.position(0)
    mps.position(len(mps), normalize=True)
    tw = mps.position(0, normalize=True, D=Dmax)
    return mps, tw

  def absorb_gates(self, Dmax=None):
    """
        absorb one- and two-body gates into a copy of self.mps 
        Args:
            Dmax (int):   the maximal bond dimension to be kept after absorbtion
        Returns:
            tuple containing 
            (FiniteMPSCentralGauge, float)
            an MPS and the truncated weight resulting from optional truncation
        
        """
    tensors = [
        misc_mps.ncon([self.mps.get_tensor(site), self.one_body_gates[site]],
                      [[-1, 1, -3], [-2, 1]]) for site in range(len(self.mps))
    ]
    mps = MPS.FiniteMPSCentralGauge(tensors)
    mps.position(0)
    mps.position(len(mps), normalize=True)
    for site in range(0, len(mps) - 1, 2):
      mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
    for site in range(1, len(mps) - 2, 2):
      mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
    mps.position(0)
    mps.position(len(mps), normalize=True)
    tw = mps.position(0, normalize=True, D=Dmax)
    return mps, tw

  def maximize_layerwise(self, ref_mps, num_sweeps, alpha=1.0, verbose=0):
    """
        deprecated
        maximize the overlap by optimizing over the even  and odd two-body unitaries,
        alternating between even and odd layer.
        minimization runs from left to right and right to left, and changes `gates` one at at time.
        this function is deprecated; use `maximize_even` and `maximize_odd` instead.
        Args:
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int):  number of iterations
            alpha (float):    determines the mixing of the update
                                   the new gate is given by `1- alpha` * old_gate + `alphas` * update
            verbose (int):         verbosity flag; larger means more output
        """
    assert (alpha <= 1.0)
    assert (alpha >= 0)
    [
        self.add_unitary_right(site, self.right_envs, self.mps, ref_mps,
                               self.one_body_gates, self.two_body_gates)
        for site in reversed(range(1, len(self.mps)))
    ]
    for it in range(num_sweeps):
      for site in range(len(self.mps) - 1):
        env = self.two_body_gates[
            (site, site + 1)] * (1 - alpha) + alpha * self.get_two_body_env(
                (site, site + 1), selff.left_envs, self.right_envs,
                one_body_gates, self.mps, ref_mps)
        self.two_body_gates[(site,
                             site + 1)] = self.two_body_update_svd_numpy(env)
        self.add_unitary_left(site, self.left_envs, self.mps, ref_mps,
                              self.one_body_gates, self.two_body_gates)
        if verbose > 0 and site > 0:
          overlap = self.overlap(site, self.left_envs, self.right_envs,
                                 self.one_body_gates, self.mps, ref_mps)
          stdout.write("\r iteration  %i/%i, overlap = %.6f" %
                       (it, num_sweeps, np.abs(np.real(overlap))))
          stdout.flush()
        if verbose > 1:
          print()

      for site in reversed(range(1, len(self.mps) - 1)):
        env = self.two_body_gates[
            (site, site + 1)] * (1 - alpha) + alpha * self.get_two_body_env(
                (site, site + 1), self.left_envs, self.right_envs,
                one_body_gates, self.mps, ref_mps)
        self.two_body_gates[(site,
                             site + 1)] = self.two_body_update_svd_numpy(env)
        self.add_unitary_right(site + 1, self.right_envs, self.mps, ref_mps,
                               self.one_body_gates, self.two_body_gates)
        if verbose > 0 and site > 0:
          overlap = self.overlap(site, self.left_envs, self.right_envs,
                                 self.one_body_gates, self.mps, ref_mps)
          stdout.write("\r iteration  %i/%i, overlap = %.6f" %
                       (it, num_sweeps, np.abs(np.real(overlap))))
          stdout.flush()
        if verbose > 1:
          print()

  def maximize_sequentially(self, ref_mps, num_sweeps, alpha=1.0, verbose=0):
    """
        deprecated; use maximize_two_body instead
        maximize the overlap w9th `ref_mps` 
        by optimizing over the all two-body unitaries sequentially, running from left to right and right to left
        Args:
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int):  number of iterations
            alpha (float):    determines the mixing of the update
                                   the new gate is given by `1- alpha` * old_gate + `alphas` * update
            verbose (int):         verbosity flag; larger means more output
        """
    assert (alpha <= 1.0)
    assert (alpha >= 0)

    self.left_envs = {}
    self.right_envs = {}
    [
        self.add_unitary_right(site, self.right_envs, self.mps, ref_mps,
                               self.one_body_gates, self.two_body_gates)
        for site in reversed(range(1, len(self.mps)))
    ]
    for it in range(num_sweeps):
      for site in range(0, len(self.mps) - 1):
        env = self.two_body_gates[
            (site, site + 1)] * (1 - alpha) + alpha * self.get_two_body_env(
                (site, site + 1), self.left_envs, self.right_envs,
                self.one_body_gates, self.mps, ref_mps)
        self.two_body_gates[(site,
                             site + 1)] = self.two_body_update_svd_numpy(env)
        self.add_unitary_left(site, self.left_envs, self.mps, ref_mps,
                              self.one_body_gates, self.two_body_gates)
        if verbose > 0 and site > 0:
          overlap = self.overlap(site, self.left_envs, self.right_envs,
                                 self.one_body_gates, self.mps, ref_mps)
          stdout.write("\r iteration  %i/%i at site %i , overlap = %.6f" %
                       (it, num_sweeps, site, np.abs(np.real(overlap))))
          stdout.flush()
        if verbose > 1:
          print()

      self.right_envs = {}
      self.add_unitary_right(
          len(self.mps) - 1, self.right_envs, self.mps, ref_mps,
          self.one_body_gates, self.two_body_gates)
      for site in reversed(range(len(self.mps) - 2)):
        env = self.two_body_gates[
            (site, site + 1)] * (1 - alpha) + alpha * self.get_two_body_env(
                (site, site + 1), self.left_envs, self.right_envs,
                self.one_body_gates, self.mps, ref_mps)
        self.two_body_gates[(site,
                             site + 1)] = self.two_body_update_svd_numpy(env)
        self.add_unitary_right(site + 1, self.right_envs, self.mps, ref_mps,
                               self.one_body_gates, self.two_body_gates)
        if verbose > 0 and site > 0:
          overlap = self.overlap(site, self.left_envs, self.right_envs,
                                 self.one_body_gates, self.mps, ref_mps)
          stdout.write("\r iteration  %i/%i at site %i , overlap = %.6f" %
                       (it, num_sweeps, site, np.abs(np.real(overlap))))
          stdout.flush()
        if verbose > 1:
          print()

  def maximize_one_body(self,
                        samples=None,
                        ref_mps=None,
                        num_sweeps=10,
                        sites=None,
                        alpha_gates=0.0,
                        alpha_samples=1.0,
                        alpha_ref_mps=1.0,
                        normalize=False,
                        verbose=0):
    """
        maximize the overlap by optimizing over the even two-body unitaries.
        maximization runs from left to right and changes even gates one at at time.
        One can either optimize the overlap with `samples`, where `samples` is a (n_samples, N) tensor
        of samples, for example obtained from FiniteMPSCentralGauge.generate_samples(...). 
        In this case the method optimizes the overlap with 1/sqrt(n_samples)\sum_n |`samples[n,:]`>.
        If `ref_mps` is given (a FiniteMPSCentralGauge), the routine optimizes the overlap with |`ref_mps`>.
        If `samples` and `ref_mps` are given the method optnimizes the overlap with 
        1/sqrt(n_samples)\sum_n |`samples[n,:]`> + |`ref_mps`>
        Args:
            samples (tf.Tensor of shape (n_samples, N):    basis-state samples
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int): number of optimiztion sweeps
            sites (iterable): the sites that should be optimized, e.g. `sites=range(0,N-1,2)` optimizes all even sites
            alpha_gates (float): see below
            alpha_samples (float): see below
            alpha_ref_mps (float): the three `alpha_` arguments determine the mixing of the update
                                   the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
            normalize (bool):      if `True`, normalize environments
            verbose (int):         verbosity flag; larger means more output

        Returns:
            c1, c2:   convergence of overlap with ref_mps and/or samples
        """
    assert (alpha_samples <= 1.0)
    assert (alpha_samples >= 0)
    assert (alpha_ref_mps <= 1.0)
    assert (alpha_ref_mps >= 0)
    assert (alpha_gates <= 1.0)
    assert (alpha_gates >= 0)

    self.left_envs_batched = {}
    self.right_envs_batched = {}
    self.left_envs = {}
    self.right_envs = {}
    if sites is None:
      sites = range(len(self.mps))

    #fixme: do right sweeps as well
    ds = self.mps.d
    convs_1, convs_2 = [0] * len(self.mps), [0] * len(self.mps)
    if samples != None:
      [
          self.add_unitary_batched_right(site, self.right_envs_batched,
                                         self.mps, samples, self.one_body_gates,
                                         self.two_body_gates)
          for site in reversed(range(1, len(self.mps)))
      ]
    if ref_mps != None:
      [
          self.add_unitary_right(
              site,
              self.right_envs,
              self.mps,
              ref_mps,
              self.one_body_gates,
              self.two_body_gates,
              normalize=normalize)
          for site in reversed(range(1, len(self.mps)))
      ]

    for it in range(num_sweeps):
      for site in range(len(self.mps) - 1):
        if site in sites:
          env = self.one_body_gates[site] * alpha_gates
          if samples != None:
            env += (
                alpha_samples * tf.math.reduce_mean(
                    self.get_one_body_env_batched(site, self.left_envs_batched,
                                                  self.right_envs_batched,
                                                  self.mps, samples),
                    axis=0))
          if ref_mps != None:
            env += (
                alpha_ref_mps * self.get_one_body_env(
                    site, self.left_envs, self.right_envs, self.mps, ref_mps))
          self.one_body_gates[site] = self.one_body_update_svd_numpy(env)

        if samples != None:
          overlap_1 = self.overlap_batched(
              site, self.left_envs_batched, self.right_envs_batched,
              self.one_body_gates, self.mps, samples)
          if site <= min(sites):
            overlap_1_old = overlap_1
          else:
            conv_1 = np.abs(overlap_1 - overlap_1_old)
            convs_1[site] = conv_1
            overlap_1_old = overlap_1

        if ref_mps != None:
          overlap_2 = self.overlap(site, self.left_envs, self.right_envs,
                                   self.one_body_gates, self.mps, ref_mps)
          if site <= min(sites):
            overlap_2_old = overlap_2
          else:
            conv_2 = np.abs(overlap_2 - overlap_2_old)
            convs_2[site] = conv_2
            overlap_2_old = overlap_2

        if site < (len(self.mps) - 1):
          if samples != None:
            self.add_unitary_batched_left(
                site, self.left_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
          if ref_mps != None:
            self.add_unitary_left(
                site,
                self.left_envs,
                self.mps,
                ref_mps,
                self.one_body_gates,
                self.two_body_gates,
                normalize=normalize)

        if verbose > 0:
          if (ref_mps != None) and (samples != None):
            stdout.write(
                "\r  overlap_samples = %.6f + %.6f i, overlap_ref_mps %.6f + %.6f i, conv_1=%.16f, "
                "conv_2=%.16f, iteration  %i/%i at site %i ," %
                (np.real(overlap_1), np.imag(overlap_1), np.real(overlap_2),
                 np.imag(overlap_2), convs_1[site], convs_2[site], it,
                 num_sweeps, site))

          elif (ref_mps == None) and (samples != None):
            stdout.write(
                "\r overlap_samples = %.6f + %.6f i, conv_1=%.16f,iteration  %i/%i at site %i"
                % (np.real(overlap_1), np.imag(overlap_1), convs_1[site], it,
                   num_sweeps, site))
          if (ref_mps != None) and (samples == None):
            stdout.write(
                "\r overlap_ref_mps = %.6f + %.6f i conv_2=%.6E, iteration  %i/%i at site %i"
                % (np.real(overlap_2), np.abs(np.imag(overlap_2)),
                   convs_2[site], it, num_sweeps, site))
          stdout.flush()
        if verbose > 1:
          print()

      for site in reversed(range(1, len(self.mps))):
        if site in sites:
          env = self.one_body_gates[site] * alpha_gates
          if samples != None:
            env += (
                alpha_samples * tf.math.reduce_mean(
                    self.get_one_body_env_batched(site, self.left_envs_batched,
                                                  self.right_envs_batched,
                                                  self.mps, samples),
                    axis=0))
          if ref_mps != None:
            env += (
                alpha_ref_mps * self.get_one_body_env(
                    site, self.left_envs, self.right_envs, self.mps, ref_mps))
          self.one_body_gates[site] = self.one_body_update_svd_numpy(env)

        if samples != None:
          overlap_1 = self.overlap_batched(
              site, self.left_envs_batched, self.right_envs_batched,
              self.one_body_gates, self.mps, samples)
          conv_1 = np.abs(overlap_1 - overlap_1_old)
          convs_1[site] = conv_1
          overlap_1_old = overlap_1

        if ref_mps != None:
          overlap_2 = self.overlap(site, self.left_envs, self.right_envs,
                                   self.one_body_gates, self.mps, ref_mps)
          conv_2 = np.abs(overlap_2 - overlap_2_old)
          convs_2[site] = conv_2
          overlap_2_old = overlap_2

        if site > 0:
          if samples != None:
            self.add_unitary_batched_right(
                site, self.right_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
          if ref_mps != None:
            self.add_unitary_right(
                site,
                self.right_envs,
                self.mps,
                ref_mps,
                self.one_body_gates,
                self.two_body_gates,
                normalize=normalize)

        if verbose > 0:
          #print(convs_1, convs_2)
          if (ref_mps != None) and (samples != None):
            stdout.write(
                "\r  overlap_samples = %.6f + %.6f i, overlap_ref_mps %.6f + %.6f i, conv_1=%.16f, "
                "conv_2=%.16f, iteration  %i/%i at site %i ," %
                (np.real(overlap_1), np.imag(overlap_1), np.real(overlap_2),
                 np.imag(overlap_2), convs_1[site], convs_2[site], it,
                 num_sweeps, site))

          elif (ref_mps == None) and (samples != None):
            stdout.write(
                "\r overlap_samples = %.6f + %.6f i, conv_1=%.16f,iteration  %i/%i at site %i"
                % (np.real(overlap_1), np.imag(overlap_1), convs_1[site], it,
                   num_sweeps, site))
          if (ref_mps != None) and (samples == None):
            stdout.write(
                "\r overlap_ref_mps = %.6f + %.6f i conv_2=%.6E, iteration  %i/%i at site %i"
                % (np.real(overlap_2), np.abs(np.imag(overlap_2)),
                   convs_2[site], it, num_sweeps, site))
          stdout.flush()
        if verbose > 1:
          print()

    if (ref_mps != None) and (samples != None):
      return np.max(convs_1), np.max(convs_2)
    elif (ref_mps == None) and (samples == None):
      return np.max(convs_1), None
    elif (ref_mps != None) and (samples == None):
      return np.max(convs_2), None
    elif (ref_mps == None) and (samples == None):
      return None, None

  def maximize_two_body(self,
                        samples=None,
                        ref_mps=None,
                        num_sweeps=10,
                        sites=None,
                        alpha_gates=0.0,
                        alpha_samples=1.0,
                        alpha_ref_mps=1.0,
                        normalize=False,
                        verbose=0):
    """
        maximize the overlap by optimizing over the even two-body unitaries.
        minimization runs from left to right and changes even gates one at at time.
        One can either optimize the overlap with `samples`, where `samples` is a (n_samples, N) tensor
        of snamples, for example obtained from FiniteMPSCentralGauge.generate_samples(...). 
        In this case the method optimizes the overlap with 1/sqrt(n_samples)\sum_n |`samples[n,:]`>.
        If `ref_mps` is given (a FiniteMPSCentralGauge), the routine optimizes the overlap with |`ref_mps`>.
        If `samples` and `ref_mps` are given the method optnimizes the overlap with 
        1/sqrt(n_samples)\sum_n |`samples[n,:]`> + |`ref_mps`>
        Args:
            samples (tf.Tensor of shape (n_samples, N):    basis-state samples
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int): number of optimiztion sweeps
            sites (iterable):  the sites which should be optimized. Gates are optimized at (s,s+1) for s in `sites`
                               thus if `sites` has only even sites, only the even gates are updated.
            alpha_gates (float): see below
            alpha_samples (float): see below
            alpha_ref_mos (float): the three `alpha_` arguments determine the mixing of the update
                                   the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
            verbose (int):         verbosity flag; larger means more output
        """
    assert (alpha_samples <= 1.0)
    assert (alpha_samples >= 0)
    assert (alpha_ref_mps <= 1.0)
    assert (alpha_ref_mps >= 0)
    assert (alpha_gates <= 1.0)
    assert (alpha_gates >= 0)

    self.left_envs_batched = {}
    self.right_envs_batched = {}
    self.left_envs = {}
    self.right_envs = {}

    #fixme: do right sweeps as well
    ds = self.mps.d
    if sites is None:
      sites = range(len(self.mps) - 1)

    convs_1, convs_2 = [0] * (len(self.mps) - 1), [0] * (len(self.mps) - 1)

    for it in range(num_sweeps):
      if samples != None:
        [
            self.add_unitary_batched_right(
                site, self.right_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
            for site in reversed(range(1, len(self.mps)))
        ]
      if ref_mps != None:
        [
            self.add_unitary_right(
                site,
                self.right_envs,
                self.mps,
                ref_mps,
                self.one_body_gates,
                self.two_body_gates,
                normalize=normalize)
            for site in reversed(range(1, len(self.mps)))
        ]

      for site in range(0, len(self.mps) - 1):
        if site in sites:
          env = self.two_body_gates[(site, site + 1)] * alpha_gates
          if samples != None:
            tmp = self.get_two_body_env_batched(
                (site, site + 1), self.left_envs_batched,
                self.right_envs_batched, self.one_body_gates, self.mps, samples)
            env += (alpha_samples * tf.math.reduce_mean(tmp, axis=0)
                   )  #/np.sqrt(samples.shape[0]))
          if ref_mps != None:
            env += alpha_ref_mps * self.get_two_body_env(
                (site, site + 1), self.left_envs, self.right_envs,
                self.one_body_gates, self.mps, ref_mps)
          self.two_body_gates[(site,
                               site + 1)] = self.two_body_update_svd_numpy(env)
        if site < len(self.mps) - 1:
          if samples != None:
            self.add_unitary_batched_left(
                site, self.left_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
          if ref_mps != None:
            self.add_unitary_left(
                site,
                self.left_envs,
                self.mps,
                ref_mps,
                self.one_body_gates,
                self.two_body_gates,
                normalize=normalize)

        if samples != None:
          overlap_1 = self.overlap_batched(
              site, self.left_envs_batched, self.right_envs_batched,
              self.one_body_gates, self.mps, samples)
          if site <= min(sites):
            overlap_1_old = overlap_1
          else:
            conv_1 = np.abs(overlap_1 - overlap_1_old)
            convs_1[site] = conv_1
            overlap_1_old = overlap_1

        if ref_mps != None:
          overlap_2 = self.overlap(site, self.left_envs, self.right_envs,
                                   self.one_body_gates, self.mps, ref_mps)
          if site <= min(sites):
            overlap_2_old = overlap_2
          else:
            conv_2 = np.abs(overlap_2 - overlap_2_old)
            convs_2[site] = conv_2
            overlap_2_old = overlap_2

        if verbose > 0:
          if (ref_mps != None) and (samples != None):
            stdout.write(
                "\r  overlap_samples = %.6E + %.6E i, overlap_ref_mps %.6E + %.6fE i, conv_1=%.6E, "
                "conv_2=%.6E, iteration  %i/%i at site %i ," %
                (np.real(overlap_1), np.imag(overlap_1), np.real(overlap_2),
                 np.imag(overlap_2), convs_1[site], convs_2[site], it,
                 num_sweeps, site))
          elif (ref_mps == None) and (samples != None):
            stdout.write(
                "\r overlap_samples = %.6E + %.6E i, conv_1=%.6E,iteration  %i/%i at site %i"
                % (np.real(overlap_1), np.imag(overlap_1), convs_1[site], it,
                   num_sweeps, site))
          if (ref_mps != None) and (samples == None):
            stdout.write(
                "\r overlap_ref_mps = %.6f + %.6f i conv_2=%.6E, iteration  %i/%i at site %i"
                % (np.real(overlap_2), np.abs(np.imag(overlap_2)),
                   convs_2[site], it, num_sweeps, site))
          stdout.flush()

        if verbose > 1:
          print()

      # for site in reversed(range(1, len(self.mps) - 1)):
      #     if site in sites:
      #         env = self.two_body_gates[(site, site + 1)] * alpha_gates
      #         if samples != None:
      #             tmp = self.get_two_body_env_batched(
      #                 (site, site + 1), self.left_envs_batched,
      #                 self.right_envs_batched, self.one_body_gates,
      #                 self.mps, samples)
      #             env += (alpha_samples * tf.math.reduce_mean(tmp, axis=0)
      #                    )  #/np.sqrt(samples.shape[0]))
      #         if ref_mps != None:
      #             env += alpha_ref_mps * self.get_two_body_env(
      #                 (site, site + 1), self.left_envs, self.right_envs,
      #                 self.one_body_gates, self.mps, ref_mps)
      #         self.two_body_gates[(
      #             site, site + 1)] = self.two_body_update_svd_numpy(env)
      #     if site > 0:
      #         if samples != None:
      #             self.add_unitary_batched_right(site,
      #                                            self.right_envs_batched,
      #                                            self.mps, samples,
      #                                            self.one_body_gates,
      #                                            self.two_body_gates)
      #         if ref_mps != None:
      #             self.add_unitary_right(site,
      #                                    self.right_envs,
      #                                    self.mps,
      #                                    ref_mps,
      #                                    self.one_body_gates,
      #                                    self.two_body_gates,
      #                                    normalize=normalize)

      #     if samples != None:
      #         overlap_1 = self.overlap_batched(site,
      #                                          self.left_envs_batched,
      #                                          self.right_envs_batched,
      #                                          self.one_body_gates,
      #                                          self.mps, samples)
      #         if site <= min(sites):
      #             overlap_1_old = overlap_1
      #         else:
      #             conv_1 = np.abs(overlap_1 - overlap_1_old)
      #             convs_1[site] = conv_1
      #             overlap_1_old = overlap_1

      #     if ref_mps != None:
      #         overlap_2 = self.overlap(site, self.left_envs,
      #                                  self.right_envs,
      #                                  self.one_body_gates, self.mps,
      #                                  ref_mps)
      #         if site <= min(sites):
      #             overlap_2_old = overlap_2
      #         else:
      #             conv_2 = np.abs(overlap_2 - overlap_2_old)
      #             convs_2[site] = conv_2
      #             overlap_2_old = overlap_2

      #     if verbose > 0:
      #         if (ref_mps != None) and (samples != None):
      #             stdout.write(
      #                 "\r  overlap_samples = %.6E + %.6E i, overlap_ref_mps %.6E + %.6fE i, conv_1=%.6E, "
      #                 "conv_2=%.6E, iteration  %i/%i at site %i ,"
      #                 % (np.real(overlap_1), np.imag(overlap_1),
      #                    np.real(overlap_2), np.imag(overlap_2),
      #                    convs_1[site], convs_2[site], it, num_sweeps,
      #                    site))
      #         elif (ref_mps == None) and (samples != None):
      #             stdout.write(
      #                 "\r overlap_samples = %.6E + %.6E i, conv_1=%.6E,iteration  %i/%i at site %i"
      #                 % (np.real(overlap_1), np.imag(overlap_1),
      #                    convs_1[site], it, num_sweeps, site))
      #         if (ref_mps != None) and (samples == None):
      #             stdout.write(
      #                 "\r overlap_ref_mps = %.6f + %.6f i conv_2=%.6E, iteration  %i/%i at site %i"
      #                 % (np.real(overlap_2), np.abs(np.imag(overlap_2)),
      #                    convs_2[site], it, num_sweeps, site))
      #         stdout.flush()

      #     if verbose > 1:
      #         print()

    if (ref_mps != None) and (samples != None):
      return np.max(convs_1), np.max(convs_2)
    elif (ref_mps == None) and (samples == None):
      return np.max(convs_1), None
    elif (ref_mps != None) and (samples == None):
      return np.max(convs_2), None
    elif (ref_mps == None) and (samples == None):
      return None, None

  def maximize_even(self,
                    samples=None,
                    ref_mps=None,
                    num_sweeps=10,
                    alpha_gates=0.0,
                    alpha_samples=1.0,
                    alpha_ref_mps=1.0,
                    verbose=0):
    """
        maximize the overlap by optimizing over the even two-body unitaries.
        minimization runs from left to right and changes even gates one at at time.
        One can either optimize the overlap with `samples`, where `samples` is a (n_samples, N) tensor
        of snamples, for example obtained from FiniteMPSCentralGauge.generate_samples(...). 
        In this case the method optimizes the overlap with 1/sqrt(n_samples)\sum_n |`samples[n,:]`>.
        If `ref_mps` is given (a FiniteMPSCentralGauge), the routine optimizes the overlap with |`ref_mps`>.
        If `samples` and `ref_mps` are given the method optnimizes the overlap with 
        1/sqrt(n_samples)\sum_n |`samples[n,:]`> + |`ref_mps`>
        Args:
            samples (tf.Tensor of shape (n_samples, N):    basis-state samples
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int): number of optimiztion sweeps
            alpha_gates (float): see below
            alpha_samples (float): see below
            alpha_ref_mos (float): the three `alpha_` arguments determine the mixing of the update
                                   the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
            verbose (int):         verbosity flag; larger means more output
        """
    sites = range(0, len(self.mps), 2)
    self.maximize_two_body(
        samples=samples,
        ref_mps=ref_mps,
        num_sweeps=num_sweeps,
        sites=sites,
        alpha_gates=alpha_gates,
        alpha_samples=alpha_samples,
        alpha_ref_mps=alpha_ref_mps,
        verbose=verbose)

  def maximize_odd(self,
                   samples=None,
                   ref_mps=None,
                   num_sweeps=10,
                   alpha_gates=0.0,
                   alpha_samples=1.0,
                   alpha_ref_mps=1.0,
                   verbose=0):
    """
        maximize the overlap by optimizing over the odd two-body unitaries.
        minimization runs from left to right and changes odd gates one at at time.
        One can either optimize the overlap with `samples`, where `samples` is a (n_samples, N) tensor
        of samples, for example obtained from FiniteMPSCentralGauge.generate_samples(...). 
        In this case the method optimizes the overlap with 1/sqrt(n_samples)\sum_n |`samples[n,:]`>.
        If `ref_mps` is given (a FiniteMPSCentralGauge), the routine optimizes the overlap with |`ref_mps`>.
        If `samples` and `ref_mps` are given the method optimizes the overlap with 
        1/sqrt(n_samples)\sum_n |`samples[n,:]`> + |`ref_mps`>
        Args:
            samples (tf.Tensor of shape (n_samples, N):    basis-state samples
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int): number of optimiztion sweeps
            alpha_gates (float): see below
            alpha_samples (float): see below
            alpha_ref_mos (float): the three `alpha_` arguments determine the mixing of the update
                                   the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
            verbose (int):         verbosity flag; larger means more output
        """
    sites = range(1, len(self.mps), 2)
    self.maximize_two_body(
        samples=samples,
        ref_mps=ref_mps,
        num_sweeps=num_sweeps,
        sites=sites,
        alpha_gates=alpha_gates,
        alpha_samples=alpha_samples,
        alpha_ref_mps=alpha_ref_mps,
        verbose=verbose)

  def gradient_minimization_two_body(self,
                                     two_body_generators,
                                     one_body_generators=None,
                                     opt_type='sequential',
                                     samples=None,
                                     ref_mps=None,
                                     alpha=1E-5,
                                     num_sweeps=10,
                                     sites=None,
                                     activation=None,
                                     gamma=0.1,
                                     verbose=0):
    """
        maximize the overlap by optimizing over all two-body unitaries.
        minimization runs from left to right and changes all two-body gates, either one at at time (for `opt_type == 'sequential')
        or all simultaneously (for `opt_type == 'simultaneous').
        One can either optimize the overlap with `samples`, where `samples` is a (n_samples, N) tensor
        of samples, for example obtained from FiniteMPSCentralGauge.generate_samples(...). 
        In this case the method optimizes the overlap with 1/sqrt(n_samples)\sum_n |`samples[n,:]`>.
        If `ref_mps` is given (a FiniteMPSCentralGauge), the routine optimizes the overlap with |`ref_mps`>.
        If `samples` and `ref_mps` are given the method optnimizes the overlap with 
        1/sqrt(n_samples)\sum_n |`samples[n,:]`> + |`ref_mps`>
        Args:
            two_body_generators (list of tf.Tensor of shape (d**2,d**2)): the generators `g` for the two-body unitaries
                                                                          note that U = expm(`g` - herm(`g`))

            one_body_generators (None or list of tf.Tensor of shape (d,d)): the generators `g` for the one-body unitaries
                                                                            note that U = expm(`g` - herm(`g`))
                                                                            if not `None`, self.one_body_gates are initialized from generators
                                                                            else, the current values in self.one_body_gates are used
            opt_type (str):  the type of optimization. `opt_type` == 'sequential' optimizes the generators one at a time
                                                       `opt_type` == 'simultaneous' optimizes the generators simultaneously
            samples (tf.Tensor of shape (n_samples, N):    basis-state samples
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int): number of optimiztion sweeps
            sites (iterable): the sites that should be optimized, e.g. `sites=range(0,N-1,2)` optimizes all even sites
            alpha_gates (float): see below
            alpha_samples (float): see below
            alpha_ref_mos (float): the three `alpha_` arguments determine the mixing of the update
                                   the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
            verbose (int):         verbosity flag; larger means more output
        """
    self.left_envs_batched = {}
    self.right_envs_batched = {}
    self.left_envs = {}
    self.right_envs = {}
    if sites is None:
      sites = range(len(self.mps))

    #fixme: do right sweeps as well
    ds = self.mps.d
    if one_body_generators is not None:
      self.one_body_gates = initialize_gates_from_generators(
          one_body_generators, ds)

    self.two_body_gates = initialize_gates_from_generators(
        two_body_generators, ds)

    max_site = np.max(sites)
    for it in range(num_sweeps):
      if samples != None:
        [
            self.add_unitary_batched_right(
                site, self.right_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
            for site in reversed(range(1, len(self.mps)))
        ]
      if ref_mps != None:
        [
            self.add_unitary_right(site, self.right_envs, self.mps, ref_mps,
                                   self.one_body_gates, self.two_body_gates)
            for site in reversed(range(1, len(self.mps)))
        ]

      grads = []

      for site in range(len(self.mps) - 1):
        if site > max_site:  #stop if we reached the largest of the sites we want to optimize
          break
        if site in sites:
          grad = tf.zeros(
              shape=[ds[site] * ds[site + 1], ds[site] * ds[site + 1]],
              dtype=self.mps.dtype)
          if samples != None:
            gradient, avsign, c1 = self.two_body_gradient_batched(
                (site, site + 1),
                self.left_envs_batched,
                self.right_envs_batched,
                self.one_body_gates,
                self.mps,
                samples,
                two_body_generators,
                activation=activation,
                gamma=gamma)
            grad += gradient
          if ref_mps != None:
            grad += self.two_body_gradient(
                (site, site + 1), self.left_envs, self.right_envs,
                self.one_body_gates, self.mps, ref_mps, two_body_generators)
          if opt_type in ('sequential', 'seq'):
            grad /= tf.linalg.norm(grad)
            two_body_generators[(site, site + 1)] += (
                grad * alpha
            )  #we are trying to maximize, not minimize, hence the + operation
            self.two_body_gates[(site, site + 1)] = get_gate_from_generator(
                two_body_generators[(site, site + 1)],
                (ds[site], ds[site + 1], ds[site], ds[site + 1]))
          elif opt_type in ('simultaneous', 'sim'):
            grads.append(grad)
          else:
            raise ValueError('unknown value {} for opt_type'.format(opt_type))
        if site < (len(self.mps) - 1):
          if samples != None:
            self.add_unitary_batched_left(
                site, self.left_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
          if ref_mps != None:
            self.add_unitary_left(site, self.left_envs, self.mps, ref_mps,
                                  self.one_body_gates, self.two_body_gates)
        if (verbose > 0) and (site not in (0, len(self.mps) - 1)):
          if samples != None:
            pass
            #overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.one_body_gates, self.mps, samples)
            #c1 = np.real(overlap_1) - np.imag(overlap_1)
          if ref_mps != None:
            overlap_2 = self.overlap(site, self.left_envs, self.right_envs,
                                     self.one_body_gates, self.mps, ref_mps)
            c2 = np.real(overlap_2) - np.imag(overlap_2)

          if (ref_mps != None) and (samples != None):
            stdout.write(
                "\r iteration  %i/%i at site %i , C = %.6f + %.6f i, <sgn> = %.6f + %.6f i, "
                "Re(overlap_ref_mps) - Im(overlap_ref_mps) %.6f + %.6f i" %
                (it, num_sweeps, site, np.rea(c1), np.imag(c1), np.real(avsign),
                 np.imag(avsign), np.real(c2), np.imag(c2)))

          elif (ref_mps == None) and (samples != None):
            stdout.write(
                "\r iteration  %i/%i at site %i , C = %.6f + %.6f i, <sgn> = %.6f + %.6f i"
                % (it, num_sweeps, site, np.real(c1), np.imag(c1),
                   np.real(avsign), np.imag(avsign)))
          if (ref_mps != None) and (samples == None):
            stdout.write(
                "\r iteration  %i/%i at site %i , Re(overlap_ref_mps) - Im(overlap_ref_mps) = %.6f + %.6f i"
                % (it, num_sweeps, site, np.real(c2), np.imag(c2)))
          stdout.flush()
        if verbose > 1:
          print()

      if opt_type in ('sequentual', 'seq'):
        pass
      elif opt_type in ('simultaneous', 'sim'):
        Zs = []
        for site in sites:
          Z = np.linalg.norm(grads[site].numpy())
          Zs.append(Z)
        Zmax = np.max(Zs)
        for site in sites:
          two_body_generators[(site, site + 1)] += (
              grad * alpha / Zmax
          )  #we are trying to maximize, not minimize, hence the + operation
          self.two_body_gates[(site, site + 1)] = get_gate_from_generator(
              two_body_generators[(site, site + 1)],
              (ds[site], ds[site + 1], ds[site], ds[site + 1]))
      else:
        raise ValueError('unknown value {} for opt_type'.format(opt_type))

  def gradient_minimization_overlap_one_body(self,
                                             one_body_generators,
                                             ref_mps,
                                             two_body_generators=None,
                                             opt_type='sequential',
                                             alpha=1E-5,
                                             num_sweeps=10,
                                             sites=None,
                                             verbose=0):
    """
        maximize the distance to `ref_mps` by optimizing over the one-body unitaries.
        This implements a gradient optimization on one-body unitaries by updating the generators of the unitaries.
        minimization runs from left to right and changes generator either one at at time or simultaneously, depending
        on the value of `opt_type`.
        Args:
            one_body_generators (list of tf.Tensor of shape (d,d)): the generators `g` for the one-body unitaries
                                                                            note that U = expm(`g` - herm(`g`))
                                                                            if not `None`, self.one_body_gates are initialized from generators
                                                                            else, the current values in self.one_body_gates are used
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            two_body_generators (None or list of tf.Tensor of shape (d**2,d**2)): the generators `g` for the two-body unitaries
                                                                          note that U = expm(`g` - herm(`g`))

            opt_type (str):  the type of optimization. `opt_type` == 'sequential' optimizes the generators one at a time
                                                       `opt_type` == 'simultaneous' optimizes the generators simultaneously

            alpha (float): stepsize
            num_sweeps (int): number of optimiztion sweeps
            sites (iterable): the sites that should be optimized, e.g. `sites=range(0,N-1,2)` optimizes all even sites
            verbose (int):         verbosity flag; larger means more output
        """
    self.left_envs_batched = {}
    self.right_envs_batched = {}
    self.left_envs = {}
    self.right_envs = {}
    if sites is None:
      sites = range(len(self.mps))

    #fixme: do right sweeps as well
    ds = self.mps.d
    if two_body_generators is not None:
      self.two_body_gates = initialize_gates_from_generators(
          two_body_generators, ds)

    self.one_body_gates = initialize_gates_from_generators(
        one_body_generators, ds)
    max_site = np.max(sites)
    for it in range(num_sweeps):
      [
          self.add_unitary_right(site, self.right_envs, self.mps, ref_mps,
                                 self.one_body_gates, self.two_body_gates)
          for site in reversed(range(1, len(self.mps)))
      ]

      grads = []
      for site in range(len(self.mps)):
        if site > max_site:  #stop if we reached the largest of the sites we want to optimize
          break

        if site in sites:
          grad, overlap = self.one_body_gradient(site, self.left_envs,
                                                 self.right_envs, self.mps,
                                                 ref_mps, one_body_generators)
          if opt_type in ('sequential', 's', 'seq'):
            grad /= tf.linalg.norm(grad)
            one_body_generators[site] += (
                grad * alpha
            )  #we are trying to maximize, not minimize, hence the + operation
            self.one_body_gates[site] = get_gate_from_generator(
                one_body_generators[site], (ds[site], ds[site]))
          elif opt_type in ('simultaneous', 'sim'):
            grads.append(grad)
          else:
            raise ValueError('unknown value {} for opt_type'.format(opt_type))

        if site < (len(self.mps) - 1):
          self.add_unitary_left(site, self.left_envs, self.mps, ref_mps,
                                self.one_body_gates, self.two_body_gates)
        if (verbose > 0) and (site not in (0, len(self.mps) - 1)):
          stdout.write(
              "\r iteration  %i/%i at site %i , overlap = %.6f + %.6f i" %
              (it, num_sweeps, site, np.real(overlap), np.imag(overlap)))
          stdout.flush()
        if verbose > 1:
          print()

      if opt_type in ('sequential', 'seq'):
        pass
      elif opt_type in ('simultaneous', 'sim'):
        Zs = []
        for site in sites:
          Z = np.linalg.norm(grads[site].numpy())
          Zs.append(Z)
        Zmax = np.max(Zs)
        for site in sites:
          one_body_generators[site] += (
              grads[site] * alpha / Zmax
          )  #we are trying to maximize, not minimize, hence the + operation
          self.one_body_gates[site] = get_gate_from_generator(
              one_body_generators[site], (ds[site], ds[site]))
      else:
        raise ValueError('unknown value {} for opt_type'.format(opt_type))

  def gradient_minimization_cost_function_one_body_batched(
      self,
      one_body_generators,
      samples,
      two_body_generators=None,
      opt_type='sequential',
      alpha=1E-5,
      num_sweeps=10,
      sites=None,
      activation=None,
      gamma=0.1,
      verbose=0):
    """
        positivizes self.mps by optimizing the one-body unitaries.
        This implements a gradient optimization on one-body unitaries by updating the generators of the unitaries.
        Optimization runs from left to right and changes generators either one at at time or simultaneously, depending
        on the value of `opt_type`. The cost function that is maximized is given by (see https://arxiv.org/abs/1906.04654)

                  Cost = \sum_{\sigma} \partial C_{\sigma} + 2 * C_{\sigma} * \Re \partial(\log(\psi_{\sigma}))
        with 
                  C_{\sigma} = ((1-\gamma) `activation`(\Re(\psi_{\sigma}))) - \gamma * |\Im(\psi_{\sigma})|

        Args:
            one_body_generators (dict mapping integer `site` -> tf.Tensor of shape `(mps.d[site],mps.d[site])`):  the generators `g` of the one-body unitaries
                                                                                                                  U = expm(`g`-herm(`g`))
            samples (tf.Tensor of shape (Nt, len(mps)):   the samples
            two_body_generators (None or dict mapping tuple (site, site+1) -> tf.Tensor 
                                 of shape `(mps.d[sites[0]] * mps.d[sites[1]], mps.d[sites[0]] * mps.d[sites[1]])`):  the generators `g` of the two-body unitaries
                                                                                                                      U = expm(`g`-herm(`g`))
                                                                                                                      if given, two-body gates are initialized from generators
            opt_type (str):  the type of optimization. `opt_type` == 'sequential' optimizes the generators one at a time
                                                       `opt_type` == 'simultaneous' optimizes the generators simultaneously
            alpha (float): step size
            num_sweeps (int): number of optimiztion sweeps
            sites (iterable): the sites that should be optimized, e.g. `sites=range(0,N-1,2)` optimizes all even sites
            activation (callable):  activation function, see above
            gamma (float):          see above
            verbose (int):         verbosity flag; larger means more output
        """
    self.left_envs_batched = {}
    self.right_envs_batched = {}
    self.left_envs = {}
    self.right_envs = {}
    if sites is None:
      sites = range(len(self.mps))

    #fixme: do right sweeps as well
    ds = self.mps.d
    if two_body_generators is not None:
      self.two_body_gates = initialize_gates_from_generators(
          two_body_generators, ds)

    self.one_body_gates = initialize_gates_from_generators(
        one_body_generators, ds)
    max_site = np.max(sites)
    for it in range(num_sweeps):
      [
          self.add_unitary_batched_right(site, self.right_envs_batched,
                                         self.mps, samples, self.one_body_gates,
                                         self.two_body_gates)
          for site in reversed(range(1, len(self.mps)))
      ]

      grads = []
      for site in range(len(self.mps)):
        if site > max_site:  #stop if we reached the largest of the sites we want to optimize
          break

        if site in sites:
          grad, avsign, c1 = self.one_body_gradient_cost_function_batched(
              site,
              self.left_envs_batched,
              self.right_envs_batched,
              self.mps,
              samples,
              one_body_generators,
              activation,
              gamma=gamma)
          if opt_type in ('sequential', 's', 'seq'):
            grad /= tf.linalg.norm(grad)
            one_body_generators[site] += (
                grad * alpha
            )  #we are trying to maximize, not minimize, hence the + operation
            self.one_body_gates[site] = get_gate_from_generator(
                one_body_generators[site], (ds[site], ds[site]))
          else:
            grads.append(grad)

        if site < (len(self.mps) - 1):
          self.add_unitary_batched_left(site, self.left_envs_batched, self.mps,
                                        samples, self.one_body_gates,
                                        self.two_body_gates)
        if (verbose > 0) and (site not in (0, len(self.mps) - 1)):
          stdout.write(
              "\r iteration  %i/%i at site %i , C = %.6f + %.6f i, <sgn> = %.6f + %.6f i"
              % (it, num_sweeps, site, np.real(c1), np.imag(c1),
                 np.real(avsign), np.imag(avsign)))
          stdout.flush()
        if verbose > 1:
          print()

      if opt_type in ('sequential', 'seq'):
        pass
      elif opt_type in ('simultaneous', 'sim'):
        Zs = []
        for site in sites:
          Z = np.linalg.norm(grads[site].numpy())
          Zs.append(Z)
        Zmax = np.max(Zs)
        for site in sites:
          one_body_generators[site] += (
              grads[site] * alpha / Zmax
          )  #we are trying to maximize, not minimize, hence the + operation
          self.one_body_gates[site] = get_gate_from_generator(
              one_body_generators[site], (ds[site], ds[site]))
      else:
        raise ValueError('unknown value {} for opt_type'.format(opt_type))

  def gradient_minimization_cost_function_overlap_one_body_batched(
      self,
      one_body_generators,
      two_body_generators=None,
      opt_type='sequential',
      samples=None,
      ref_mps=None,
      alpha=1E-5,
      num_sweeps=10,
      sites=None,
      activation=None,
      gamma=0.1,
      verbose=0):
    """
        This is experimental
        combines  `gradient_minimization_overlap_one_body` and `gradient_minimization_cost_function_one_body_batched`
        into a single function and combines the gradients of the two into a single one. 
        """
    self.left_envs_batched = {}
    self.right_envs_batched = {}
    self.left_envs = {}
    self.right_envs = {}
    if sites is None:
      sites = range(len(self.mps))

    #fixme: do right sweeps as well
    ds = self.mps.d
    if two_body_generators is not None:
      self.two_body_gates = initialize_gates_from_generators(
          two_body_generators, ds)

    self.one_body_gates = initialize_gates_from_generators(
        one_body_generators, ds)
    max_site = np.max(sites)
    for it in range(num_sweeps):
      if samples != None:
        [
            self.add_unitary_batched_right(
                site, self.right_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
            for site in reversed(range(1, len(self.mps)))
        ]
      if ref_mps != None:
        [
            self.add_unitary_right(site, self.right_envs, self.mps, ref_mps,
                                   self.one_body_gates, self.two_body_gates)
            for site in reversed(range(1, len(self.mps)))
        ]

      grads = []
      for site in range(len(self.mps)):
        if site > max_site:  #stop if we reached the largest of the sites we want to optimize
          break

        if site in sites:
          grad = tf.zeros(shape=[ds[site], ds[site]], dtype=self.mps.dtype)
          if samples != None:
            gradient, avsign, c1 = self.one_body_gradient_cost_function_batched(
                site,
                self.left_envs_batched,
                self.right_envs_batched,
                self.mps,
                samples,
                one_body_generators,
                activation,
                gamma=gamma)
            grad += gradient
          if ref_mps != None:
            gradient, cost = self.one_body_gradient(
                site, self.left_envs, self.right_envs, self.mps, ref_mps,
                one_body_generators)
            grad += gradient
          if opt_type in ('sequential', 's', 'seq'):
            grad /= tf.linalg.norm(grad)
            one_body_generators[site] += (
                grad * alpha
            )  #we are trying to maximize, not minimize, hence the + operation
            self.one_body_gates[site] = get_gate_from_generator(
                one_body_generators[site], (ds[site], ds[site]))
          else:
            grads.append(grad)

        if site < (len(self.mps) - 1):
          if samples != None:
            self.add_unitary_batched_left(
                site, self.left_envs_batched, self.mps, samples,
                self.one_body_gates, self.two_body_gates)
          if ref_mps != None:
            self.add_unitary_left(site, self.left_envs, self.mps, ref_mps,
                                  self.one_body_gates, self.two_body_gates)
        if (verbose > 0) and (site not in (0, len(self.mps) - 1)):
          if samples != None:
            pass
            #overlap_1 = self.overlap_batched(site, self.left_envs_batched,
            #self.right_envs_batched, self.one_body_gates, self.mps, samples)
            #c1 = np.real(overlap_1) - np.imag(overlap_1)
          if ref_mps != None:
            overlap_2 = self.overlap(site, self.left_envs, self.right_envs,
                                     self.one_body_gates, self.mps, ref_mps)
            c2 = np.real(overlap_2) - np.imag(overlap_2)
          if (ref_mps != None) and (samples != None):
            stdout.write(
                "\r iteration  %i/%i at site %i , C = %.6f + %.6f i, <sgn> = %.6f + %.6f i, "
                "Re(overlap_ref_mps) - Im(overlap_ref_mps) %.6f + %.6f i" %
                (it, num_sweeps, site, np.rea(c1), np.imag(c1), np.real(avsign),
                 np.imag(avsign), np.real(c2), np.imag(c2)))

          elif (ref_mps == None) and (samples != None):
            stdout.write(
                "\r iteration  %i/%i at site %i , C = %.6f + %.6f i, <sgn> = %.6f + %.6f i"
                % (it, num_sweeps, site, np.real(c1), np.imag(c1),
                   np.real(avsign), np.imag(avsign)))
          if (ref_mps != None) and (samples == None):
            stdout.write(
                "\r iteration  %i/%i at site %i , Re(overlap_ref_mps) - Im(overlap_ref_mps) = %.6f + %.6f i"
                % (it, num_sweeps, site, np.real(c2), np.imag(c2)))
          stdout.flush()
        if verbose > 1:
          print()

      if opt_type in ('sequentual', 'seq'):
        pass
      elif opt_type in ('simultaneous', 'sim'):
        Zs = []
        for site in sites:
          Z = np.linalg.norm(grads[site].numpy())
          Zs.append(Z)
        Zmax = np.max(Zs)
        for site in sites:
          one_body_generators[site] += (
              grads[site] * alpha / Zmax
          )  #we are trying to maximize, not minimize, hence the + operation
          self.one_body_gates[site] = get_gate_from_generator(
              one_body_generators[site], (ds[site], ds[site]))
      else:
        raise ValueError('unknown value {} for opt_type'.format(opt_type))


class OneBodyStoquastisizer:

  def __init__(self,
               mpo,
               one_body_gates=None,
               name='OneBodyStoquastisizer',
               backend='tensorflow'):
    """
        uses a unitary circuit with three layers:
        the layer contains `N` one-body unitaries

        index conventions for one-body unitaries:

               1
               |
              ___
             |   |
             |___|
               |
               0

        index conventions for two-body unitaries:

               2   3
               |   |
              _______
             |       |
             |_______|
               |   |
               0   1
    
        2,3 are the physical outgoing and incoming indices, respectively. The conjugated 
        side of the MPS is on the bottom (at index 2)

        MPS index convention:
              ___
             |   |
         0---     ---2
             |___|
               |
               1



        MPO index convention:

               3
              _|_
             |   |
         0---     ---1
             |___|
               |
               2

    
        An MPO by this convention is contracted from above:
                     
                    _|_
                 --|___|--
                     |
                    ___
                   |   |
                   |___|
                     |

                _|_     _|_    
             --|   | --|   |--
                ---     ---   
                 |       |     
                ___________ 
               |           |
               |___________|
                 |       |  


    
        Args:
            mpo (FiniteMPSCentralGauge):       an mps of even length
            one_body_gates (iterable or None): an iterable mapping sites to matrices
                                               `one_body_gates[site]` is the one-body unitary  at site `site
                                               if `None`, one-body gates are initialized with identities
            two_body_gates (dict or None):     dictionary mapping tuples `(site`, site2)` to rank-4 unitary tensors
                                               the convention 
                                               `one_body_gates[site]` is the one-body unitary  at site `site
                                               if `None`, two-body gates are initialized with identities
            name (str):                        an optional name for the object
        """

    self.name = name
    self.mpo = mpo
    self.right_envs = {}
    self.left_envs = {}
    self.backend = backend

    if (one_body_gates == None) or (len(one_body_gates) == 0):
      ds = [self.mpo.get_tensor(site).shape[2] for site in range(len(mpo))]
      self.gates = initialize_one_body_gates(
          ds, self.mpo.dtype, which='e', noise=0.0)
    else:
      self.gates = one_body_gates

  def add_unitary_left(self, site, reference_mps, normalize=False):
    net = tn.TensorNetwork(backend=self.backend)
    gate = net.add_node(self.gates[site])
    mps = net.add_node(reference_mps.get_tensor(site))
    mpo = net.add_node(self.mpo.get_tensor(site))
    conj_mps = net.add_node(net.backend.conj(reference_mps.get_tensor(site)))
    conj_gate = net.add_node(net.backend.conj(self.gates[site]))
    if site == 0:
      L = net.add_node(net.backend.ones((1, 1, 1), dtype=self.mpo.dtype))
      self.left_envs[site] = L.tensor

    L = net.add_node(self.left_envs[site])
    L[0] ^ mps[0]
    L[1] ^ conj_mps[0]
    L[2] ^ mpo[0]

    gate[1] ^ mps[1]
    gate[0] ^ mpo[3]

    conj_gate[1] ^ conj_mps[1]
    conj_gate[0] ^ mpo[2]
    output_order = [mps[2], conj_mps[2], mpo[1]]
    out = L @ mps @ gate @ mpo @ conj_gate @ conj_mps

    out.reorder_edges(output_order)
    if normalize:
      out.tensor /= tf.linalg.norm(out.tensor)
    self.left_envs[site + 1] = out.tensor

  def add_unitary_right(self, site, reference_mps, normalize=False):
    net = tn.TensorNetwork(backend=self.backend)
    gate = net.add_node(self.gates[site])
    mps = net.add_node(reference_mps.get_tensor(site))
    mpo = net.add_node(self.mpo.get_tensor(site))
    conj_mps = net.add_node(net.backend.conj(reference_mps.get_tensor(site)))
    conj_gate = net.add_node(net.backend.conj(self.gates[site]))
    if site == len(self.mpo) - 1:
      R = net.add_node(net.backend.ones((1, 1, 1), dtype=self.mpo.dtype))
      self.right_envs[site] = R.tensor

    R = net.add_node(self.right_envs[site])

    R[0] ^ mps[2]
    R[1] ^ conj_mps[2]
    R[2] ^ mpo[1]
    gate[1] ^ mps[1]
    gate[0] ^ mpo[3]
    conj_gate[1] ^ conj_mps[1]
    conj_gate[0] ^ mpo[2]

    output_order = [mps[0], conj_mps[0], mpo[0]]
    out = R @ mps @ gate @ mpo @ conj_gate @ conj_mps

    out.reorder_edges(output_order)
    if normalize:
      out.tensor /= tf.linalg.norm(out.tensor)
    self.right_envs[site - 1] = out.tensor

  def compute_left_envs(self, reference_mps, normalize=False):
    for site in range(len(self.mpo)):
      self.add_unitary_left(site, reference_mps, normalize=normalize)

  def compute_right_envs(self, reference_mps, normalize=False):
    for site in reversed(range(len(self.mpo))):
      self.add_unitary_right(site, reference_mps, normalize=normalize)

  def get_environment(self, site, reference_mps):
    net = tn.TensorNetwork(backend=self.backend)
    L = net.add_node(self.left_envs[site])
    R = net.add_node(self.right_envs[site])
    mpo = net.add_node(self.mpo[site])
    mps = net.add_node(reference_mps.get_tensor(site))
    conj_mps = net.add_node(net.backend.conj(reference_mps.get_tensor(site)))
    conj_gate = net.add_node(net.backend.conj(self.gates[site]))

    L[0] ^ mps[0]
    R[0] ^ mps[2]

    L[1] ^ conj_mps[0]
    R[1] ^ conj_mps[2]

    L[2] ^ mpo[0]
    R[2] ^ mpo[1]

    conj_gate[1] ^ conj_mps[1]
    conj_gate[0] ^ mpo[2]
    output_order = [mpo[3], mps[1]]
    out = L @ conj_mps @ conj_gate @ mpo @ R @ mps
    out.reorder_edges(output_order)
    return out.tensor

  @staticmethod
  def update_svd_numpy(env):
    """
        obtain the update to the disentangler using numpy svd
        Fixme: this currently only works with numpy arrays
        Args:
            wIn (np.ndarray or Tensor):  unitary tensor of rank 4
        Returns:
            The svd update of `wIn`
        """
    ut, st, vt = np.linalg.svd(env, full_matrices=False)
    return -misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1, -2]])

  def reset_gates(self, which='eye', noise=0.0):
    """
        reset the one-body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
            noise (float): nose parameter; if nonzero, add noise to the identities
        Returns:
            dict:          maps (s,s+1) to gate for s even
        Raises:
            ValueError
        """

    if which in ('e', 'eye', 'h', 'haar', 'i', 'identities'):
      ds = [self.mpo.get_tensor(site).shape[2] for site in range(len(self.mpo))]
      self.gates = initialize_one_body_gates(
          ds, self.mpo.dtype, which, noise=noise)
    else:
      raise ValueError('wrong value {} for argument `which`'.format(which))

  def absorb_gates(self):
    final_mpo = copy.deepcopy(self.mpo)
    for site in range(len(self.mpo)):
      net = tn.TensorNetwork(backend=self.backend)
      mpo = net.add_node(self.mpo[site])
      gate = net.add_node(self.gates[site])
      conj_gate = net.add_node(net.backend.conj(self.gates[site]))
      mpo[2] ^ conj_gate[0]
      mpo[3] ^ gate[0]
      output_order = [mpo[0], mpo[1], conj_gate[1], gate[1]]
      out = mpo @ gate @ conj_gate
      out.reorder_edges(output_order)
      final_mpo._tensors[site] = out.tensor
    return final_mpo

  def stoquastisize(self, reference_mps, num_steps, normalize=False):
    self.compute_right_envs(reference_mps, normalize=normalize)
    self.add_unitary_left(
        0, reference_mps,
        normalize=normalize)  #needed for initialization of left_envs[0]
    for step in range(num_steps):

      for site in range(len(self.mpo)):
        env = self.get_environment(site, reference_mps)
        cost = misc_mps.ncon([env, self.gates[site]], [[1, 2], [1, 2]])
        if tf.real(cost) > 0:
          return False
        self.gates[site] = self.update_svd_numpy(env)
        self.add_unitary_left(site, reference_mps, normalize=normalize)
        stdout.write("\r step %i/%i cost: %.6E" % (step + 1, num_steps, cost))
        stdout.flush()

      for site in reversed(range(len(self.mpo))):
        env = self.get_environment(site, reference_mps)
        cost = misc_mps.ncon([env, self.gates[site]], [[1, 2], [1, 2]])
        if tf.real(cost) > 0:
          return False
        self.gates[site] = self.update_svd_numpy(env)
        self.add_unitary_right(site, reference_mps, normalize=normalize)
        stdout.write("\r step %i/%i cost: %.6E" % (step + 1, num_steps, cost))
        stdout.flush()


class TwoBodyStoquastisizer:

  def __init__(self,
               mpo,
               gates=None,
               name='TwoBodyStoquastisizer',
               backend='tensorflow'):
    """
        uses a unitary circuit with three layers:
        the first layer 1 contains `N/2` two-body unitaries on sites (site1, site2) with site1 even 
        the second layer 2 contains `N/2 - 1` two-body unitaries on sites (site1, site2) with site1 odd 

        index conventions for one-body unitaries:

               1
               |
              ___
             |   |
             |___|
               |
               0

        index conventions for two-body unitaries:

               2   3
               |   |
              _______
             |       |
             |_______|
               |   |
               0   1
    
        2,3 are the physical outgoing and incoming indices, respectively. The conjugated 
        side of the MPS is on the bottom (at index 2)

        MPS index convention:
              ___
             |   |
         0---     ---2
             |___|
               |
               1



        MPO index convention:

               3
              _|_
             |   |
         0---     ---1
             |___|
               |
               2

    
        An MPO by this convention is contracted from above:
                     
                    _|_
                 --|___|--
                     |
                    ___
                   |   |
                   |___|
                     |

                _|_     _|_    
             --|   | --|   |--
                ---     ---   
                 |       |     
                ___________ 
               |           |
               |___________|
                 |       |  


    
        Args:
            mpo (FiniteMPSCentralGauge):       an mps of even length
            one_body_gates (iterable or None): an iterable mapping sites to matrices
                                               `one_body_gates[site]` is the one-body unitary  at site `site
                                               if `None`, one-body gates are initialized with identities
            two_body_gates (dict or None):     dictionary mapping tuples `(site`, site2)` to rank-4 unitary tensors
                                               the convention 
                                               `one_body_gates[site]` is the one-body unitary  at site `site
                                               if `None`, two-body gates are initialized with identities
            name (str):                        an optional name for the object
        """

    self.name = name
    self.mpo = mpo
    self.right_envs = {}
    self.left_envs = {}
    self.backend = backend
    if (gates == None) or (len(gates) == 0):
      ds = [self.mpo.get_tensor(site).shape[2] for site in range(len(mpo))]
      self.gates = initialize_even_two_body_gates(
          ds, self.mpo.dtype, which='e', noise=0.0)
      self.gates.update(
          initialize_odd_two_body_gates(
              ds, self.mpo.dtype, which='e', noise=0.0))
    else:
      self.gates = gates

    if len(self.gates) != len(mpo) - 1:
      raise ValueError('len(gates) != len(mpo) - 1')

  def add_unitary_left(self, sites, reference_mps, normalize=False):
    site = sites[0]
    if site == 0:
      net = tn.TensorNetwork(backend=self.backend)
      L = net.add_node(net.backend.ones((1, 1, 1), dtype=self.mpo.dtype))
      mps = net.add_node(reference_mps.get_tensor(site))
      mpo = net.add_node(self.mpo.get_tensor(site))
      conj_mps = net.add_node(net.backend.conj(reference_mps.get_tensor(site)))
      L[0] ^ mps[0]
      L[1] ^ conj_mps[0]
      L[2] ^ mpo[0]
      output_order = [
          mps[2], conj_mps[2], mps[1], conj_mps[1], mpo[2], mpo[3], mpo[1]
      ]
      result = L @ mps @ mpo @ conj_mps
      result.reorder_edges(output_order)
      self.left_envs[(site, site + 1)] = result.tensor

    net = tn.TensorNetwork(backend=self.backend)
    gate = net.add_node(self.gates[(site, site + 1)])
    mps = net.add_node(reference_mps.get_tensor(site + 1))
    mpo = net.add_node(self.mpo.get_tensor(site + 1))
    conj_mps = net.add_node(
        net.backend.conj(reference_mps.get_tensor(site + 1)))
    conj_gate = net.add_node(net.backend.conj(self.gates[(site, site + 1)]))

    L = net.add_node(self.left_envs[(site, site + 1)])
    L[0] ^ mps[0]
    L[1] ^ conj_mps[0]
    L[2] ^ gate[2]
    L[3] ^ conj_gate[2]
    L[4] ^ conj_gate[0]
    L[5] ^ gate[0]
    L[6] ^ mpo[0]
    if site % 2 == 1:
      mps[1] ^ gate[3]
      conj_mps[1] ^ conj_gate[3]
      output_order = [
          mps[2], conj_mps[2], gate[1], conj_gate[1], mpo[2], mpo[3], mpo[1]
      ]
      out = L @ mps @ gate @ conj_mps @ conj_gate @ mpo
    else:
      mpo[2] ^ conj_gate[1]
      mpo[3] ^ gate[1]
      output_order = [
          mps[2], conj_mps[2], mps[1], conj_mps[1], conj_gate[3], gate[3],
          mpo[1]
      ]
      out = L @ mpo @ gate @ conj_gate @ conj_mps @ mps

    out.reorder_edges(output_order)
    self.left_envs[(site + 1, site + 2)] = out.tensor

  def add_unitary_right(self, sites, reference_mps, normalize=False):
    site = sites[1]
    if site == len(self.mpo) - 1:
      net = tn.TensorNetwork(backend=self.backend)
      R = net.add_node(net.backend.ones((1, 1, 1), dtype=self.mpo.dtype))

      mps = net.add_node(reference_mps.get_tensor(site))
      mpo = net.add_node(self.mpo.get_tensor(site))
      conj_mps = net.add_node(net.backend.conj(reference_mps.get_tensor(site)))
      R[0] ^ mps[2]
      R[1] ^ conj_mps[2]
      R[2] ^ mpo[1]
      output_order = [
          mps[0], conj_mps[0], mps[1], conj_mps[1], mpo[2], mpo[3], mpo[0]
      ]
      result = R @ mps @ mpo @ conj_mps
      result.reorder_edges(output_order)
      self.right_envs[(site - 1, site)] = result.tensor

    net = tn.TensorNetwork(backend=self.backend)
    gate = net.add_node(self.gates[(site - 1, site)])
    mps = net.add_node(reference_mps.get_tensor(site - 1))
    mpo = net.add_node(self.mpo.get_tensor(site - 1))
    conj_mps = net.add_node(
        net.backend.conj(reference_mps.get_tensor(site - 1)))
    conj_gate = net.add_node(net.backend.conj(self.gates[(site - 1, site)]))

    R = net.add_node(self.right_envs[(site - 1, site)])
    R[0] ^ mps[2]
    R[1] ^ conj_mps[2]
    R[2] ^ gate[3]
    R[3] ^ conj_gate[3]
    R[4] ^ conj_gate[1]
    R[5] ^ gate[1]
    R[6] ^ mpo[1]
    if site % 2 == 0:
      mps[1] ^ gate[2]
      conj_mps[1] ^ conj_gate[2]
      output_order = [
          mps[0], conj_mps[0], gate[0], conj_gate[0], mpo[2], mpo[3], mpo[0]
      ]
      out = R @ mps @ gate @ conj_mps @ conj_gate @ mpo
    else:
      mpo[2] ^ conj_gate[0]
      mpo[3] ^ gate[0]
      output_order = [
          mps[0], conj_mps[0], mps[1], conj_mps[1], conj_gate[2], gate[2],
          mpo[0]
      ]
      out = R @ mpo @ gate @ conj_gate @ conj_mps @ mps

    out.reorder_edges(output_order)
    self.right_envs[(site - 2, site - 1)] = out.tensor

  def compute_left_envs(self, reference_mps, normalize=False):
    for site in range(len(self.mpo) - 1):
      self.add_unitary_left((site, site + 1),
                            reference_mps,
                            normalize=normalize)

  def compute_right_envs(self, reference_mps, normalize=False):
    for site in reversed(range(len(self.mpo) - 1)):
      self.add_unitary_right((site, site + 1),
                             reference_mps,
                             normalize=normalize)

  def get_environment(self, sites):
    net = tn.TensorNetwork(backend=self.backend)
    L = net.add_node(self.left_envs[sites])
    R = net.add_node(self.right_envs[sites])
    conj_gate = net.add_node(net.backend.conj(self.gates[sites]))

    L[0] ^ R[0]
    L[1] ^ R[1]
    L[3] ^ conj_gate[2]
    L[4] ^ conj_gate[0]
    L[6] ^ R[6]
    R[3] ^ conj_gate[3]
    R[4] ^ conj_gate[1]
    output_order = [L[5], R[5], L[2], R[2]]
    out = L @ R @ conj_gate
    out.reorder_edges(output_order)
    return out.tensor

  @staticmethod
  def update_svd_numpy(wIn):
    """
        obtain the update to the disentangler using numpy svd
        Fixme: this currently only works with numpy arrays
        Args:
            wIn (np.ndarray or Tensor):  unitary tensor of rank 4
        Returns:
            The svd update of `wIn`
        """
    shape = tf.shape(wIn)
    ut, st, vt = np.linalg.svd(
        np.reshape(wIn, (shape[0] * shape[1], shape[2] * shape[3])),
        full_matrices=False)
    mat = -misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1, -2]])
    return tf.reshape(mat, shape)

  def reset_gates(self, which='eye', noise=0.0):
    """
        reset the one-body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
            noise (float): nose parameter; if nonzero, add noise to the identities
        Returns:
            dict:          maps (s,s+1) to gate for s even
        Raises:
            ValueError
        """

    if which in ('e', 'eye', 'h', 'haar', 'i', 'identities'):
      ds = [self.mpo.get_tensor(site).shape[2] for site in range(len(self.mpo))]
      self.gates.update(
          initialize_odd_two_body_gates(
              ds, dtype=self.mpo.dtype, which=which, noise=noise))
    else:
      raise ValueError('wrong value {} for argument `which`'.format(which))

  def absorb_gates(self):
    net = tn.TensorNetwork(backend=self.backend)
    gates = {k: net.add_node(v) for k, v in self.gates.items()}
    top_edges = {}
    bottom_edges = {}
    N = len(self.mpo)
    for sites, gate in gates.items():
      site = sites[0]
      if site % 2 == 1:
        top_edges[site] = gate[2]
        top_edges[site + 1] = gate[3]
      if site % 2 == 0:
        bottom_edges[site] = gate[0]
        bottom_edges[site + 1] = gate[1]

    top_edges[0] = gates[(0, 1)][2]
    top_edges[N - 1] = gates[(N - 2, N - 1)][3]

    for sites, gate in gates.items():
      site = sites[0]
      if site % 2 == 0 and site != 0 and site != len(self.mpo) - 2:
        gates[sites][2] ^ gates[(site - 1, site)][1]
        gates[sites][3] ^ gates[(site + 1, site + 2)][0]
      elif site == 0:
        gates[sites][3] ^ gates[(site + 1, site + 2)][0]
      elif site == len(self.mpo) - 2:
        gates[sites][2] ^ gates[(site - 1, site)][1]

    new_gates = []
    anc = {}
    for site in range(len(self.mpo) - 1):
      gate = gates[(site, site + 1)]
      L, R, _ = net.split_node(gate, [gate[0], gate[2]], [gate[1], gate[3]])
      anc[site] = L[2]
      new_gates.append(L)
      new_gates.append(R)
    tensors = [new_gates[0].tensor]
    out_nodes = {}
    out_nodes[0] = new_gates[0]
    out_nodes[N - 1] = new_gates[-1]
    for n in range(1, len(new_gates) - 1, 2):
      out_nodes[(n + 1) // 2] = new_gates[n] @ new_gates[n + 1]

    out_nodes[0].reorder_edges([anc[0], bottom_edges[0], top_edges[0]])
    out_nodes[N - 1].reorder_edges(
        [anc[N - 2], bottom_edges[N - 1], top_edges[N - 1]])
    for s, node in out_nodes.items():
      if s > 0 and s < N - 1:
        node.reorder_edges([anc[s - 1], anc[s], bottom_edges[s], top_edges[s]])
    tensors = [out_nodes[s].tensor for s in sorted(out_nodes.keys())]
    tensors[0] = tf.expand_dims(tensors[0], 0)
    tensors[-1] = tf.expand_dims(tensors[-1], 1)
    final = []
    for site in range(len(self.mpo)):
      net = tn.TensorNetwork(backend=self.backend)
      gate = net.add_node(tensors[site])
      conj_gate = net.add_node(net.backend.conj(tensors[site]))
      mpo = net.add_node(self.mpo[site])
      gate[2] ^ mpo[3]
      conj_gate[2] ^ mpo[2]
      out_order = [
          gate[0], mpo[0], conj_gate[0], gate[1], mpo[1], conj_gate[1],
          conj_gate[3], gate[3]
      ]
      out = gate @ mpo @ conj_gate
      out.reorder_edges(out_order)
      d1, d2, d3, d4, d5, d6, d7, d8 = out.shape
      final.append(tf.reshape(out.tensor, (d1 * d2 * d3, d4 * d5 * d6, d7, d8)))
    return MPO.FiniteMPO(final)

  def stoquastisize(self, reference_mps, num_steps, normalize=False):
    self.compute_right_envs(reference_mps)
    self.add_unitary_left(
        (0, 1), reference_mps)  #needed for initialization of left_envs[0]
    for step in range(num_steps):

      for site in range(0, len(self.mpo) - 1):
        env = self.get_environment((site, site + 1))
        cost = misc_mps.ncon([env, self.gates[(site, site + 1)]],
                             [[1, 2, 3, 4], [1, 2, 3, 4]])
        if tf.real(cost) > 0:
          return False
        self.gates[(site, site + 1)] = self.update_svd_numpy(env)
        self.add_unitary_left((site, site + 1), reference_mps)
        stdout.write("\r step %i/%i cost: %.6E" % (step + 1, num_steps, cost))
        stdout.flush()

      for site in reversed(range(0, len(self.mpo) - 1)):
        env = self.get_environment((site, site + 1))
        cost = misc_mps.ncon([env, self.gates[(site, site + 1)]],
                             [[1, 2, 3, 4], [1, 2, 3, 4]])
        if tf.real(cost) > 0:
          return False
        self.gates[(site, site + 1)] = self.update_svd_numpy(env)
        self.add_unitary_right((site, site + 1), reference_mps)
        stdout.write("\r step %i/%i cost: %.6E" % (step + 1, num_steps, cost))
        stdout.flush()
    return True

  @staticmethod
  def mat_vec(left_env, right_env, left_gate, right_gate, mpo_tensor, backend,
              site, mps_tensor):
    net = tn.TensorNetwork(backend=backend)
    L = net.add_node(left_env)
    R = net.add_node(right_env)
    LGATE = net.add_node(left_gate)
    CONJ_LGATE = net.add_node(net.backend.conj(left_gate))
    RGATE = net.add_node(right_gate)
    CONJ_RGATE = net.add_node(net.backend.conj(right_gate))
    MPS = net.add_node(mps_tensor)
    CONJ_MPS = net.add_node(net.backend.conj(mps_tensor))
    MPO = net.add_node(mpo_tensor)
    if site % 2 == 1:
      L[0] ^ MPS[0]
      L[2] ^ LGATE[2]
      L[3] ^ CONJ_LGATE[2]
      L[4] ^ CONJ_LGATE[0]
      L[5] ^ LGATE[0]
      L[6] ^ MPO[0]
      RGATE[2] ^ MPS[1]
      RGATE[0] ^ LGATE[3]
      LGATE[1] ^ MPO[3]
      CONJ_LGATE[1] ^ MPO[2]
      CONJ_LGATE[3] ^ CONJ_RGATE[0]
      R[0] ^ MPS[2]
      R[2] ^ RGATE[3]
      R[3] ^ CONJ_RGATE[3]
      R[4] ^ CONJ_RGATE[1]
      R[5] ^ RGATE[1]
      R[6] ^ MPO[1]

      output_order = [L[1], CONJ_RGATE[2], R[1]]
      out = (((((
          (L @ LGATE) @ CONJ_LGATE) @ MPO) @ MPS) @ RGATE) @ R) @ CONJ_RGATE
    else:
      L[0] ^ MPS[0]
      L[2] ^ LGATE[2]
      L[3] ^ CONJ_LGATE[2]
      L[4] ^ CONJ_LGATE[0]
      L[5] ^ LGATE[0]
      L[6] ^ MPO[0]
      LGATE[3] ^ MPS[1]
      RGATE[2] ^ LGATE[1]
      RGATE[0] ^ MPO[3]

      CONJ_RGATE[0] ^ MPO[2]
      CONJ_LGATE[1] ^ CONJ_RGATE[2]
      R[0] ^ MPS[2]
      R[2] ^ RGATE[3]
      R[3] ^ CONJ_RGATE[3]
      R[4] ^ CONJ_RGATE[1]
      R[5] ^ RGATE[1]
      R[6] ^ MPO[1]

      output_order = [L[1], CONJ_LGATE[3], R[1]]
      out = (((((
          (L @ LGATE) @ CONJ_LGATE) @ MPO) @ MPS) @ RGATE) @ R) @ CONJ_RGATE

    out.reorder_edges(output_order)
    return out.tensor
