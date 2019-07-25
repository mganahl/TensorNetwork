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
import time
import tensornetwork as tn
import numpy as np
import tensorflow as tf
import experiments.MPS_classifier.batchtensornetwork as btn
import experiments.MPS.misc_mps as misc_mps
from sys import stdout
import experiments.MPS.matrixproductstates as MPS
import experiments.MPS.matrixproductstates as positivize
import functools as fct
from experiments.MPS.matrixproductstates import InfiniteMPSCentralGauge, FiniteMPSCentralGauge



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
    for n in range(len(shape)//2):
        dout *= shape[n]
    for n in range(len(shape)//2, len(shape)):
        din *= shape[n]
    if g.dtype in (tf.complex128, tf.complex64):
        return tf.linalg.logm(tf.reshape(g, (dout, din)))/4
    elif g.dtype in (tf.float32, tf.float64):
        matrix =  tf.linalg.logm(tf.complex(tf.reshape(g, (dout, din)),tf.zeros(shape=[dout,din], dtype=g.dtype)))
        return (matrix - tf.transpose(tf.conj(matrix)))/4


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
        for site in range(0,len(ds) - 1, 2):
            if dtype in (tf.float32, tf.float64):
                two_body_gates[(site, site+1)] = tf.reshape(tf.eye(ds[site]*ds[site+1], dtype=dtype), 
                                                                 (ds[site], ds[site+1], ds[site], ds[site+1]))
            elif dtype in (tf.complex128, tf.complex64):
                two_body_gates[(site, site+1)] = tf.reshape(tf.complex(tf.eye(ds[site]*ds[site+1], dtype=dtype.real_dtype),
                                                                            noise * tf.random_uniform(shape=[ds[site]*ds[site+1],ds[site]*ds[site+1]],
                                                                                                      dtype=dtype.real_dtype, minval=-1, maxval=1)), 
                                                                 (ds[site], ds[site+1], ds[site], ds[site+1]))

    elif which in ('h','haar'):
        for site in range(0,len(ds) - 1, 2):
            two_body_gates[(site, site+1)] = tf.reshape(misc_mps.haar_random_unitary((ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype=dtype), 
                                                             (ds[site], ds[site+1], ds[site], ds[site+1]))
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
                two_body_gates[(site, site+1)] = tf.reshape(tf.complex(tf.eye(ds[site]*ds[site+1], dtype=dtype.real_dtype),
                                                                            noise * tf.random_uniform(shape=[ds[site]*ds[site+1],ds[site]*ds[site+1]],
                                                                                                      dtype=dtype.real_dtype, minval=-1, maxval=1)),
                                                                 (ds[site], ds[site+1], ds[site], ds[site+1]))

    elif which in ('h','haar'):
        for site in range(1, len(ds) - 2, 2):
            two_body_gates[(site, site+1)] = tf.reshape(misc_mps.haar_random_unitary((ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype), 
                                                             (ds[site], ds[site+1], ds[site], ds[site+1]))
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
    elif which in ('h','haar'):
        one_body_gates = {site: misc_mps.haar_random_unitary((ds[site], ds[site]), dtype) for site in range(0,len(ds))}
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
    two_body_generators.update(initialize_odd_two_body_generators(ds, dtype, which))
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
        for site in range(0,len(ds) - 1, 2):
            if dtype in (tf.float32, tf.float64):                
                two_body_generators[(site, site+1)] = tf.eye(ds[site]*ds[site+1], dtype=dtype)
            elif dtype in (tf.complex128, tf.complex64):
                two_body_generators[(site, site+1)] = tf.complex(tf.eye(ds[site]*ds[site+1], dtype=dtype.real_dtype),
                                                                 tf.zeros(shape=[ds[site]*ds[site+1],ds[site]*ds[site+1]],dtype=dtype.real_dtype))
    elif which in ('r','random'):
        for site in range(0,len(ds) - 1, 2):        
            if dtype in (tf.float32, tf.float64):                            
                two_body_generators[(site, site+1)] = tf.random_uniform(shape=(ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype=dtype)
            elif dtype in (tf.complex128, tf.complex64):
                two_body_generators[(site, site+1)] = tf.complex(tf.random_uniform(shape=(ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype=dtype.real_dtype),
                                                                 tf.random_uniform(shape=(ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype=dtype.real_dtype))

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
                two_body_generators[(site, site+1)] = tf.eye(ds[site]*ds[site+1], dtype=dtype)
            elif dtype in (tf.complex128, tf.complex64):
                two_body_generators[(site, site+1)] = tf.complex(tf.eye(ds[site]*ds[site+1], dtype=dtype.real_dtype),
                                                                 tf.zeros(shape=[ds[site]*ds[site+1],ds[site]*ds[site+1]],dtype=dtype.real_dtype))
    elif which in ('r','random'):
        for site in range(1, len(ds) - 2, 2):
            if dtype in (tf.float32, tf.float64):                            
                two_body_generators[(site, site+1)] = tf.random_uniform(shape=(ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype=dtype)
            elif dtype in (tf.complex128, tf.complex64):
                two_body_generators[(site, site+1)] = tf.complex(tf.random_uniform(shape=(ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype=dtype.real_dtype),
                                                                 tf.random_uniform(shape=(ds[site] * ds[site+1], ds[site] * ds[site+1]), dtype=dtype.real_dtype))
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
            one_body_generators = {site: tf.eye(ds[site], dtype=dtype)                              
                                   for site in range(0,len(ds))}
        elif dtype in (tf.float64,tf.complex128):
            one_body_generators = {site: tf.complex(tf.eye(ds[site], dtype=dtype.real_dtype), tf.zeros(ds[site], dtype=dtype.real_dtype))
                                   for site in range(0,len(ds))}
    elif which in ('r','random'):
        if dtype in (tf.float32, tf.float64):                        
            one_body_generators = {site: tf.random_uniform(shape=(ds[site], ds[site]), dtype=dtype.real_dtype) for site in range(0,len(ds))}
        elif dtype in (tf.float64,tf.complex128):            
            one_body_generators = {site: tf.complex(tf.random_uniform(shape=(ds[site], ds[site]), dtype=dtype.real_dtype),
                                                    tf.random_uniform(shape=(ds[site], ds[site]), dtype=dtype.real_dtype))
                                   for site in range(0,len(ds))}
    else:
        raise ValueError('unrecognized value which = {0}'.format(which))            
    return one_body_generators            



class OverlapMinimizer:
    """
    minimizes the overlap between `mps` and a given reference mps, using a double layer of two-body unitaries.
    For now `mps` has to have even length

    """
    def __init__(self, mps, one_body_gates=None, two_body_gates=None, name='overlap_minimizer'):
        """
        initialize an OverlapMinimizer object
        This object minimizes the overlab between `mps` and a reference mps 
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


    def reset_two_body_gates(self, which='identities', dtype=None, noise=0.0):
        """
        reset the two body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
        Raises:
            ValueError
        """
        if not dtype:
            dtype = self.mps.dtype
        
            
        self.two_body_gates = initialize_even_two_body_gates(self.mps.d, self.mps.dtype, which, noise=noise)
        self.two_body_gates.update(initialize_odd_two_body_gates(self.mps.d, self.mps.dtype, which,noise=noise))
        
    def reset_even_two_body_gates(self, which='identities', dtype=None, noise=0.0):
        """
        reset the two body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
        Raises:
            ValueError
        """
        self.two_body_gates.update(initialize_even_two_body_gates(self.mps.d, self.mps.dtype, which, noise=noise))

    def reset_odd_two_body_gates(self, which='identities', dtype=None, noise=0.0):
        """
        reset the two body gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
        Raises:
            ValueError
        """
        self.two_body_gates.update(initialize_odd_two_body_gates(self.mps.d, self.mps.dtype, which, noise=noise))        
        
    def reset_one_body_gates(self, which='eye', dtype=None, noise=0.0):
        """
        reset the two one gates
        Args:
            which (str):   the type to which gates should be reset
                           `which` can take values in {'eye','e', 'identities', 'i'} for identity operators
                           or in ('h','haar') for Haar random unitaries
        Raises:
            ValueError
        """

        if which in ('e','eye','h','haar', 'i', 'identities'):
            self.one_body_gates = initialize_one_body_gates(self.mps.d, self.mps.dtype, which, noise=noise)
        else:
            ds = self.mps.ds            
            self.one_body_gates = {}
            for site in range(len(ds)):
                self.one_body_gates[site] = get_gate_from_generator(gen[site],(ds[site], ds[site]))
            
    @staticmethod
    def add_unitary_batched_right(site, right_envs, mps, samples, one_body_gates, two_body_gates):
        """ 
        samples (tf.Tensor of shape (Nt, N)
        """
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        #fixme: calling tf.one_hot could be slowing things down. 
        assert(site>0)
        assert(len(mps)%2==0)
        if site == (len(mps) - 1):
            right_envs[site - 1] = tf.squeeze(misc_mps.ncon([mps.get_tensor(site), one_body_gates[site],
                                                             two_body_gates[(site-1,site)], tf.one_hot(samples[:,site], mps.d[site], dtype=mps.dtype)],
                                                            [[-2,1,-5], [2, 1], [-4, 3, -3, 2],[-1, 3]]), 4)
        elif (site < len(mps) - 1) and (site % 2 == 0):
            tmp = misc_mps.ncon([right_envs[site], mps.get_tensor(site), one_body_gates[site], two_body_gates[(site-1, site)]],
                                [[-1,1,3,4], [-2,2,1], [3, 2], [-4, -5, -3, 4]]) #shape (Nt, D, din, dout, d)
            Nt, D, din, dout, d = tmp.shape
            tmp = tf.reshape(tmp, (Nt, D * din * dout, d)) #(Nt, D * din * dout, d)
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2) #(Nt, d, 1)
            right_envs[site - 1] = tf.reshape(tf.matmul(tmp, tmp2),(Nt, D, din, dout))            
            
        elif (site < len(mps) - 1) and (site % 2 == 1):
            tmp = misc_mps.ncon([right_envs[site], mps.get_tensor(site), one_body_gates[site], two_body_gates[(site-1,site)]],
                                [[-1, 1, 4, -5],[-2, 2, 1], [3, 2], [-4, 4, -3, 3]]) #has shape (Nt, Dl, din1, dout1, d)
            Nt, D, din, dout, d = tmp.shape
            tmp = tf.reshape(tmp, (Nt, D * din * dout, d))
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2) #has dim (Nt, d, 1)
            right_envs[site - 1] = tf.reshape(tf.matmul(tmp, tmp2),(Nt, D, din, dout))

    @staticmethod            
    def add_unitary_batched_left(site, left_envs, mps, samples, one_body_gates, two_body_gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site<len(mps))
        assert(len(mps)%2==0)
            
        if site == 0:
            left_envs[site + 1] = tf.squeeze(misc_mps.ncon([mps.get_tensor(site), one_body_gates[site], two_body_gates[(site, site + 1)],
                                                            tf.one_hot(samples[:,site], mps.d[site], dtype=mps.dtype)],
                                                           [[-5, 1, -2], [2, 1], [3, -4, 2, -3], [-1, 3]]), 4)
        elif (site > 0) and (site % 2 == 0):
            tmp = misc_mps.ncon([left_envs[site], mps.get_tensor(site), one_body_gates[site], two_body_gates[(site, site + 1)]],
                                [[-1, 1, 4, -5], [1, 2, -2], [3, 2], [4, -4, 3, -3]]) #has shape (Nt, D, di, do, d)
            Nt, D, di, do, d = tmp.shape
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2)#has dim (Nt, d, 1)
            tmp = tf.reshape(tmp,(Nt, D * di * do, d))
            left_envs[site + 1] = tf.reshape(tf.matmul(tmp, tmp2), (Nt, D, di, do))
            
        elif (site > 0) and (site % 2 == 1):
            tmp = misc_mps.ncon([left_envs[site], mps.get_tensor(site), one_body_gates[site], two_body_gates[(site, site + 1)]],
                                [[-1, 1, 3, 4], [1, 2, -2], [3, 2], [-5, -4, 4, -3]]) #has shape (Nt, D, di, do, d)
            Nt, D, di, do, d = tmp.shape
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2)#has dim (Nt, d, 1)
            tmp = tf.reshape(tmp,(Nt, D * di * do, d))
            left_envs[site + 1] = tf.reshape(tf.matmul(tmp, tmp2), (Nt, D, di, do))
            
    @staticmethod            
    def get_two_body_env_batched(sites, left_envs, right_envs, one_body_gates, mps, samples):
        assert((len(mps) % 2) == 0)
        assert(sites[0] >= 0)
        assert(sites[1] < len(mps))
        if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(left_envs[sites[0]])
            renv_node = bnet.add_node(right_envs[sites[1]])
            right_sample_node = bnet.add_node(tf.one_hot(samples[:, sites[1]], ds[sites[1]], dtype=mps.dtype))
            left_sample_node = bnet.add_node(tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype))
            right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
            left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
            right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
            left_unitary_node = bnet.add_node(one_body_gates[sites[0]])            
            
            order = [lenv_node[0], lenv_node[2], renv_node[2], left_unitary_node[0], right_unitary_node[0]]
            e1 = bnet.connect(lenv_node[3], left_sample_node[1])
            e2 = bnet.connect(lenv_node[1], left_tensor_node[0])
            e3 = bnet.connect(renv_node[3], right_sample_node[1])
            e4 = bnet.connect(renv_node[1], right_tensor_node[2])
            e5 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            e6 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
            e7 = bnet.connect(right_tensor_node[1], right_unitary_node[1])            
            
            bnet.batched_contract_between(lenv_node, left_sample_node, lenv_node[0], left_sample_node[0])
            bnet.batched_contract_between(renv_node, right_sample_node, renv_node[0], right_sample_node[0])
            bnet.contract(e6)
            bnet.contract(e7)            
            rtmp = bnet.contract(e4)
            ltmp = bnet.contract(e2)
            out = bnet.batched_contract_between(ltmp, rtmp, lenv_node[0], renv_node[0])
            out = out.reorder_edges(order)
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0]) 

        #odd sites
        elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
            ds = mps.d                        
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[sites[0]],4)) #add fake legs to the tensors
            renv_node = bnet.add_node(tf.expand_dims(right_envs[sites[1]],4))
            right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
            left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
            left_sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype), 2))
            right_sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, sites[1]], ds[sites[0]], dtype=mps.dtype), 2))            
            right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
            right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
            left_unitary_node = bnet.add_node(one_body_gates[sites[0]])            
            
            order = [lenv_node[0], left_sample_node[1], right_sample_node[1], lenv_node[3],  renv_node[3]]
            
            e1 = bnet.connect(lenv_node[1], left_tensor_node[0])
            e2 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
            e3 = bnet.connect(lenv_node[2], left_unitary_node[0])
            e4 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            e5 = bnet.connect(right_tensor_node[1], right_unitary_node[1])                        
            e6 = bnet.connect(renv_node[2], right_unitary_node[0])
            e7 = bnet.connect(renv_node[1], right_tensor_node[2])
            e8 = bnet.connect(lenv_node[4], left_sample_node[2])
            e9 = bnet.connect(renv_node[4], right_sample_node[2])
            
            ltmp = bnet.contract(e2)#contract one-body unitaries
            rtmp = bnet.contract(e5)
            
            ltmp = bnet.contract_between(lenv_node, ltmp)
            rtmp = bnet.contract_between(renv_node, rtmp)
            
            ctmp = bnet.batched_contract_between(ltmp, rtmp, lenv_node[0], renv_node[0])
            ctmp = bnet.batched_contract_between(ctmp, left_sample_node, lenv_node[0], left_sample_node[0])
            out = bnet.batched_contract_between(ctmp, right_sample_node, lenv_node[0], right_sample_node[0])            
            out = out.reorder_edges(order)
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])

            
        elif sites[0] == 0:
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            renv_node = bnet.add_node(tf.expand_dims(right_envs[sites[1]],4))
            right_sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, sites[1]], ds[sites[1]], dtype=mps.dtype), 2))
            left_sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype), 2))
            right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
            left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
            right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
            left_unitary_node = bnet.add_node(one_body_gates[sites[0]])            

            
            order = [renv_node[0], left_sample_node[1], renv_node[2], left_unitary_node[0], right_unitary_node[0]]
            
            e1 = bnet.connect(left_tensor_node[0], left_sample_node[2])
            e2 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            e3 = bnet.connect(renv_node[1], right_tensor_node[2])
            e4 = bnet.connect(renv_node[3], right_sample_node[1])
            e5 = bnet.connect(renv_node[4], right_sample_node[2])
            e6 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
            e7 = bnet.connect(right_tensor_node[1], right_unitary_node[1])            

            bnet.contract(e6)
            bnet.contract(e7)
            
            bnet.batched_contract_between(renv_node, right_sample_node, renv_node[0], right_sample_node[0])
            bnet.contract(e3)
            tmp = bnet.contract(e2)
            out = bnet.batched_contract_between(tmp, left_sample_node, renv_node[0], left_sample_node[0])
            out = out.reorder_edges(order)
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])


        elif sites[1] == (len(mps) - 1):
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[sites[0]],4))
            right_sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, sites[1]], ds[sites[1]], dtype=mps.dtype), 2))
            left_sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype), 2))
            right_tensor_node = bnet.add_node(mps.get_tensor(sites[1]))
            left_tensor_node = bnet.add_node(mps.get_tensor(sites[0]))
            right_unitary_node = bnet.add_node(one_body_gates[sites[1]])
            left_unitary_node = bnet.add_node(one_body_gates[sites[0]])            
            
            order = [lenv_node[0], lenv_node[2], right_sample_node[1], left_unitary_node[0], right_unitary_node[0]]
            e1 = bnet.connect(right_tensor_node[2], right_sample_node[2])
            e2 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            e3 = bnet.connect(lenv_node[1], left_tensor_node[0])
            e4 = bnet.connect(lenv_node[3], left_sample_node[1])
            e5 = bnet.connect(lenv_node[4], left_sample_node[2])
            e6 = bnet.connect(left_tensor_node[1], left_unitary_node[1])
            e7 = bnet.connect(right_tensor_node[1], right_unitary_node[1])            

            bnet.contract(e6)
            bnet.contract(e7)

            bnet.batched_contract_between(lenv_node, left_sample_node, lenv_node[0], left_sample_node[0])
            bnet.contract(e3)
            tmp = bnet.contract(e2)
            out = bnet.batched_contract_between(tmp, right_sample_node, lenv_node[0], right_sample_node[0])
            out = out.reorder_edges(order)
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])

    @staticmethod
    def get_one_body_env_batched(site,left_envs,right_envs, mps, samples):
        if (site not in (0, len(mps) - 1))  and (site % 2 == 1):
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
            renv_node = bnet.add_node(right_envs[site])
            sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
            tensor_node = bnet.add_node(mps.get_tensor(site))
            
            order = [lenv_node[0], lenv_node[2], tensor_node[1]]
            
            e1 = bnet.connect(lenv_node[1], tensor_node[0])
            e2 = bnet.connect(renv_node[1], tensor_node[2])
            e3 = bnet.connect(lenv_node[3], renv_node[2])
            e4 = bnet.connect(renv_node[3], sample_node[1])
            e5 = bnet.connect(lenv_node[4], sample_node[2])
            
            tmp = bnet.contract_between(lenv_node, tensor_node)
            tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
            out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
            out = out.reorder_edges(order)
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])

        elif (site not in (0, len(mps) - 1)) and (site % 2) == 0:
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
            renv_node = bnet.add_node(right_envs[site])
            sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
            tensor_node = bnet.add_node(mps.get_tensor(site))
            
            order = [lenv_node[0], renv_node[2], tensor_node[1]]
            
            e1 = bnet.connect(lenv_node[1], tensor_node[0])
            e2 = bnet.connect(renv_node[1], tensor_node[2])
            e3 = bnet.connect(lenv_node[2], renv_node[3])
            e4 = bnet.connect(lenv_node[3], sample_node[1])
            e5 = bnet.connect(lenv_node[4], sample_node[2])
            
            tmp = bnet.contract_between(lenv_node, tensor_node)
            tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
            out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
            out = out.reorder_edges(order)            
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])

        elif site == 0:
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            renv_node = bnet.add_node(right_envs[site])
            sample_node = bnet.add_node(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype))
            tensor_node = bnet.add_node(tf.squeeze(mps.get_tensor(site),0)) #remove trailing 0th dimension of the mps-tensor at the left boundary
            
            order = [renv_node[0], renv_node[2], tensor_node[0]]
            
            e1 = bnet.connect(renv_node[1], tensor_node[1])
            e2 = bnet.connect(renv_node[3], sample_node[1])

            
            tmp = bnet.contract_between(renv_node, tensor_node)
            out = bnet.batched_contract_between(tmp, sample_node, renv_node[0], sample_node[0])
            out = out.reorder_edges(order)
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])
            
        elif site == (len(mps) - 1):
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(left_envs[site])
            sample_node = bnet.add_node(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype))
            tensor_node = bnet.add_node(tf.squeeze(mps.get_tensor(site), 2)) #remove trailing dimension from mps tensor
            
            order = [lenv_node[0], lenv_node[2], tensor_node[1]]
            
            e1 = bnet.connect(lenv_node[1], tensor_node[0])
            e2 = bnet.connect(lenv_node[3], sample_node[1])
            
            tmp = bnet.contract_between(lenv_node, tensor_node)
            out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
            out = out.reorder_edges(order)            
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])
            
        
    @staticmethod        
    def check_overlap_batched(site,left_envs,right_envs, mps, samples):
        """
        a check ; this should return one mps was normalized and samples
        are all basis states
        """
        if site%2 == 1:
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
            renv_node = bnet.add_node(right_envs[site])
            sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
            tensor_node = bnet.add_node(mps.get_tensor(site))

            e1 = bnet.connect(lenv_node[1], tensor_node[0])
            e2 = bnet.connect(renv_node[1], tensor_node[2])
            e3 = bnet.connect(lenv_node[2], tensor_node[1])
            e4 = bnet.connect(lenv_node[3], renv_node[2])
            e5 = bnet.connect(renv_node[3], sample_node[1])
            e6 = bnet.connect(lenv_node[4], sample_node[2])
            tmp = bnet.contract_between(lenv_node, tensor_node)
            tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
            out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
            return tf.math.reduce_sum(tf.pow(out.tensor,2), axis=0)

        elif site%2 == 0:
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
            renv_node = bnet.add_node(right_envs[site])
            sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
            tensor_node = bnet.add_node(mps.get_tensor(site))

            e1 = bnet.connect(lenv_node[1], tensor_node[0])
            e2 = bnet.connect(renv_node[1], tensor_node[2])
            e3 = bnet.connect(renv_node[2], tensor_node[1])
            e4 = bnet.connect(lenv_node[2], renv_node[3])
            e5 = bnet.connect(lenv_node[3], sample_node[1])
            e6 = bnet.connect(lenv_node[4], sample_node[2])
            tmp = bnet.contract_between(lenv_node, tensor_node)
            tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
            out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
            return tf.math.reduce_sum(tf.pow(out.tensor,2), axis=0)
        
    @staticmethod        
    def overlap_batched(site, left_envs, right_envs, one_body_gates, mps, samples):
        if site%2 == 1:
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
            renv_node = bnet.add_node(right_envs[site])
            unitary_node = bnet.add_node(one_body_gates[site])
            sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
            tensor_node = bnet.add_node(mps.get_tensor(site))

            e1 = bnet.connect(lenv_node[1], tensor_node[0])
            e2 = bnet.connect(renv_node[1], tensor_node[2])
            e3 = bnet.connect(unitary_node[1], tensor_node[1])
            e4 = bnet.connect(lenv_node[2], unitary_node[0])
            e5 = bnet.connect(lenv_node[3], renv_node[2])
            e6 = bnet.connect(renv_node[3], sample_node[1])
            e7 = bnet.connect(lenv_node[4], sample_node[2])
            
            tmp = bnet.contract_between(tensor_node, unitary_node)
            tmp = bnet.contract_between(lenv_node, tmp)
            tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
            out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])

        elif site%2 == 0:
            ds = mps.d            
            bnet = btn.BatchTensorNetwork()
            lenv_node = bnet.add_node(tf.expand_dims(left_envs[site],4))
            renv_node = bnet.add_node(right_envs[site])
            unitary_node = bnet.add_node(one_body_gates[site])
            sample_node = bnet.add_node(tf.expand_dims(tf.one_hot(samples[:, site], ds[site], dtype=mps.dtype),2))
            tensor_node = bnet.add_node(mps.get_tensor(site))

            e1 = bnet.connect(lenv_node[1], tensor_node[0])
            e2 = bnet.connect(renv_node[1], tensor_node[2])
            e3 = bnet.connect(tensor_node[1], unitary_node[1])
            e4 = bnet.connect(renv_node[2], unitary_node[0])
            e5 = bnet.connect(lenv_node[2], renv_node[3])
            e6 = bnet.connect(lenv_node[3], sample_node[1])
            e7 = bnet.connect(lenv_node[4], sample_node[2])
            
            tmp = bnet.contract_between(tensor_node, unitary_node)
            tmp = bnet.contract_between(lenv_node, tmp)
            tmp = bnet.batched_contract_between(tmp,renv_node, lenv_node[0], renv_node[0])
            out = bnet.batched_contract_between(tmp, sample_node, lenv_node[0], sample_node[0])
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])
            
    @staticmethod
    def add_unitary_right(site, right_envs, mps, conj_mps, one_body_gates, two_body_gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site>0)
        assert(len(mps)%2==0)
        tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]], [[-1,1,-3],[-2, 1]])        
        if site == (len(mps) - 1):
            right_envs[site - 1] = misc_mps.ncon([tensor, tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site-1,site)]],
                                                 [[-1,2,1],[-2,3,1],[-4,3,-3,2]])
        elif (site < len(mps) - 1) and (site % 2 == 0):
            right_envs[site - 1] = misc_mps.ncon([right_envs[site], tensor,
                                                  tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site-1,site)]],
                                                 [[1,3,2,5],[-1,2,1],[-2,4,3],[-4,4,-3,5]])
        elif (site < len(mps) - 1) and (site % 2 == 1):
            right_envs[site - 1] = misc_mps.ncon([right_envs[site], tensor,
                                                  tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site-1,site)]],
                                                 [[1,4,3,5],[-1,2,1],[-2,5,4],[-4,3,-3,2]])
    @staticmethod            
    def add_unitary_left(site, left_envs, mps, conj_mps, one_body_gates, two_body_gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site<len(mps) - 1)
        assert(len(mps)%2==0)
        tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]], [[-1,1,-3],[-2, 1]])
        if site == 0:
            left_envs[site + 1] = misc_mps.ncon([tensor, tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site, site + 1)]],
                                                [[1,2,-1],[1,3, -2],[3, -4, 2, -3]])        
        elif (site > 0) and (site % 2 == 0):
            left_envs[site + 1] = misc_mps.ncon([left_envs[site], tensor, 
                                                 tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site, site + 1)]],
                                                [[1,4,3,5], [1,2,-1], [4,5,-2], [3,-4,2,-3]])                                          
        elif (site > 0) and (site % 2 == 1):
            left_envs[site + 1] = misc_mps.ncon([left_envs[site], tensor, 
                                                 tf.conj(conj_mps.get_tensor(site)), two_body_gates[(site, site + 1)]],
                                                [[1,3,2,5],[1,2,-1],[3,4,-2],[4,-4,5,-3]])


    def compute_right_envs(self, ref_mps):
        [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
    def compute_left_envs(self, ref_mps):
        [self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in range(len(self.mps) - 1)]

    def compute_right_envs_batched(self, samples):
        [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
    def compute_left_envs_batched(self, samples):
        [self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in range(len(self.mps) - 1)]

            
    @staticmethod            
    def get_two_body_env(sites,left_envs,right_envs, one_body_gates, mps, conj_mps):
        """
        """
        assert((len(mps) % 2) == 0)
        tensor_left = misc_mps.ncon([mps.get_tensor(sites[0]), one_body_gates[sites[0]]], [[-1,1,-3],[-2, 1]])
        tensor_right = misc_mps.ncon([mps.get_tensor(sites[1]), one_body_gates[sites[1]]], [[-1,1,-3],[-2, 1]])
        
        if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
            return misc_mps.ncon([left_envs[sites[0]], tensor_left, tensor_right, right_envs[sites[1]], 
                                  tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                                 [[7,1,-1,2], [7,-3,6], [6,-4,8], [8,4,-2,3], [1,2,5], [5,3,4]])
        elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
            return misc_mps.ncon([left_envs[sites[0]], tensor_left, 
                                  tensor_right, right_envs[sites[1]], 
                                  tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                                 [[1, 7, 2, -3], [1, 2, 3], [3, 4, 5], [5, 8, 4, -4], [7, -1, 6], [6, -2, 8]])    
        elif sites[0] == 0:
            return misc_mps.ncon([tensor_left, tensor_right, right_envs[sites[1]], 
                                  tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                                 [[1, -3, 2], [2, -4, 3], [3, 4, -2, 5], [1, -1, 6], [6, 5, 4]])  
        elif sites[1] == (len(mps) - 1):
            return misc_mps.ncon([left_envs[sites[0]], tensor_left,tensor_right,
                                  tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                                 [[5,4, -1,3], [5, -3, 1], [1,-4, 6], [4,3,2], [2,-2,6]])


    @staticmethod        
    def get_one_body_env(site,left_envs,right_envs, mps, conj_mps):
        
        if (site not in (0,len(mps) - 1)) and (site%2 == 1):
            return misc_mps.ncon([left_envs[site], mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site)),
                                  right_envs[site]],
                                 [[1,5,-1,4], [1,-2,3], [5,6,7], [3,7,4,6]])
        elif (site not in (0,len(mps) - 1)) and (site%2 == 0):
            return misc_mps.ncon([left_envs[site], mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site)),
                                  right_envs[site]],
                                 [[1,5,4,6], [1,-2,3], [5,6,7], [3,7,-1,4]])
        elif site == 0:
            return misc_mps.ncon([mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site)), right_envs[site]],
                                 [[1, -2, 2], [1, 3, 4], [2, 4, -1, 3]])
        elif site  == (len(mps) - 1):
            return misc_mps.ncon([left_envs[site], mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site))],
                                 [[1, 3, -1, 4], [1, -2, 2], [3, 4, 2]])



    @staticmethod
    def one_body_gradient(site,left_envs,right_envs, mps, conj_mps, one_body_generators):
        """
        gradient with respect to the one-body generators.
        U = tf.linalg.expm(g - herm(g)) with g an arbitrary real or complex
        The gradient takes this into account
        """
        env = OverlapMinimizer.get_one_body_env(site,left_envs,right_envs, mps, conj_mps)
        with tf.GradientTape() as tape:
            g = one_body_generators[site]
            tape.watch(g)
            if mps.dtype in (tf.complex128, tf.complex64):
                tmp = misc_mps.ncon([env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])
                res = (1.0 + 1j)/2 * tmp + (1.0 - 1j)/2 * tf.conj(tmp) #maximize tf.real(tmp) - tf.imag(tmp)
                #res = tf.real(tmp) - tf.imag(tmp)
            elif mps.dtype in (tf.float64, tf.float32):
                res = misc_mps.ncon([env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])
            else:
                raise TypeError('unsupported dtype {0}'.format(dtype))
                
                
        return tape.gradient(res, g)


    @staticmethod
    def two_body_gradient(sites, left_envs, right_envs, one_body_generators,  mps, conj_mps, two_body_generators):
        env = OverlapMinimizer.get_two_body_env(sites, left_envs, right_envs, one_body_generators, mps, conj_mps)
        ds = mps.d
        with tf.GradientTape() as tape:
            g = two_body_generators[sites]
            tape.watch(g)
            if mps.dtype in (tf.complex128, tf.complex64):            
                tmp = misc_mps.ncon([tf.reshape(env,(ds[sites[0]] * ds[sites[1]], ds[sites[0]] * ds[sites[1]])),
                                     tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])
                res = (1.0 + 1j)/2 * tmp + (1.0 - 1j)/2 * tf.conj(tmp)                
            elif mps.dtype in (tf.float64, tf.float32):
                res = misc_mps.ncon([tf.reshape(env,(ds[sites[0]] * ds[sites[1]], ds[sites[0]] * ds[sites[1]])),
                                     tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])
            else:
                raise TypeError('unsupported dtype {0}'.format(dtype))

                
        return tape.gradient(res, g)

    @staticmethod
    def one_body_gradient_batched(site,left_envs,right_envs, mps, samples, one_body_generators):
        env = OverlapMinimizer.get_one_body_env_batched(site,left_envs,right_envs, mps, samples)
        with tf.GradientTape() as tape:
            g = one_body_generators[site]
            tape.watch(g)
            if mps.dtype in (tf.complex128, tf.complex64):                        
                res = misc_mps.ncon([env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])
                #res = (1.0 + 1j)/2 * tmp + (1.0 - 1j)/2 * tf.conj(tmp)                                
            elif mps.dtype in (tf.float64, tf.float32):
                res = misc_mps.ncon([env, tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])                
            else:
                raise TypeError('unsupported dtype {0}'.format(dtype))
        return tape.gradient(res, g)


    @staticmethod
    def two_body_gradient_batched(sites, left_envs, right_envs, one_body_generators,  mps, samples, two_body_generators):
        env = OverlapMinimizer.get_two_body_env_batched(sites, left_envs, right_envs, one_body_generators, mps, samples)
        ds = mps.d
        with tf.GradientTape() as tape:
            g = two_body_generators[sites]
            tape.watch(g)
            if mps.dtype in (tf.complex128, tf.complex64):                        
                res = misc_mps.ncon([tf.reshape(env,(ds[sites[0]] * ds[sites[1]], ds[sites[0]] * ds[sites[1]])),
                                     tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])
                #res = (1.0 + 1j)/2 * tmp + (1.0 - 1j)/2 * tf.conj(tmp)                                
            elif mps.dtype in (tf.float64, tf.float32):
                res = misc_mps.ncon([tf.reshape(env,(ds[sites[0]] * ds[sites[1]], ds[sites[0]] * ds[sites[1]])),
                                     tf.linalg.expm(g - tf.conj(tf.transpose(g)))],[[1,2],[1,2]])
            else:
                raise TypeError('unsupported dtype {0}'.format(dtype))
                
        return tape.gradient(res, g)

    @staticmethod        
    def overlap(site,left_envs,right_envs, one_body_gates, mps, conj_mps):
        assert(site>0)
        assert(site<len(mps))
        if site%2 == 1:
            tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]], [[-1,1,-3],[-2, 1]])
            return misc_mps.ncon([left_envs[site], tensor, tf.conj(conj_mps.get_tensor(site)),
                                  right_envs[site]],
                                 [[1,5,2,4], [1,2,3], [5,6,7], [3,7,4,6]])
        elif site%2 == 0:
            tensor = misc_mps.ncon([mps.get_tensor(site), one_body_gates[site]], [[-1,1,-3],[-2, 1]])            
            return misc_mps.ncon([left_envs[site], tensor, tf.conj(conj_mps.get_tensor(site)),
                            right_envs[site]],
                           [[1,5,4,6], [1,2,3], [5,6,7], [3,7,2,4]])
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
        mat = misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1,-2]])        
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
        return misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1,-2]])        

    
    def absorb_one_body_gates(self):
        """
        absorb the one-body gates into a copy of self.mps 
        Args:
            no args
        Returns:
            FiniteMPSCentralGauge
        
        """
        tensors = [misc_mps.ncon([self.mps.get_tensor(site), self.one_body_gates[site]],[[-1,1,-3], [-2, 1]]) for site in range(len(self.mps))]
        mps = MPS.FiniteMPSCentralGauge(tensors)
        mps.position(0)
        mps.position(len(mps), normalize=True)
        return mps
    
    def absorb_two_body_gates(self,Dmax=None):
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
        for site in range(0,len(mps)-1,2):
            mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
        for site in range(1,len(mps)-2,2):
            mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
        mps.position(0)
        mps.position(len(mps), normalize=True)
        tw = mps.position(0, normalize=True, D=Dmax)
        return mps, tw
    
    def absorb_gates(self,Dmax=None):
        """
        absorb one- and two-body gates into a copy of self.mps 
        Args:
            Dmax (int):   the maximal bond dimension to be kept after absorbtion
        Returns:
            tuple containing 
            (FiniteMPSCentralGauge, float)
            an MPS and the truncated weight resulting from optional truncation
        
        """
        tensors = [misc_mps.ncon([self.mps.get_tensor(site), self.one_body_gates[site]],[[-1,1,-3], [-2, 1]]) for site in range(len(self.mps))]
        mps = MPS.FiniteMPSCentralGauge(tensors)
        mps.position(0)
        mps.position(len(mps), normalize=True)
        for site in range(0,len(mps)-1,2):
            mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
        for site in range(1,len(mps)-2,2):
            mps.apply_2site(self.two_body_gates[(site, site + 1)], site)
        mps.position(0)
        mps.position(len(mps), normalize=True)
        tw = mps.position(0, normalize=True, D=Dmax)
        return mps, tw

    def minimize_layerwise(self, ref_mps, num_sweeps, alpha = 1.0, verbose=0):
        """

        minimize the overlap by optimizing over the even  and odd two-body unitaries,
        alternating between even and odd layer.
        minimization runs from left to right and right to left, and changes `gates` one at at time.
        this function is deprecated; use `minimize_even` and `minimize_odd` instead.
        Args:
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int):  number of iterations
            alpha (float):    determines the mixing of the update
                                   the new gate is given by `1- alpha` * old_gate + `alphas` * update
            verbose (int):         verbosity flag; larger means more output
        """
        assert(alpha <= 1.0)
        assert(alpha >= 0)
        [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]        
        for it in range(num_sweeps):
            for site in range(len(self.mps) - 1):
                env = self.two_body_gates[(site, site+1)] * (1-alpha) + alpha * self.get_two_body_env((site, site + 1), selff.left_envs, self.right_envs, one_body_gates,
                                                                                                      self.mps, ref_mps)
                self.two_body_gates[(site, site+1)] = self.two_body_update_svd_numpy(env)
                self.add_unitary_left(site,self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)
                    stdout.write(
                        "\r iteration  %i/%i, overlap = %.16f" %
                        (it, num_sweeps, np.abs(np.real(overlap))))
                    stdout.flush()
                if verbose > 1:
                    print()

            for site in reversed(range(1, len(self.mps) - 1)):
                env = self.two_body_gates[(site, site+1)] * (1-alpha) + alpha * self.get_two_body_env((site, site + 1), self.left_envs, self.right_envs, one_body_gates,
                                                                                                      self.mps, ref_mps)
                self.two_body_gates[(site, site+1)] = self.two_body_update_svd_numpy(env)
                self.add_unitary_right(site + 1, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)
                    stdout.write(
                        "\r iteration  %i/%i, overlap = %.16f" %
                        (it, num_sweeps, np.abs(np.real(overlap))))
                    stdout.flush()
                if verbose > 1:
                    print()

    def minimize_sequentially(self, ref_mps, num_sweeps, alpha=1.0, verbose=0):
        """
        deprecated; use minimize_two_body instead
        minimize the overlap w9th `ref_mps` 
        by optimizing over the all two-body unitaries sequentially, running from left to right and right to left
        Args:
            ref_mps (FiniteMPSCentralGauge):               a reference mps 
            num_sweeps (int):  number of iterations
            alpha (float):    determines the mixing of the update
                                   the new gate is given by `1- alpha` * old_gate + `alphas` * update
            verbose (int):         verbosity flag; larger means more output
        """
        assert(alpha <= 1.0)
        assert(alpha >= 0)
        
        self.left_envs = {}
        self.right_envs = {}        
        [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
        for it in range(num_sweeps):
            for site in range(0,len(self.mps) - 1):
                env = self.two_body_gates[(site, site+1)] * (1-alpha) + alpha * self.get_two_body_env((site, site + 1), self.left_envs, self.right_envs, self.one_body_gates,
                                                                                                      self.mps, ref_mps)
                self.two_body_gates[(site, site+1)] = self.two_body_update_svd_numpy(env)
                self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)                    
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_sweeps, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()
                    
            self.right_envs = {}                    
            self.add_unitary_right(len(self.mps) - 1, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
            for site in reversed(range(len(self.mps) - 2)):
                env = self.two_body_gates[(site, site+1)] * (1-alpha) + alpha * self.get_two_body_env((site, site + 1), self.left_envs, self.right_envs, self.one_body_gates,
                                                                                                      self.mps, ref_mps)
                self.two_body_gates[(site, site+1)] = self.two_body_update_svd_numpy(env)
                self.add_unitary_right(site + 1, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)                    
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_sweeps, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()

                    
    def minimize_one_body(self, samples=None, ref_mps=None, num_sweeps=10,  sites=None, alpha_gates=0.0, alpha_samples=1.0, alpha_ref_mps = 1.0, verbose=0):
        """
        minimize the overlap by optimizing over the even two-body unitaries.
        minimization runs from left to right and changes even gates one at at time.
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
            alpha_ref_mos (float): the three `alpha_` arguments determine the mixing of the update
                                   the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
            verbose (int):         verbosity flag; larger means more output
        """
        assert(alpha_samples <= 1.0)
        assert(alpha_samples >= 0)
        assert(alpha_ref_mps <= 1.0)
        assert(alpha_ref_mps >= 0)
        assert(alpha_gates <= 1.0)
        assert(alpha_gates >= 0)

        self.left_envs_batched = {}
        self.right_envs_batched = {}
        self.left_envs = {}
        self.right_envs = {}
        if sites is None:
            sites = range(len(self.mps))
            
        #fixme: do right sweeps as well
        ds = self.mps.d
        for it in range(num_sweeps):
            if samples != None:
                [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
            if ref_mps !=None:
                [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]                        
            for site in range(len(self.mps)):
                if site in sites:
                    env = self.one_body_gates[site] * alpha_gates
                    if samples != None:                
                        env += (alpha_samples * self.get_one_body_env_batched(site, self.left_envs_batched, self.right_envs_batched,
                                                                              self.mps, samples))
                    if ref_mps !=None:
                        env += (alpha_ref_mps * self.get_one_body_env(site, self.left_envs, self.right_envs, self.mps, ref_mps))
                    self.one_body_gates[site] = self.one_body_update_svd_numpy(env)
                if site < (len(self.mps) - 1):                    
                    if samples != None:                                
                        self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
                    if ref_mps !=None:
                        self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
                if (verbose > 0) and (site not in (0, len(self.mps) -1)):
                    if samples != None:
                        overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.one_body_gates, self.mps, samples)
                    if ref_mps != None:
                        overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)
                    if (ref_mps != None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i, overlap_ref_mps %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(overlap_1), np.imag(overlap_1),np.real(overlap_2), np.imag(overlap_2)))
                        
                    elif (ref_mps == None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(overlap_1),np.imag(overlap_1)))
                    if (ref_mps != None) and (samples ==None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_ref_mps = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(overlap_2), np.imag(overlap_2)))
                    stdout.flush()
                if verbose > 1:
                    print()


    def minimize_two_body(self, samples=None, ref_mps=None, num_sweeps=10, sites=None, alpha_gates=0.0, alpha_samples=1.0, alpha_ref_mps = 1.0, verbose=0):
        """
        minimize the overlap by optimizing over the even two-body unitaries.
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
        assert(alpha_samples <= 1.0)
        assert(alpha_samples >= 0)
        assert(alpha_ref_mps <= 1.0)
        assert(alpha_ref_mps >= 0)
        assert(alpha_gates <= 1.0)
        assert(alpha_gates >= 0)

        self.left_envs_batched = {}
        self.right_envs_batched = {}
        self.left_envs = {}
        self.right_envs = {}
        
        #fixme: do right sweeps as well
        ds = self.mps.d
        if sites is None:
            sites = range(len(self.mps) - 1)

        for it in range(num_sweeps):
            if samples != None:
                [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
            if ref_mps !=None:
                [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]                        
            for site in range(0,len(self.mps) - 1):
                if site in sites:
                    env = self.two_body_gates[(site, site+1)] * alpha_gates                
                    if samples != None:                
                        env += (alpha_samples * self.get_two_body_env_batched((site, site + 1), self.left_envs_batched, self.right_envs_batched,
                                                                              self.one_body_gates, self.mps, samples))
                    if ref_mps !=None:
                        env += alpha_ref_mps * self.get_two_body_env((site, site + 1), self.left_envs, self.right_envs, self.one_body_gates,
                                                                     self.mps, ref_mps)
                    self.two_body_gates[(site, site+1)] = self.two_body_update_svd_numpy(env)
                if site < len(self.mps) - 1:                    
                    if samples != None:                                
                        self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
                    if ref_mps !=None:                
                        self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)

                if verbose > 0 and site > 0:
                    if samples != None:
                        overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.one_body_gates, self.mps, samples)
                    if ref_mps != None:
                        overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)
                    if (ref_mps != None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i, overlap_ref_mps %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(overlap_1), np.imag(overlap_1),np.real(overlap_2), np.imag(overlap_2)))
                        
                    elif (ref_mps == None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(overlap_1),np.imag(overlap_1)))
                    if (ref_mps != None) and (samples ==None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_ref_mps = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(overlap_2), np.imag(overlap_2)))
                    stdout.flush()

                if verbose > 1:
                    print()

                    
    def minimize_even(self, samples=None, ref_mps=None, num_sweeps=10,  alpha_gates=0.0, alpha_samples=1.0, alpha_ref_mps = 1.0, verbose=0):
        """
        minimize the overlap by optimizing over the even two-body unitaries.
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
        sites = range(0,len(self.mps), 2)
        self.minimize_two_body(samples=samples, ref_mps=ref_mps, num_sweeps=num_sweeps, sites=sites,
                               alpha_gates=alpha_gates, alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps,
                               verbose=verbose)
        # assert(alpha_samples <= 1.0)
        # assert(alpha_samples >= 0)
        # assert(alpha_ref_mps <= 1.0)
        # assert(alpha_ref_mps >= 0)
        # assert(alpha_gates <= 1.0)
        # assert(alpha_gates >= 0)

        # self.left_envs_batched = {}
        # self.right_envs_batched = {}
        # self.left_envs = {}
        # self.right_envs = {}
        
        # #fixme: do right sweeps as well
        # ds = self.mps.d
        # for it in range(num_sweeps):
        #     if samples != None:
        #         [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
        #     if ref_mps !=None:
        #         [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]                        
        #     for site in range(0,len(self.mps) - 1, 2):
        #         env = self.two_body_gates[(site, site+1)] * alpha_gates                
        #         if samples != None:                
        #             env += (alpha_samples * self.get_two_body_env_batched((site, site + 1), self.left_envs_batched, self.right_envs_batched,
        #                                                                   self.one_body_gates, self.mps, samples))
        #         if ref_mps !=None:
        #             env += alpha_ref_mps * self.get_two_body_env((site, site + 1), self.left_envs, self.right_envs, self.one_body_gates,
        #                                                          self.mps, ref_mps)
        #         self.two_body_gates[(site, site+1)] = self.two_body_update_svd_numpy(env)
        #         if samples != None:                                
        #             self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
        #         if ref_mps !=None:                
        #             self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
        #         if site + 1 < len(self.mps) - 1:
        #             if samples != None:                
        #                 self.add_unitary_batched_left(site + 1,self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
        #             if ref_mps !=None:                                    
        #                 self.add_unitary_left(site + 1,self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)                    
        #         if verbose > 0 and site > 0:
        #             if samples != None:
        #                 overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.one_body_gates, self.mps, samples)
        #             if ref_mps != None:
        #                 overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)
        #             if (ref_mps != None) and (samples !=None): 
        #                 stdout.write(
        #                     "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i, overlap_ref_mps %.16f + %.16f i" %
        #                     (it, num_sweeps, site, np.real(overlap_1), np.imag(overlap_1),np.real(overlap_2), np.imag(overlap_2)))
                        
        #             elif (ref_mps == None) and (samples !=None): 
        #                 stdout.write(
        #                     "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i" %
        #                     (it, num_sweeps, site, np.real(overlap_1),np.imag(overlap_1)))
        #             if (ref_mps != None) and (samples ==None): 
        #                 stdout.write(
        #                     "\r iteration  %i/%i at site %i , overlap_ref_mps = %.16f + %.16f i" %
        #                     (it, num_sweeps, site, np.real(overlap_2), np.imag(overlap_2)))
        #             stdout.flush()

        #         if verbose > 1:
        #             print()


    def minimize_odd(self, samples=None,  ref_mps=None, num_sweeps=10, alpha_gates=0.0, alpha_samples=1.0, alpha_ref_mps=1.0, verbose=0):
        """
        minimize the overlap by optimizing over the odd two-body unitaries.
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
        sites = range(1,len(self.mps), 2)
        self.minimize_two_body(samples=samples, ref_mps=ref_mps, num_sweeps=num_sweeps, sites=sites,
                               alpha_gates=alpha_gates, alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps,
                               verbose=verbose)
        
        # assert(alpha_samples <= 1.0)
        # assert(alpha_samples >= 0)
        # assert(alpha_ref_mps <= 1.0)
        # assert(alpha_ref_mps >= 0)
        # assert(alpha_gates <= 1.0)
        # assert(alpha_gates >= 0)

        # self.left_envs_batched = {}
        # self.right_envs_batched = {}
        # self.left_envs = {}
        # self.right_envs = {}
        # ds = self.mps.d        
        # for it in range(num_sweeps):
        #     if samples != None:
        #         [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
        #         self.add_unitary_batched_left(0, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
            
        #     if ref_mps != None:
        #         [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
        #         self.add_unitary_left(0, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)

        #     for site in range(1,len(self.mps) - 2, 2):
        #         env = self.two_body_gates[(site, site+1)] * alpha_gates
        #         if samples != None:
        #             env += (alpha_samples * self.get_two_body_env_batched((site, site + 1), self.left_envs_batched, self.right_envs_batched,
        #                                                                   self.one_body_gates, self.mps, samples))
        #         if ref_mps !=None:
        #             env += alpha_ref_mps * self.get_two_body_env((site, site + 1), self.left_envs, self.right_envs, self.one_body_gates,
        #                                                          self.mps, ref_mps)
                    
        #         self.two_body_gates[(site, site+1)] = self.two_body_update_svd_numpy(env)
        #         if samples != None:                
        #             self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
        #         if ref_mps != None:
        #             self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
        #         if site + 1 < len(self.mps) - 1:
        #             if samples != None:
        #                 self.add_unitary_batched_left(site + 1,self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
        #             if ref_mps != None:
        #                 self.add_unitary_left(site + 1,self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
        #         if verbose > 0 and site > 0:
        #             if samples != None:
        #                 overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.one_body_gates, self.mps, samples)
        #             if ref_mps != None:
        #                 overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)

        #             if (ref_mps != None) and (samples !=None): 
        #                 stdout.write(
        #                     "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i, overlap_ref_mps %.16f + %.16f i" %
        #                     (it, num_sweeps, site, np.real(overlap_1), np.imag(overlap_1),np.real(overlap_2), np.imag(overlap_2)))
                        
        #             elif (ref_mps == None) and (samples !=None): 
        #                 stdout.write(
        #                     "\r iteration  %i/%i at site %i , overlap_samples = %.16f + %.16f i" %
        #                     (it, num_sweeps, site, np.real(overlap_1),np.imag(overlap_1)))
        #             if (ref_mps != None) and (samples ==None): 
        #                 stdout.write(
        #                     "\r iteration  %i/%i at site %i , overlap_ref_mps = %.16f + %.16f i" %
        #                     (it, num_sweeps, site, np.real(overlap_2), np.imag(overlap_2)))
        #             stdout.flush()
        #         if verbose > 1:
        #             print()
                    

    def gradient_minimization_two_body(self, two_body_generators, one_body_generators=None, opt_type='sequential',
                                       samples=None, ref_mps=None, alpha=1E-5, num_sweeps=10,  sites=None, verbose=0):
        """
        minimize the overlap by optimizing over all two-body unitaries.
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
            self.one_body_gates = initialize_gates_from_generators(one_body_generators, ds)
            
        self.two_body_gates = initialize_gates_from_generators(two_body_generators, ds)
            

        for it in range(num_sweeps):
            if samples != None:
                [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
            if ref_mps !=None:
                [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]

            grads = []                
            for site in range(len(self.mps)):
                if site in sites:
                    grad = tf.zeros(shape=[ds[site],ds[site + 1], ds[site],ds[site + 1]], dtype=self.mps.dtype)
                    if samples != None:
                        grad += self.two_body_gradient_batched(site,self.left_envs_batched, self.right_envs_batched, self.mps, samples, one_body_generators) 
                    if ref_mps !=None:
                        grad += self.two_body_gradient(site,self.left_envs, self.right_envs, self.mps, ref_mps, one_body_generators)
                    if opt_type in ('sequential','seq'):
                        grad /= tf.linalg.norm(grad)
                        two_body_generators[(site, site + 1)] += (grad * alpha)  #we are trying to maximize, not minimize, hence the + operation
                        self.two_body_gates[(site, site + 1)] = get_gate_from_generator(two_body_generators[(site, site + 1)],(ds[site], ds[site + 1], ds[site], ds[site + 1]))
                    elif opt_type in ('simultaneous','sim'):
                        grads.append(grad)
                    else:
                        raise ValueError('unknown value {} for opt_type'.format(opt_type))
                if site < (len(self.mps) - 1):                    
                    if samples != None:                                
                        self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
                    if ref_mps !=None:
                        self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
                if (verbose > 0) and (site not in (0, len(self.mps) -1)):
                    if samples != None:
                        overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.one_body_gates, self.mps, samples)
                    if ref_mps != None:
                        overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)
                    if (ref_mps != None) and (samples !=None):
                        c1 = np.real(overlap_1) - np.imag(overlap_1)
                        c2 = np.real(overlap_2) - np.imag(overlap_2)
                        stdout.write(
                            "\r iteration  %i/%i at site %i , Re(overlap_samples) - Im(overlap_samples) = %.16f + %.16f i, Re(overlap_ref_mps) - Im(overlap_ref_mps) %.16f + %.16f i" %
                            (it, num_sweeps, site, np.rea(c1),np.imag(c1),np.real(c2), np.imag(c2)))
                        
                    elif (ref_mps == None) and (samples !=None):
                        c1 = np.real(overlap_1) - np.imag(overlap_1)
                        stdout.write(
                            "\r iteration  %i/%i at site %i , Re(overlap_samples) - Im(overlap_samples) = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(c1),np.imag(c1)))
                    if (ref_mps != None) and (samples ==None):
                        c2 = np.real(overlap_2) - np.imag(overlap_2)                        
                        stdout.write(
                            "\r iteration  %i/%i at site %i , Re(overlap_ref_mps) - Im(overlap_ref_mps) = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(c2), np.imag(c2)))
                    stdout.flush()
                if verbose > 1:
                    print()

            if opt_type not in ('sequential','s','seq'):
                Zs = []
                for g in grads:
                    Zs.append(np.linalg.norm(g.numpy()))
                Z = np.max(Zs)
                for site in sites:
                    two_body_generators[(site, site + 1)] += (grad * alpha)  #we are trying to maximize, not minimize, hence the + operation
                    self.two_body_gates[(site, site + 1)] = get_gate_from_generator(two_body_generators[(site, site + 1)],(ds[site], ds[site + 1], ds[site], ds[site + 1]))


    def gradient_minimization_one_body(self, one_body_generators, two_body_generators=None, opt_type='sequential',
                                       samples=None, ref_mps=None, alpha=1E-5, num_sweeps=10,  sites=None, verbose=0):
        """
        minimize the overlap by optimizing over the even two-body unitaries.
        minimization runs from left to right and changes even gates one at at time.
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
        if two_body_generators is not None:
            self.two_body_gates = initialize_gates_from_generators(two_body_generators, ds)
            
        self.one_body_gates = initialize_gates_from_generators(one_body_generators, ds)
        for it in range(num_sweeps):
            if samples != None:
                [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]
            if ref_mps !=None:
                [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates) for site in reversed(range(1,len(self.mps)))]

            grads = []                
            for site in range(len(self.mps)):
                if site in sites:
                    grad = tf.zeros(shape=[ds[site], ds[site]], dtype=self.mps.dtype)
                    if samples != None:
                        grad += self.one_body_gradient_batched(site,self.left_envs_batched, self.right_envs_batched, self.mps, samples, one_body_generators) 
                    if ref_mps !=None:
                        grad += self.one_body_gradient(site,self.left_envs, self.right_envs, self.mps, ref_mps, one_body_generators)
                    if opt_type in ('sequential','s','seq'):
                        grad /= tf.linalg.norm(grad)
                        one_body_generators[site] += (grad * alpha)  #we are trying to maximize, not minimize, hence the + operation
                        self.one_body_gates[site] = get_gate_from_generator(one_body_generators[site],(ds[site], ds[site]))
                    else:
                        grads.append(grad)
                    
                if site < (len(self.mps) - 1):                    
                    if samples != None:                                
                        self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.one_body_gates, self.two_body_gates)
                    if ref_mps !=None:
                        self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.one_body_gates, self.two_body_gates)
                if (verbose > 0) and (site not in (0, len(self.mps) -1)):
                    if samples != None:
                        overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.one_body_gates, self.mps, samples)
                    if ref_mps != None:
                        overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.one_body_gates, self.mps, ref_mps)
                    if (ref_mps != None) and (samples !=None):
                        c1 = np.real(overlap_1) - np.imag(overlap_1)
                        c2 = np.real(overlap_2) - np.imag(overlap_2)
                        stdout.write(
                            "\r iteration  %i/%i at site %i , Re(overlap_samples) - Im(overlap_samples) = %.16f + %.16f i, Re(overlap_ref_mps) - Im(overlap_ref_mps) %.16f + %.16f i" %
                            (it, num_sweeps, site, np.rea(c1),np.imag(c1),np.real(c2), np.imag(c2)))
                        
                    elif (ref_mps == None) and (samples !=None):
                        c1 = np.real(overlap_1) - np.imag(overlap_1)
                        stdout.write(
                            "\r iteration  %i/%i at site %i , Re(overlap_samples) - Im(overlap_samples) = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(c1),np.imag(c1)))
                    if (ref_mps != None) and (samples ==None):
                        c2 = np.real(overlap_2) - np.imag(overlap_2)                        
                        stdout.write(
                            "\r iteration  %i/%i at site %i , Re(overlap_ref_mps) - Im(overlap_ref_mps) = %.16f + %.16f i" %
                            (it, num_sweeps, site, np.real(c2), np.imag(c2)))
                    stdout.flush()
                if verbose > 1:
                    print()

            if opt_type in ('sequentual','seq'):
                pass
            elif opt_type in ('simultaneous','sim'):
                Zs = []
                for g in grads:
                    Zs.append(np.linalg.norm(g.numpy()))
                Z = np.max(Zs)
                for site in sites:
                    one_body_generators[site] += (grads[site] * alpha/Z)  #we are trying to maximize, not minimize, hence the + operation
                    self.one_body_gates[site] = get_gate_from_generator(one_body_generators[site],(ds[site], ds[site]))
            else:
                raise ValueError('unknown value {} for opt_type'.format(opt_type))
                
                    


