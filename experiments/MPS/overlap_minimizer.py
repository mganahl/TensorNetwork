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
import experiments.MPS.Lanczos as LZ
from sys import stdout
from experiments.MPS import misc_mps
import experiments.MPS.DMRG as DMRG
import functools as fct
from experiments.MPS.matrixproductstates import InfiniteMPSCentralGauge, FiniteMPSCentralGauge


class OverlapMinimizer:
    def __init__(self, mps, conj_mps, gates, name='overlap_minimizer'):
        self.name = name
        self.mps = mps
        self.conj_mps = conj_mps
        self.gates = gates
        self.right_envs = {}
        self.left_envs = {}        


    @staticmethod
    def add_unitary_right(site, right_envs, mps, conj_mps, gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site>0)
        assert(len(mps)%2==0)
        if site == (len(mps) - 1):
            right_envs[site - 1] = tn.ncon([mps.get_tensor(site), 
                                            tf.conj(conj_mps.get_tensor(site)), gates[(site-1,site)]],
                                           [[-1,2,1],[-2,3,1],[-4,3,-3,2]])        
        elif (site < len(mps) - 1) and (site % 2 == 0):
            right_envs[site - 1] = tn.ncon([right_envs[site], mps.get_tensor(site), 
                                            tf.conj(conj_mps.get_tensor(site)), gates[(site-1,site)]],
                                           [[1,3,2,5],[-1,2,1],[-2,4,3],[-4,4,-3,5]])
        elif (site < len(mps) - 1) and (site % 2 == 1):
            right_envs[site - 1] = tn.ncon([right_envs[site], mps.get_tensor(site), 
                                            tf.conj(conj_mps.get_tensor(site)), gates[(site-1,site)]],
                                           [[1,4,3,5],[-1,2,1],[-2,5,4],[-4,3,-3,2]])
    @staticmethod            
    def add_unitary_left(site, left_envs, mps, conj_mps, gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site<len(mps))
        assert(len(mps)%2==0)
            
        if site == 0:
            left_envs[site + 1] = tn.ncon([mps.get_tensor(site), 
                                             tf.conj(conj_mps.get_tensor(site)), gates[(site, site + 1)]],
                                            [[1,2,-1],[1,3, -2],[3, -4, 2, -3]])        
        elif (site > 0) and (site % 2 == 0):
            left_envs[site + 1] = tn.ncon([left_envs[site], mps.get_tensor(site), 
                                           tf.conj(conj_mps.get_tensor(site)), gates[(site, site + 1)]],
                                          [[1,4,3,5], [1,2,-1], [4,5,-2], [3,-4,2,-3]])                                          
        elif (site > 0) and (site % 2 == 1):
            left_envs[site + 1] = tn.ncon([left_envs[site], mps.get_tensor(site), 
                                             tf.conj(conj_mps.get_tensor(site)), gates[(site, site + 1)]],
                                             [[1,3,2,5],[1,2,-1],[3,4,-2],[4,-4,5,-3]])  
    @staticmethod            
    def get_env(sites,left_envs,right_envs, mps, conj_mps):
        assert((len(mps) % 2) == 0)
        if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
            return tn.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                       mps.get_tensor(sites[1]), right_envs[sites[1]], 
                       tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                      [[7,1,-1,2], [7,-3,6], [6,-4,8], [8,4,-2,3], [1,2,5], [5,3,4]])
        elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
            return tn.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]), right_envs[sites[1]], 
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[1, 7, 2, -3], [1, 2, 3], [3, 4, 5], [5, 8, 4, -4], [7, -1, 6], [6, -2, 8]])    
        elif sites[0] == 0:
            return tn.ncon([mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]), right_envs[sites[1]], 
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[1, -3, 2], [2, -4, 3], [3, 4, -2, 5], [1, -1, 6], [6, 5, 4]])  
        elif sites[1] == (len(mps) - 1):
            return tn.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]),
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[5,4, -1,3], [5, -3, 1], [1,-4, 6], [4,3,2], [2,-2,6]])
    @staticmethod        
    def overlap(site,left_envs,right_envs, mps, conj_mps):
        if site%2 == 1:
            return tn.ncon([left_envs[site], mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site)),
                            right_envs[site]],
                           [[1,5,2,4], [1,2,3], [5,6,7], [3,7,4,6]])
        elif site%2 == 0:
            return tn.ncon([left_envs[site], mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site)),
                            right_envs[site]],
                           [[1,5,4,6], [1,2,3], [5,6,7], [3,7,2,4]])
    @staticmethod
    def u_update_svd_numpy(wIn):
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
        return tf.convert_to_tensor(tf.reshape(tn.ncon([ut, vt], [[-1, 1], [1, -2]]), shape))
        

    def minimize(self,num_iterations, verbose=0):
        [self.add_unitary_right(site, self.right_envs, self.mps, self.conj_mps, self.gates) for site in reversed(range(1,len(self.mps)))]        
        for it in range(num_iterations):
            for site in range(len(self.mps) - 1):
                env = self.get_env((site,site+1), self.left_envs, self.right_envs, self.mps, self.conj_mps)
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site,self.left_envs, self.mps, self.conj_mps, self.gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)
                    stdout.write(
                        "\r iteration  %i/%i, overlap = %.16f" %
                        (it, num_iterations, np.abs(np.real(overlap))))
                    stdout.flush()
                if verbose > 1:
                    print()

            for site in reversed(range(1, len(self.mps) - 1)):
                env = self.get_env((site,site+1), self.left_envs, self.right_envs, self.mps, self.conj_mps)
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_right(site + 1, self.right_envs, self.mps, self.conj_mps, self.gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)
                    stdout.write(
                        "\r iteration  %i/%i, overlap = %.16f" %
                        (it, num_iterations, np.abs(np.real(overlap))))
                    stdout.flush()
                if verbose > 1:
                    print()

    def minimize_even(self,num_iterations, thresh=1.0, verbose=0):
        self.left_envs = {}
        self.right_envs = {}        

        for it in range(num_iterations):
            [self.add_unitary_right(site, self.right_envs, self.mps, self.conj_mps, self.gates) for site in reversed(range(1,len(self.mps)))]
            for site in range(0,len(self.mps) - 1, 2):
                env = self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site, self.left_envs, self.mps, self.conj_mps, self.gates)
                if site + 1 < len(self.mps) - 1:
                    self.add_unitary_left(site + 1,self.left_envs, self.mps, self.conj_mps, self.gates)
                if site > 0:
                    
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)
                    if np.abs(overlap)>thresh:
                        return 
                if verbose > 0 and site > 0:
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_iterations, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()


    def minimize_odd(self,num_iterations, thresh=1.0, verbose=0):
        self.left_envs = {}
        self.right_envs = {}        
        for it in range(num_iterations):
            [self.add_unitary_right(site, self.right_envs, self.mps, self.conj_mps, self.gates) for site in reversed(range(1,len(self.mps)))]
            self.add_unitary_left(0, self.left_envs, self.mps, self.conj_mps, self.gates)
            
            for site in range(1,len(self.mps) - 2, 2):
                env = self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site, self.left_envs, self.mps, self.conj_mps, self.gates)
                if site + 1 < len(self.mps) - 1:
                    self.add_unitary_left(site + 1,self.left_envs, self.mps, self.conj_mps, self.gates)

                if site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)                    
                    if np.abs(overlap)>thresh:
                        return 
                if verbose > 0 and site > 0:
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_iterations, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()

