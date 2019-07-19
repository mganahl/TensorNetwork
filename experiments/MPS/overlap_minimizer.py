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
from experiments.MPS import misc_mps
import experiments.MPS.DMRG as DMRG
import functools as fct
from experiments.MPS.matrixproductstates import InfiniteMPSCentralGauge, FiniteMPSCentralGauge


class OverlapMinimizer:
    """
    minimizes the overlap between `mps` and `conj_mps` using a double layer of two-body unitaries.
    For now `mps` and `conj_mps` have to have even length
    """
    def __init__(self, mps, conj_mps, gates, name='overlap_minimizer'):
        self.name = name
        self.mps = mps
        self.conj_mps = conj_mps
        self.gates = gates
        self.right_envs = {}
        self.left_envs = {}        

    @staticmethod
    def add_unitary_batched_right(site, right_envs, mps, samples, gates):
        """ 
        samples (tf.Tensor of shape (Nt, N)
        """
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        #fixme: calling tf.one_hot could be slowing things down. 
        assert(site>0)
        assert(len(mps)%2==0)
        if site == (len(mps) - 1):
            right_envs[site - 1] = tf.squeeze(misc_mps.ncon([mps.get_tensor(site), gates[(site-1,site)], tf.one_hot(samples[:,site], mps.d[site], dtype=mps.dtype)],
                                                      [[-2,1,-5],[-4, 2, -3, 1],[-1,2]]), 4)
        elif (site < len(mps) - 1) and (site % 2 == 0):
            tmp = misc_mps.ncon([right_envs[site], mps.get_tensor(site), gates[(site-1,site)]],
                          [[-1,1,2,3],[-2,2,1],[-4, -5, -3, 3]]) #shape (Nt, D, din, dout, d)
            Nt, D, din, dout, d = tmp.shape
            tmp = tf.reshape(tmp, (Nt, D * din * dout, d)) #(Nt, D * din * dout, d)
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2) #(Nt, d, 1)
            right_envs[site - 1] = tf.reshape(tf.matmul(tmp, tmp2),(Nt, D, din, dout))            
            
        elif (site < len(mps) - 1) and (site % 2 == 1):
            tmp = misc_mps.ncon([right_envs[site], mps.get_tensor(site), gates[(site-1,site)]],
                           [[-1, 1, 3, -5],[-2, 2, 1],[-4, 3, -3, 2]]) #has shape (Nt, Dl, din1, dout1, d)
            Nt, D, din, dout, d = tmp.shape
            tmp = tf.reshape(tmp, (Nt, D * din * dout, d))
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2) #has dim (Nt, d, 1)
            right_envs[site - 1] = tf.reshape(tf.matmul(tmp, tmp2),(Nt, D, din, dout))

    @staticmethod            
    def add_unitary_batched_left(site, left_envs, mps, samples, gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site<len(mps))
        assert(len(mps)%2==0)
            
        if site == 0:
            left_envs[site + 1] = tf.squeeze(misc_mps.ncon([mps.get_tensor(site), gates[(site, site + 1)], tf.one_hot(samples[:,site], mps.d[site], dtype=mps.dtype)],
                                                     [[-5, 1, -2],[2, -4, 1, -3], [-1, 2]]), 4)
        elif (site > 0) and (site % 2 == 0):
            tmp = misc_mps.ncon([left_envs[site], mps.get_tensor(site), gates[(site, site + 1)]],
                          [[-1, 1, 3, -5], [1, 2, -2], [3, -4, 2, -3]]) #has shape (Nt, D, di, do, d)
            Nt, D, di, do, d = tmp.shape
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2)#has dim (Nt, d, 1)
            tmp = tf.reshape(tmp,(Nt, D * di * do, d))
            left_envs[site + 1] = tf.reshape(tf.matmul(tmp, tmp2), (Nt, D, di, do))
            
        elif (site > 0) and (site % 2 == 1):
            tmp = misc_mps.ncon([left_envs[site], mps.get_tensor(site), gates[(site, site + 1)]],
                          [[-1, 1, 2, 3], [1, 2, -2], [-5, -4, 3, -3]]) #has shape (Nt, D, di, do, d)
            Nt, D, di, do, d = tmp.shape
            tmp2 = tf.expand_dims(tf.one_hot(samples[:, site], mps.d[site], dtype=mps.dtype),2)#has dim (Nt, d, 1)
            tmp = tf.reshape(tmp,(Nt, D * di * do, d))
            left_envs[site + 1] = tf.reshape(tf.matmul(tmp, tmp2), (Nt, D, di, do))
            
    @staticmethod            
    def get_env_batched(sites,left_envs,right_envs, mps, samples):
        assert((len(mps) % 2) == 0)
        if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
            raise NotImplementedError
            return misc_mps.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                       mps.get_tensor(sites[1]), right_envs[sites[1]], 
                       tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                      [[7,1,-1,2], [7,-3,6], [6,-4,8], [8,4,-2,3], [1,2,5], [5,3,4]])
        elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
            ds = mps.d
            lnet = tn.TensorNetwork()
            lenv = lnet.add_node(left_envs[sites[0]])
            tl = lnet.add_node(mps.get_tensor(sites[0]))
            e1 = lnet.connect(lenv[1], tl[0])
            e2 = lnet.connect(lenv[2], tl[1])
            lorder = [lenv[0], lenv[3], tl[2]]
            left = lnet.contract_between(lenv, tl)
            left.reorder_edges(lorder)
    
            rnet = tn.TensorNetwork()
            renv = rnet.add_node(right_envs[sites[1]])
            tr = rnet.add_node(mps.get_tensor(sites[1]))
            rorder = [renv[0], renv[3], tr[0]]
            e3 = rnet.connect(renv[1], tr[2])
            e4 = rnet.connect(renv[2], tr[1])
            right = rnet.contract_between(renv, tr)
            right.reorder_edges(rorder)
    
            bnet = btn.BatchTensorNetwork()
            batched_left = bnet.add_node(left.tensor)
            batched_right = bnet.add_node(right.tensor)
            order = [batched_left[0], batched_left[1], batched_right[1]]
            bnet.connect(batched_left[2], batched_right[2])
            out = bnet.batched_contract_between(batched_left, batched_right, batched_left[0], batched_right[0])
            out.reorder_edges(order)
            a = mm.batched_kronecker(tf.one_hot(samples[:, sites[0]], ds[sites[0]], dtype=mps.dtype),
                                     tf.one_hot(samples[:, sites[1]], ds[sites[1]], dtype=mps.dtype))
            Nt, dol ,dor = out.tensor.shape
            
            result = tf.reshape(mm.batched_kronecker(a, tf.reshape(out.tensor,(Nt, dol*dor))),
                                (Nt, ds[sites[0]], ds[sites[1]], dol, dor))
            
            return result
            
        elif sites[0] == 0:
            raise NotImplementedError            
            return misc_mps.ncon([mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]), right_envs[sites[1]], 
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[1, -3, 2], [2, -4, 3], [3, 4, -2, 5], [1, -1, 6], [6, 5, 4]])  
        elif sites[1] == (len(mps) - 1):
            raise NotImplementedError                        
            return misc_mps.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]),
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[5,4, -1,3], [5, -3, 1], [1,-4, 6], [4,3,2], [2,-2,6]])
            
    @staticmethod
    def add_unitary_right(site, right_envs, mps, conj_mps, gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site>0)
        assert(len(mps)%2==0)
        if site == (len(mps) - 1):
            right_envs[site - 1] = misc_mps.ncon([mps.get_tensor(site), 
                                            tf.conj(conj_mps.get_tensor(site)), gates[(site-1,site)]],
                                           [[-1,2,1],[-2,3,1],[-4,3,-3,2]])
        elif (site < len(mps) - 1) and (site % 2 == 0):
            right_envs[site - 1] = misc_mps.ncon([right_envs[site], mps.get_tensor(site), 
                                            tf.conj(conj_mps.get_tensor(site)), gates[(site-1,site)]],
                                           [[1,3,2,5],[-1,2,1],[-2,4,3],[-4,4,-3,5]])
        elif (site < len(mps) - 1) and (site % 2 == 1):
            right_envs[site - 1] = misc_mps.ncon([right_envs[site], mps.get_tensor(site), 
                                            tf.conj(conj_mps.get_tensor(site)), gates[(site-1,site)]],
                                           [[1,4,3,5],[-1,2,1],[-2,5,4],[-4,3,-3,2]])
    @staticmethod            
    def add_unitary_left(site, left_envs, mps, conj_mps, gates):
        #if site is even, add an odd gate
        #if site is odd, add an even gate
        assert(site<len(mps))
        assert(len(mps)%2==0)
            
        if site == 0:
            left_envs[site + 1] = misc_mps.ncon([mps.get_tensor(site), 
                                             tf.conj(conj_mps.get_tensor(site)), gates[(site, site + 1)]],
                                            [[1,2,-1],[1,3, -2],[3, -4, 2, -3]])        
        elif (site > 0) and (site % 2 == 0):
            left_envs[site + 1] = misc_mps.ncon([left_envs[site], mps.get_tensor(site), 
                                           tf.conj(conj_mps.get_tensor(site)), gates[(site, site + 1)]],
                                          [[1,4,3,5], [1,2,-1], [4,5,-2], [3,-4,2,-3]])                                          
        elif (site > 0) and (site % 2 == 1):
            left_envs[site + 1] = misc_mps.ncon([left_envs[site], mps.get_tensor(site), 
                                             tf.conj(conj_mps.get_tensor(site)), gates[(site, site + 1)]],
                                             [[1,3,2,5],[1,2,-1],[3,4,-2],[4,-4,5,-3]])  
    @staticmethod            
    def get_env(sites,left_envs,right_envs, mps, conj_mps):
        """
        """
        assert((len(mps) % 2) == 0)
        if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
            return misc_mps.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                       mps.get_tensor(sites[1]), right_envs[sites[1]], 
                       tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                      [[7,1,-1,2], [7,-3,6], [6,-4,8], [8,4,-2,3], [1,2,5], [5,3,4]])
        elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
            return misc_mps.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]), right_envs[sites[1]], 
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[1, 7, 2, -3], [1, 2, 3], [3, 4, 5], [5, 8, 4, -4], [7, -1, 6], [6, -2, 8]])    
        elif sites[0] == 0:
            return misc_mps.ncon([mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]), right_envs[sites[1]], 
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[1, -3, 2], [2, -4, 3], [3, 4, -2, 5], [1, -1, 6], [6, 5, 4]])  
        elif sites[1] == (len(mps) - 1):
            return misc_mps.ncon([left_envs[sites[0]], mps.get_tensor(sites[0]), 
                            mps.get_tensor(sites[1]),
                            tf.conj(conj_mps.get_tensor(sites[0])), tf.conj(conj_mps.get_tensor(sites[1]))],
                           [[5,4, -1,3], [5, -3, 1], [1,-4, 6], [4,3,2], [2,-2,6]])


    @staticmethod            
    def get_single_env(sites,left_env,right_env, mps_tensor_l, mps_tensor_r, conj_mps_tensor_l,  conj_mps_tensor_r):
        """
        """
        assert((len(mps) % 2) == 0)
        if (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 0):
            return misc_mps.ncon([left_env, mps_tensor_l, 
                            mps_tensor_r, right_env,
                            tf.conj(conj_mps_tensor_l), tf.conj(conj_mps_tensor_r)],
                           [[7,1,-1,2], [7,-3,6], [6,-4,8], [8,4,-2,3], [1,2,5], [5,3,4]])
        
        elif (sites[0] > 0) and (sites[1] < len(mps) - 1) and (sites[0] % 2 == 1):
            return misc_mps.ncon([left_env, mps_tensor_l, 
                            mps_tensor_r, right_env, 
                            tf.conj(conj_mps_tensor_l), tf.conj(conj_mps_tensor_r)],
                            [[1, 7, 2, -3], [1, 2, 3], [3, 4, 5], [5, 8, 4, -4], [7, -1, 6], [6, -2, 8]])    
        elif sites[0] == 0:
            return misc_mps.ncon([mps_tensor_l,
                            mps_tensor_r, right_env,
                            tf.conj(conj_mps_tensor_l), tf.conj(conj_mps_tensor_r)],
                           [[1, -3, 2], [2, -4, 3], [3, 4, -2, 5], [1, -1, 6], [6, 5, 4]])  
        elif sites[1] == (len(mps) - 1):
            return misc_mps.ncon([left_env, mps_tensor_l,
                            mps_tensor_r,
                            tf.conj(conj_mps_tensor_l), tf.conj(conj_mps_tensor_r)],
                           [[5,4, -1,3], [5, -3, 1], [1,-4, 6], [4,3,2], [2,-2,6]])
        
    @staticmethod        
    def overlap(site,left_envs,right_envs, mps, conj_mps):
        if site%2 == 1:
            return misc_mps.ncon([left_envs[site], mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site)),
                            right_envs[site]],
                           [[1,5,2,4], [1,2,3], [5,6,7], [3,7,4,6]])
        elif site%2 == 0:
            return misc_mps.ncon([left_envs[site], mps.get_tensor(site), tf.conj(conj_mps.get_tensor(site)),
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
        #mat = misc_mps.ncon([ut, vt], [[-1, 1], [1, -2]])
        mat = misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1,-2]])        
        return tf.reshape(mat, shape)
    #@staticmethod
    def gradient_update_unitary(sites,left_env,right_env, mps_tensor, conj_mps_tensor, gate):
        left_env = left_envs[sites[0]]
        right_env = right_envs[sites[1]]
        mps_tensor_l = self.mps.get_tensor(sites[0])
        mps_tensor_r = self.mps.get_tensor(sites[1])
        conj_mps_tensor_l = self.conj_mps.get_tensor(sites[0])
        conj_mps_tensor_r = self.conj_mps.get_tensor(sites[1])        
        
        env = self.get_single_env(sites,left_env,right_env, mps_tensor_l, mps_tensor_r, conj_mps_tensor_l, conj_mps_tensor_r)
        print(env)
        
    def absorb_gates(self):
        for site in range(0,len(self.mps)-1,2):
            self.mps.apply_2site(self.gates[(site, site + 1)], site)
        for site in range(1,len(self.mps)-2,2):
            self.mps.apply_2site(self.gates[(site, site + 1)], site)
        self.mps.position(0)
        self.mps.position(len(self.mps))
        self.mps.position(0)
        
    def minimize_layerwise(self,num_iterations, alpha = 1.0, verbose=0):
        """
        minimize the overlap by optimizing over the even  and odd two-body unitaries,
        alternating between even and odd layer.
        minimization runs from left to right and right to left, and changes `gates` one at at time.
        Args:
            num_iterations (int):  number of iterations
            thresh (float):        if overlap is larger than `thresh`, optimization stops
            verbose (int):         verbosity flag; larger means more output
        """
        assert(alpha <= 1.0)
        assert(alpha >= 0)
        [self.add_unitary_right(site, self.right_envs, self.mps, self.conj_mps, self.gates) for site in reversed(range(1,len(self.mps)))]        
        for it in range(num_iterations):
            for site in range(len(self.mps) - 1):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)                
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
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)                                
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

    def minimize_even(self,num_iterations, alpha=1.0, thresh=1.0, verbose=0):
        """
        minimize the overlap by optimizing over the even two-body unitaries.
        minimization runs from left to right and changes `gates` one at at time.
        Args:
            num_iterations (int):  number of iterations
            thresh (float):        if overlap is larger than `thresh`, optimization stops
            verbose (int):         verbosity flag; larger means more output
        """
        assert(alpha <= 1.0)
        assert(alpha >= 0)
        
        self.left_envs = {}
        self.right_envs = {}        
        #fixme: do right sweeps as well
        for it in range(num_iterations):
            [self.add_unitary_right(site, self.right_envs, self.mps, self.conj_mps, self.gates) for site in reversed(range(1,len(self.mps)))]            
            for site in range(0,len(self.mps) - 1, 2):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site, self.left_envs, self.mps, self.conj_mps, self.gates)
                if site + 1 < len(self.mps) - 1:
                    self.add_unitary_left(site + 1,self.left_envs, self.mps, self.conj_mps, self.gates)
                # if site > 0:
                #     overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)
                #     if np.abs(overlap)>thresh:
                #         return 
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)                                        
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_iterations, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()

    def minimize_sequentially(self,num_iterations, alpha=1.0, thresh=1.0, verbose=0):
        """
        minimize the overlap by optimizing over the even two-body unitaries.
        minimization runs from left to right and changes `gates` one at at time.
        Args:
            num_iterations (int):  number of iterations
            thresh (float):        if overlap is larger than `thresh`, optimization stops
            verbose (int):         verbosity flag; larger means more output
        """
        assert(alpha <= 1.0)
        assert(alpha >= 0)
        
        self.left_envs = {}
        self.right_envs = {}        
        [self.add_unitary_right(site, self.right_envs, self.mps, self.conj_mps, self.gates) for site in reversed(range(1,len(self.mps)))]
        for it in range(num_iterations):
            for site in range(0,len(self.mps) - 1):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)                                
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site, self.left_envs, self.mps, self.conj_mps, self.gates)
                # if site > 0:
                #     overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)
                #     if np.abs(overlap)>thresh:
                #         return 
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)                    
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_iterations, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()
                    
            self.right_envs = {}                    
            self.add_unitary_right(len(self.mps) - 1, self.right_envs, self.mps, self.conj_mps, self.gates)
            for site in reversed(range(len(self.mps) - 2)):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)                
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_right(site + 1, self.right_envs, self.mps, self.conj_mps, self.gates)
                # if site > 0:
                #     overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)
                #     if np.abs(overlap)>thresh:
                #         return 
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)                    
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_iterations, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()
                    

    def minimize_odd(self,num_iterations, alpha=1.0, thresh=1.0, verbose=0):
        """
        minimize the overlap by optimizing over the odd two-body unitaries. 
        minimization runs from left to right and changes `gates` one at at time.
        Args:
            num_iterations (int):  number of iterations
            thresh (float):        if overlap is larger than `thresh`, optimization stops
            verbose (int):         verbosity flag; larger means more output
        """
        assert(alpha <= 1.0)
        assert(alpha >= 0)
        self.left_envs = {}
        self.right_envs = {}
        
        for it in range(num_iterations):
            [self.add_unitary_right(site, self.right_envs, self.mps, self.conj_mps, self.gates) for site in reversed(range(1,len(self.mps)))]
            self.add_unitary_left(0, self.left_envs, self.mps, self.conj_mps, self.gates)
            
            for site in range(1,len(self.mps) - 2, 2):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, self.conj_mps)
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site, self.left_envs, self.mps, self.conj_mps, self.gates)
                if site + 1 < len(self.mps) - 1:
                    self.add_unitary_left(site + 1,self.left_envs, self.mps, self.conj_mps, self.gates)

                # if site > 0:
                #     overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)                    
                #     if np.abs(overlap)>thresh:
                #         return 
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, self.conj_mps)                                        
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_iterations, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()

