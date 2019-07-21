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
    minimizes the overlap between `mps` and a given reference mps, using a double layer of two-body unitaries.
    For now `mps` has to have even length
    """
    def __init__(self, mps, gates, name='overlap_minimizer'):
        self.name = name
        self.mps = mps
        self.gates = gates
        self.right_envs = {}
        self.left_envs = {}        
        self.right_envs_batched = {}
        self.left_envs_batched = {}        

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
            order = [lenv_node[0], lenv_node[2], renv_node[2], left_tensor_node[1], right_tensor_node[1]]
            e1 = bnet.connect(lenv_node[3], left_sample_node[1])
            e2 = bnet.connect(lenv_node[1], left_tensor_node[0])
            e3 = bnet.connect(renv_node[3], right_sample_node[1])
            e4 = bnet.connect(renv_node[1], right_tensor_node[2])
            e5 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            
            bnet.batched_contract_between(lenv_node, left_sample_node, lenv_node[0], left_sample_node[0])
            bnet.batched_contract_between(renv_node, right_sample_node, renv_node[0], right_sample_node[0])
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
            order = [lenv_node[0], left_sample_node[1], right_sample_node[1], lenv_node[3],  renv_node[3]]
            
            e1 = bnet.connect(lenv_node[1], left_tensor_node[0])
            e2 = bnet.connect(lenv_node[2], left_tensor_node[1])
            
            e3 = bnet.connect(renv_node[1], right_tensor_node[2])
            e4 = bnet.connect(renv_node[2], right_tensor_node[1])
            
            e5 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            e6 = bnet.connect(lenv_node[4], left_sample_node[2])
            e7 = bnet.connect(renv_node[4], right_sample_node[2])            

            ltmp = bnet.contract_between(lenv_node, left_tensor_node)
            rtmp = bnet.contract_between(renv_node, right_tensor_node)
            
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
            order = [renv_node[0], left_sample_node[1], renv_node[2], left_tensor_node[1], right_tensor_node[1]]
            e1 = bnet.connect(left_tensor_node[0], left_sample_node[2])
            e2 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            e3 = bnet.connect(renv_node[1], right_tensor_node[2])
            e4 = bnet.connect(renv_node[3], right_sample_node[1])
            e5 = bnet.connect(renv_node[4], right_sample_node[2])

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
            order = [lenv_node[0], lenv_node[2], right_sample_node[1], left_tensor_node[1], right_tensor_node[1]]
            e1 = bnet.connect(right_tensor_node[2], right_sample_node[2])
            e2 = bnet.connect(left_tensor_node[2], right_tensor_node[0])
            e3 = bnet.connect(lenv_node[1], left_tensor_node[0])
            e4 = bnet.connect(lenv_node[3], left_sample_node[1])
            e5 = bnet.connect(lenv_node[4], left_sample_node[2])

            bnet.batched_contract_between(lenv_node, left_sample_node, lenv_node[0], left_sample_node[0])
            bnet.contract(e3)
            tmp = bnet.contract(e2)
            out = bnet.batched_contract_between(tmp, right_sample_node, lenv_node[0], right_sample_node[0])
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
    def overlap_batched(site,left_envs,right_envs, mps, samples):
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
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])

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
            return tf.math.reduce_sum(out.tensor, axis=0)/np.sqrt(samples.shape[0])
            
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
        mat = misc_mps.ncon([np.conj(ut), np.conj(vt)], [[-1, 1], [1,-2]])        
        return tf.reshape(mat, shape)

    #@staticmethod
    def gradient_update_unitary(sites,left_env,right_env, mps_tensor, conj_mps_tensor, gate):
        raise NotImplementedError
        left_env = left_envs[sites[0]]
        right_env = right_envs[sites[1]]
        mps_tensor_l = self.mps.get_tensor(sites[0])
        mps_tensor_r = self.mps.get_tensor(sites[1])
        conj_mps_tensor_l = self.conj_mps.get_tensor(sites[0])
        conj_mps_tensor_r = self.conj_mps.get_tensor(sites[1])        
        
        env = self.get_single_env(sites,left_env,right_env, mps_tensor_l, mps_tensor_r, conj_mps_tensor_l, conj_mps_tensor_r)
        
    def absorb_gates(self,Dmax=None):
        """
        absorb the gates into a copy of self.mps 
        Args:
            Dmax (int):   the maximal bond dimension to be kept after absorbtion
        Returns:
            FiniteMPSCentralGauge
        
        """
        mps = copy.deepcopy(self.mps)
        for site in range(0,len(mps)-1,2):
            mps.apply_2site(self.gates[(site, site + 1)], site)
        for site in range(1,len(mps)-2,2):
            mps.apply_2site(self.gates[(site, site + 1)], site)
        mps.position(0)
        mps.position(len(mps), normalize=True)
        mps.position(0, normalize=True, D=Dmax)
        return mps

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
        [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.gates) for site in reversed(range(1,len(self.mps)))]        
        for it in range(num_sweeps):
            for site in range(len(self.mps) - 1):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, ref_mps)                
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site,self.left_envs, self.mps, ref_mps, self.gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, ref_mps)
                    stdout.write(
                        "\r iteration  %i/%i, overlap = %.16f" %
                        (it, num_sweeps, np.abs(np.real(overlap))))
                    stdout.flush()
                if verbose > 1:
                    print()

            for site in reversed(range(1, len(self.mps) - 1)):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, ref_mps)                                
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_right(site + 1, self.right_envs, self.mps, ref_mps, self.gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, ref_mps)
                    stdout.write(
                        "\r iteration  %i/%i, overlap = %.16f" %
                        (it, num_sweeps, np.abs(np.real(overlap))))
                    stdout.flush()
                if verbose > 1:
                    print()

    def minimize_sequentially(self, ref_mps, num_sweeps, alpha=1.0, verbose=0):
        """
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
        [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.gates) for site in reversed(range(1,len(self.mps)))]
        for it in range(num_sweeps):
            for site in range(0,len(self.mps) - 1):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, ref_mps)                                
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, ref_mps)                    
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_sweeps, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()
                    
            self.right_envs = {}                    
            self.add_unitary_right(len(self.mps) - 1, self.right_envs, self.mps, ref_mps, self.gates)
            for site in reversed(range(len(self.mps) - 2)):
                env = self.gates[(site, site+1)] * (1-alpha) + alpha * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, ref_mps)                
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                self.add_unitary_right(site + 1, self.right_envs, self.mps, ref_mps, self.gates)
                if verbose > 0 and site > 0:
                    overlap = self.overlap(site, self.left_envs, self.right_envs, self.mps, ref_mps)                    
                    stdout.write(
                        "\r iteration  %i/%i at site %i , overlap = %.16f" %
                        (it, num_sweeps, site, np.abs(np.real(overlap))))                        
                    stdout.flush()
                if verbose > 1:
                    print()
                    
    def minimize_even(self, samples=None, ref_mps=None, num_sweeps=10,  alpha_gates=0.0, alpha_samples=1.0, alpha_ref_mps = 1.0, verbose=0):
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
        for it in range(num_sweeps):
            if samples != None:
                [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.gates) for site in reversed(range(1,len(self.mps)))]
            if ref_mps !=None:
                [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.gates) for site in reversed(range(1,len(self.mps)))]                        
            for site in range(0,len(self.mps) - 1, 2):
                if (site >= 0) and (site <=(len(self.mps)-2)):
                    env = self.gates[(site, site+1)] * alpha_gates                
                    if samples != None:                
                        env += (alpha_samples * self.get_env_batched((site, site + 1), self.left_envs_batched, self.right_envs_batched, self.mps, samples))
                    if ref_mps !=None:
                        env += alpha_ref_mps * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, ref_mps)
                    self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                if samples != None:                                
                    self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.gates)
                if ref_mps !=None:                
                    self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.gates)
                if site + 1 < len(self.mps) - 1:
                    if samples != None:                
                        self.add_unitary_batched_left(site + 1,self.left_envs_batched, self.mps, samples, self.gates)
                    if ref_mps !=None:                                    
                        self.add_unitary_left(site + 1,self.left_envs, self.mps, ref_mps, self.gates)                    
                if verbose > 0 and site > 0:
                    if samples != None:
                        overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.mps, samples)
                    if ref_mps != None:
                        overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.mps, ref_mps)
                    if (ref_mps != None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f, overlap_ref_mps %.16f" %
                            (it, num_sweeps, site, np.abs(np.real(overlap_1)),np.abs(np.real(overlap_2))))
                    elif (ref_mps == None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f" %
                            (it, num_sweeps, site, np.abs(np.real(overlap_1))))
                    if (ref_mps != None) and (samples ==None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_ref_mps = %.16f" %
                            (it, num_sweeps, site, np.abs(np.real(overlap_2))))
                    stdout.flush()
                if verbose > 1:
                    print()


    def minimize_odd(self, samples,  ref_mps=None, num_sweeps=10, alpha_gates=0.0, alpha_samples=1.0, alpha_ref_mps=1.0, verbose=0):
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
        ds = self.mps.d        
        for it in range(num_sweeps):
            if samples != None:
                [self.add_unitary_batched_right(site, self.right_envs_batched, self.mps, samples, self.gates) for site in reversed(range(1,len(self.mps)))]
                self.add_unitary_batched_left(0, self.left_envs_batched, self.mps, samples, self.gates)
            
            if ref_mps != None:
                [self.add_unitary_right(site, self.right_envs, self.mps, ref_mps, self.gates) for site in reversed(range(1,len(self.mps)))]
                self.add_unitary_left(0, self.left_envs, self.mps, ref_mps, self.gates)

            for site in range(1,len(self.mps) - 2, 2):
                env = self.gates[(site, site+1)] * alpha_gates
                if samples != None:
                    env += (alpha_samples * self.get_env_batched((site, site + 1), self.left_envs_batched, self.right_envs_batched, self.mps, samples))
                if ref_mps !=None:
                    env += alpha_ref_mps * self.get_env((site, site + 1), self.left_envs, self.right_envs, self.mps, ref_mps)
                    
                self.gates[(site, site+1)] = self.u_update_svd_numpy(env)
                if samples != None:                
                    self.add_unitary_batched_left(site, self.left_envs_batched, self.mps, samples, self.gates)
                if ref_mps != None:
                    self.add_unitary_left(site, self.left_envs, self.mps, ref_mps, self.gates)
                if site + 1 < len(self.mps) - 1:
                    if samples != None:
                        self.add_unitary_batched_left(site + 1,self.left_envs_batched, self.mps, samples, self.gates)
                    if ref_mps != None:
                        self.add_unitary_left(site + 1,self.left_envs, self.mps, ref_mps, self.gates)
                if verbose > 0 and site > 0:
                    if samples != None:
                        overlap_1 = self.overlap_batched(site, self.left_envs_batched, self.right_envs_batched, self.mps, samples)
                    if ref_mps != None:
                        overlap_2 = self.overlap(site, self.left_envs, self.right_envs, self.mps, ref_mps)
                    if (ref_mps != None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f, overlap_ref_mps %.16f" %
                            (it, num_sweeps, site, np.abs(np.real(overlap_1)),np.abs(np.real(overlap_2))))
                    elif (ref_mps == None) and (samples !=None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_samples = %.16f" %
                            (it, num_sweeps, site, np.abs(np.real(overlap_1))))
                    if (ref_mps != None) and (samples ==None): 
                        stdout.write(
                            "\r iteration  %i/%i at site %i , overlap_ref_mps = %.16f" %
                            (it, num_sweeps, site, np.abs(np.real(overlap_2))))
                    stdout.flush()
                if verbose > 1:
                    print()
                    
