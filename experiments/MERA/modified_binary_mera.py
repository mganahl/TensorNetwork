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


""" 
modified binary MERA optimization
parts of the following code are based on code written by Glen Evenbly (c) for www.tensors.net, (v1.1) 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../../')
NUM_THREADS = 4
import os
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
import tensorflow as tf
import copy
import numpy as np
import time
import pickle
import ncon
import misc_mera
import modified_binary_mera_lib as mbml
from sys import stdout


config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1
tf.enable_eager_execution(config=config)
tf.enable_v2_behavior()



def run_mod_binary_mera_optimization_TFI(chis=[8, 12, 16],
                                         niters=[200, 300, 1000],
                                         embeddings=None,
                                         dtype=tf.float64,  
                                         verbose=1,
                                         refsym=True,
                                         nsteps_steady_state=4,opt_u_after=9,
                                         noise=0.0):
    """
    modified binary mera optimization
    Parameters:
    -------------------
    chis:                 list of int 
                          bond dimension of successive MERA simulations 
    niters:               list of int 
                          number of optimization steps of successive MERA optimizations 
    embeddings:           list of str or None
                          type of embeddings scheme used to embed mera into the next larger bond dimension 
                          entries can be: 'p' or 'pad' for padding with zeros without, if possible, adding new layers 
                                          'a' or 'add' for adding new layer with increased  bond dimension
    dtype:                tensorflow dtype 
    verbose:              int 
                          verbosity flag 
    refsym:               bool 
                          if `True`, impose reflection symmetry 
    nsteps_steady_state:  int 
                          number power iteration of steps used to obtain the steady state reduced 
                          density matrix
    noise:                float 
                          noise amplitude for initializing new layers and/or padding existing ones
    Returns: 
    --------------------
    (energies, walltimes, wC, vC, uC)
    energies:   list of tf.Tensor of shape () 
                energies at iterations steps
    walltimes:  list of float 
                walltimes per iteration step 
    wC, vC, uC: list of tf.Tensor
                isometries (wC, vC) and disentanglers (uC)
    """
    
    if not embeddings:
        embeddings = ['p']*len(chi)
    wC, vC, uC, rhoAB_0, rhoBA_0 = mbml.initialize_mod_binary_MERA(phys_dim=4, chi=chis[0],dtype=dtype)
    hamAB_0, hamBA_0 = mbml.initialize_TFI_hams(dtype=dtype)
    energies = []
    walltimes = []
    init = True    
    if not ([len(chis), len(niters), len(embeddings)] == [len(chis)] * 3):
        raise ValueError('`chis`, `niter` and `embeddings` need to be of same lengths')
    for chi, niter, which in zip(chis, niters, embeddings):
        if not init:
            if which in ('a','add') :            
                wC, vC, uC = mbml.unlock_layer(wC, vC, uC, noise=noise)
                wC, vC, uC = mbml.pad_mera_tensors(chi, wC, vC, uC, noise=noise)
            elif which in ('p','pad') :
                wC, vC, uC = mbml.pad_mera_tensors(chi, wC, vC, uC, noise=noise)

            
        rhoAB_0, rhoBA_0 = misc_mera.pad_tensor(rhoAB_0, [chi,chi,chi,chi]), misc_mera.pad_tensor(rhoBA_0, [chi,chi,chi,chi])
        wC, vC, uC, rhoAB_0, rhoBA_0, times, es = mbml.optimize_mod_binary_mera(hamAB_0=hamAB_0, hamBA_0=hamBA_0, rhoAB_0=rhoAB_0, 
                                                                           rhoBA_0=rhoBA_0, wC=wC,
                                                                           vC=vC, uC=uC, verbose = verbose,
                                                                           numiter=niter,opt_u=True, opt_vw=True, refsym=refsym,
                                                                           nsteps_steady_state=nsteps_steady_state, opt_u_after=opt_u_after)
        energies.extend(es)
        walltimes.extend(times)
        init = False
    return energies, walltimes, wC, vC, uC
    

def benchmark_ascending_operator(hab, hba, w, v, u, num_layers):
    t1 = time.time()
    for t in range(num_layers):
        hab, hba = mbml.ascending_super_operator(
            hab, hba, w, v, u, refsym=False)
    return time.time() - t1

def benchmark_descending_operator(rhoab, rhoba, w, v, u, num_layers):
    t1 = time.time()
    for p in range(num_layers):
        rhoab, rhoba = mbml.descending_super_operator(rhoab,rhoba,w,v,u, refsym=False)
    return time.time() - t1



def run_ascending_operator_benchmark(filename,
                                     chis=[4, 8, 16, 32],
                                     num_layers=8,
                                     dtype=tf.float64,
                                     device=None):
    walltimes = {'warmup': {}, 'profile': {}}
    for chi in chis:
        print('running ascending-operator benchmark for chi = {0} benchmark'.
              format(chi))
        with tf.device(device):
            w = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)
            v = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)                
            u = tf.random_uniform(shape=[chi, chi, chi, chi], dtype=dtype)

            hab = tf.random_uniform(shape = [chi,chi,chi,chi], dtype=dtype)
            hba = tf.random_uniform(shape = [chi,chi,chi,chi], dtype=dtype)
            walltimes['warmup'][chi] = benchmark_ascending_operator(
                hab, hba, w, v, u, num_layers)
            print('     warmup took {0} s'.format(walltimes['warmup'][chi]))
            walltimes['profile'][chi] = benchmark_ascending_operator(
                hab, hba, w, v, u, num_layers)                
            print('     profile took {0} s'.format(walltimes['profile'][chi]))

    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(walltimes, f)
    return walltimes

def run_descending_operator_benchmark(filename,
                                      chis=[4, 8, 16, 32],
                                      num_layers=8,
                                      dtype=tf.float64,
                                      device=None):
    walltimes = {'warmup': {}, 'profile': {}}
    for chi in chis:
        print('running descending-operator benchmark for chi = {0} benchmark'.
              format(chi))
        with tf.device(device):
            w = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)
            v = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)            
            u = tf.random_uniform(shape=[chi, chi, chi, chi], dtype=dtype)

            rhoAB = tf.random_uniform(shape = [chi,chi,chi,chi], dtype=dtype)
            rhoBA = tf.random_uniform(shape = [chi,chi,chi,chi], dtype=dtype)

            walltimes['warmup'][chi] = benchmark_descending_operator(
                rhoAB, rhoBA, w, v, u, num_layers = num_layers)
            print('     warmup took {0} s'.format(walltimes['warmup'][chi]))
            walltimes['profile'][chi] = benchmark_descending_operator(
                rhoAB, rhoBA, w, v, u, num_layers = num_layers)
            print('     profile took {0} s'.format(walltimes['profile'][chi]))

    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(walltimes, f)
    return walltimes

def run_naive_optimization_benchmark(filename,
                                     chis=[4, 8, 16, 32],
                                     dtype=tf.float64,
                                     numiter=30,
                                     device=None,
                                     opt_u=True, opt_vw=True,
                                     numpy_update=True, refsym=True,opt_u_after=9):
    
    walltimes = {'profile': {}, 'energies' : {}}    
    with tf.device(device):        
        for chi in chis:
            print('running naive optimization benchmark for chi = {0}'.
                  format(chi))

            wC, vC, uC, rhoAB_0, rhoBA_0 = mbml.initialize_mod_binary_MERA(phys_dim=4, chi=chi, dtype=dtype)
            hamAB_0, hamBA_0 = mbml.initialize_TFI_hams(dtype=dtype)
            wC, vC, uC, rhoAB_0, rhoBA_0, runtimes, energies = mbml.optimize_mod_binary_mera(hamAB_0=hamAB_0, hamBA_0=hamBA_0, rhoAB_0=rhoAB_0, 
                                                                                             rhoBA_0=rhoBA_0, wC=wC,
                                                                                             vC=vC, uC=uC, verbose=1,
                                                                                             numiter=numiter,opt_u=True, opt_vw=True,
                                                                                             numpy_update=numpy_update,
                                                                                             refsym=refsym,opt_u_after=opt_u_after)
            walltimes['profile'][chi] = runtimes
            walltimes['energies'][chi] = energies
            print('     steps took {0} s'.format(walltimes['profile'][chi]))
            with open(filename + '.pickle', 'wb') as f:
                pickle.dump(walltimes, f)

    return walltimes
            

def run_optimization_benchmark(filename,
                               chis=[4, 8, 16, 32],
                               numiters=[200, 200, 400, 800],
                               embeddings=None,
                               dtype=tf.float64,
                               device=None, 
                               refsym=True, verbose=1):
    walltimes = {}
    with tf.device(device):    
        print('running optimization benchmark')
        print(' ###########################   hello')
        energies, runtimes, wC, vC, uC = run_mod_binary_mera_optimization_TFI(chis=chis, niters=numiters, embeddings=embeddings,
                                                                              dtype=dtype, verbose=verbose, refsym=refsym)
        walltimes['profile'] = runtimes
        walltimes['energies'] = energies
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(walltimes, f)
    return walltimes
    
if __name__ == "__main__":
    if not tf.executing_eagerly():
        pass

    else:
        fname =  'mod_binary_mera_benchmarks'            
        if not os.path.exists(fname):
            os.mkdir(fname)
        os.chdir(fname)
        rootdir = os.getcwd()
        
        # benchmarks = {'ascend' : {'chis' :  [4, 6, 8, 12, 16],
        #                           'dtype' : tf.float32,
        #                           'num_layers' : 2},
        #               'descend' : {'chis' :  [4, 6, 8, 12, 16],
        #                            'dtype' : tf.float32,
        #                            'num_layers' : 2},
        #               'optimize_naive' : {'chis' :  [4, 6, 8],
        #                                   'dtype' : tf.float64,
        #                                   'opt_u' : True,
        #                                   'opt_vw' : True,
        #                                   'numpy_update' : True,
        #                                   'refsym' : True,
        #                                   'numiter' : 5}}
        benchmarks = {'optimize' : {'chis' :  [6, 8, 10, 12],
                                    'numiters' : [2000, 2000, 2000, 1400],
                                    'embeddings' : ['p', 'a','a', 'p'],                                                            
                                    'dtype' : tf.float64,
                                    'refsym' : True}}
        # benchmarks = {'optimize_naive' : {'chis' :  [16, 32, 40],
        #                                   'dtype' : tf.float64,
        #                                   'opt_u' : True,
        #                                   'opt_vw' : True,
        #                                   'numpy_update' : True,
        #                                   'refsym' : True,
        #                                   'numiter' : 5}}

        
        use_gpu = False
        DEVICES = tf.contrib.eager.list_devices()
        print("Available devices:")
        for i, device in enumerate(DEVICES):
            print("%d) %s" % (i, device))
        CPU = '/device:CPU:0'
        GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
        if use_gpu:
            specified_device_type = GPU
            name = 'GPU'
        else:
            specified_device_type = CPU
            name = 'CPU'


        if 'ascend' in benchmarks:
            
            filename = name + 'modified_binary_mera_ascending_benchmark'
            for key, val in benchmarks['ascend'].items():
                if hasattr(val, 'name'):
                    val = val.name
                filename = filename + '_' + str(key) + str(val)
            filename = filename.replace(' ', '')

            fname =  'ascending_benchmarks'            
            if not os.path.exists(fname):
                os.mkdir(fname)
            os.chdir(fname)
            run_ascending_operator_benchmark(
                filename,
                device=specified_device_type,
                **benchmarks['ascend'])
            os.chdir(rootdir)
            
        if 'descend' in benchmarks:
            filename = name + 'modified_binary_mera_descending_benchmark'
            for key, val in benchmarks['descend'].items():
                if hasattr(val, 'name'):                
                    val = val.name
                
                filename = filename + '_' + str(key) + str(val)
            filename= filename.replace(' ', '')
            
            fname =  'descending_benchmarks'
            if not os.path.exists(fname):
                os.mkdir(fname)
            os.chdir(fname)
            
            run_descending_operator_benchmark(
                filename,
                device=specified_device_type,
                **benchmarks['descend'])
            os.chdir(rootdir)
            
        if 'optimize_naive' in benchmarks:
            filename = name + 'modified_binary_mera_naive_optimization_benchmark_Nthreads{}'.format(NUM_THREADS)
            keys = sorted(benchmarks['optimize_naive'].keys())
            for key in keys:
                val = benchmarks['optimize_naive'][key]
                if hasattr(val, 'name'):                                
                    val = val.name
                
                filename = filename + '_' + str(key) + str(val)
            filename = filename.replace(' ', '')
            
            fname = 'benchmarks_optimize_naive'
            if not os.path.exists(fname):
                os.mkdir(fname)
                
            os.chdir(fname)
            run_naive_optimization_benchmark(
                filename,
                **benchmarks['optimize_naive'],
                opt_u_after=0,
                device=specified_device_type)
            os.chdir(rootdir)            

        if 'optimize' in benchmarks:
            filename = name + 'modified_binary_mera_optimization_benchmark_Nthreads{}'.format(NUM_THREADS)
            
            fname = 'benchmarks_optimize'
            if not os.path.exists(fname):
                os.mkdir(fname)
            os.chdir(fname)
            
            
            for key, val in benchmarks['optimize'].items():
                if hasattr(val, 'name'):                                                    
                    val =val.name
                filename = filename + '_' + str(key) + str(val)
            filename = filename.replace(' ', '')
            
            
            run_optimization_benchmark(filename,
                                       device=specified_device_type,
                                       verbose=1,
                                       **benchmarks['optimize'])
                                       
            os.chdir(rootdir)                        

