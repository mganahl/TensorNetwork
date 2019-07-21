import tensorflow as tf
tf.enable_v2_behavior()
import copy
from sys import stdout
import tensornetwork as tn
import experiments.MPS.misc_mps as misc_mps
import experiments.MPS.matrixproductstates as MPS
import experiments.MPS.matrixproductoperators as MPO
import experiments.MPS.DMRG as DMRG
import experiments.MERA.misc_mera as misc_mera
import experiments.MPS_classifier.MPSMNIST as mm
import experiments.MPS.overlap_minimizer as OM
from scipy.linalg import expm
import numpy as  np
from lib.mpslib.Tensor import Tensor
import lib.mpslib.TensorNetwork as TN
import copy
import os
import itertools
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import sign_mps_data.readMPS as readMPS
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
misc_mps.compile_ncon(True)

plt.ion()

def generate_basis(N):
    basis = []
    for n in range(2**N):
        text =  '{:0' + str(N) + 'b}'
        basis.append(np.array([int(t) for t in text.format(n)]))
    basis = np.stack(basis,axis=0)
    return basis

def haar_random_unitary(shape):
    Q, R = np.linalg.qr(np.random.random_sample(shape) - 0.5)
    diagr = np.diag(R)
    Lambda = np.diag(diagr/np.abs(diagr))
    return Q.dot(Lambda)

def get_product_states(sigmas,d):
    mpss = []
    N = sigmas.shape[1]
    for m in  range(sigmas.shape[0]):
        tensors = [np.zeros((1,1,d)).view(Tensor) for _ in range(sigmas.shape[1])]
        for n in  range(len(tensors)):
            tensors[n][0,0,sigmas[m,n]] = 1
        mpss.append(TN.FiniteMPS(tensors))
    return mpss

def get_sample_mps(mps, num_samples,Dmax):
    """
    compute the mps obtained from sampling `num_samples` states from `mps` and 
    taking an equal superposition of all 
    the use of this in OverlapMinimizer is deprecated; OverlapMinimizer can directly optimize from a 
    list of samples
    Args:
        mps (FiniteMPSCentralGauge)
        num_samples (int): number of samples to draw from `mps`
        Dmax (int):  the maximum bond dimension to be kept when building the sampled mps
    Returns:
        FiniteMPSCentralGauge
    """
    if not Dmax:
        Dmax=40
    samples = mps.generate_samples(num_samples)
    mpss=get_product_states(samples.numpy(), 2)
    for m in range(1,len(mpss)):
        mpss[0] = mpss[0] + mpss[m]
        if m % Dmax == 0:
            mpss[0].position(0)
            mpss[0].position(len(mpss[0]))
            mpss[0].position(0,D=Dmax)
            #print(mpss[0].D)
    tensors = [tf.convert_to_tensor(np.transpose(mpss[0].get_tensor(n),(0,2,1))) for n in  range(len(mpss[0]))]
    mps = MPS.FiniteMPSCentralGauge(tensors=tensors)
    mps.position(0) #normalize the state
    mps.position(len(mps),normalize=True,D=Dmax)
    return mps

def apply_random_positive_gates(mps):
    for site in range(0,len(mps)-1,2):
        mps.apply_2site(tf.random_uniform(shape=[mps.d[site], mps.d[site+1],mps.d[site], mps.d[site+1]], 
                                          dtype = mps.dtype), site)
    for site in range(1,len(mps)-2,2):
        mps.apply_2site(tf.random_uniform(shape=[mps.d[site], mps.d[site+1],mps.d[site], mps.d[site+1]], 
                                          dtype = mps.dtype), site)
    mps.position(0)
    mps.position(len(mps))
    mps.position(0)
    return mps

def read_and_convert_to_pyten(prefix,dtype=tf.float64):
    """
    function to read Giacomo's mps data
    Args: 
         prefix (str):  the path + the prefix to  the data; the routine appends
                        '_tensors.txt' and '_index.txt' to `prefix`
    Returns:
        list of PyTen.lib.mpslib.Tensor.Tensor objects (essentially np.ndarrays)
    """
    tensors = readMPS.readMPS(prefix + '_tensors.txt',prefix + '_index.txt',N=8,convert=True)
    tensors_new=[]
    for n in range(len(tensors)):
        if n ==0:
            tensors_new.append(np.transpose(np.expand_dims(tensors[n],0),(0,2,1)).astype(dtype.as_numpy_dtype))
        elif n > 0 and n < len(tensors) - 1:
            tensors_new.append(np.transpose(tensors[n], (0,2,1)).astype(dtype.as_numpy_dtype))
        elif n == len(tensors) -1 :
            tensors_new.append(np.transpose(np.expand_dims(tensors[n],2), (1,2,0)).astype(dtype.as_numpy_dtype))
    return [t.view(Tensor) for t in tensors_new]

def read_and_convert_to_tf(prefix,  dtype=tf.float64):
    """
    function to read Giacomo's mps data
    Args: 
         prefix (str):  the path + the prefix to  the data; the routine appends
                        '_tensors.txt' and '_index.txt' to `prefix`
    Returns:
        list of tf.Tensor objects
    """
    tensors = read_and_convert_to_pyten(prefix,dtype)
    return [tf.transpose(tf.convert_to_tensor(tensors[n]),(0,2,1)) for n in range(len(tensors))] #the index order of the PyTeN and TensorNetwork MPS is different

#%matplotlib qt
def analyze_positivity(mps, nsamples=1000, discard_threshold=1E-13, show_real=True,show_imag=False, fignum_real=0, fignum_imag=1):
    """
    analyze the MPS wavefunction by  sampling `nsamples` and measuring the average sign
    """
    samples=mps.generate_samples(nsamples)
    amps = np.array(mps.get_amplitude(samples))
    amps[np.abs(amps)<discard_threshold] = 0.0
    #print(np.real(amps))
    if show_real:
        plt.figure(fignum_real)

        plt.clf()
        plt.title('real part')
        plt.hist(np.real(amps), bins=100)
        plt.draw()
        plt.show()
        plt.pause(0.01)        
    if show_imag:
        if np.linalg.norm(np.imag(amps)>1E-10):
            plt.figure(fignum_imag)
            plt.clf()
            plt.title('imag part')
            plt.hist(np.imag(amps), bins=100)
            plt.draw()
            plt.show()
            plt.pause(0.01)

    av_real_sign = np.mean(np.sign(np.real(amps[amps!=0])))
    #av_imag_sign = np.mean(np.sign(np.imag(amps[amps!=0])))
    print('#############################')
    print('   all samples > 0: ',np.all(amps>=0))
    print('   all samples < 0: ', np.all(amps<=0))
    print('   <sgn>={0}'.format(av_real_sign))#,av_imag_sign))
    print('#############################')
    print()
    return av_real_sign# + 1j* av_imag_sign

def equal_superposition(ds,dtype=np.float64):
    """
    return mps tensors correpsonding to the equal superposition of all basis states
    """
    return [np.expand_dims(np.expand_dims(np.ones(ds[n]), 0),0).astype(dtype).view(Tensor) for n in range(len(ds))]
def equal_superposition_tf(ds,dtype=tf.float64):
    """
    return mps tensors correpsonding to the equal superposition of all basis states
    """
    return [np.expand_dims(np.expand_dims(np.ones(ds[n]), 0),2).astype(dtype.as_numpy_dtype) for n in range(len(ds))]



def positivize(minimizer, samples, ref_mps, num_its=100, 
               num_minimzer_sweeps=2,alpha_gates=0.0,
               alpha_samples=1, alpha_ref_mps=1):
    """
    runs a positivation using `samples` and `ref_mps`
    one layer of even and one layer of odd unitaries are optimized layer by layer
    see OverlapMinimizer docstring for more information
    Args:
        minimizer (OverlapMinimizer):     the overlap minimizer object
        samples (tf.Tensor of shape (n_samples, N):    basis-state samples
        ref_mps (FiniteMPSCentralGauge):               a reference mps 
        num_its (int): number of iterations; one iteration consists of an even and an odd layer update
        num_sweeps (int): number of optimiztion sweeps of minimizer
        alpha_gates (float): see below
        alpha_samples (float): see below
        alpha_ref_mos (float): the three `alpha_` arguments determine the mixing of the update
                               the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
        verbose (int):         verbosity flag; larger means more output
        
    """
    for it in range(num_its):
        minimizer.minimize_even_batched(samples= samples,num_sweeps=num_minimzer_sweeps, ref_mps=ref_mps, 
                                        alpha_gates=alpha_gates, 
                                        alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps, verbose=1)
        minimizer.minimize_odd_batched(samples=samples, num_sweeps=num_minimzer_sweeps, ref_mps=ref_mps,  
                                       alpha_gates=alpha_gates, 
                                       alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps, verbose=1)
        
    return minimizer

def positivize_from_self_sampling(minimizer, ref_mps=None, Dmax=20,num_its=100, num_minimzer_sweeps=2, n_samples=1000,
                                  alpha_gates=0.0,
                                  alpha_samples=1, alpha_ref_mps=1):
    """
    runs a self sampling optimization:
    one layer of even and one layer of odd 
    """

    pos_mps = minimizer.absorb_gates(Dmax=Dmax)
    for it in range(num_its):
        samples = pos_mps.generate_samples(n_samples)
        minimizer.minimize_even_batched(samples= samples,num_sweeps=num_minimzer_sweeps, ref_mps=ref_mps, 
                                        alpha_gates=alpha_gates, 
                                        alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps, verbose=1)  
        pos_mps = minimizer.absorb_gates(Dmax=Dmax)
        samples = pos_mps.generate_samples(n_samples)
        minimizer.minimize_odd_batched(samples=samples, num_sweeps=num_minimzer_sweeps, ref_mps=ref_mps,  
                                       alpha_gates=alpha_gates, 
                                       alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps, verbose=1)
        pos_mps = minimizer.absorb_gates(Dmax=Dmax)
        
    return minimizer

# def optimal_positivizer(mps_in, num_sweeps=10, num_opt_sweeps = 10, num_its=4, num_samples=100,Dmax=None,Dmax_sample=None, alpha=1.0, verbose=1):
#     mps = copy.deepcopy(mps_in)
#     #first converge with respect to the totally mixed state
#     ds = mps.d
#     gates={(site, site+1): tf.reshape(tf.eye(ds[site]*ds[site+1], dtype=dtype), 
#                                       (ds[site], ds[site+1], ds[site], ds[site+1]))
#             for site in range(0,N-1)}
#     ref_mps = MPS.FiniteMPSCentralGauge(tensors=equal_superposition_tf(ds))
#     ref_mps.position(0,normalize=True)
#     ref_mps.position(len(ref_mps),normalize=True)  
#     minimizer = OM.OverlapMinimizer(mps,ref_mps, gates)
#     for _ in range(num_sweeps):
#         minimizer.minimize_even(num_iterations=num_its,verbose=1)
#         minimizer.minimize_odd(num_iterations=num_its,verbose=1)
   
#     #now the gates should be close to he ideal gates
#     #switch to the ideal optimization scheme:
#     for sweep in range(num_opt_sweeps):
#         for site in range(len(minimizer.mps) - 1):
#             #reset all envs to empty
#             minimizer.left_envs = {}
#             minimizer.right_envs = {}            
#             [minimizer.add_unitary_right(pos, minimizer.right_envs, minimizer.mps, minimizer.conj_mps, minimizer.gates) for pos in reversed(range(site+1,len(minimizer.mps)))]
#             [minimizer.add_unitary_left(pos, minimizer.left_envs, minimizer.mps, minimizer.conj_mps, minimizer.gates) for pos in range(site)]
            
#             env = minimizer.gates[(site, site+1)] * (1-alpha) + alpha * minimizer.get_env((site, site + 1),
#                                                                                           minimizer.left_envs, minimizer.right_envs, minimizer.mps, minimizer.conj_mps)
#             minimizer.gates[(site, site+1)] = minimizer.u_update_svd_numpy(env)
#             minimizer.add_unitary_left(site,minimizer.left_envs, minimizer.mps, minimizer.conj_mps, minimizer.gates)
#             if verbose > 0 and site > 0:
#                 overlap = minimizer.overlap(site, minimizer.left_envs, minimizer.right_envs, minimizer.mps, minimizer.conj_mps)
#                 stdout.write("\r site %i, overlap = %.16f" %
#                     (site, np.abs(np.real(overlap))))
#                 stdout.flush()
#             if verbose > 1:
#                 print()

#             minimizer.absorb_gates() #absorb the current gates into the state 
#             minimizer.mps.position(0) #normalize the state
#             minimizer.mps.position(len(minimizer.mps),normalize=True)
#             minimizer.mps.position(0,normalize=True,D=Dmax) #normalize the state
#             analyze_positivity(minimizer.mps,nsamples=2000, show_imag=False)
#             #minimizer = OM.OverlapMinimizer(copy.deepcopy(mps_in),get_sample_mps(minimizer.mps, num_samples) , gates)
#             #generate a new ref state by sampling from U *minimizer.mps
#             minimizer.conj_mps = get_sample_mps(minimizer.mps, num_samples,Dmax=Dmax_sample)     #returns normalized state
#             minimizer.mps = copy.deepcopy(mps_in)#reset the input state