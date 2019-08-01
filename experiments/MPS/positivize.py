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
import copy
import os
import itertools
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import sign_mps_data.readMPS as readMPS
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
misc_mps.compile_ncon(True)
plt.ion()

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


def convert_to_tf(tensors):
    """
    permutes the indices of the tensors from PyTeN ordering to TensorNetwork ordering
    Args: 
         tensors (list of np.ndarrau):  

    Returns:
        list of tf.Tensor objects
    """
    return [tf.transpose(tf.convert_to_tensor(tensors[n]),(0,2,1)) for n in range(len(tensors))] #the index order of the PyTeN and TensorNetwork MPS is different

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
def analyze_positivity(mps, samples=None, nsamples=1000, show=True, fignum=0, bins=100, verbose=0):
    """
    analyze the MPS wavefunction by  sampling `nsamples` and measuring the average sign

    Returns:
        for float dtype: av_sign, av_amp

        for complex dtype: av_real_sign, av_abs_imag_amp
    """
    if mps.dtype in (tf.float64, tf.float32):
        if samples is None:
            samples=mps.generate_samples(nsamples, verbose=verbose)
        signs, log_amps = mps.get_log_amplitude(samples)
        log_amps = log_amps.numpy()
        signs = signs.numpy()
        if show:
            plt.figure(fignum)
            plt.clf()
            plt.title(r'log-amplitudes; min ($\log$(amps))={0}, max($\log$(amp))={1}'.format(np.round(np.min(log_amps), 2),np.round(np.max(log_amps), 2)))
            plt.hist(-signs*log_amps, bins=bins)
            plt.xlabel(r'-Sign(amp) $\log (\vert$amp$\vert)$')
            plt.ylabel('frequency')
            plt.tight_layout()            
            plt.draw()
            plt.show()
            plt.pause(0.01)        
        av_sign = np.mean(signs)
        av_amp = np.mean(np.exp(log_amps * signs))
        print()
        print('#############################')
        print('   all samples >= 0: ',np.all(signs>=0))
        print('   all samples <= 0: ', np.all(signs<=0))
        print('   <sgn> = {0}'.format(av_sign))
        print('   <amp> = {0}'.format(av_amp))                        
        print('#############################')
        print()
        return av_sign, av_amp

    if mps.dtype in (tf.complex128, tf.complex64):
        if samples is None:
            samples=mps.generate_samples(nsamples, verbose=verbose)
        phases, log_amps = mps.get_log_amplitude(samples)
        log_amps = log_amps.numpy()
        phases = phases.numpy()
        if show:
            # plt.figure(fignum)
            # plt.clf()
            # plt.title(r'log-amplitudes; min ($\log$(amps))={0}, max($\log$(amp))={1}'.format(np.round(np.min(log_amps), 2),np.round(np.max(log_amps), 2)))
            # plt.scatter(np.real(phases), np.imag(phases))
            # plt.xlim([-1.2, 1.2])
            # plt.ylim([-1.2, 1.2])
            # plt.draw()
            # plt.show()
            # plt.pause(0.01)
            
            plt.figure(fignum)
            plt.clf()            
            ax = plt.subplot(2,1,1)
            plt.title(r'log-amplitudes; min ($\log$(amps))={0}, max($\log$(amp))={1}'.format(np.round(np.min(log_amps), 2),np.round(np.max(log_amps), 2)))
            plt.hist(np.real(np.exp(log_amps)*phases), bins=bins)
            plt.xlabel(r'$\Re(amp)$')                        
            plt.ylabel('frequency')

            ax = plt.subplot(2,1,2)
            plt.hist(np.imag(np.exp(log_amps)*phases), bins=bins)
            plt.xlabel(r'$\Im(amp)$')            
            plt.ylabel('frequency')
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.01)        
        av_real_sign = np.mean(np.sign(np.real(np.exp(log_amps)*phases)))
        av_real = np.mean(np.real(np.exp(log_amps)*phases))
        av_imag = np.mean(np.imag(np.exp(log_amps)*phases))
        av_abs_imag = np.mean(np.abs(np.imag(np.exp(log_amps)*phases)))

        print()
        print()
        print('#############################')
        print('   all samples >= 0: N/A')
        print('   all samples <= 0: N/A')
        print('   <Sgn(Re(amp))> = {0}'.format(av_real_sign))
        print('   <Re(amp)> = {0}'.format(av_real))                
        print('   <Im(amp)> = {0}'.format(av_imag))
        print('   <|Im(amp)|> = {0}'.format(av_abs_imag))                
        print('#############################')
        print()
        return av_real_sign, av_abs_imag

    

def equal_superposition_tf(ds,dtype=tf.float64):
    """
    return mps tensors correpsonding to the equal superposition of all basis states
    """
    return [np.expand_dims(np.expand_dims(np.ones(ds[n]), 0),2).astype(dtype.as_numpy_dtype) for n in range(len(ds))]



def positivize(minimizer, ref_mps=None, num_its=100, num_minimizer_sweeps_even=2, num_minimizer_sweeps_odd=2,
               num_minimizer_sweeps_one_body=2,
               alpha_gates=0.0, sites=None,
               alpha_ref_mps=1):
    """
    runs a positivation using `samples` and `ref_mps`
    one layer of even and one layer of odd unitaries are optimized layer by layer
    see OverlapMinimizer docstring for more information
    Args:
        minimizer (OverlapMinimizer):     the overlap minimizer object
        ref_mps (FiniteMPSCentralGauge):               a reference mps 
        num_its (int): number of iterations; one iteration consists of an even and an odd layer update
        num_minimizer_sweeps_even (int): number of optimization sweeps over even gates in minimizer
        num_minimizer_sweeps_odd (int): number of optimization sweeps over odd gates in minimizer
        num_minimizer_sweeps_one_body (int): number of optimization sweeps over one-body gates in minimizer
        sites (iterable or None):  the sites on which one-body gates should be optimized
        alpha_gates (float): see below
        alpha_ref_mps (float): the three `alpha_` arguments determine the mixing of the update
                               the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
    """
    for it in range(num_its):
        if num_minimizer_sweeps_one_body > 0:
            minimizer.minimize_one_body(samples=None, num_sweeps=num_minimizer_sweeps_one_body, ref_mps=ref_mps,  
                                        alpha_gates=alpha_gates, sites=sites,
                                        alpha_ref_mps=alpha_ref_mps, verbose=1)
        if num_minimizer_sweeps_even >0:
            minimizer.minimize_even(samples=None,num_sweeps=num_minimizer_sweeps_even, ref_mps=ref_mps, 
                                    alpha_gates=alpha_gates, 
                                    alpha_ref_mps=alpha_ref_mps, verbose=1)  
        if num_minimizer_sweeps_odd >0:
            minimizer.minimize_odd(samples=None, num_sweeps=num_minimizer_sweeps_odd, ref_mps=ref_mps,  
                                   alpha_gates=alpha_gates, 
                                   alpha_ref_mps=alpha_ref_mps, verbose=1)
    return minimizer

def positivize_from_self_sampling(minimizer, ref_mps=None, Dmax=20,num_its=100, num_minimizer_sweeps_even=2, num_minimizer_sweeps_odd=2,
                                  num_minimizer_sweeps_one_body=2, n_samples=1000,
                                  alpha_gates=0.0, sites=None,
                                  alpha_samples=1, alpha_ref_mps=1):
    """
    Args:
        minimizer (OverlapMinimizer):     the overlap minimizer object
        ref_mps (FiniteMPSCentralGauge):               a reference mps 
        Dmax (int):  maximum bond dimension kept when contracting the gates to the the new mps to sample from
        num_its (int): number of iterations; one iteration consists of an even and an odd layer update
        num_minimizer_sweeps_even (int): number of optimization sweeps over even gates in minimizer
        num_minimizer_sweeps_odd (int): number of optimization sweeps over odd gates in minimizer
        num_minimizer_sweeps_one_body (int): number of optimization sweeps over one-body gates in minimizer
        n_samples (int):              number of samples to draw from U*mps for optimization
        sites (iterable or None):  the sites on which one-body gates should be optimized
        alpha_gates (float): see `alpha_ref_mps`
        alpha_samples (float): see `alpha_ref_mps`
        alpha_ref_mps (float): the three `alpha_` arguments determine the mixing of the update
                               the new gate is given by `alpha_gate` * old_gate + `alpha_samples` * sample_update + `alpha_ref_mps` * ref_mps_udate
    """

    pos_mps, tw = minimizer.absorb_gates(Dmax=Dmax)
    for it in range(num_its):
        if num_minimizer_sweeps_one_body > 0:
            samples = pos_mps.generate_samples(n_samples)
            minimizer.minimize_one_body(samples=samples, num_sweeps=num_minimizer_sweeps_one_body, ref_mps=ref_mps,  
                                        alpha_gates=alpha_gates, sites=sites,
                                        alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps, verbose=1)
            pos_mps, tw = minimizer.absorb_gates(Dmax=Dmax)
        

        if num_minimizer_sweeps_even >0:
            samples = pos_mps.generate_samples(n_samples)
            minimizer.minimize_even(samples= samples,num_sweeps=num_minimizer_sweeps_even, ref_mps=ref_mps, 
                                    alpha_gates=alpha_gates, 
                                    alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps, verbose=1)  
            pos_mps, tw = minimizer.absorb_gates(Dmax=Dmax)
        if num_minimizer_sweeps_odd >0:
            samples = pos_mps.generate_samples(n_samples)
            minimizer.minimize_odd(samples=samples, num_sweeps=num_minimizer_sweeps_odd, ref_mps=ref_mps,  
                                   alpha_gates=alpha_gates, 
                                   alpha_samples=alpha_samples, alpha_ref_mps=alpha_ref_mps, verbose=1)
            pos_mps, tw = minimizer.absorb_gates(Dmax=Dmax)
        
    return minimizer

