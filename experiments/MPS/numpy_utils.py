import numpy as  np
import sign_mps_data.readMPS as readMPS
from lib.mpslib.Tensor import Tensor
import lib.mpslib.TensorNetwork as TN
import sign_mps_data.readMPS as readMPS


def generate_basis(N):
    basis = []
    for n in range(2**N):
        text =  '{:0' + str(N) + 'b}'
        basis.append(np.array([int(t) for t in text.format(n)]))
    basis = np.stack(basis,axis=0)
    return basis

def get_product_states(sigmas,d):
    mpss = []
    N = sigmas.shape[1]
    for m in  range(sigmas.shape[0]):
        tensors = [np.zeros((1,1,d)).view(Tensor) for _ in range(sigmas.shape[1])]
        for n in  range(len(tensors)):
            tensors[n][0,0,sigmas[m,n]] = 1
        mpss.append(TN.FiniteMPS(tensors))
    return mpss

def equal_superposition(ds,dtype=np.float64):
    """
    return mps tensors correpsonding to the equal superposition of all basis states
  n  """
    return [np.expand_dims(np.expand_dims(np.ones(ds[n]), 0),0).astype(dtype).view(Tensor) for n in range(len(ds))]

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

def read_and_convert_to_pyten(prefix,dtype=np.float64):
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
            tensors_new.append(np.transpose(np.expand_dims(tensors[n],0),(0,2,1)).astype(dtype))
        elif n > 0 and n < len(tensors) - 1:
            tensors_new.append(np.transpose(tensors[n], (0,2,1)).astype(dtype))
        elif n == len(tensors) -1 :
            tensors_new.append(np.transpose(np.expand_dims(tensors[n],2), (1,2,0)).astype(dtype))
    return [t.view(Tensor) for t in tensors_new]
