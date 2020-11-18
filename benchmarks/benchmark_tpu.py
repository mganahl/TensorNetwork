from cloud_tpu_client import Client
c = Client('tpu-node-2')
c.configure_tpu_version('tpu_driver_nightly')

import tensornetwork as tn
from tensornetwork.matrixproductstates.dmrg import FiniteDMRG
from tensornetwork.matrixproductstates.mpo import FiniteXXZ
import numpy as np
import jax
import pickle
import jax.config as config
import tensornetwork.timer as timer
config.update("jax_enable_x64", False)
config.FLAGS.jax_xla_backend = "tpu_driver"
#config.FLAGS.jax_backend_target = "grpc://10.203.32.162:8470" #tpu-node-1 (v2-8)
config.FLAGS.jax_backend_target = "grpc://10.149.215.90:8470" #tpu-node-2 (v3-8)
backend = 'jax'
tn.set_default_backend(backend)
backend = tn.backends.backend_factory.get_backend('jax')
backend.jax_precision = jax.lax.Precision.HIGHEST
N = 30
Jz = np.ones(N - 1)
Jxy = np.ones(N - 1)
Bz = np.zeros(N)
for dtype in [np.float32]:
  mpo = FiniteXXZ(Jz, Jxy, Bz, dtype=dtype)
  for D in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    mps = tn.FiniteMPS.random(
        [2] * N, [D] * (N - 1), dtype=dtype, canonicalize=True)
    dmrg = FiniteDMRG(mps, mpo)
    dmrg.run_one_site_timing(
        13,
        17,
        num_sweeps=1,
        verbose=1,
        num_krylov_vecs=10,
        delta=1E-16,
        tol=1E-16)
    with open(
        'first_run_timings_D{}_N_{}_dtype{}.pickle'.format(
            D, N,
            np.dtype(dtype).name), 'wb') as f:
      pickle.dump(timer.timings, f)
    timer.reset_all()
    dmrg.run_one_site_timing(
        13,
        17,
        num_sweeps=10,
        verbose=1,
        num_krylov_vecs=10,
        delta=1E-16,
        tol=1E-16)
    with open(
        'second_run_timings_D{}_N_{}_dtype{}.pickle'.format(
            D, N,
            np.dtype(dtype).name), 'wb') as f:
      pickle.dump(timer.timings, f)
