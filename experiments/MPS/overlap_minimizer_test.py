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
unittests
"""
import tensorflow as tf
import tensornetwork as tn

import numpy as np
import itertools
import experiments.MPS.matrixproductstates as MPS
import experiments.MPS.DMRG as DMRG
import experiments.MPS.matrixproductoperators as MPO
import experiments.MPS.overlap_minimizer as OM
import pytest
import numpy as np
tn.set_default_backend('tensorflow')


def test_matvec():
  J1, J2 = 1, 1
  N1, N2 = 10, 10
  dtype = tf.float64
  block_length = 2
  mpo = MPO.Finite2D_J1J2(J1, J2, N1, N2, dtype=dtype)
  mpo_blocked = OM.block_MPO(mpo, block_length)
  stoq = OM.TwoBodyStoquastisizer(mpo_blocked)
  stoq.reset_gates('haar')
  D = 4
  d = [mpo_blocked[0].shape[2]] * len(mpo_blocked)
  Ds = [D] * (len(mpo_blocked) - 1)
  mps = MPS.FiniteMPSCentralGauge.random(d=d, D=Ds, dtype=dtype)
  mps.position(0)
  mps.position(len(mps))
  mps.position(0)
  stoq.compute_right_envs(mps)

  site = 4
  stoq.position(mps, site)

  L = stoq.left_envs[(site - 1, site)]  #contains the mps tensor site site - 1
  R = stoq.right_envs[(site, site + 1)]  #contains the MPS tensor at site + 1
  Lgate = stoq.gates[(site - 1, site)]
  Rgate = stoq.gates[(site, site + 1)]
  mpotensor = mpo_blocked[site]
  shape = mps[site].shape
  dim = np.prod(shape)

  def matvec(mps_tensor):
    return stoq.mat_vec(
        L,
        R,
        Lgate,
        Rgate,
        mpo_tensor=mpo_blocked[site],
        backend='tensorflow',
        site=site,
        mps_tensor=mps_tensor)

  def dotprod(a, b):
    return tn.ncon([a, b], [[1, 2, 3], [1, 2, 3]], backend='tensorflow')

  H = np.zeros((dim, dim))
  basis = np.eye(dim)
  for n in range(dim):
    t = np.reshape(basis[n, :], shape)
    Hx = matvec(t)
    for m in range(dim):
      t2 = np.reshape(basis[m, :], shape)
      H[m, n] = tn.ncon([t2, Hx], [[1, 2, 3], [1, 2, 3]],
                        backend='tensorflow').numpy()
  assert np.linalg.norm(H - H.T) < 1E-10
  eta, U = np.linalg.eigh(H)
  stoq.position(mps, site)
  e, opt = stoq._optimize_1s_local(
      mps, site, sweep_dir='r', ncv=40, precision=1E-13, delta=1E-10)
  np.testing.assert_allclose(e, min(eta))


def test_matvec_complex():
  J1, J2 = 1, 1
  N1, N2 = 10, 10
  dtype = tf.complex128
  block_length = 2
  mpo = MPO.Finite2D_J1J2(J1, J2, N1, N2, dtype=dtype)
  mpo_blocked = OM.block_MPO(mpo, block_length)
  stoq = OM.TwoBodyStoquastisizer(mpo_blocked)
  stoq.reset_gates('haar')
  D = 4
  d = [mpo_blocked[0].shape[2]] * len(mpo_blocked)
  Ds = [D] * (len(mpo_blocked) - 1)
  mps = MPS.FiniteMPSCentralGauge.random(d=d, D=Ds, dtype=dtype)
  mps.position(0)
  mps.position(len(mps))
  mps.position(0)
  stoq.compute_right_envs(mps)

  site = 4
  stoq.position(mps, site)

  L = stoq.left_envs[(site - 1, site)]  #contains the mps tensor site site - 1
  R = stoq.right_envs[(site, site + 1)]  #contains the MPS tensor at site + 1
  Lgate = stoq.gates[(site - 1, site)]
  Rgate = stoq.gates[(site, site + 1)]
  mpotensor = mpo_blocked[site]
  shape = mps[site].shape
  dim = np.prod(shape)

  def matvec(mps_tensor):
    return stoq.mat_vec(
        L,
        R,
        Lgate,
        Rgate,
        mpo_tensor=mpo_blocked[site],
        backend='tensorflow',
        site=site,
        mps_tensor=mps_tensor)

  def dotprod(a, b):
    return tn.ncon([tf.math.conj(a), b], [[1, 2, 3], [1, 2, 3]],
                   backend='tensorflow')

  H = np.zeros((dim, dim), dtype=dtype.as_numpy_dtype)
  basis = np.eye(dim).astype(dtype.as_numpy_dtype)
  for n in range(dim):
    t = np.reshape(basis[n, :], shape)
    Hx = matvec(t)
    for m in range(dim):
      t2 = np.reshape(basis[m, :], shape)
      H[m, n] = tn.ncon([t2, Hx], [[1, 2, 3], [1, 2, 3]],
                        backend='tensorflow').numpy()
  assert np.linalg.norm(H - np.conj(H.T)) < 1E-10
  eta, U = np.linalg.eigh(H)
  stoq.position(mps, site)
  e, opt = stoq._optimize_1s_local(
      mps, site, sweep_dir='r', ncv=100, precision=1E-13, delta=1E-10)
  np.testing.assert_allclose(e, min(eta))


def test_J1J2_mpo(N1=4, N2=2, J1=1, J2=1):
  N1, N2 = 4, 2
  J1, J2 = np.random.rand(2)
  H, neighbors = OM.J1J2_exact(N1=N1, N2=N2, J1=J1, J2=J2)
  eta, U = np.linalg.eigh(H)

  D = 300
  dtype = tf.float64
  mpo_points = []
  ncv = 30
  precision = 1E-12
  mps = MPS.FiniteMPSCentralGauge.random(
      d=[2] * N1 * N2, D=[D] * (N1 * N2 - 1), dtype=dtype)
  mpo = MPO.Finite2D_J1J2(J1=J1, J2=J2, N1=N1, N2=N2, dtype=dtype)
  dmrg = DMRG.FiniteDMRGEngine(mps, mpo)
  e_dmrg = dmrg.run_one_site(
      verbose=1, Nsweeps=20, ncv=ncv, precision=precision)
  np.testing.assert_allclose(e_dmrg, min(eta))


def test_stoq_dmrg():
  Jz, Jxy = 1, 1
  N = 12
  D = 256
  ncv = 10
  precision = 1E-6
  dtype = tf.float64
  mpo = MPO.FiniteXXZ(
      Jz=Jz * np.ones(N - 1),
      Jxy=np.ones(N - 1),
      Bz=np.zeros(N),
      dtype=tf.float64)
  d = [mpo[0].shape[2]] * N
  Ds = [D] * (N - 1)
  mps = MPS.FiniteMPSCentralGauge.random(d=d, D=Ds, dtype=dtype)
  mps.position(len(mps))
  mps.position(0)
  stoq = OM.TwoBodyStoquastisizer(mpo)
  stoq.reset_gates('haar')
  stoq.compute_right_envs(mps)
  e1 = stoq.do_dmrg(mps, num_sweeps=6, precision=precision, verbose=1, ncv=ncv)

  mps2 = MPS.FiniteMPSCentralGauge.random(d=d, D=Ds, dtype=dtype)
  mps2.position(len(mps))
  mps2.position(0)
  dmrg = DMRG.FiniteDMRGEngine(mps2, mpo)
  e2 = dmrg.run_one_site(verbose=1, Nsweeps=10, ncv=ncv, precision=precision)
  np.testing.assert_allclose(e1, e2)


#
