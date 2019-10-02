import tensornetwork as tn
import tensorflow as tf
tf.enable_v2_behavior()
import experiments.MPS.matrixproductstates as MPSmodule
import experiments.MPS.matrixproductoperators as MPOmodule
import experiments.MPS.DMRG as DMRG
import experiments.MPS.overlap_minimizer as OM
import numpy as np
import positivize as pv
import experiments.MPS.misc_mps as misc_mps
import os
import pickle
import argparse
from sys import stdout
misc_mps.compile_ncon(
    True)  #compiles ncon calls into graphs; use `True` for better performance
misc_mps.compile_decomps(
    True
)  #compiles matrix decomposition calls into graphs; use `True` for better performance
tn.set_default_backend('tensorflow')

parser = argparse.ArgumentParser(description='J1-J2 stoquastization')
parser.add_argument('--N1', type=int, help='height')
parser.add_argument('--N2', type=int, help='width')
parser.add_argument('--J1', type=float, help='nearest neighbor coupling')
parser.add_argument(
    '--J2', type=float, help='next-to-nearest neighbor coupling')
parser.add_argument(
    '--save_gate_filename',
    type=str,
    help="A file name for saving the optimized gates. "
    "This will save a dictionary mapping (site, site + 1) -> tf.Tensor (i.e. a pair of neighboring sites "
    " to a unitary gate) into a .pickle file with the given name.")
parser.add_argument(
    '--overwrite',
    action='store_true',
    help="If `True`, existing files will be overwritten (False)",
    default=False)

parser.add_argument(
    '--blocklength',
    type=int,
    help='blocking size for MPO blocking (2)',
    default=2)
parser.add_argument(
    '--num_stoq_steps',
    type=int,
    help='number of stoquastization steps (1500)',
    default=1500)

parser.add_argument(
    '--D', type=int, help='MPS bond dimension (None)', default=None)
parser.add_argument(
    '--save_mps_filename',
    type=str,
    help="A file name for saving the optimized mps (None).",
    default=None)

parser.add_argument(
    '--verbosity', type=int, help='verbosity flag (1)', default=1)

parser.add_argument(
    '--num_dmrg_sweeps',
    type=int,
    help='Number of dmrg-sweeps to get ground state after stoquastization (0)',
    default=0)

parser.add_argument(
    '--dmrg_precision',
    type=float,
    help='Precision of the DMRG solver (1E-10).',
    default=1E-10)
parser.add_argument(
    '--ncv',
    type=int,
    help='Number of krylov vectors for DMRG (20).',
    default=20)

parser.add_argument(
    '--load_gate_filename',
    type=str,
    help="A .pickle file containing the gates to be loaded. "
    "The file has to be a pickled python dict mapping (site, site + 1) -> tf.Tensor (a pair of neighboring sites "
    " to a unitary gate) (None).",
    default=None)

parser.add_argument(
    '--load_mps_filename',
    type=str,
    help="A .pickle file containing an mps to be loaded (None).",
    default=None)

args = parser.parse_args()
print(args.overwrite)
if not args.N1:
  raise ValueError('no value for N1 given')
if not args.N2:
  raise ValueError('no value for N2 given')
if not args.J1:
  raise ValueError('no value for J1 given')
if not args.J2:
  raise ValueError('no value for J2 given')
if not args.save_gate_filename:
  raise ValueError('no value for save_gate_filename given')

if args.save_gate_filename[-7::] == '.pickle':
  raise ValueError('--save_gate_filename should not have a .pickle file-ending')
if args.save_mps_filename and (args.save_mps_filename[-7::] == '.pickle'):
  raise ValueError('--save_mps_filename should not have a .pickle file-ending')
if args.load_gate_filename and (args.load_gate_filename[-7::] != '.pickle'):
  raise ValueError('--load_gate_filename should have a .pickle file-ending')

if args.load_mps_filename and (args.load_mps_filename[-7::] != '.pickle'):
  raise ValueError('--load_mps_filename should have a .pickle file-ending')

if not args.overwrite and args.save_mps_filename and os.path.exists(
    args.save_mps_filename + '.pickle'):
  raise ValueError(
      'filename {} already exists'.format(args.save_mps_filename + '.pickle'))

if not args.overwrite and os.path.exists(args.save_gate_filename + '.pickle'):
  raise ValueError(
      'filename {} already exists'.format(args.save_gate_filename + '.pickle'))

if args.num_dmrg_sweeps > 0:
  if not args.D:
    raise ValueError('no value for D given')
  if not args.save_mps_filename:
    raise ValueError('no value for save_mps_filename given')

J1, J2 = args.J1, args.J2
N1, N2 = args.N1, args.N2
block_length = args.blocklength
D = args.D
dtype = tf.float64

mpo = OM.block_MPO(
    MPOmodule.Finite2D_J1J2(J1, J2, N1, N2, dtype=dtype), block_length)
stoq = OM.TwoBodyStoquastisizer(mpo)
ds = [t.shape[3] for t in mpo._tensors]
ref_mps = MPSmodule.FiniteMPSCentralGauge(
    tensors=pv.equal_superposition_tf(ds, dtype=dtype))
ref_mps.position(
    0, normalize=True
)  #mps.position is by default not normalizing the wavefunction

#=========  load gates =====================================
if args.load_gate_filename:
  with open(args.load_gate_filename, 'rb') as f:
    gates = pickle.load(f)
  stoq.gates = gates

#==========  stoquastisize  ================================
if args.num_stoq_steps > 0:
  OK = False
  while not OK:
    if not args.load_gate_filename:
      stoq.reset_gates(which='haar')
    OK = stoq.stoquastisize(ref_mps, num_steps=args.num_stoq_steps)
  #===========   save gates   ==============================
  with open(args.save_gate_filename + '.pickle', 'wb') as f:
    pickle.dump(stoq.gates, f)

#==================    do dmrg   ===========================
if args.num_dmrg_sweeps > 0:
  d = [mpo[0].shape[2]] * len(mpo)
  Ds = [args.D] * (len(mpo) - 1)
  #============= load an mps ============================
  if args.load_mps_filename:
    with open(args.load_mps_filename, 'rb') as f:
      mps = pickle.load(f)
  else:
    mps = MPSmodule.FiniteMPSCentralGauge.random(d=d, D=Ds, dtype=dtype)
  mps.position(0)
  mps.position(len(mps))
  mps.position(0)
  stoq.compute_right_envs(mps)
  stoq.do_dmrg(
      mps,
      num_sweeps=args.num_dmrg_sweeps,
      precision=args.dmrg_precision,
      verbose=args.verbosity,
      ncv=args.ncv)

  #========================== save mps ====================================
  with open(args.save_mps_filename + '.pickle', 'wb') as f:
    pickle.dump(mps, f)
  #==================================================================
