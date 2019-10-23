import tensornetwork as tn
import tensorflow as tf
tn.set_default_backend('tensorflow')
import experiments.MPS.matrixproductstates as MPSmodule
import experiments.MPS.matrixproductoperators as MPOmodule
import experiments.MPS.DMRG as DMRG
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

parser = argparse.ArgumentParser(description='regular DMRG for the J1-J2 model')
parser.add_argument('--N1', type=int, help='height')
parser.add_argument('--N2', type=int, help='width')
parser.add_argument('--J1', type=float, help='nearest neighbor coupling')
parser.add_argument(
    '--J2', type=float, help='next-to-nearest neighbor coupling')
parser.add_argument(
    '--overwrite',
    action='store_true',
    help="If `True`, existing files will be overwritten (False)",
    default=False)
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
    '--load_mps_filename',
    type=str,
    help="A .pickle file containing an mps to be loaded (None).",
    default=None)

args = parser.parse_args()
if not args.N1:
  raise ValueError('no value for N1 given')
if not args.N2:
  raise ValueError('no value for N2 given')
if not args.J1:
  raise ValueError('no value for J1 given')
if not args.J2:
  raise ValueError('no value for J2 given')

if not args.save_mps_filename:
  raise ValueError('no value for save_mps_filename given')

if args.save_mps_filename[-7::] == '.pickle':
  raise ValueError('--save_mps_filename should not have a .pickle file-ending')

if args.load_mps_filename and (args.load_mps_filename[-7::] != '.pickle'):
  raise ValueError('--load_mps_filename should have a .pickle file-ending')

if not args.overwrite and args.save_mps_filename and os.path.exists(
    args.save_mps_filename + '.pickle'):
  raise ValueError(
      'filename {} already exists'.format(args.save_mps_filename + '.pickle'))

if not args.D:
  raise ValueError('no value for D given')

J1, J2 = args.J1, args.J2
N1, N2 = args.N1, args.N2

D = args.D
dtype = tf.float64

mpo = MPOmodule.Finite2D_J1J2(J1, J2, N1, N2, dtype=dtype)
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
dmrg = DMRG.FiniteDMRGEngine(mps, mpo)
dmrg.run_one_site(
    Nsweeps=args.num_dmrg_sweeps,
    precision=args.dmrg_precision,
    verbose=args.verbosity,
    ncv=args.ncv,
    filename=args.save_mps_filename)
#========================== save mps ====================================
with open(args.save_mps_filename + '.pickle', 'wb') as f:
  pickle.dump(mps, f)
  #==================================================================
