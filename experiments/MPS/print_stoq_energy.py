import tensornetwork as tn
import tensorflow as tf
tn.set_default_backend('tensorflow')
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

parser = argparse.ArgumentParser(description='print energy')
parser.add_argument('--N1', type=int, help='height')
parser.add_argument('--N2', type=int, help='width')
parser.add_argument('--J1', type=float, help='nearest neighbor coupling')
parser.add_argument(
    '--J2', type=float, help='next-to-nearest neighbor coupling')
parser.add_argument(
    '--blocklength',
    type=int,
    help='blocking size for MPO blocking (2)',
    default=2)
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
if not args.N1:
  raise ValueError('no value for N1 given')
if not args.N2:
  raise ValueError('no value for N2 given')
if not args.J1:
  raise ValueError('no value for J1 given')
if not args.J2:
  raise ValueError('no value for J2 given')
if args.load_gate_filename and (args.load_gate_filename[-7::] != '.pickle'):
  raise ValueError('--load_gate_filename should have a .pickle file-ending')

if args.load_mps_filename and (args.load_mps_filename[-7::] != '.pickle'):
  raise ValueError('--load_mps_filename should have a .pickle file-ending')

J1, J2 = args.J1, args.J2
N1, N2 = args.N1, args.N2
block_length = args.blocklength

print('(N1={}, N2={}, j1={}, j2={}): energy = {}'.format(
    N1, N2, args.J1, args.J2, OM.get_energy(J1, J2, N1, N2, block_length, args.load_gate_filename, args.load_mps_filename)))
