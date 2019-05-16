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
functions for binary MERA optimization
"""
import tensorflow as tf
import copy
import numpy as np
import time
import pickle
import tensornetwork as tn
import experiments.MERA.misc_mera as misc_mera
from sys import stdout
import scipy as sp
from scipy.sparse.linalg import eigs, LinearOperator


def right_matvec(isometry, unitary, density):
    """
    This function computes the sum of the left 
    and right descending super-operators. Needed
    for calculating scaling dimensions for the binary mera.

    Args:
        isometry (tf.Tensor): isometry of the binary mera 
        unitary  (tf.Tensor): disentanlger of the mera
        density  (tf.Tensor): reduced density matrix

    Returns: 
        tf.Tensor
    """
    left = left_descending_super_operator(density, isometry, unitary)
    right = right_descending_super_operator(density, isometry, unitary)
    return left + right


def left_matvec(isometry, unitary, density):
    """
    wrapper around ascending_super_operator. Needed
    for calculating scaling dimensions for the binary mera.

    Args:
        isometry (tf.Tensor): isometry of the binary mera 
        unitary  (tf.Tensor): disentanlger of the mera
        density  (tf.Tensor): reduced density matrix

    Returns: 
        tf.Tensor
    """
    return ascending_super_operator(density, isometry, unitary)


def get_scaling_dimensions(isometry, unitary, k=4):
    """
    calculate the scaling dimensions of a binary mera
    Args:
        isometry (tf.Tensor): isometry of the mera
        unitary  (tf.Tensor): disentangler of the mera
        k        (int):       number of scaling dimensions
    Returns:
        tf.Tensor of shape (k,): first k scaling dimensions
    """
    chi = isometry.shape[2]
    dtype = isometry.dtype

    def lmv(vec):
        dens = np.reshape(vec, [chi] * 6).astype(dtype.as_numpy_dtype)
        out = ascending_super_operator(dens, isometry, unitary)
        return np.reshape(np.array(out).astype(dtype.as_numpy_dtype), chi**6)

    def rmv(vec):
        dens = np.reshape(vec, [chi] * 6).astype(dtype.as_numpy_dtype)
        o1 = left_descending_super_operator(dens, isometry, unitary)
        o2 = right_descending_super_operator(dens, isometry, unitary)
        return np.reshape(
            np.array(o1 + o2).astype(dtype.as_numpy_dtype), chni**6)

    A = LinearOperator(shape=(chi**6, chi**6), matvec=lmv, rmatvec=rmv)
    eta, U = sp.sparse.linalg.eigs(A, k=k, which='LM')
    scdims = -np.log2(np.abs(eta))
    return scdims - scdims[0]


def eigs(isometry, unitary, N=10, thresh=1E-6):
    """
    non-hermitian lanczos method for diagonalizing 
    the super-operator of a binary mera network
    Args:
        isometry (tf.Tensor): isometry of the binary mera 
        unitary (tf.Tensor): disentanlger of binary mera
        N (int): number of scaling dimensions
        thresh (float): precision of eigensolver
    Returns:
        (list,list,list): central, lower and upper diagonal part of the tridiagonal matrix
    """
    chi = isometry.shape[2]
    dtype = isometry.dtype
    q_j = tf.random_uniform(shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
    q_j = misc_mera.symmetrize(q_j)
    p_j = copy.copy(q_j)
    Z = misc_mera.scalar_product(p_j, q_j)
    q_j /= tf.sqrt(Z)
    p_j /= tf.sqrt(Z)
    r = right_matvec(isometry, unitary, q_j)
    s = left_matvec(isometry, unitary, p_j)
    alphas, betas, gammas = [], [], []

    for n in range(N):
        alpha_j = misc_mera.scalar_product(p_j, r)
        alphas.append(alpha_j)
        betas.append(0.0)
        gammas.append(0.0)
        r = r - q_j * alpha_j
        s = s - p_j * tf.conj(alpha_j)
        if (tf.norm(r) < thresh) or (tf.norm(s) < thresh):
            break
        w_j = misc_mera.scalar_product(r, s)
        if tf.norm(w_j) < thresh:
            break

        beta_jp1 = tf.sqrt(tf.abs(w_j))
        gamma_jp1 = tf.conj(w_j) / beta_jp1
        betas[-1] = beta_jp1
        gammas[-1] = gamma_jp1
        q_jp1 = r / beta_jp1
        p_jp1 = s / tf.conj(gamma_jp1)
        r = right_matvec(isometry, unitary, q_jp1)
        s = left_matvec(isometry, unitary, p_jp1)
        r = r - q_j * gamma_jp1
        s = s - p_j * tf.conj(beta_jp1)
        q_j = q_jp1
        p_j = p_jp1
    return alphas, betas[0:-1], gammas[0:-1]


#@tf.contrib.eager.defun
def ascending_super_operator(ham, isometry, unitary):
    """
    binary mera ascending super opertor
    Args:
        ham (tf.Tensor): hamiltonian
        isometry (tf.Tensor): isometry of the binary mera 
        unitary  (tf.Tensor): disentanlger of the mera
    Returns:
        tf.Tensor
    """
    return left_ascending_super_operator(
        ham, isometry, unitary) + right_ascending_super_operator(
            ham, isometry, unitary)

@tf.contrib.eager.defun
def two_site_ascending_super_operator(operator, isometry, unitary):
    net = tn.TensorNetwork()
    
    iso_l = net.add_node(isometry)
    iso_r = net.add_node(isometry)
    iso_l_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    op = net.add_node(operator)
    un = net.add_node(unitary)
    un_con = net.add_node(tf.conj(unitary))
    out_order = [iso_l_con[2], iso_r_con[2], iso_l[2], iso_r[2]]
    
    edges = {}
    edges[0] = net.connect(iso_l[0],iso_l_con[0])
    edges[1] = net.connect(iso_r[1],iso_r_con[1])
    edges[2] = net.connect(un_con[0], op[0])
    edges[3] = net.connect(un_con[1], op[1])
    edges[4] = net.connect(op[2], un[0])
    edges[5] = net.connect(op[3], un[1])
    edges[6] = net.connect(iso_l[1], un[2])
    edges[7] = net.connect(iso_r[0], un[3])
    edges[8] = net.connect(iso_l_con[1], un_con[2])
    edges[9] = net.connect(un_con[3], iso_r_con[0])

    out = net.contract_between(un_con, op)
    out = net.contract_between(un,out)
    left = net.contract(edges[0])
    right = net.contract(edges[1])
    out = net.contract_between(left, out)
    out = net.contract_between(right, out)
    
    out.reorder_edges(out_order)
    out.axis_names = [out[n].name for n in range(len(out.get_tensor().shape))]        
    return out

@tf.contrib.eager.defun
def two_site_descending_super_operator(rho, isometry, unitary):
    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry)
    iso_r = net.add_node(isometry)
    iso_l_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    rho = net.add_node(reduced_density)
    un = net.add_node(unitary)
    un_con = net.add_node(tf.conj(unitary))
    
    out_order = [un_con[0], un_con[1], un[0], un[1]]
    
    edges[0] = net.connect(iso_l[0], iso_l_con[0])
    edges[1] = net.connect(iso_r[1], iso_r_con[1])
    edges[2] = net.connect(iso_l[2], rho[0])
    edges[3] = net.connect(iso_l_con[2], rho[2])
    edges[4] = net.connect(iso_r[2], rho[1])
    edges[5] = net.connect(iso_r_con[2], rho[3])
    edges[6] = net.connect(iso_l[1], un[2])
    edges[7] = net.connect(iso_r[0], un[3])
    edges[8] = net.connect(iso_l_con[1], un_con[2])
    edges[9] = net.connect(iso_r_con[0], un_con[3])
    
    left = net.contract(edges[0])
    temp = net.contract_between(left, rho)
    right = net.contract(edges[1])
    temp = net.contract_between(temp,right)
    temp = net.contract_between(temp, un)
    out = net.contract_between(temp,un_con)
    
    out = out.reorder_edges(out_order)
    out.axis_names = [out[n].name for n in range(len(out.get_tensor().shape))]    
    return out

@tf.contrib.eager.defun
def left_ascending_super_operator(hamiltonian, isometry, unitary):
    net = tn.TensorNetwork()
    
    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)
    iso_r = net.add_node(isometry)
    
    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    
    op = net.add_node(hamiltonian)

    un_l = net.add_node(unitary)
    un_l_con = net.add_node(tf.conj(unitary))
    
    un_r = net.add_node(unitary)
    un_r_con = net.add_node(tf.conj(unitary))
    
    out_order = [iso_l_con[2], iso_c_con[2], iso_r_con[2],
                 iso_l[2], iso_c[2], iso_r[2]]
    edges = {}

    edges[1] = net.connect(iso_l[0], iso_l_con[0])
    edges[2] = net.connect(iso_r[1], iso_r_con[1])
    edges[3] = net.connect(un_l[0], op[3])
    edges[4] = net.connect(un_l[1], op[4])    
    edges[5] = net.connect(un_l_con[0], op[0])
    edges[6] = net.connect(un_l_con[1], op[1])
    edges[7] = net.connect(iso_c_con[1], un_r_con[2])
    edges[8] = net.connect(iso_c_con[0], un_l_con[3])
    edges[9] = net.connect(un_r_con[0], op[2])
    edges[10] = net.connect(iso_c[1], un_r[2])
    edges[11] = net.connect(un_l[3], iso_c[0])
    edges[12] = net.connect(un_r[0], op[5])
    edges[13] = net.connect(un_r[1], un_r_con[1])
    edges[14] = net.connect(un_r[3], iso_r[0])
    edges[15] = net.connect(un_r_con[3], iso_r_con[0])
    edges[16] = net.connect(iso_l[1], un_l[2])
    edges[17] = net.connect(iso_l_con[1], un_l_con[2])

    op = net.contract_between(op, un_l)
    op = net.contract_between(op, un_l_con)
    
    lower = net.contract(edges[7])
    op = net.contract_between(lower, op)
    del lower
    
    upper= net.contract(edges[10])
    op = net.contract_between(upper, op)
    del upper

    right = net.contract(edges[2])
    op = net.contract_between(right, op)
    del right

    left = net.contract(edges[1])
    op = net.contract_between(left, op)
    del left

    op.reorder_edges(out_order)
    return op.get_tensor()


@tf.contrib.eager.defun
def right_ascending_super_operator(hamiltonian, isometry, unitary):
    
    net = tn.TensorNetwork()
    
    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)
    iso_r = net.add_node(isometry)
    
    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    
    op = net.add_node(hamiltonian)

    un_l = net.add_node(unitary)
    un_l_con = net.add_node(tf.conj(unitary))
    
    un_r = net.add_node(unitary)
    un_r_con = net.add_node(tf.conj(unitary))
    
    out_order = [iso_l_con[2], iso_c_con[2], iso_r_con[2],
                 iso_l[2], iso_c[2], iso_r[2]]
    
    edges = {}

    edges[1] = net.connect(iso_l[0], iso_l_con[0])
    edges[2] = net.connect(iso_r[1], iso_r_con[1])
    edges[3] = net.connect(un_r[0], op[4])
    edges[4] = net.connect(un_r[1], op[5])    
    edges[5] = net.connect(un_r_con[0], op[1])
    edges[6] = net.connect(un_r_con[1], op[2])
    edges[7] = net.connect(iso_c_con[0], un_l_con[3])
    edges[8] = net.connect(iso_c_con[1], un_r_con[2])
    edges[9] = net.connect(un_l_con[1], op[0])
    edges[10] = net.connect(iso_c[0], un_l[3])
    edges[11] = net.connect(un_r[2], iso_c[1])
    edges[12] = net.connect(un_l[1], op[3])
    edges[13] = net.connect(un_l[0], un_l_con[0])
    edges[14] = net.connect(un_l[2], iso_l[1])
    edges[15] = net.connect(un_l_con[2], iso_l_con[1])
    edges[16] = net.connect(iso_r[0], un_r[3])
    edges[17] = net.connect(iso_r_con[0], un_r_con[3])

    op = net.contract_between(op, un_r)
    op = net.contract_between(op, un_r_con)
    
    lower = net.contract(edges[7])
    op = net.contract_between(lower, op)
    del lower
    
    upper= net.contract(edges[10])
    op = net.contract_between(upper, op)
    del upper

    right = net.contract(edges[2])
    op = net.contract_between(right, op)
    del right

    left = net.contract(edges[1])
    op = net.contract_between(left, op)
    del left

    op.reorder_edges(out_order)
    return op.get_tensor()



@tf.contrib.eager.defun
def left_descending_super_operator(reduced_density, isometry, unitary):
    net = tn.TensorNetwork()
    
    iso_l = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])
    iso_c = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])
    iso_r = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])
    
    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [un_l[0], un_l[1], un_r[0], un_l_con[0], un_l_con[1], un_r_con[0]]
    
    edges={}
    edges[1] = net.connect(iso_l[0],iso_l_con[0])
    edges[2] = net.connect(iso_r[1],iso_r_con[1])
    edges[3] = net.connect(iso_l[2], rho[0])
    edges[4] = net.connect(iso_l_con[2], rho[3])
    edges[5] = net.connect(iso_r[2], rho[2])
    edges[6] = net.connect(iso_r_con[2], rho[5])
    edges[7] = net.connect(iso_c[1], un_r[2])
    edges[8] = net.connect(iso_c[2], rho[1])
    edges[9] = net.connect(un_r[3], iso_r[0])
    edges[10] = net.connect(iso_c_con[1], un_r_con[2])
    edges[11]= net.connect(iso_c_con[2], rho[4])
    edges[12]= net.connect(un_r_con[3], iso_r_con[0])
    edges[13]= net.connect(un_r[1], un_r_con[1])
    edges[14]= net.connect(iso_l[1], un_l[2])
    edges[15]= net.connect(un_l[3], iso_c[0])
    edges[16]= net.connect(iso_l_con[1], un_l_con[2])
    edges[17]= net.connect(un_l_con[3], iso_c_con[0])
    
    left = net.contract(edges[1])
    e = net.flatten_edges_between(rho, left)
    out = net.contract(e)
    del left, rho
    
    right = net.contract(edges[2])
    e = net.flatten_edges_between(out, right)
    out = net.contract(e)
    del right
    
    upper = net.contract(edges[7])
    e = net.flatten_edges_between(out, upper)
    out = net.contract(e)
    del upper

    lower = net.contract(edges[10])
    out = net.contract_between(out, lower)
    del lower

    
    out = net.contract_between(out, un_l)
    
    out = net.contract_between(out, un_l_con)
    out.reorder_edges(out_order)
    return out.get_tensor()

@tf.contrib.eager.defun
def right_descending_super_operator(reduced_density, isometry, unitary):
    net = tn.TensorNetwork()
    
    iso_l = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])
    iso_c = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])
    iso_r = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])
    
    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [un_l[1], un_r[0], un_r[1], un_l_con[1], un_r_con[0], un_r_con[1]]
    
    edges={}
    edges[1] = net.connect(iso_l[0],iso_l_con[0])
    edges[2] = net.connect(iso_r[1],iso_r_con[1])
    edges[3] = net.connect(iso_l[2], rho[0])
    edges[4] = net.connect(iso_l_con[2], rho[3])
    edges[5] = net.connect(iso_r[2], rho[2])
    edges[6] = net.connect(iso_r_con[2], rho[5])
    edges[7] = net.connect(iso_c[1], un_r[2])
    edges[8] = net.connect(iso_c[2], rho[1])
    edges[9] = net.connect(un_r[3], iso_r[0])
    edges[10] = net.connect(iso_c_con[1], un_r_con[2])
    edges[11]= net.connect(iso_c_con[2], rho[4])
    edges[12]= net.connect(un_r_con[3], iso_r_con[0])
    edges[13]= net.connect(un_l[0], un_l_con[0])
    edges[14]= net.connect(iso_l[1], un_l[2])
    edges[15]= net.connect(un_l[3], iso_c[0])
    edges[16]= net.connect(iso_l_con[1], un_l_con[2])
    edges[17]= net.connect(un_l_con[3], iso_c_con[0])
    
    left = net.contract(edges[1])
    e = net.flatten_edges_between(rho, left)
    out = net.contract(e)
    del left, rho
    
    right = net.contract(edges[2])
    out = net.contract_between(out, right)
    del right
    
    upper = net.contract(edges[15])
    out = net.contract_between(out, upper)
    del upper

    lower = net.contract(edges[17])
    out = net.contract_between(out, lower)
    del lower
    
    out = net.contract_between(out, un_r)
    
    out = net.contract_between(out, un_r_con)
    out.reorder_edges(out_order)
    return out.get_tensor()




#@tf.contrib.eager.defun
def descending_super_operator(rho, isometry, unitary):
    rho_1 = right_descending_super_operator(rho, isometry, unitary)
    rho_2 = left_descending_super_operator(rho, isometry, unitary)
    rho = 0.5 * (rho_1 + rho_2)
    rho = misc_mera.symmetrize(rho)
    rho = rho / misc_mera.trace(rho)
    return rho



@tf.contrib.eager.defun
def get_env_disentangler_1(hamiltonian, reduced_density, isometry, unitary):
    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)    
    iso_r = net.add_node(isometry)


    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    
    op = net.add_node(hamiltonian)
    rho = net.add_node(reduced_density)

    
    un_l_con = net.add_node(tf.conj(unitary))
    
    un_r = net.add_node(unitary)
    un_r_con = net.add_node(tf.conj(unitary))
    
    out_order = [op[3], op[4], iso_l[1], iso_c[0] ]
    
    edges={}
    edges[1] = net.connect(iso_r[1],iso_r_con[1])
    edges[2] = net.connect(iso_l[0],iso_l_con[0])
    edges[3] = net.connect(iso_l[2], rho[0])
    edges[4] = net.connect(iso_l_con[2], rho[3])
    edges[5] = net.connect(iso_r[2], rho[2])
    edges[6] = net.connect(iso_r_con[2], rho[5])
    edges[7] = net.connect(iso_c[1],un_r[2])
    edges[8] = net.connect(iso_c[2],rho[1])
    edges[9] = net.connect(op[0],un_l_con[0])
    edges[10] = net.connect(op[1],un_l_con[1])
    edges[11]= net.connect(iso_c_con[1],un_r_con[2])
    edges[12]= net.connect(iso_c_con[2],rho[4])
    edges[13]= net.connect(un_r_con[3],iso_r_con[0])
    edges[14]= net.connect(un_r[3], iso_r[0])
    edges[15]= net.connect(un_r[1], un_r_con[1])
    edges[16]= net.connect(iso_l_con[1], un_l_con[2])
    edges[17]= net.connect(un_l_con[3], iso_c_con[0])
    edges[18]= net.connect(op[2],un_r_con[0])
    edges[19]= net.connect(un_r[0], op[5])


    out = net.contract(edges[1])
    out = net.contract_between(out, rho)
    del rho
    
    left = net.contract(edges[2])
    out = net.contract_between(out, left)    
    del left
    
    lower = net.contract(edges[11])
    out = net.contract_between(out, lower)        
    del lower

    upper = net.contract(edges[7])
    out = net.contract_between(out, upper)            
    del upper

    op = net.contract_between(un_l_con,op)
    out = net.contract_between(out, op)
    del op
    
    out.reorder_edges(out_order)
    return out.get_tensor()







@tf.contrib.eager.defun
def get_env_disentangler_2(hamiltonian, reduced_density, isometry, unitary):
    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)    
    iso_r = net.add_node(isometry)


    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    
    op = net.add_node(hamiltonian)
    rho = net.add_node(reduced_density)

    
    un_l_con = net.add_node(tf.conj(unitary))
    
    un_r = net.add_node(unitary)
    un_r_con = net.add_node(tf.conj(unitary))
    
    out_order = [un_l_con[0], op[3], iso_l[1], iso_c[0]]
    
    edges={}
    edges[1] = net.connect(iso_r[1],iso_r_con[1])
    edges[2] = net.connect(iso_l[0],iso_l_con[0])
    edges[3] = net.connect(iso_l[2], rho[0])
    edges[4] = net.connect(iso_l_con[2], rho[3])
    edges[5] = net.connect(iso_r[2], rho[2])
    edges[6] = net.connect(iso_r_con[2], rho[5])
    edges[7] = net.connect(iso_c[1],un_r[2])
    edges[8] = net.connect(iso_c[2],rho[1])
    edges[9] = net.connect(un_r[1], op[5])
    edges[10] = net.connect(op[0],un_l_con[1])
    edges[11]= net.connect(iso_c_con[1],un_r_con[2])
    edges[12]= net.connect(iso_c_con[2],rho[4])
    edges[13]= net.connect(un_r_con[3],iso_r_con[0])
    edges[14]= net.connect(un_r[3], iso_r[0])
    edges[15]= net.connect(un_r_con[1], op[2])
    edges[16]= net.connect(iso_l_con[1], un_l_con[2])
    edges[17]= net.connect(un_l_con[3], iso_c_con[0])
    edges[18]= net.connect(op[1],un_r_con[0])
    edges[19]= net.connect(un_r[0], op[4])


    out = net.contract(edges[1])
    out = net.contract_between(out, rho)
    del rho
    
    left = net.contract(edges[2])
    out = net.contract_between(out, left)    
    del left

    op = net.contract_between(un_r_con,op)
    op = net.contract_between(un_r, op)
    lower = net.contract(edges[17])
    op = net.contract_between(op, lower) #op is D^7 object now!
    del lower

    out = net.contract_between(out, op)        
    del op 

    out = net.contract_between(out, iso_c)
    
    out.reorder_edges(out_order)
    return out.get_tensor()




@tf.contrib.eager.defun
def get_env_disentangler_3(hamiltonian, reduced_density, isometry, unitary):
    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)
    iso_r = net.add_node(isometry)


    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    
    op = net.add_node(hamiltonian)
    rho = net.add_node(reduced_density)

    un_l = net.add_node(unitary)
    un_l_con = net.add_node(tf.conj(unitary))
    

    un_r_con = net.add_node(tf.conj(unitary))
    
    out_order = [op[5], un_r_con[1], iso_c[1], iso_r[0]]
    
    edges={}
    edges[1] = net.connect(iso_r[1],iso_r_con[1])
    edges[2] = net.connect(iso_l[0],iso_l_con[0])
    edges[3] = net.connect(iso_l[2], rho[0])
    edges[4] = net.connect(iso_l_con[2], rho[3])
    edges[5] = net.connect(iso_r[2], rho[2])
    edges[6] = net.connect(iso_r_con[2], rho[5])
    edges[7] = net.connect(iso_l[1],un_l[2])
    edges[8] = net.connect(iso_c[2],rho[1])
    edges[9] = net.connect(op[0],un_l_con[0])
    edges[10] = net.connect(op[1],un_l_con[1])
    edges[11]= net.connect(iso_c_con[1],un_r_con[2])
    edges[12]= net.connect(iso_c_con[2],rho[4])
    edges[13]= net.connect(un_r_con[3],iso_r_con[0])
    edges[14]= net.connect(un_l[0], op[3])
    edges[15]= net.connect(un_l[1], op[4])
    edges[16]= net.connect(iso_l_con[1], un_l_con[2])
    edges[17]= net.connect(un_l_con[3], iso_c_con[0])
    edges[18]= net.connect(op[2],un_r_con[0])
    edges[19]= net.connect(un_l[3], iso_c[0])

    out = net.contract(edges[1])
    out = net.contract_between(out, rho)
    del rho
    
    left = net.contract(edges[2])
    out = net.contract_between(out, left)    
    del left

    op = net.contract_between(un_l_con,op)
    op = net.contract_between(un_l, op)
    lower = net.contract(edges[11])
    op = net.contract_between(op, lower) #op is D^7 object now!
    del lower

    out = net.contract_between(out, op)        
    del op 

    out = net.contract_between(out, iso_c)
    
    out.reorder_edges(out_order)
    return out.get_tensor()





@tf.contrib.eager.defun
def get_env_disentangler_4(hamiltonian, reduced_density, isometry, unitary):
    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)    
    iso_r = net.add_node(isometry)


    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))
    
    op = net.add_node(hamiltonian)
    rho = net.add_node(reduced_density)

    un_l = net.add_node(unitary)
    un_l_con = net.add_node(tf.conj(unitary))
    
    un_r_con = net.add_node(tf.conj(unitary))
    
    out_order = [op[4], op[5], iso_c[1], iso_r[0]]
    
    edges={}
    edges[1] = net.connect(iso_r[1],iso_r_con[1])
    edges[2] = net.connect(iso_l[0],iso_l_con[0])
    edges[3] = net.connect(iso_l[2], rho[0])
    edges[4] = net.connect(iso_l_con[2], rho[3])
    edges[5] = net.connect(iso_r[2], rho[2])
    edges[6] = net.connect(iso_r_con[2], rho[5])
    edges[7] = net.connect(iso_l[1],un_l[2])
    edges[8] = net.connect(iso_c[2],rho[1])
    edges[9] = net.connect(un_l[1], op[3])
    edges[10] = net.connect(op[0],un_l_con[1])
    edges[11]= net.connect(iso_c_con[1],un_r_con[2])
    edges[12]= net.connect(iso_c_con[2],rho[4])
    edges[13]= net.connect(un_r_con[3],iso_r_con[0])
    edges[14]= net.connect(un_l[3], iso_c[0])
    edges[15]= net.connect(un_r_con[1], op[2])
    edges[16]= net.connect(iso_l_con[1], un_l_con[2])
    edges[17]= net.connect(un_l_con[3], iso_c_con[0])
    edges[18]= net.connect(op[1],un_r_con[0])
    edges[19]= net.connect(un_l[0], un_l_con[0])


    out = net.contract(edges[1])
    out = net.contract_between(out, rho)
    del rho
    
    left = net.contract(edges[2])
    out = net.contract_between(out, left)    
    del left

    op = net.contract_between(un_r_con,op)
    
    lower = net.contract(edges[17])
    out = net.contract_between(out, lower)
    del lower

    upper = net.contract(edges[14])
    out = net.contract_between(out, upper)
    del upper
    
    out = net.contract_between(out, op)        
    del op 

    out.reorder_edges(out_order)
    return out.get_tensor()





def get_env_disentangler(ham, rho, isometry, unitary):
    env_1 = get_env_disentangler_1(ham, rho, isometry, unitary)
    env_2 = get_env_disentangler_2(ham, rho, isometry, unitary)
    env_3 = get_env_disentangler_3(ham, rho, isometry, unitary)
    env_4 = get_env_disentangler_4(ham, rho, isometry, unitary)
    return env_1 + env_2 + env_3 + env_4

@tf.contrib.eager.defun
def get_env_isometry_1(hamiltonian, reduced_density, isometry, unitary):

    net = tn.TensorNetwork()

    iso_c = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])
    iso_r = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])
    
    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    op = net.add_node(hamiltonian,axis_names = ['op_0', 'op_1', 'op_2','op_3','op_4','op_5'])
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    
    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [iso_l_con[0], un_l[2], rho[0]]
    
    edges={}
    edges[1] = net.connect(iso_r[1],iso_r_con[1])
    edges[2] = net.connect(op[3],un_l[0])
    edges[3] = net.connect(op[4],un_l[1])
    edges[4] = net.connect(op[0],un_l_con[0])
    edges[5] = net.connect(op[1],un_l_con[1]) 
    edges[6] = net.connect(rho[2],iso_r[2]) 
    edges[7] = net.connect(rho[5],iso_r_con[2]) 
    edges[8] = net.connect(iso_c_con[1], un_r_con[2])
    edges[9] = net.connect(iso_c[1], un_r[2])
    edges[10] = net.connect(rho[4],iso_c_con[2])
    edges[11]= net.connect(iso_r_con[0],un_r_con[3])
    edges[12]= net.connect(rho[1],iso_c[2])
    edges[13]= net.connect(un_r[3],iso_r[0])
    edges[14]= net.connect(un_r[1],un_r_con[1])
    edges[15]= net.connect(un_l[3],iso_c[0])
    edges[16]= net.connect(op[5],un_r[0])
    edges[17]= net.connect(un_l_con[3],iso_c_con[0])
    edges[18]= net.connect(op[2],un_r_con[0])
    edges[19]= net.connect(iso_l_con[2],rho[3])
    edges[20]= net.connect(iso_l_con[1],un_l_con[2])
    

    out = net.contract(edges[1])
    e = net.flatten_edges_between(out, rho)
    out = net.contract(e)
    del rho
    
    lower = net.contract(edges[8])
    e = net.flatten_edges_between(lower, out)
    out = net.contract(e)
    del lower
    
    upper = net.contract(edges[9])
    e = net.flatten_edges_between(upper, out)
    out = net.contract(e)    
    del upper
    
    op = net.contract_between(un_l,op)
    op = net.contract_between(op,un_l_con)
    e = net.flatten_edges_between(op, out)
    out = net.contract(e)
    del op
    
    out = net.contract_between(iso_l_con,out)
    out.reorder_edges(out_order)
    return out.get_tensor()



@tf.contrib.eager.defun
def get_env_isometry_2(hamiltonian, reduced_density, isometry, unitary):

    net = tn.TensorNetwork()

    iso_c = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])
    iso_r = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])

    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    op = net.add_node(hamiltonian,axis_names = ['op_0', 'op_1', 'op_2','op_3','op_4','op_5'])
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    
    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [iso_l_con[0], un_l[2], rho[0]]
    
    edges={}

    edges[1] = net.connect(iso_r[1], iso_r_con[1])
    edges[2] = net.connect(iso_r[2], rho[2])
    edges[3] = net.connect(iso_l_con[2], rho[3])
    edges[4] = net.connect(op[4], un_r[0])
    edges[5] = net.connect(op[5], un_r[1])    
    edges[6] = net.connect(op[1], un_r_con[0])
    edges[7] = net.connect(op[2], un_r_con[1])
    edges[8] = net.connect(iso_c[0], un_l[3])
    edges[9]= net.connect(iso_c_con[0], un_l_con[3])
    edges[10]= net.connect(iso_l_con[1],un_l_con[2])
    edges[11]= net.connect(iso_c_con[2], rho[4])
    edges[12]= net.connect(iso_c[2], rho[1])
    edges[13]= net.connect(iso_r[0], un_r[3])
    edges[14]= net.connect(un_l[0], un_l_con[0])
    edges[15] = net.connect(un_l[1], op[3])
    edges[16] = net.connect(iso_c[1], un_r[2])
    edges[17]= net.connect(un_l_con[1], op[0])
    edges[18]= net.connect(un_r_con[2], iso_c_con[1])
    edges[19]= net.connect(un_r_con[3], iso_r_con[0])
    edges[20]= net.connect(iso_r_con[2], rho[5])
    
    upper = net.contract(edges[8])
    op = net.contract_between(un_r,op)
    op = net.contract_between(op,un_r_con)
    e = net.flatten_edges_between(op, upper)
    op = net.contract(e)
    del upper
    
    lower = net.contract(edges[9])  
    e = net.flatten_edges_between(op, lower)
    op = net.contract(e)
    del lower

    left = net.contract(edges[1])
    e = net.flatten_edges_between(left, rho)
    out = net.contract(e)
    del rho
    
    e = net.flatten_edges_between(out, op)
    out = net.contract(e)
    del op
    
    e = net.flatten_edges_between(out, iso_l_con)
    out = net.contract(e)

    out.reorder_edges(out_order)    
    return out.get_tensor()


@tf.contrib.eager.defun
def get_env_isometry_3(hamiltonian, reduced_density, isometry, unitary):

    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])
    iso_r = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])


    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    op = net.add_node(hamiltonian,axis_names = ['op_0', 'op_1', 'op_2','op_3','op_4','op_5'])
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    
    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [un_l[3], un_r[2], rho[1]]
    
    edges={}
    edges[1] = net.connect(iso_r[1],iso_r_con[1])
    edges[2] = net.connect(iso_l[0],iso_l_con[0])
    edges[3] = net.connect(iso_l[2], rho[0])
    edges[4] = net.connect(iso_l_con[2], rho[3])
    edges[5] = net.connect(iso_r[2], rho[2])
    edges[6] = net.connect(iso_r_con[2], rho[5])
    edges[7] = net.connect(op[3],un_l[0])
    edges[8] = net.connect(op[4],un_l[1])    
    edges[9] = net.connect(op[0],un_l_con[0])
    edges[10] = net.connect(op[1],un_l_con[1])
    edges[11]= net.connect(iso_c_con[1],un_r_con[2])
    edges[12]= net.connect(iso_c_con[2],rho[4])
    edges[13]= net.connect(un_r_con[3],iso_r_con[0])
    edges[14]= net.connect(un_r[3], iso_r[0])
    edges[15]= net.connect(un_r[1], un_r_con[1])
    edges[16]= net.connect(iso_l_con[1], un_l_con[2])
    edges[17]= net.connect(un_l_con[3], iso_c_con[0])
    edges[18]= net.connect(op[2],un_r_con[0])
    edges[19]= net.connect(iso_l[1], un_l[2])
    edges[20]= net.connect(op[5], un_r[0])
    

    out = net.contract(edges[1])
    e = net.flatten_edges_between(out, rho)
    out = net.contract(e)

    left = net.contract(edges[2])    
    e = net.flatten_edges_between(out, left)
    out = net.contract(e)
    del left
    
    lower = net.contract(edges[11])
    e = net.flatten_edges_between(lower, out)
    out = net.contract(e)
    del lower
    
    out = net.contract_between(out, un_r)
    op = net.contract_between(un_l,op)
    op = net.contract_between(op, un_l_con)
    out = net.contract_between(op, out)
    del op
    
    out.reorder_edges(out_order)
    return out.get_tensor()


@tf.contrib.eager.defun
def get_env_isometry_4(hamiltonian, reduced_density, isometry, unitary):

    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])
    iso_r = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])


    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    op = net.add_node(hamiltonian,axis_names = ['op_0', 'op_1', 'op_2','op_3','op_4','op_5'])
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    
    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [un_l[3], un_r[2], rho[1]]
    
    edges={}

    edges[1] = net.connect(iso_l[0], iso_l_con[0])
    edges[2] = net.connect(iso_l[2], rho[0])
    edges[3] = net.connect(iso_l_con[2], rho[3])
    edges[4] = net.connect(op[4], un_r[0])
    edges[5] = net.connect(op[5], un_r[1])    
    edges[6] = net.connect(op[1], un_r_con[0])
    edges[7] = net.connect(op[2], un_r_con[1])
    edges[8] = net.connect(iso_r[0], un_r[3])
    edges[9]= net.connect(iso_c_con[0], un_l_con[3])
    edges[10]= net.connect(iso_l_con[1],un_l_con[2])
    edges[11]= net.connect(iso_c_con[2], rho[4])
    edges[12]= net.connect(iso_r[2], rho[2])
    edges[13]= net.connect(iso_l[1], un_l[2])
    edges[14]= net.connect(un_l[0], un_l_con[0])
    edges[15] = net.connect(un_l[1], op[3])
    edges[16] = net.connect(iso_r[1], iso_r_con[1])
    edges[17]= net.connect(un_l_con[1], op[0])
    edges[18]= net.connect(un_r_con[2], iso_c_con[1])
    edges[19]= net.connect(un_r_con[3], iso_r_con[0])
    edges[20]= net.connect(iso_r_con[2], rho[5])
    

    
    op = net.contract_between(un_r,op)
    op = net.contract_between(op,un_r_con)

    left = net.contract(edges[1])
    e = net.flatten_edges_between(left, rho)
    out = net.contract(e)
    del left, rho
    
    right = net.contract(edges[16])
    e = net.flatten_edges_between(right, out)
    out = net.contract(e)    
    del right

    lower = net.contract(edges[9])      
    e = net.flatten_edges_between(out, lower)
    out = net.contract(e)
    del lower
    
    e = net.flatten_edges_between(out, un_l)
    out = net.contract(e)
    e = net.flatten_edges_between(out, op)
    out = net.contract(e)
    
    out.reorder_edges(out_order)    
    return out.get_tensor()


@tf.contrib.eager.defun
def get_env_isometry_5(hamiltonian, reduced_density, isometry, unitary):

    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])
    iso_c = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])


    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    op = net.add_node(hamiltonian,axis_names = ['op_0', 'op_1', 'op_2','op_3','op_4','op_5'])
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    
    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [un_r[3], iso_r_con[1], rho[2]]
    
    edges={}

    edges[1] = net.connect(iso_l[0], iso_l_con[0])
    edges[2] = net.connect(iso_l[2], rho[0])
    edges[3] = net.connect(iso_l_con[2], rho[3])
    edges[4] = net.connect(op[3], un_l[0])
    edges[5] = net.connect(op[4], un_l[1])    
    edges[6] = net.connect(op[0], un_l_con[0])
    edges[7] = net.connect(op[1], un_l_con[1])
    edges[8] = net.connect(iso_c[1], un_r[2])
    edges[9] = net.connect(iso_c[0], un_l[3])    
    edges[10] = net.connect(un_r[0], op[5])
    edges[11]= net.connect(un_r_con[2], iso_c_con[1])
    edges[12]= net.connect(iso_c_con[0], un_l_con[3])    
    edges[13]= net.connect(op[2], un_r_con[0])
    edges[14]= net.connect(un_r[1], un_r_con[1])
    edges[15]= net.connect(iso_l[1], un_l[2])
    edges[16]= net.connect(iso_c[2], rho[1])
    edges[17]= net.connect(iso_l_con[1],un_l_con[2])
    edges[18]= net.connect(iso_c_con[2], rho[4])
    edges[19]= net.connect(un_r_con[3], iso_r_con[0])
    edges[20]= net.connect(iso_r_con[2], rho[5])
    


    
    op = net.contract_between(un_l,op)
    op = net.contract_between(op,un_l_con)
    upper = net.contract(edges[8])    
    e = net.flatten_edges_between(op, upper)
    op = net.contract(e)
    del upper

    lower = net.contract(edges[11])    
    e = net.flatten_edges_between(op,  lower)
    op = net.contract(e)
    del lower
    
    left = net.contract(edges[1])
    e = net.flatten_edges_between(left, rho)
    left = net.contract(e)
    del rho
    
    e = net.flatten_edges_between(left, op)
    out = net.contract(e)
    del left, op
    
    e = net.flatten_edges_between(out, iso_r_con)
    out = net.contract(e)
    
    out.reorder_edges(out_order)
    return out.get_tensor()


@tf.contrib.eager.defun
def get_env_isometry_6(hamiltonian, reduced_density, isometry, unitary):

    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry,axis_names = ['ic_0', 'ic_1', 'ic_2'])
    iso_c = net.add_node(isometry,axis_names = ['ir_0', 'ir_1', 'ir_2'])


    iso_l_con = net.add_node(tf.conj(isometry),axis_names = ['il_c_0', 'il_c_1', 'il_c_2'])
    iso_c_con = net.add_node(tf.conj(isometry),axis_names = ['ic_c_0', 'ic_c_1', 'ic_c_2'])
    iso_r_con = net.add_node(tf.conj(isometry),axis_names = ['ir_c_0', 'ir_c_1', 'ir_c_2'])
    
    op = net.add_node(hamiltonian,axis_names = ['op_0', 'op_1', 'op_2','op_3','op_4','op_5'])
    rho = net.add_node(reduced_density,axis_names = ['rho_0', 'rho_1', 'rho_2','rho_3','rho_4','rho_5'])

    
    un_l = net.add_node(unitary,axis_names = ['ul_0', 'ul_1', 'ul_2','ul_3'])
    un_l_con = net.add_node(tf.conj(unitary),axis_names = ['ul_c_0', 'ul_c_1', 'ul_c_2','ul_c_3'])
    
    un_r = net.add_node(unitary,axis_names = ['ur_0', 'ur_1', 'ur_2','ur_3'])
    un_r_con = net.add_node(tf.conj(unitary),axis_names = ['ur_c_0', 'ur_c_1', 'ur_c_2','ur_c_3'])
    
    out_order = [un_r[3], iso_r_con[1], rho[2]]
    
    edges={}

    edges[1] = net.connect(iso_l[0], iso_l_con[0])
    edges[2] = net.connect(iso_l[2], rho[0])
    edges[3] = net.connect(iso_l_con[2], rho[3])
    edges[4] = net.connect(op[4], un_r[0])
    edges[5] = net.connect(op[5], un_r[1])    
    edges[6] = net.connect(op[1], un_r_con[0])
    edges[7] = net.connect(op[2], un_r_con[1])
    edges[8] = net.connect(iso_c[0], un_l[3])
    edges[9]= net.connect(iso_c_con[0], un_l_con[3])
    edges[10]= net.connect(iso_l_con[1],un_l_con[2])
    edges[11]= net.connect(iso_c_con[2], rho[4])
    edges[12]= net.connect(iso_c[2], rho[1])
    edges[13]= net.connect(iso_l[1], un_l[2])
    edges[14]= net.connect(un_l[0], un_l_con[0])
    edges[15] = net.connect(un_l[1], op[3])
    edges[16] = net.connect(iso_c[1], un_r[2])
    edges[17]= net.connect(un_l_con[1], op[0])
    edges[18]= net.connect(un_r_con[2], iso_c_con[1])
    edges[19]= net.connect(un_r_con[3], iso_r_con[0])
    edges[20]= net.connect(iso_r_con[2], rho[5])
    


    
    op = net.contract_between(un_r,op)
    op = net.contract_between(op,un_r_con)

    left = net.contract(edges[1])
    e = net.flatten_edges_between(left, rho)
    out = net.contract(e)
    del left, rho

    lower = net.contract(edges[9])      
    e = net.flatten_edges_between(out, lower)
    out = net.contract(e)
    del lower

    upper = net.contract(edges[8])    
    e = net.flatten_edges_between(out, upper)
    out = net.contract(e)
    del upper
    
    e = net.flatten_edges_between(out, op)
    out = net.contract(e)
    del op
    
    e = net.flatten_edges_between(out, iso_r_con)
    out = net.contract(e)

    out.reorder_edges(out_order)    
    return out.get_tensor()



def get_env_isometry(ham, rho, isometry, unitary):
    env_1 = get_env_isometry_1(ham, rho, isometry, unitary)
    env_2 = get_env_isometry_2(ham, rho, isometry, unitary)
    env_3 = get_env_isometry_3(ham, rho, isometry, unitary)
    env_4 = get_env_isometry_4(ham, rho, isometry, unitary)
    env_5 = get_env_isometry_5(ham, rho, isometry, unitary)
    env_6 = get_env_isometry_6(ham, rho, isometry, unitary)
    return env_1 + env_2 + env_3 + env_4 + env_5 + env_6


def steady_state_density_matrix(nsteps, rho, isometry, unitary, verbose=0):
    """
    obtain steady state density matrix of the scale invariant binary MERA
    Parameters:
    ------------------------
    nsteps:     int 
    rho:        tf.Tensor 
                reduced density matrix
    isometry:   tf.Tensor 
                isometry of the mera
    unitary:    tf.Tensor 
                disentangler of the mera
    verbose:    int 
                verbosity flag
    Returns: 
    ------------------------
    tf.Tensor
    """
    for n in range(nsteps):
        if verbose > 0:
            stdout.write('\r binary-mera stead-state rdm iteration = %i' % (n))
            stdout.flush()
        rho_new = descending_super_operator(rho, isometry, unitary)
        rho_new = misc_mera.symmetrize(rho_new)
        rho_new = rho_new /misc_mera.trace(rho_new)
        rho = rho_new
    return rho


def unlock_layer(wC, uC, noise=0.0):
    wC.append(copy.copy(wC[-1]))
    uC.append(copy.copy(uC[-1]))
    wC[-1] += (tf.random_uniform(
        shape=wC[-1].shape, minval=-1, maxval=1, dtype=wC[-1].dtype) * noise)
    uC[-1] += (tf.random_uniform(
        shape=uC[-1].shape, minval=-1, maxval=1, dtype=uC[-1].dtype) * noise)
    return wC, uC


def increase_bond_dimension_by_adding_layers(chi_new, wC, uC, noise=0.0):
    """
    increase the bond dimension of the MERA to `chi_new`
    by padding tensors in the last layer with zeros. If the desired `chi_new` cannot
    be obtained from padding, adds layers of Tensors
    the last layer is guaranteed to have uniform bond dimension
    Parameters:
    --------------------------------
    chi_new:         int 
                     new bond dimenion
    wC, uC:         list of tf.Tensor 
                     MERA isometries and disentanglers
    Returns: 
    --------------------------------
    (wC, uC):       list of tf.Tensors
    """
    if misc_mera.all_same_chi(wC[-1], uC[-1]) and (wC[-1].shape[2] >= chi_new):
        #nothing to do here
        return wC, uC
    elif misc_mera.all_same_chi(wC[-1], uC[-1]) and (wC[-1].shape[2] < chi_new):
        chi = min(chi_new, wC[-1].shape[0] * wC[-1].shape[1])
        wC[-1] = misc_mera.pad_tensor(wC[-1],
                                      [wC[-1].shape[0], wC[-1].shape[1], chi])
        wC_temp = copy.deepcopy(wC[-1])
        uC_temp = copy.deepcopy(uC[-1])
        wC.append(misc_mera.pad_tensor(wC_temp, [chi, chi, chi]))
        uC.append(misc_mera.pad_tensor(uC_temp, [chi, chi, chi, chi]))
        wC[-1] += (tf.random_uniform(
            shape=wC[-1].shape, minval=-1, maxval=1, dtype=wC[-1].dtype) *
                   noise)
        uC[-1] += (tf.random_uniform(
            shape=uC[-1].shape, minval=-1, maxval=1, dtype=uC[-1].dtype) *
                   noise)
        return increase_bond_dimension_by_adding_layers(chi_new, wC, uC)

    elif not misc_mera.all_same_chi(wC[-1], uC[-1]):
        raise ValueError('chis of last layer have to be all the same!')


def pad_mera_tensors(chi_new, wC, uC, noise=0.0):
    """
    increase the bond dimension of the MERA to `chi_new`
    by padding tensors in all layers with zeros. If the desired `chi_new` cannot
    be obtained from padding, adds layers of Tensors
    the last layer is guaranteed to have uniform bond dimension
    Parameters:
    --------------------------------
    chi_new:         int 
                     new bond dimenion
    wC, uC:          list of tf.Tensor 
                     MERA isometries and disentanglers
    Returns: 
    --------------------------------
    (wC, uC):       list of tf.Tensors
    """

    all_chis = [t.shape[n] for t in wC for n in range(len(t.shape))]
    if not np.all([c <= chi_new for c in all_chis]):
        #nothing to increase
        return wC, uC

    chi_0 = wC[0].shape[0]
    wC[0] = misc_mera.pad_tensor(wC[0], [chi_0, chi_0, min(chi_new, chi_0**2)])

    for n in range(1, len(wC)):
        wC[n] = misc_mera.pad_tensor(wC[n], [
            min(chi_new, chi_0**(2**n)),
            min(chi_new, chi_0**(2**n)),
            min(chi_new, chi_0**(2**(n + 1)))
        ])
        uC[n] = misc_mera.pad_tensor(uC[n], [
            min(chi_new, chi_0**(2**n)),
            min(chi_new, chi_0**(2**n)),
            min(chi_new, chi_0**(2**n)),
            min(chi_new, chi_0**(2**n))
        ])

        wC[n] += (
            tf.random_uniform(shape=wC[n].shape, dtype=wC[n].dtype) * noise)
        uC[n] += (
            tf.random_uniform(shape=uC[n].shape, dtype=uC[n].dtype) * noise)
    n = len(wC)
    while not misc_mera.all_same_chi(wC[-1]):
        wC.append(
            misc_mera.pad_tensor(wC[-1], [
                min(chi_new, chi_0**(2**n)),
                min(chi_new, chi_0**(2**n)),
                min(chi_new, chi_0**(2**(n + 1)))
            ]))
        uC.append(
            misc_mera.pad_tensor(uC[-1], [
                min(chi_new, chi_0**(2**n)),
                min(chi_new, chi_0**(2**n)),
                min(chi_new, chi_0**(2**n)),
                min(chi_new, chi_0**(2**n))
            ]))

        wC[-1] += (tf.random_uniform(
            shape=wC[-1].shape, minval=-1, maxval=1, dtype=wC[-1].dtype) *
                   noise)
        uC[-1] += (tf.random_uniform(
            shape=uC[-1].shape, minval=-1, maxval=1, dtype=uC[-1].dtype) *
                   noise)
        n += 1

    return wC, uC


def initialize_binary_MERA(phys_dim, chi, dtype=tf.float64):
    """
    initialize a binary MERA network of bond dimension `chi`
    isometries and disentanglers are initialized with identies
    
    Parameters:
    -------------------
    phys_dim:         int 
                      Hilbert space dimension of the bottom layer
    chi:              int 
                      maximum bond dimension
    dtype:            tensorflow dtype
                      dtype of the MERA tensors
    Returns:
    -------------------
    (wC, uC, rho)
    wC, uC:      list of tf.Tensor
    rho:         tf.Tensor
    """
    wC = []
    uC = []
    n = 0
    if chi < phys_dim:
        raise ValueError(
            'cannot initialize a MERA with chi < physical dimension!')
    while True:
        wC.append(
            tf.reshape(
                tf.eye(
                    min(phys_dim**(2**(n + 1)), chi**2),
                    min(phys_dim**(2**(n + 1)), chi),
                    dtype=dtype),
                (min(phys_dim**(2**n), chi), min(phys_dim**(2**n), chi),
                 min(phys_dim**(2**(n + 1)), chi))))
        uC.append(
            tf.reshape(
                tf.eye(min(phys_dim**(2**(n + 1)), chi**2), dtype=dtype),
                (min(phys_dim**(2**n), chi), min(phys_dim**(2**n), chi),
                 min(phys_dim**(2**n), chi), min(phys_dim**(2**n), chi))))
        n += 1
        if misc_mera.all_same_chi(wC[-1]):
            break

    chi_top = wC[-1].shape[2]
    rho = tf.reshape(
        tf.eye(chi_top * chi_top * chi_top, dtype=dtype),
        (chi_top, chi_top, chi_top, chi_top, chi_top, chi_top))

    return wC, uC, rho / misc_mera.trace(rho)


def initialize_TFI_hams(dtype=tf.float64):
    """
    initialize a transverse field ising hamiltonian
    Args:
      dtype:  tensorflow dtype
    Returns:
      tf.Tensor
    """
    sX = np.array([[0, 1], [1, 0]]).astype(dtype.as_numpy_dtype)
    sZ = np.array([[1, 0], [0, -1]]).astype(dtype.as_numpy_dtype)
    eye = np.eye(2).astype(dtype.as_numpy_dtype)
    
    net = tn.TensorNetwork()
    X1 = net.add_node(sX)
    X2 = net.add_node(sX)
    I3 = net.add_node(eye)
    out_order = [X1[0], X2[0], I3[0], X1[1], X2[1], I3[1]]
    t1 = net.outer_product(net.outer_product(X1,X2), I3)
    t1 = t1.reorder_edges(out_order).get_tensor()
    
    net = tn.TensorNetwork()
    Z1 = net.add_node(sZ)
    I2 = net.add_node(eye)    
    I3 = net.add_node(eye)
    out_order = [Z1[0], I2[0], I3[0], Z1[1], I2[1], I3[1]]
    t2 = net.outer_product(net.outer_product(Z1, I2), I3)
    t2 = t2.reorder_edges(out_order).get_tensor()/2
                 
    net = tn.TensorNetwork()
    I1 = net.add_node(eye)    
    Z2 = net.add_node(sZ)
    I3 = net.add_node(eye)
    out_order = [I1[0], Z2[0], I3[0], I1[1], Z2[1], I3[1]]
    t3 = net.outer_product(net.outer_product(I1, Z2), I3)
    t3 = t3.reorder_edges(out_order).get_tensor()/2
                 
    ham = t1 + t2 + t3
    
    ####################   equivalent using ncon   ##################
    # sX = np.array([[0, 1], [1, 0]]).astype(dtype.as_numpy_dtype)
    # sZ = np.array([[1, 0], [0, -1]]).astype(dtype.as_numpy_dtype)
    # eye = np.eye(2).astype(dtype.as_numpy_dtype)
    # ham = tn.ncon([sX, sX, eye],[[-4, -1], [-5, -2], [-6, -3]])+\
    #     tn.ncon([sZ, eye, eye],[[-4, -1], [-5, -2], [-6, -3]])/2+\
    #     tn.ncon([eye, sZ, eye],[[-4, -1], [-5, -2], [-6, -3]])/2
    #################################################################
    return ham


def optimize_binary_mera(ham_0,
                         wC,
                         uC,
                         rho_0=0,
                         numiter=1000,
                         nsteps_steady_state=8,
                         verbose=0,
                         opt_u=True,
                         opt_w=True,
                         numpy_update=True,
                         opt_all_layers=False,
                         opt_u_after=40):
    """
    ------------------------
    optimization of a scale invariant binary MERA tensor network
    Parameters:
    ----------------------------
    ham_0:                 tf.Tensor
                           bottom-layer Hamiltonian
    wC, uC:                list of tf.Tensor 
                           isometries (wC) and disentanglers (uC) of the MERA, with 
                           bottom layers first 
    rho_0:                 tf.Tensor 
                           initial value for steady-state density matrix
    numiter:               int 
                           number of iteration steps 
    nsteps_steady_state:   int 
                           number of power-methodf iteration steps for calculating the 
                           steady state density matrices 
    verbose:               int 
                           verbosity flag 
    opt_u, opt_uv:         bool 
                           if False, skip unitary or isometry optimization 
    numpy_update:          bool
                           if True, use numpy svd to calculate update of disentanglers and isometries
    opt_all_layers:        bool
                           if True, optimize all layers
                           if False, optimize only truncating layers
    opt_u_after:           int 
                           start optimizing disentangler only after `opt_u_after` initial optimization steps
    Returns: 
    -------------------------------
    (wC, uC, rho, run_times, Energies)
    wC, uC:                 list of tf.Tensor 
                            obtimized MERA tensors
    rho:                    tf.Tensor 
                            steady state density matrices at the top layer 
    run_times:              list 
                            run times per iteration step 
    Energies:               list 
                            energies at each iteration step
    """
    dtype = ham_0.dtype

    ham = [0 for x in range(len(wC) + 1)]
    rho = [0 for x in range(len(wC) + 1)]
    ham[0] = ham_0

    chi1 = ham[0].shape[0]

    bias = tf.math.reduce_max(
        tf.linalg.eigvalsh(
            tf.reshape(ham[0], (chi1 * chi1 * chi1, chi1 * chi1 * chi1)))) / 2
    ham[0] = ham[0] - bias * tf.reshape(
        tf.eye(chi1 * chi1 * chi1, dtype=dtype),
        (chi1, chi1, chi1, chi1, chi1, chi1))

    skip_layer = [misc_mera.skip_layer(w) for w in wC]
    for p in range(len(wC)):
        if skip_layer[p]:
            ham[p + 1] = ascending_super_operator(ham[p], wC[p], uC[p])

    Energies = []
    run_times = []

    if rho_0 == 0:
        chi_max = wC[-1].shape[2]
        rho_0 = tf.reshape(
            tf.eye(chi_max**3, dtype=dtype),
            (chi_max, chi_max, chi_max, chi_max, chi_max, chi_max))

    for k in range(numiter):
        t1 = time.time()
        rho_0 = steady_state_density_matrix(nsteps_steady_state, rho_0, wC[-1],
                                            uC[-1])
        rho[-1] = rho_0
        for p in range(len(rho) - 2, -1, -1):
            rho[p] = descending_super_operator(rho[p + 1], wC[p], uC[p])

        if verbose > 0:
            if np.mod(k, 10) == 1:
                Z = misc_mera.trace(rho[0])
                net = tn.TensorNetwork()
                r = net.add_node(rho[0])
                h = net.add_node(ham[0])
                edges = [net.connect(r[n], h[n]) for n in range(6)]
                Energies.append(net.contract_between(r,h).get_tensor()/Z + bias)
                stdout.write(
                    '\r     Iteration: %i of %i: E = %.8f, err = %.16f at D = %i with %i layers'
                    % (int(k), int(numiter), float(Energies[-1]),
                       float(Energies[-1] + 4 / np.pi,), int(wC[-1].shape[2]),
                       len(wC)))
                stdout.flush()

        for p in range(len(wC)):

            if (not opt_all_layers) and skip_layer[p]:
                continue
            if k >= opt_u_after:
                uEnv = get_env_disentangler(ham[p], rho[p + 1], wC[p], uC[p])
                if opt_u:
                    if numpy_update:
                        uC[p] = misc_mera.u_update_svd_numpy(uEnv)
                    else:
                        uC[p] = misc_mera.u_update_svd(uEnv)

            wEnv = get_env_isometry(ham[p], rho[p + 1], wC[p], uC[p])
            if opt_w:
                if numpy_update:
                    wC[p] = misc_mera.w_update_svd_numpy(wEnv)
                else:
                    wC[p] = misc_mera.w_update_svd(wEnv)

            ham[p + 1] = ascending_super_operator(ham[p], wC[p], uC[p])

        run_times.append(time.time() - t1)
        if verbose > 2:
            print('time per iteration: ', run_times[-1])

    return wC, uC, rho[-1], run_times, Energies


def optimize_binary_mera_scale_invariant(ham_0,
                                         wC,
                                         uC,
                                         rho_0=0,
                                         numiter=1000,
                                         nsteps_steady_state=8,
                                         verbose=0,
                                         opt_u=True,
                                         opt_w=True,
                                         numpy_update=True,
                                         opt_all_layers=False,
                                         opt_u_after=40):
    """
    ------------------------
    optimization of a scale invariant binary MERA tensor network
    Parameters:
    ----------------------------
    ham_0:                 tf.Tensor
                           bottom-layer Hamiltonian
    wC, uC:                list of tf.Tensor 
                           isometries (wC) and disentanglers (uC) of the MERA, with 
                           bottom layers first 
    rho_0:                 tf.Tensor 
                           initial value for steady-state density matrix
    numiter:               int 
                           number of iteration steps 
    nsteps_steady_state:   int 
                           number of power-methodf iteration steps for calculating the 
                           steady state density matrices 
    verbose:               int 
                           verbosity flag 
    opt_u, opt_uv:         bool 
                           if False, skip unitary or isometry optimization 
    numpy_update:          bool
                           if True, use numpy svd to calculate update of disentanglers and isometries
    opt_all_layers:        bool
                           if True, optimize all layers
                           if False, optimize only truncating layers
    opt_u_after:           int 
                           start optimizing disentangler only after `opt_u_after` initial optimization steps
    Returns: 
    -------------------------------
    (wC, uC, rho, run_times, Energies)
    wC, uC:                 list of tf.Tensor 
                            obtimized MERA tensors
    rho:                    tf.Tensor 
                            steady state density matrices at the top layer 
    run_times:              list 
                            run times per iteration step 
    Energies:               list 
                            energies at each iteration step
    """
    dtype = ham_0.dtype

    ham = ham_0

    chi1 = ham[0].shape[0]

    bias = tf.math.reduce_max(
        tf.linalg.eigvalsh(
            tf.reshape(ham, (chi1 * chi1 * chi1, chi1 * chi1 * chi1)))) / 2
    ham = ham - bias * tf.reshape(
        tf.eye(chi1 * chi1 * chi1, dtype=dtype),
        (chi1, chi1, chi1, chi1, chi1, chi1))
    ham_0 = copy.deepcopy(ham)

    skip_layer = [misc_mera.skip_layer(w) for w in wC]
    p = 0

    while (p < len(skip_layer)) and skip_layer[p]:
        if skip_layer[p]:
            ham = ascending_super_operator(ham, wC[p], uC[p])
            ham /= 2
        p += 1
    Linit = p
    isometry = wC[p]
    unitary = uC[p]
    Energies = []
    run_times = []
    ham_init = copy.deepcopy(ham)
    if rho_0 == 0:
        chi_max = isometry.shape[2]
        rho = tf.reshape(
            tf.eye(chi_max**3, dtype=dtype),
            (chi_max, chi_max, chi_max, chi_max, chi_max, chi_max))
    else:
        rho = rho_0

    for k in range(numiter):
        t1 = time.time()
        rho = steady_state_density_matrix(nsteps_steady_state, rho, isometry,
                                          unitary)
        ham = ham_init
        for b in range(10):
            ham = ascending_super_operator(ham, isometry, unitary)
            ham /= 2

        if verbose > 0:
            rho_temp = rho
            for p in range(Linit - 1, -1, -1):
                rho_temp = descending_super_operator(rho_temp, wC[p], uC[p])

            if np.mod(k, 10) == 1:
                Z = misc_mera.trace(rho_temp)
                net = tn.TensorNetwork()
                r = net.add_node(rho_temp)
                h = net.add_node(ham_0)
                edges = [net.connect(r[n], h[n]) for n in range(6)]
                Energies.append((net.contract_between(r,h).get_tensor() + bias)/Z)
                stdout.write(
                    '\r     Iteration: %i of %i: E = %.8f, err = %.16f at D = %i with %i layers'
                    % (int(k), int(numiter), float(Energies[-1]),
                       float(Energies[-1] + 4 / np.pi,), int(wC[-1].shape[2]),
                       len(wC)))
                stdout.flush()

        for bla in range(1):
            uEnv = get_env_disentangler(ham, rho, isometry, unitary)
            if opt_u:
                if numpy_update:
                    unitary = misc_mera.u_update_svd_numpy(uEnv)
                else:
                    unitary = misc_mera.u_update_svd(uEnv)

            wEnv = get_env_isometry(ham, rho, isometry, unitary)
            if opt_w:
                if numpy_update:
                    isometry = misc_mera.w_update_svd_numpy(wEnv)
                else:
                    isometry = misc_mera.w_update_svd(wEnv)

        run_times.append(time.time() - t1)
        if verbose > 2:
            print('time per iteration: ', run_times[-1])

    return isometry, unitary, rho, run_times, Energies
