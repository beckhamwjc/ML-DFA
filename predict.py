from pyscf import gto
from pyscf import dft
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from natsort import natsorted
from os import listdir
from os.path import join

from snn_3h import SNN

input_dim = 3
output_dim = 1
depth = 4
lamda = 1e-5
beta = 1.5
use_cuda=False
device = torch.device("cuda" if use_cuda else "cpu")
hidden = [20,20,20]

s_nn = SNN(input_dim,output_dim,hidden,lamda,beta,use_cuda)
snn_para = s_nn.state_dict()
s_nn.to(device)


def eval_xc_ml(xc_code, rho, spin, relativity=0, deriv=1, verbose=None):
    # Definition of ML-DFA

    if spin == 0:
        rho0, dx, dy, dz= rho[:4]
        gamma1=gamma2=gamma12=(dx**2+dy**2+dz**2)*0.25+1e-10
        rho01=rho02=rho0*0.5
    else:
        rho1=rho[0]
        rho2=rho[1]
        rho01, dx1, dy1, dz1 = rho1[:4]
        rho02, dx2, dy2, dz2 = rho2[:4]
        gamma1=dx1**2+dy1**2+dz1**2+1e-10
        gamma2=dx2**2+dy2**2+dz2**2+1e-10
        gamma12=dx1*dx2+dy1*dy2+dz1*dz2+1e-10
        
    ml_in_ = np.concatenate((rho01.reshape((-1,1)),rho02.reshape((-1,1)),gamma1.reshape((-1,1)),gamma2.reshape((-1,1)),gamma12.reshape((-1,1))),axis=1)
    #print("ml_in: ", ml_in_)
    ml_in = torch.Tensor(ml_in_)
    ml_in.requires_grad = True
    num_r = ml_in.shape[0]
    dim_in = ml_in.shape[1]
    exc_ml_out = s_nn(ml_in, is_training_data=False)
    ml_exc = exc_ml_out.detach().numpy()
    exc_ml = torch.sum(exc_ml_out)
    exc_ml.backward()
    grad=ml_in.grad.data.numpy()
    
    
    if spin!=0:
        vrho_ml=np.hstack((grad[:,0].reshape((-1,1)),grad[:,1].reshape((-1,1))))
        vgamma_ml=np.hstack((grad[:,2].reshape((-1,1)),grad[:,4].reshape((-1,1)),grad[:,3].reshape((-1,1))))

    else:
        vrho_ml=(grad[:,0]+grad[:,1])*0.5
        vgamma_ml=(grad[:,2]+grad[:,3]+grad[:,4])*0.25
    

    # Mix with existing functionals
    b3lyp_xc = dft.libxc.eval_xc('B3LYP', rho, spin, relativity, deriv, verbose)
    b3lyp_exc = np.array(b3lyp_xc[0])
    b3lyp_vrho = np.array(b3lyp_xc[1][0])
    b3lyp_vgamma = np.array(b3lyp_xc[1][1])
    exc = (b3lyp_exc.reshape(-1,1) + ml_exc).reshape(-1)
    vrho = b3lyp_vrho + vrho_ml
    vgamma = b3lyp_vgamma + vgamma_ml


    vlapl = b3lyp_xc[1][2]
    vtau = b3lyp_xc[1][3]
    vxc = (vrho, vgamma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    
    return exc, vxc, fxc, kxc


if __name__ == "__main__":
    weight = np.load('de_best.npz')
    k = 0
    for i00 in range(hidden[0]):
        for j in range(input_dim):
            snn_para['model.0.weight'][i00,j] = weight['p_h1'][k]
            k += 1
    for i01 in range(hidden[0]):
        snn_para['model.0.bias'][i01] = weight['p_h1_bias'][i01]
    k = 0
    for i10 in range(hidden[1]):
        for j in range(hidden[0]):
            snn_para['model.2.weight'][i10,j] = weight['p_h1_h2'][k]
            k += 1
    for i11 in range(hidden[1]):
        snn_para['model.2.bias'][i11] = weight['p_h2_bias'][i11]
    k = 0
    for i20 in range(hidden[2]):
        for j in range(hidden[1]):
            snn_para['model.4.weight'][i20,j] = weight['p_h2_h3'][k]
            k += 1
    for i21 in range(hidden[2]):
        snn_para['model.4.bias'][i21] = weight['p_h3_bias'][i21]
    for i3 in range(hidden[2]):
        snn_para['model.6.weight'][0,i3] = weight['p_out'][i3]
    s_nn.load_state_dict(snn_para)
    #print(snn_para)
    
    # pred_atom = pd.DataFrame([],columns=['spin','y'])
    # pred_atom['spin'] = pd.read_table('../dataset/atoms/multi-file',header=None)
    # pred_atom['spin'] = pred_atom['spin'] - 1
    # atm_path = '../dataset/atoms'
    # dd = listdir(atm_path)
    # dd = natsorted(dd)
    # ii = 0
    
    # print("atom calculation starts")
    # for files in dd:
        # if files.endswith((".xyz")):
            # #print(files)
            # atm_file = join(atm_path,files)
            # atm1         = gto.Mole()
            # atm1.verbose = 1
            # atm1.atom    = open(atm_file)
            # atm1.charge  = 0
            # atm1.spin    = int(pred_atom.loc[ii,'spin'])
            # atm1.basis   = "def2qzvpd"
            # atm1.build()
            
            # if atm1.spin == 0:
                # mlpbe = dft.RKS(atm1)
            # else:
                # mlpbe = dft.UKS(atm1)

            # mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0,0.0,0.2])
            # mlpbe.grids.level = 5
            # mlpbe.max_cycle=100
            # pred_atom.loc[ii,'y'] = mlpbe.kernel() * 627.5095
            # print("Atom No.",ii+1," converged: ",mlpbe.converged)
            # ii += 1
            
    # pred_atom['y'].to_csv('qzvpd-pred_g2-atom.txt',sep='\t',header=False,index=False)
    
    
    pred = pd.DataFrame([],columns=['spin','y'])
    pred['spin'] = pd.read_table('../dataset/g3/multi.txt',header=None)
    pred['spin'] = pred['spin'] - 1
    mol_path = '../dataset/g3'
    dd = listdir(mol_path)
    dd = natsorted(dd)
    ii = 0
    
    print("mol calculation starts")
    for files in dd:
        if files.endswith((".xyz")):
            #print(files)
            mol_file = join(mol_path,files)
            mol1         = gto.Mole()
            mol1.verbose = 1
            mol1.atom    = open(mol_file)
            mol1.charge  = 0
            mol1.spin    = int(pred.loc[ii,'spin'])
            mol1.basis   = "def2qzvpd"
            mol1.build()
            
            if mol1.spin == 0:
                mlpbe = dft.RKS(mol1)
            else:
                mlpbe = dft.UKS(mol1)

            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0,0.0,0.2])
            mlpbe.grids.level = 5
            mlpbe.max_cycle=100
            pred.loc[ii,'y'] = mlpbe.kernel() * 627.5095
            print("Molecule No.",ii+1," converged: ",mlpbe.converged)
            ii += 1
    
    pred['y'].to_csv('qzvpd-pred_g3.txt',sep='\t',header=False,index=False)
    


