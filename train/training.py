from pyscf import gto
from pyscf import dft
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import nevergrad as ng
from concurrent import futures
from natsort import natsorted,ns
from os import listdir
from os.path import join

from snn_3h import SNN

scaling_factor0 = 1.0
scaling_factor1 = 1.0
scaling_factor2 = 1.0
scaling_factor3 = 0.001
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


def loss1(weight, is_eval=False):
    ## units in kcal/mol !!!
    ## do not forget to convert SCF energies from PySCF to kcal/mol !!!
    ## order of TE: h2o, cn, so2, H, C, N, O, S

    TE_ccsd = np.array([-47963.0, -48524.5, -344244.0, -313.8, -23745.8, -47102.5, -249778.1])
    AE_ccsd = np.array([233.0, 405.3, 260.9])
    
    
    verbosity = 1
    basis = 'def2tzvpd'
    ## PySCF initialization
    mol1 = gto.Mole()
    mol1.verbose = verbosity
    mol1.atom    = open('../data_training/H2O.xyz')
    mol1.charge  = 0
    mol1.spin    = 0
    mol1.basis   = basis
    mol1.build()
    
    mol2 = gto.Mole()
    mol2.verbose = verbosity
    mol2.atom    = open('../data_training/C2H2.xyz')
    mol2.charge  = 0
    mol2.spin    = 0
    mol2.basis   = basis
    mol2.build()
    
    #mol3 = gto.Mole()
    #mol3.verbose = verbosity
    #mol3.atom    = open('../data_training/CN.xyz')
    #mol3.charge  = 0
    #mol3.spin    = 1
    #mol3.basis   = basis
    #mol3.build()

    mol4 = gto.Mole()
    mol4.verbose = verbosity
    mol4.atom    = open('../data_training/SO2.xyz')
    mol4.charge  = 0
    mol4.spin    = 0
    mol4.basis   = basis
    mol4.build()

    
    atm_1 = gto.Mole()
    atm_1.verbose = verbosity
    atm_1.atom    = 'H 0 0 0'
    atm_1.charge  = 0
    atm_1.spin    = 1
    atm_1.basis   = basis
    atm_1.build()
    
    atm_2 = gto.Mole()
    atm_2.verbose = verbosity
    atm_2.atom    = 'C 0 0 0'
    atm_2.charge  = 0
    atm_2.spin    = 2
    atm_2.basis   = basis
    atm_2.build()
    
    #atm_3 = gto.Mole()
    #atm_3.verbose = verbosity
    #atm_3.atom    = 'N 0 0 0'
    #atm_3.charge  = 0
    #atm_3.spin    = 3
    #atm_3.basis   = basis
    #atm_3.build()
    
    atm_4 = gto.Mole()
    atm_4.verbose = verbosity
    atm_4.atom    = 'O 0 0 0'
    atm_4.charge  = 0
    atm_4.spin    = 2
    atm_4.basis   = basis
    atm_4.build()

    atm_5 = gto.Mole()
    atm_5.verbose = verbosity
    atm_5.atom    = 'S 0 0 0'
    atm_5.charge  = 0
    atm_5.spin    = 2
    atm_5.basis   = basis
    atm_5.build()
    
    mol_list = [mol1, mol2, mol4]
    atm_list = [atm_1, atm_2, atm_4, atm_5]

    TE_mlpbe = np.zeros(3)
    TE_atoms = np.zeros(4)
    AE_mlpbe = np.zeros(3)

    
    w = torch.Tensor(weight)
    #print("weight in: ","\n",w)
    k = 0
    for i00 in range(hidden[0]):
        for j in range(input_dim):
            snn_para['model.0.weight'][i00,j] = w[k] * scaling_factor0
            k += 1
    for i01 in range(hidden[0]):
        snn_para['model.0.bias'][i01] = w[k+i01] * scaling_factor0
    k = hidden[0]*(input_dim+1)
    for i10 in range(hidden[1]):
        for j in range(hidden[0]):
            snn_para['model.2.weight'][i10,j] = w[k] * scaling_factor1
            k += 1
    for i11 in range(hidden[1]):
        snn_para['model.2.bias'][i11] = w[k+i11] * scaling_factor1
    k = hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)
    for i20 in range(hidden[2]):
        for j in range(hidden[1]):
            snn_para['model.4.weight'][i20,j] = w[k] * scaling_factor2
            k += 1
    for i21 in range(hidden[2]):
        snn_para['model.4.bias'][i21] = w[k+i21] * scaling_factor2
    k = hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1) + hidden[2]*(hidden[1]+1)
    for i3 in range(hidden[2]):
        snn_para['model.6.weight'][0,i3] = w[k+i3] * scaling_factor3
    s_nn.load_state_dict(snn_para)
    #print("weight loaded: ", snn_para)
    
    indicator = 0
    print("indicator before SCF: ", indicator)
    print("mol calculation starts")
    for i in range(len(TE_mlpbe)):
        if mol_list[i].spin == 0:
            mlpbe = dft.RKS(mol_list[i])
        else:
            mlpbe = dft.UKS(mol_list[i])
            
        mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0,0.0,0.2])
        #mlpbe.xc = 'b3lyp'
        #mlpbe.grids.atom_grid = (99,590)
        mlpbe.grids.level = 5
        if is_eval:
            mlpbe.max_cycle=50
        else:
            mlpbe.max_cycle=50
        #mlpbe.diis_space = 24
        #mlpbe.diis_start_cycle=5
        TE_mlpbe[i] = mlpbe.kernel() * 627.5095
        print("Molecule No.",i+1," converged: ",mlpbe.converged)
        if not mlpbe.converged:
            indicator = 1
            
    
    print("atom calculation starts")
    for i in range(len(TE_atoms)):
        if atm_list[i].spin == 0:
            mlpbe = dft.RKS(atm_list[i])
        else:
            mlpbe = dft.UKS(atm_list[i])

        mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0,0.0,0.2])
        #mlpbe.xc = 'b3lyp'
        #mlpbe.grids.atom_grid = (99,590)
        mlpbe.grids.level = 5
        if is_eval:
            mlpbe.max_cycle=50
        else:
            mlpbe.max_cycle=50
        #mlpbe.diis_space = 24
        #mlpbe.diis_start_cycle=5
        TE_atoms[i] = mlpbe.kernel() * 627.5095
        print("Atom No.",i," converged: ",mlpbe.converged)
        if not mlpbe.converged:
            indicator = 1
    
    AE_mlpbe[0] = TE_atoms[2] + TE_atoms[0]*2 - TE_mlpbe[0]
    AE_mlpbe[1] = TE_atoms[1]*2 + TE_atoms[0]*2 - TE_mlpbe[1]
    #AE_mlpbe[1] = TE_atoms[1] + TE_atoms[2] - TE_mlpbe[1]
    AE_mlpbe[2] = TE_atoms[2]*2 + TE_atoms[3] - TE_mlpbe[2]
    E_ml = np.append(np.append(TE_mlpbe, TE_atoms), AE_mlpbe)
    E_ref = np.append(TE_ccsd, AE_ccsd)
    #print("ML_pred: ",E_ml)
    #print("Ref: ",E_ref)
    loss = np.mean(np.abs(E_ref - E_ml))
    #print("weights of snn:", s_nn.state_dict())
    #print("E_atoms: ", E_atoms)
    #print("E_molecules: ", E_mol)
    print("TE_mol: ",TE_mlpbe,"TE_atoms: ",TE_atoms,"AE_ml: ",AE_mlpbe)
    print("loss: ",loss)
    print("indicator after SCF: ", indicator)
    
    return loss, indicator



if __name__ == "__main__":

    param = ng.p.Array(shape=(hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)+hidden[2]*(hidden[1]+2),)).set_bounds(-20,1)
    

    de_opt = ng.optimization.optimizerlib.ConfiguredPSO(popsize=100,omega=0.9,phip=0.95,phig=0.9)
    de1 = de_opt(param, budget=3000, num_workers=2)


    de1.suggest(np.zeros(hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)+hidden[2]*(hidden[1]+2)))
    indicator_scf = 0
    for i in range(3000):
        print(i, "-epoch starts")
        x1 = de1.ask()
        y1,indicator_scf = loss1(*x1.args)
        print("SCF convered? ",not bool(indicator_scf))
        if not indicator_scf:
            de1.tell(x1,y1)
            print(i, "-epoch ends \n")
        else:
            print("\n")
        
    recommendation = de1.recommend()
    de_best = recommendation.value
    loss_best = loss1(de_best,is_eval=True)
    print("final:", loss_best)
    #print("best weights: ", de_best)
    np.savez("de_best", 
             p_h1 = de_best[0:input_dim*hidden[0]] * scaling_factor0, 
             p_h1_bias = de_best[input_dim*hidden[0]:(input_dim+1)*hidden[0]] * scaling_factor0, 
             p_h1_h2 = de_best[(input_dim+1)*hidden[0]:hidden[0]*(input_dim+1)+hidden[1]*hidden[0]] * scaling_factor1, 
             p_h2_bias = de_best[hidden[0]*(input_dim+1)+hidden[1]*hidden[0]:hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)] * scaling_factor1,  
             p_h2_h3 = de_best[hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1):hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)+hidden[1]*hidden[2]] * scaling_factor2, 
             p_h3_bias = de_best[hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)+hidden[1]*hidden[2]:hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)+hidden[2]*(hidden[1]+1)] * scaling_factor2, 
             p_out = de_best[hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)+hidden[2]*(hidden[1]+1):hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)+hidden[2]*(hidden[1]+2)] * scaling_factor3, 
             allow_pickle=False, 
             fix_imports=False)
    de1.dump("de_optimizer.data")
    print("Best loss is: ", (loss_best), "\n", "Best params are: ", de_best, file = open('de.log','a'))

######################Validation#####################

    w = torch.Tensor(de_best)
    k = 0
    for i00 in range(hidden[0]):
        for j in range(input_dim):
            snn_para['model.0.weight'][i00,j] = w[k] * scaling_factor0
            k += 1
    for i01 in range(hidden[0]):
        snn_para['model.0.bias'][i01] = w[k+i01] * scaling_factor0
    k = hidden[0]*(input_dim+1)
    for i10 in range(hidden[1]):
        for j in range(hidden[0]):
            snn_para['model.2.weight'][i10,j] = w[k] * scaling_factor1
            k += 1
    for i11 in range(hidden[1]):
        snn_para['model.2.bias'][i11] = w[k+i11] * scaling_factor1
    k = hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1)
    for i20 in range(hidden[2]):
        for j in range(hidden[1]):
            snn_para['model.4.weight'][i20,j] = w[k] * scaling_factor2
            k += 1
    for i21 in range(hidden[2]):
        snn_para['model.4.bias'][i21] = w[k+i21] * scaling_factor2
    k = hidden[0]*(input_dim+1)+hidden[1]*(hidden[0]+1) + hidden[2]*(hidden[1]+1)
    for i3 in range(hidden[2]):
        snn_para['model.6.weight'][0,i3] = w[k+i3] * scaling_factor3
    s_nn.load_state_dict(snn_para)
    #print("weight loaded: ", snn_para)

    pred_atom = pd.DataFrame([],columns=['spin','y'])
    pred_atom['spin'] = np.loadtxt('../dataset/atoms/multi-file',dtype=int)    
    pred_atom['spin'] = pred_atom['spin'] - 1
    atm_path = '../dataset/atoms'
    dd = listdir(atm_path)
    dd = natsorted(dd,alg=ns.PATH)
    ii = 0
    
    print("atom calculation starts")
    for files in dd:
        if files.endswith((".xyz")):
            #print(files)
            atm_file = join(atm_path,files)
            atm1         = gto.Mole()
            atm1.verbose = 1
            atm1.atom    = open(atm_file)
            atm1.charge  = 0
            atm1.spin    = int(pred_atom.loc[ii,'spin'])
            atm1.basis   = "def2tzvpd"
            atm1.build()
            
            if atm1.spin == 0:
                mlpbe = dft.RKS(atm1)
            else:
                mlpbe = dft.UKS(atm1)
                #mlpbe.level_shift = (0.1,0.2)
            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0,0.0,0.2])
            mlpbe.grids.level = 5
            mlpbe.max_cycle=50
            pred_atom.loc[ii,'y'] = mlpbe.kernel() * 627.5095
            print("Atom No.",ii+1," converged: ",mlpbe.converged)
            ii += 1
            
    pred_atom['y'].to_csv('Atom_TE.txt',sep='\t',header=False,index=False)
    
    pred = pd.DataFrame([],columns=['spin','y'])
    pred['spin'] = np.loadtxt('../dataset/multi-file',dtype=int)
    pred['spin'] = pred['spin'] - 1
    mol_path = '../dataset'
    dd = listdir(mol_path)
    dd = natsorted(dd,alg=ns.PATH)
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
            mol1.basis   = "def2tzvpd"
            mol1.build()
            
            if mol1.spin == 0:
                mlpbe = dft.RKS(mol1)
            else:
                mlpbe = dft.UKS(mol1)
                #mlpbe.level_shift = (0.2,0.1)
            mlpbe = mlpbe.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0,0.0,0.2])
            mlpbe.grids.level = 5
            mlpbe.max_cycle=50
            pred.loc[ii,'y'] = mlpbe.kernel() * 627.5095
            print("Molecule No.",ii+1," converged: ",mlpbe.converged)
            ii += 1
    
    pred['y'].to_csv('G2_TE.txt',sep='\t',header=False,index=False)
 
