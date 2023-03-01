# ML-DFA: Machine-Learning Correction to Density Functional Approximations
Scripts and dataset for <a href="https://chemrxiv.org/engage/chemrxiv/article-details/63f8bc26937392db3dfecb86" target="_blank">A Semilocal Machine-Learning Correction to Density Functional Approximations</a>

> Use the files in the release if possible. Download and unzip the package and it's good to go.

## Dependency
ML-DFA is a custom exchange-correlation (XC) functional in <a href="https://pyscf.org/" target="_blank">PySCF</a>, with its ML model built upon <a href="https://pytorch.org/" target="_blank">PyTorch</a>.
A python library <a href="https://github.com/facebookresearch/nevergrad" target="_blank">Nevergrad</a> is required for the training.

Below is a list of primary python packages with the corresponding versions I used for developing ML-DFA:

package|version
---|---
python|3.9.12
pyscf|2.0.1
pytorch|1.10.1
nevergrad|0.4.3

No guarantees for other combination of package versions. Other packages may be necessary depending on the specific environment.

## ML-B3LYP for direct use
A set of ANN parameters has been included in the `ML-DFA/test/` directory, which can be directly used for energetic calculations as a custom XC functional of the PySCF.

For your own test sets, the usage takes simple precedures:
1. Prepare the input geometry files along with a list of spin multiplicity for each species in the same directory. Note the spin multiplicity will be converted to the number of unpaired electrons in the script for PySCF input (*line 150*). 
2. Fill in the path of the test set and the spin list on *line 149* and *line 151*.
3. You may need to create a list of charges to fill in the charge of each species on *line 164*, if you have ionic species in the test set.
4. Choose the basis set of your desire on *line 166* and change the filename of the output total energies on *line 181*.
And that's it.

It takes the same procedure for ML-DFA to output electron density on default or custom grids after SCF is done.
Refer to [PySCF Manual](https://pyscf.org/user/dft.html) and [PySCF Examples](https://github.com/pyscf/pyscf/tree/master/examples/dft) for further information.

## Train an ML-DFA
Procedures for training using `ML-DFA/train/training.py`:
1. Change the structure of the ANN in the `ML-DFA/train/snn_3h.py` if you do not want a three-hidden-layer network. Note that if you change the number of hidden layers, you have to update the weight assignment step from *line 186* to *line 208*, *line 306* and from *line 323* to *line 345*. Just follow the my example and be careful with the name of each layer in `snn_para`. Then change the shape of `param` on *line 281*. You may want to print `snn_para` to check if you have got the names right. After training, update *line 87* to *line 107* in the `ML-DFA/test/predict.py` for test.
2. Assign the scaling factor for each layer of weights starting from *line 15*.
3. Assign the number of hidden neurons in order to the variable `hidden` on *line 26*.
4. You have to configure some codes starting from *line 72* of `eval_xc_ml` on *line 33* **if you want to correct DFAs other than B3LYP**. Need to change *line 221, 244, 375, 410* accordingly.
5. Fill in the reference TEs and AEs for `TE_ccsd` and `AE_ccsd` on *line 95,96*.
6. Choose the basis set of your desire on *line 100*.
7. Fill in the path for the input geomoetry, charge and spin for each species. Configure *line 175* - *line 180* accordingly.
8. Set number of epochs for the optimization on *line 290*.
9. You can use a different validation set and change filepath from *line 320*.
After training, alter the hyperparamters if you're not satisfied with the test results. Otherwise, copy the `de_best.npz` to the `test` folder and be sure to change the settings in the `predict.py` so that it matches the settings in the training script.
