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
A set of ANN parameters has been included in the `./test/` directory, which can be directly used for energetic calculations as a custom XC functional of the PySCF.


