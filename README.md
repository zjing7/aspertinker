# aspertinker

Molecular Modeling Tools for Tinker

## Read/write chemical structure files

Supported formats: xyz, txyz

* Compute trajectories RMSDs
* Analyze structures
  * Distance
  * Angle
  * Torsion
  * Pseudo-rotation angle
  * Distance to line
  * Distance to plane
  * Angle between two planes

## Manipulate chemical structures

* Automatically fragment structures
* Find principle axes of the interface between two fragments
* Translate and/or rotate one fragment relative to the other fragment
* Delete or add trajectories or atoms

## Write input files for quantum chemistry packages

* Supported software
  * [Gaussian 16](https://gaussian.com/gaussian16/)
  * [Psi4](http://www.psicode.org/)

* Methods

  MP2, SNS-MP2
  
  B3LYP-D3, TPSS-D3, DSD-BLYP-D3BJ

* Job types

  Optimization, N-body energy

## Read output files from quantum chemistry packages

* Supported software
  * [Gaussian 16](https://gaussian.com/gaussian16/)
  * [Psi4](http://www.psicode.org/)
  * [Q-Chem](https://www.q-chem.com)
  * [Tinker](https://github.com/TinkerTools)

* Read energies, frequencies, convergence and other data from QM output files
