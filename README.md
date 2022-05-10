### TT-Gradient-Cross
Tensor Train (TT) Gradient Cross algorithm for the resolution of Hamilton Jacobi Bellman equation.

## Installation
The code is based on [TT-Toolbox](https://github.com/oseledets/TT-Toolbox) Matlab package. Numerical test scripts will probe and download it automatically using a `check_tt` function. However, if this fails, you should download and setup the TT-Toolbox yourself before running this code.

## Contents

### Numerical test scripts

* `test_high_dim_approx.m `  Approximation of a function in arbitrary dimension.
* `test_hjb_2D.m` 2-dimensional optimal control problem with exact solution.
* `test_hjb_2D_constraints.m` 2-dimensional optimal control problem with exact solution and control constraints.
* `test_hjb_Lorenz.m` Lorenz system.
* `test_hjb_cuckersmale.m` Cucker-Smale model.

The parameters required for tests are read from the keyboard. The codes provide default parameters for a quick starting experiment.

### HJB solver

* `multicontrolfun_leg.m` Computes the control signal at a particular state given the TT format of the value function.

### Discretization

* `legendre_rec.m` Computes Legendre polynomials and their derivatives
* `lgwt.m` [Legendre-Gauss Quadrature Weights and Nodes](https://uk.mathworks.com/matlabcentral/fileexchange/4540-legendre-gauss-quadrature-weights-and-nodes)

### Auxiliary

*  `parse_parameter.m` Auxiliary file to input parameters.
*  `gradient_cross.m` TT-Gradient Cross approximation
*  `yex_fun_rank_1` Reference approximation of a rank 1 function via TT-Gradient Cross with a very low tolerance
*  `yex_fun_rank_not_1` Reference approximation of a function with rank greater than 1 via TT-Gradient Cross with a very low tolerance
*  `pontrya2D`  Pontryagin solver for a 2D problem
*  `maxvol_rect`  [Rectangular MaxVol algorithm](https://doi.org/10.1016/j.laa.2017.10.014)
*  `Ax_cucker` Semilinear form for Cucker Smale
*  `DxAx_cucker`  Derivatives of the semilinear form for Cucker Smale
*  `lagrange_derivative` Derivatives of the Lagrangian basis
*  `check_tt` Check/download/add-to-path for TT-Toolbox
