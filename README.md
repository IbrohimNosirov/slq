## Performant implementation of Stochastic Lanczos Quadrature with robust
## stopping criteria.

DONE
* clean up QR tridiag so it doesn't beat up on memory.
* Finish selective orthogonalization/deflation.
* Make all of the tests pass for each instance of Lanczos.
* build a ton of different spectra and measure convergence.
* Set the converged values to 1/n.
* Pull in all of the sources. (Tyler/Tom, Chris, block lanczos (BOLT) Kingsley, Alice + kernel-based approximations,
  older work by Stefan Guttel on restarted Lanczos for f(A)b, Paige's theory (1987->2021), Davis Kahan)
* Make figures.
* finish the slides of Guilia's group.

TODO [for the manuscript.]
* Find the 4 data matrices.
* check the residual bounds I'm getting against Davis-Kahan (might need to loosen)
* Use PrecompileTools.jl to precompile when shipping package.
* LAPACK chkfinite
