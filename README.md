## Performant implementation of Stochastic Lanczos Quadrature with robust
## stopping criteria.

DONE
* clean up QR tridiag so it doesn't beat up on memory.
* Finish selective orthogonalization/deflation.
* Make all of the tests pass for each instance of Lanczos.
* build a ton of different spectra and measure convergence.

TODO [for the manuscript.]
* Set the converged values to 1/n.
* Find the 4 data matrices.
* check the residual bounds I'm getting against Davis-Kahan (might need to loosen)
* Pull in all of the sources. (Tyler/Tom, Chris, block lanczos (BOLT) Kingsley, Alice + kernel-based approximations,
  older work by Stefan Guttel on restarted Lanczos for f(A)b, Paige's theory (1987->2021), Davis Kahan)
* Use PrecompileTools.jl to precompile when shipping package.
* LAPACK chkfinite

* Make figures.
* finish the slides of Guilia's group.

Parameters

    lc = index of last fully converged Ritz vector (β\_ji < jε||A||) from the
    left end of the spectrum.

    lg = index of last good Ritz vector [lg ∈ L(k) for some k ≤ j] from the
    left.

    rc, rg as above for the right end of the spectrum.

Initialize

    lc = lg = rc = rg = |L| = 0. L = φ (empty). q\_0 = o. Pick r\_0' ≠ o.

Loop

    For j = 1, 2, ..., n repeat steps 1 through 5.

        1. If |L| > 0 then purge r' of threshold vectors to get r and set
β\_j-1 ← ||r||.

        2. If β\_j-1 = 0 then stop else normalize r to get q\_j.

        3. Take a Lanczos step to get α\_j, r', β'\_j.

        4. θ\_i^(j) ← λ\_i[T\_j] for i = lg + 1, lg+2

            and i = -(rg + 1), -(rg + 2). Compute associated s\_ji. Set |L|= 0.
        5. If β'\_ij (= β'\_js\_ji) < sqrt(ε)||T\_j|| for any of the i in step
            4 then pause.

Pause

    1. Form |L| (= {i : β'\_ij < sqrt(ε)||T\_j}). Update lg, rg.

    2. Summon Q\_j and compute y\_l^(j) = Q\_js\_l for l = lc, ..., lg
       and l = -rc, ..., -rg.

    3. Optional step: Perform modified Gram-Scmidt on the new y\_l^(j); use the
       most accurate first. Update s\_jl accordingly.

    4. If enough y's are acceptable then stop.

    5. Compute y\_l^*r' for each good y_l; if too big add l to L.
       This step allows y_l to be refined.

It is only necessary to retain the threshold vectors after the pause; the
Lanczos vectors can be rewound.

