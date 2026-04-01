## tridiag QR tests
#let
#        evals_count = 7
#        evals = zeros(evals_count)
#        make_functional_decay!(evals, Interval(0,1), matern_1_2)
#
#        evals_mine, subdiag = make_tridiag_matrix(evals)
#        evecs_mine = zeros(evals_count)
#        qr_tridiag!(evals_mine, subdiag, evecs_mine, 1)
#
#        evals_count = 1000
#        evals = zeros(evals_count)
#        make_functional_decay!(evals, Interval(0,1), matern_1_2)
#
#        evals_mine, subdiag = make_tridiag_matrix(evals)
#        evecs_mine = zeros(evals_count)
#        println("started eigensolve")
#        @time qr_tridiag!(evals_mine, subdiag, evecs_mine, 1)
#
#        evals = zeros(evals_count)
#        make_functional_decay!(evals, Interval(0,1), matern_1_2)
#
#        evals_mine, subdiag = make_tridiag_matrix(evals)
#        evecs_mine = zeros(evals_count)
#        evecs_mine = @profilehtml qr_tridiag!(evals_mine, subdiag, evecs_mine, 1)
#
#        diag, subdiag = make_tridiag_matrix(evals)
#        println("started LAPACK solver ")
#        evals_lapack, evecs_lapack = @time eigen!(SymTridiag(diag, subdiag))
#        evecs_lapack = evecs_lapack[1,:]
#
#        evec_err_lapack = sum(evecs_lapack .* evecs_lapack)
#        evec_err_mine = sum(evecs_mine .* evecs_mine)
#        println("mine QQ^T ", evec_err_mine)
#        println("lapack QQ^T ", evec_err_lapack)
#        
#        evals_lapack_err = maximum(abs.(evals .- evals_lapack) ./ abs.(evals))
#        println("max lapack eval error ", evals_lapack_err)
#
#        evals_mine_err = maximum(abs.(evals .- evals_mine) ./ abs.(evals))
#        println("max mine eval error ", evals_mine_err)
#end

