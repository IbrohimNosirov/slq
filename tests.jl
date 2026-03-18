using LinearAlgebra
#using Profile
#using PProf
#using OptimalTransport
#using Distributions
#using DataStructures
#using StatProfilerHTML

#include("multicore_slq.jl")
include("lanczos.jl")
include("qr_tridiag.jl")
include("matrix_gallery.jl")

#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), gaussian)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_1_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_3_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_5_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_7_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_9_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_11_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_13_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
## 100 eigenvalues in 1 cluster, each with radius 1e-7
#evals_count = 100
#epsilon = 1e-7
#evals = zeros(evals_count)
#make_cluster!(evals, Interval(0,1), epsilon)
#gr()
#scatter(range(1,evals_count), evals)
#end

## tridiagonal QR tests
#let
#        evals_count = 7
#        evals = zeros(evals_count)
#        make_functional_decay!(evals, Interval(0,1), matern_1_2)
#
#        evals_mine, subdiagonal = make_tridiag_matrix(evals)
#        evecs_mine = zeros(evals_count)
#        qr_tridiagonal!(evals_mine, subdiagonal, evecs_mine, 1)
#
#        evals_count = 1000
#        evals = zeros(evals_count)
#        make_functional_decay!(evals, Interval(0,1), matern_1_2)
#
#        evals_mine, subdiagonal = make_tridiag_matrix(evals)
#        evecs_mine = zeros(evals_count)
#        println("started eigensolve")
#        @time qr_tridiagonal!(evals_mine, subdiagonal, evecs_mine, 1)
#
#        evals = zeros(evals_count)
#        make_functional_decay!(evals, Interval(0,1), matern_1_2)
#
#        evals_mine, subdiagonal = make_tridiag_matrix(evals)
#        evecs_mine = zeros(evals_count)
#        evecs_mine = @profilehtml qr_tridiagonal!(evals_mine, subdiagonal, evecs_mine, 1)
#
#        diagonal, subdiagonal = make_tridiag_matrix(evals)
#        println("started LAPACK solver ")
#        evals_lapack, evecs_lapack = @time eigen!(SymTridiagonal(diagonal, subdiagonal))
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

# Need more robust test suite.

# Lanczos full orthogonalization test.
#let
#        evals_count = 3000
#        evals_true = zeros(evals_count)
#        f(x) = x
#        make_functional_decay!(evals_true, Interval(0.1,1), f)
#
#        A = make_matrix(evals_true)
#        lanczos_vec = kronecker_quasirand_vec(evals_count)
#        lanczos_vec /= sqrt(sum(lanczos_vec .* lanczos_vec))
#        context = LanczosContext(FullOrth(), A, lanczos_vec, 10)
#        @time lanczos(context, FullOrth())
#
#        lanczos_vec = kronecker_quasirand_vec(evals_count)
#        lanczos_vec /= sqrt(sum(lanczos_vec .* lanczos_vec))
#        context = LanczosContext(FullOrth(), A, lanczos_vec, evals_count)
#        lanczos(context, FullOrth())
#
#        evals_lanczos = get_diagonal(context)
#        subdiagonal = get_subdiagonal(context)
#        evec_row = zeros(size(evals_lanczos,1))
#
#        qr_tridiagonal!(evals_lanczos, subdiagonal, evec_row, 1)
#
#        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
#        println("maximum eigenvalue error ", evals_error)
#end

# Lanczos selective orthogonalization
let
        evals_count = 30
        evals_true = Array{Float64}(undef, evals_count)
        A = Array{Float64}(undef, evals_count, evals_count)
        vec_random = Array{Float64}(undef, evals_count)

        make_functional_decay!(evals_true, Interval(0.1,1000), gaussian)
        make_kronecker_quasirandom!(vec_random)
        make_matrix!(A, vec_random, evals_true)
        vec_random /= sqrt(sum(vec_random .* vec_random))

        context = LanczosContext(SO(), A, vec_random, 30)
        @time lanczos!(SO(), context)
        context.evec_row .= zeros(size(context.evec_row, 1))
        context.evec_row[1] = 1.0
        qr_tridiag!(context.diag, context.subdiag, context.evec_row)

#        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
#        println("maximum eigenvalue error ", evals_error)
end

#let
#        evals_count = 100
#        evals = zeros(evals_count)
#        make_functional_decay!(evals, Interval(0, 1), gaussian)
#        A = make_matrix(evals)
#        slq_context = SLQContext(A)
#        slq(slq_context)
#end
