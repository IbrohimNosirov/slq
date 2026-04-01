using LinearAlgebra
include("../src/lanczos.jl")
include("../src/qr_tridiag.jl")
include("../utils/matrix_gallery.jl")

# Lanczos full orthogonalization.
let
        evals_count = 100
        evals_true = Array{Float64}(undef, evals_count)
        A = Array{Float64}(undef, evals_count, evals_count)
        vec_rand = Array{Float64}(undef, evals_count)
        make_cluster!(evals_true, Interval(0.1, 1), 1.0)
        make_kronecker_quasirand!(vec_rand)
        make_matrix!(A, vec_rand, evals_true)
        vec_rand /= sqrt(sum(vec_rand .* vec_rand))
        context = FullOrthogonalization(vec_rand, evals_count)
        lanczos!(A, context)
        evec_row = zeros(evals_count)
        evec_row[1] = 1.0
        qr_tridiag!(context.diag, context.subdiag, evec_row)
        evals_lanczos = context.diag
        println("evals_lanczos ", evals_lanczos)
        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
        println("maximum eigenvalue error ", evals_error)
end

# Lanczos no orthogonalization.
let
        evals_count = 100
        evals_true = Array{Float64}(undef, evals_count)
        A = Array{Float64}(undef, evals_count, evals_count)
        vec_rand = Array{Float64}(undef, evals_count)
        make_cluster!(evals_true, Interval(0.1, 1), 1.0)
        make_kronecker_quasirand!(vec_rand)
        make_matrix!(A, vec_rand, evals_true)
        vec_rand /= sqrt(sum(vec_rand .* vec_rand))
        context = NoOrthogonalization(vec_rand, evals_count)
        lanczos!(A, context)
        evec_row = zeros(evals_count)
        evec_row[1] = 1.0
        qr_tridiag!(context.diag, context.subdiag, evec_row)
        evals_lanczos = context.diag
        println("evals_lanczos ", evals_lanczos)
        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
        println("maximum eigenvalue error", evals_error)
end

# Lanczos selective orthogonalization.
let
        evals_count = 100
        evals_true = Array{Float64}(undef, evals_count)
        A = Array{Float64}(undef, evals_count, evals_count)
        vec_rand = Array{Float64}(undef, evals_count)
        make_functional_decay!(evals_true, Interval(0.1, 1), matern_13_2)
        make_kronecker_quasirand!(vec_rand)
        make_matrix!(A, vec_rand, evals_true)
        vec_rand /= sqrt(sum(vec_rand .* vec_rand))
        context = SelectiveOrthogonalization(vec_rand, 30)
        subspace = DeflatedSubspace(evals_count, 30)
        lanczos!(A, context, subspace)
        context.evec_row .= zeros(size(context.evec_row, 1))
        context.evec_row[1] = 1.0
        qr_tridiag!(context.diag, context.subdiag, context.evec_row)
        evals_lanczos = context.diag
        println("evals_lanczos ", evals_lanczos)
        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
        println("maximum eigenvalue error ", evals_error)
end
