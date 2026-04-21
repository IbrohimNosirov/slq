using LinearAlgebra
include("../src/lanczos.jl")
include("../src/qr_tridiag.jl")
include("../utils/matrix_gallery.jl")

# Lanczos full orthogonalization.
let
        println("full orthogonalization")
        evals_count = 10
        evals_true = Array{Float64}(undef, evals_count)
        make_functional_decay!(evals_true, Interval(0.95, 1), matern_1_2)
        vec_rand = Array{Float64}(undef, evals_count)
        make_kronecker_quasirand!(vec_rand)
        A = Array{Float64}(undef, evals_count, evals_count)
        make_matrix!(A, vec_rand, evals_true)
        vec_rand /= sqrt(sum(vec_rand .* vec_rand))
        c = FullOrthogonalization(vec_rand, evals_count)
        @time s = DeflatedSubspace(evals_count, evals_count)
        lanczos!(A, c, s)
        evec_row = zeros(evals_count)
        evec_row[1] = 1.0
        qr_tridiag!(c.diagonal, c.subdiagonal, evec_row)
        evals_lanczos = c.diagonal
        println("evals_lanczos ", evals_lanczos)
        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
        println("maximum eigenvalue error ", evals_error)
end

# mul_subspace test.
let
        println("mul_subspace test.")
        # for loop that tests all numbers and all budgets.
        evals_count = 10
        subspace_dim = 1
        budget = 5
        evals_true = Array{Float64}(undef, evals_count)
        make_functional_decay!(evals_true, Interval(0.95, 1), matern_1_2)
        vec_rand = randn(evals_count)
        vec_rand /= sqrt(sum(vec_rand .* vec_rand))
        A = Array{Float64}(undef, evals_count, evals_count)
        make_matrix!(A, vec_rand, evals_true)

        vals, vecs = eigen(A)
        @assert A * vecs[:, 1] ≈ vals[1] * vecs[:, 1]
        vals[1] = 0.0
        A_deflated = vecs * diagm(vals) * vecs'
        s = DeflatedSubspace(evals_count, budget)
        s = set_ritz_vecs(s, view(vecs, :, [1]))

        source = zeros(evals_count)
        source[1] = 1.0
        source_deflated = copy(source)
        target = zeros(evals_count)
        target_deflated = zeros(evals_count)

        mul_subspace!(target, A, source, s)
        mul!(target_deflated, A_deflated, source_deflated)
        @assert target ≈ target_deflated maximum(abs.(target .- target_deflated) ./ abs.(target))
end

# Lanczos selective orthogonalization, doesn't trigger.
let
        println("selective orthogonalization; doesn't trigger")
        evals_count = 10
        evals_true = Array{Float64}(undef, evals_count)
        make_functional_decay!(evals_true, Interval(0.95, 1), matern_1_2)
        vec_rand = Array{Float64}(undef, evals_count)
        make_kronecker_quasirand!(vec_rand)
        vec_rand ./= sqrt(sum(vec_rand .* vec_rand))
        A = Array{Float64}(undef, evals_count, evals_count)
        make_matrix!(A, vec_rand, evals_true)
        vec_rand = Array{Float64}(undef, evals_count)
        make_kronecker_quasirand!(vec_rand)
        vec_rand ./= sqrt(sum(vec_rand .* vec_rand))
        c = SelectiveOrthogonalization(vec_rand, evals_count)
        s = DeflatedSubspace(evals_count, evals_count)
        deflation = true 
        while deflation
                s, deflation = lanczos!(A, c, s)
                #make_kronecker_quasirand!(c.curr)
                #c.curr .= c.curr ./ sqrt(sum(c.curr .* c.curr))
        end
        diagonal = get_diagonal(c, s, s.budget)
        subdiagonal = get_subdiagonal(c, s, s.budget-1)
        evec_row = get_evec_row(c, s, s.budget)
        fill!(evec_row, 0.0)
        evec_row[1] = 1.0
        qr_tridiag!(diagonal, subdiagonal, evec_row)
        evals_lanczos = c.diagonal
        println("evals_true ", evals_true)
        println("evals_lanczos ", evals_lanczos)
        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
        println("maximum eigenvalue error ", evals_error)
end

# Lanczos no orthogonalization. Cook an example where SO isn't triggered.
let
        println("no orthogonalization")
        evals_count = 10
        evals_true = Array{Float64}(undef, evals_count)
        make_functional_decay!(evals_true, Interval(0.95, 1), matern_1_2)
        vec_rand = Array{Float64}(undef, evals_count)
        make_kronecker_quasirand!(vec_rand)
        A = Array{Float64}(undef, evals_count, evals_count)
        make_matrix!(A, vec_rand, evals_true)
        vec_rand /= sqrt(sum(vec_rand .* vec_rand))
        context = NoOrthogonalization(vec_rand, evals_count)
        subspace = DeflatedSubspace(evals_count, evals_count)
        subspace_dim = 0
        lanczos!(A, context, subspace)
        evec_row = zeros(evals_count)
        evec_row[1] = 1.0
        qr_tridiag!(context.diagonal, context.subdiagonal, evec_row)
        evals_lanczos = context.diagonal
        println("evals_lanczos ", evals_lanczos)
        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
        println("maximum eigenvalue error", evals_error)
end

# Lanczos with selective orthogonalization; SO actually triggers.
let
        println("selective orthogonalization actually triggers.")
        evals_count = 30
        evals_true = Array{Float64}(undef, evals_count)
        make_functional_decay!(evals_true, Interval(0.95, 1), matern_1_2)
        vec_rand = Array{Float64}(undef, evals_count)
        make_kronecker_quasirand!(vec_rand)
        vec_rand ./= sqrt(sum(vec_rand .* vec_rand))
        A = Array{Float64}(undef, evals_count, evals_count)
        make_matrix!(A, vec_rand, evals_true)
        vec_rand = Array{Float64}(undef, evals_count)
        make_kronecker_quasirand!(vec_rand)
        vec_rand ./= sqrt(sum(vec_rand .* vec_rand))
        c = SelectiveOrthogonalization(vec_rand, evals_count)
        s = DeflatedSubspace(evals_count, evals_count)
        deflation = true 
        while deflation
                s, deflation = lanczos!(A, c, s)
                c.curr .= randn(evals_count)
                c.curr .= c.curr ./ sqrt(sum(c.curr .* c.curr))
        end
        diagonal = get_diagonal(c, s, s.budget-s.dim)
        subdiagonal = get_subdiagonal(c, s, s.budget-s.dim-1)
        evec_row = get_evec_row(c, s, s.budget-s.dim)
        fill!(evec_row, 0.0)
        evec_row[1] = 1.0
        qr_tridiag!(diagonal, subdiagonal, evec_row)
        evals_lanczos = c.diagonal
        println("evals_true ", evals_true)
        println("evals_lanczos ", evals_lanczos)
        evals_error = maximum(abs.(evals_true .- evals_lanczos) ./ abs.(evals_true))
        println("maximum eigenvalue error ", evals_error)
end
