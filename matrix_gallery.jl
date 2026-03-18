using LinearAlgebra
using Random

Random.seed!(3)

# Utility functions should not allocate any memory.
function make_kronecker_quasirandom!(vec_random::AbstractVector{Float64}, start=0)
        points_count = size(vec_random, 1)
        d = 1
        φ = 1.0 + 1.0/d
        for k = 1:10
                gφ = φ^(d + 1) - φ - 1
                dgφ= (d + 1)*φ^d - 1
                φ -= gφ/dgφ
        end
        αs = [mod(1.0/φ^j, 1.0) for j = 1:d]
        for j = 1:points_count
                vec_random[j] = mod(0.5 + (start+j)*αs[d], 1.0)
        end
end

function reduce_tridiag!(A::AbstractMatrix)
        n = size(A, 1)
        τ = view(A, 1:n, n)
        for k = 1:n-2
                x = view(A, k+1:n, k)
                τk = LinearAlgebra.reflector!(x)
                LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n))
                LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n)')
                τ[k] = τk
        end
end

# diagonal, not diag, because diag() is a Julia function.
function tridiag_params!(A, diagonal, subdiagonal)
        count = size(subdiagonal, 1)
        @assert size(diagonal, 1) - count == 1
        for i = 1:count
                diagonal[i] = A[i,i]
                subdiagonal[i] = A[i+1,i]
        end
        diagonal[count+1] = A[count+1, count+1]
end

# make_matrix! creates a dense matrix with the same eigenvalues as the input
# 'evals'. This is done using ten Householder reflectors.
function make_matrix!(A::AbstractMatrix{Float64}, vec_random::AbstractVector{Float64}, evals::AbstractVector{Float64})
        H = I - 2*vec_random*vec_random'./(vec_random'*vec_random)
        A .= H * diagm(evals) * H'
        for i = 1:10
                make_kronecker_quasirandom!(vec_random)
                H .= I - 2*vec_random*vec_random'./(vec_random'*vec_random)
                A .= H * A * H'
        end
end

function make_tridiag_matrix(A::AbstractMatrix{Float64}, vec_random::AbstractVector{Float64},
                                evals::AbstractVector{Float64})
        A = make_matrix!(A, evals)
        reduce_tridiag!(A)
        diagonal = diag(A)
        subdiagonal = diag(A, -1)

        diagonal, subdiagonal
end

struct Interval
        start  :: Float64
        finish :: Float64 
end

function Interval(start::Float64, finish::Float64)
        @assert start  > 0
        @assert finish > 0 
        @assert finish - start >= 0

        Interval(start, finish)
end

function make_functional_decay!(evals::AbstractVector, interval::Interval, fun::Function)
        evals_count = size(evals, 1)
        @assert evals_count > 3

        interval_range = interval.finish - interval.start
        evals .= collect(range(0, evals_count-1)) ./ evals_count .* interval_range .+ interval.start
        evals .= fun.(evals) .+ 2*sqrt(eps(Float64))*rand(evals_count)

        sort!(evals)
end

function make_cluster!(evals::AbstractVector, interval::Interval, epsilon::Float64)
        evals_count = size(evals, 1)
        @assert evals_count > 0
        @assert epsilon > 1e-8

        seed = evals_count * 42
        # there are two different kinds of partition: by index (i) and by range (e).
        interval_range = interval.finish - interval.start
        evals .= epsilon .* kronecker_quasirand_vec(evals_count) ./ interval_range .+ interval.start

        sort!(evals)
end

# gaussian kernel
gaussian(r) = exp(-r*r)

# Matérn ν = 1/2 (C^0)
matern_1_2(r, l=1.0) = exp(-r/l)

# Matérn ν = 3/2 (C^2)
matern_3_2(r, l=1.0) = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)

# Matérn ν = 5/2 (C^4)
matern_5_2(r, l=1.0) = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)

# Matérn ν = 7/2 (C^6)
matern_7_2(r, l=1.0) = (1 + sqrt(7)*r/l + 14*r^2/(5*l^2) + 7*sqrt(7)*r^3/(15*l^3)) * exp(-sqrt(7)*r/l)

# Matérn ν = 9/2 (C^8)
matern_9_2(r, l=1.0) = (1 + 3*r/l + 27*r^2/(7*l^2) + 18*r^3/(7*l^3) + 27*r^4/(35*l^4)) * exp(-3*r/l)

# Matérn ν = 11/2 (C^10)
matern_11_2(r, l=1.0) = (1 + sqrt(11)*r/l + 55*r^2/(9*l^2) + 55*sqrt(11)*r^3/(27*l^3) + 1375*r^4/(567*l^4) + 275*sqrt(11)*r^5/(1701*l^5)) * exp(-sqrt(11)*r/l)

# Matérn ν = 13/2 (C^12)
matern_13_2(r, l=1.0) = (1 + sqrt(13)*r/l + 26*r^2/(3*l^2) + 13*sqrt(13)*r^3/(9*l^3) + 169*r^4/(54*l^4) + 169*sqrt(13)*r^5/(486*l^5) + 2197*r^6/(4374*l^6)) * exp(-sqrt(13)*r/l)
