include("lanczos.jl")
include("../utils/matrix_gallery.jl")
using Base.Threads

# TODO: check to make sure s.dim works.
"""
This struct handles all memory allocation and bookkeeping for the algorithm.
Any further allocation should be deemed a bug.
The number of selective orthogonalization steps should vary depending on the matrix spectrum,
but is capped by the memory budget.
That is, each deflation operation forces us to store the last k vectors and the
next restarted iteration works in a smaller space of size (problem dimension)-(subspace dimension).
We can keep deflating until subspace dim == memory_budget.
Monte-Carlo trial count should be large enough for CLT to kick in; here it's set to 200.
The number of Lanczos steps should be set based on the compute budget
available, as measured in matvecs, after selective orthogonalization.
As it is billed as a sublinear-time algorithm, this implementation has a hard cap of matvecs;
they can't be larger than problem_dimension.
"""

struct SLQ_Context
        problem_dim   ::Int64
        MC_trial_count::Int64
        MC_step_count ::Int64
        SO_step_count ::Int64 
        worker_count  ::Int64
        SO_context    ::SelectiveOrthogonalization
        MC_context_arr::Array{NoOrthogonalization}
        subspace      ::DeflatedSubspace
end

function SLQ_Context(problem_dim::Int64, memory_budget::Int64, worker_count::Int64)
        MC_trial_count = 200
        MC_step_count = div((problem_dim - memory_budget), MC_trial_count)
        @assert MC_step_count > 0 "problem too small for SLQ."
        SO_step_count = memory_budget # TODO: there is a subtle bug lurking here.
        vec_start = Array{Float64}(undef, problem_dim)
        make_kronecker_quasirand!(vec_start)
        vec_start ./= norm(vec_start)
        SO_context = SelectiveOrthogonalization(vec_start, SO_step_count)
        MC_context_arr = Array{NoOrthogonalization}(undef, worker_count)
        for i=1:worker_count
                vec_start = Array{Float64}(undef, problem_dim)
                make_kronecker_quasirand!(vec_start)
                vec_start ./= norm(vec_start)
                MC_context_arr[i] = NoOrthogonalization(vec_start, MC_step_count)
        end
        subspace = DeflatedSubspace(problem_dim, memory_budget)
        SLQ_Context(problem_dim, MC_trial_count, MC_step_count, SO_step_count, worker_count, SO_context, MC_context_arr, subspace)
end

# should give back an n x 1 vector with eigenvalue estimates.
function slq(A::AbstractMatrix{Float64}, evals::AbstractVector{Float64}, evec_row::AbstractVector{Float64}, memory_budget::Int64, worker_count::Int64)
        problem_dim = size(A,1)
        @assert size(evec_row, 1) == problem_dim
        @assert size(evals,1) == problem_dim
        evec_row .= fill(1/problem_dim, problem_dim)
        # minimum memory budget to at least run Vanilla SLQ (worker_count * 4 matvecs)
        @assert memory_budget > worker_count * 4 "memory budget too small."
        s = SLQ_Context(problem_dim, memory_budget, worker_count)
        subspace = s.subspace
        so_context = s.SO_context
        mc_context_arr = s.MC_context_arr
        subspace_dim = 0
        subspace_dim = lanczos!(A, so_context, subspace, subspace_dim)
        println(subspace_dim)
        if subspace_dim > 1
                evals[1:subspace_dim] .= so_context.diagonal[1:subspace_dim]
        end
        Threads.@spawn for i = 1:s.MC_trial_count
                mc_context = mc_context_arr[i%worker_count + 1]
                lanczos!(A, mc_context, subspace, subspace_dim)
                evec_row_portion  = @view evec_row[subspace_dim+1:subspace_dim+s.MC_step_count]
                evec_row_portion .= zeros(s.MC_step_count)
                evec_row_portion[1] = 1.0
                qr_tridiag!(mc_context.diagonal, mc_context.subdiagonal, evec_row_portion)
                evals[subspace_dim+1:subspace_dim+s.MC_step_count] .+= mc_context.diagonal ./ s.MC_trial_count
        end
        evals = @view evals[1:subspace_dim+s.MC_step_count]
        weights = @view evec_row[1:subspace_dim+s.MC_step_count]
        # How to get estimate for the rest?
        # convolution with Gaussian kernel.
        gaussian_convolution(evals, weights)
end

function gaussian_convolution(evals::Vector{Float64}, weights::Vector{Float64})
        e_min, e_max = extrema(evals)
        grid = range(e_min, e_max, length=1000)
        density = zeros(length(grid))
        inv_sqrt2pi = 1.0 / sqrt(2 * π)
        Threads.@spawn for i in eachindex(grid)
                val = 0.0
                g_point = grid[i]
                for j in eachindex(evals)
                        diff = g_point - evals[j]
                        val += weights[j] * exp(-0.5 * diff^2)
                end
                density[i] = val * inv_sqrt2pi
        end
        grid, density
end

#function eig_lapack!(diagonal::Vector{Float64}, subdiagonal::Vector{Float64})
#    n = length(diagonal)
#    if n <= 1 return diagonal end
#
#    # dstev requires:
#    # JOBZ: 'N' (eigenvalues only) or 'V' (eigenvalues and vectors)
#    # N: Order of matrix
#    # D: Diagonal (length n)
#    # E: Subdiagonal (length n-1)
#    # Z: Eigenvector matrix (not used if 'N')
#    # LDZ: Leading dimension of Z
#    # WORK: Workspace array (length max(1, 2n-2))
#    # INFO: Status output
#    
#    jobz = 'N'
#    ldz = 1
#    z = Ref{Float64}(0.0) # Dummy for JOBZ = 'N'
#    work = Vector{Float64}(undef, max(1, 2n - 2))
#    info = Ref{Int64}(0)
#    
#    # Using the subdiagonal slice (n-1)
#    # Note: we copy the subdiagonal because LAPACK overwrites it
#    e_copy = subdiagonal[1:n-1]
#
#    ccall((:dstev_64_, Base.liblapack_name), Cvoid,
#          (Ref{UInt8}, Ref{Int64}, Ptr{Float64}, Ptr{Float64}, 
#           Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}),
#          jobz, n, diagonal, e_copy, 
#          z, ldz, work, info)
#
#    if info[] != 0
#        error("LAPACK dstev failed with info = $(info[])")
#    end
#
#    return diagonal
#end
