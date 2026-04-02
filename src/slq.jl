include("lanczos.jl")
include("../utils/matrix_gallery.jl")
using Base.Threads

#= This struct handles all memory allocation and bookkeeping for the algorithm.
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
=#
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
        MC_step_count = (problem_dim - memory_budget)/MC_trial_count
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
        evec_row .= ones(problem_dim) ./ problem_dim
        # minimum memory budget to at least run Vanilla SLQ (worker_count * 4 matvecs)
        @assert memory_budget > worker_count * 4 "memory budget too small."
        slq_context = SLQ_Context(problem_dim, memory_budget, worker_count)
        subspace = slq_context.subspace
        so_context = slq_context.SO_context
        mc_context_arr = slq_context.MC_context_arr
        lanczos!(A, so_context, subspace)
        evals[1:subspace.dim] .= so_context.diag[1:subspace.dim]
        evec_row[1:subspace.dim] .= so_context.evec_row[1:subspace.dim]
        Threads.@spawn for i = 1:MC_trial_count
                MC_context = MC_context_arr[i % worker_count]
                lanczos!(A, MC_context, slq_context.subspace)
                evals[subspace.dim+1:subspace.dim+slq_context.MC_step_count] .+= MC_context.diag ./ MC_trial_count
        end
        # How to get estimate for the rest?
end

#function eig_lapack!(diag, subdiag)
#        ccall((:dlaev2_64_, Base.liblapack_name), Cvoid,
#end
