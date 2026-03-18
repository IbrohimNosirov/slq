using LinearAlgebra
#using Base.Threads
#using ProgressBars
#using Distributed
#using Profile
using Plots
#using StatProfilerHTML
using OptimalTransport
using Distributions
using DataStructures

#= TODO: need much higher assert coverage; on the order of 40 (avg 2 per
 function.)
=#
# TODO: remove all dependencies except LinearAlgebra + Base.Threads.
#
# TODO: struct for multicore that has a Lanczos context, number of threads, memory budget, 

const MACHEPS = eps(Float64)
const NUM_THREADS = Threads.nthreads()

include("matrix_gallery.jl")
include("lanczos.jl")

struct SLQContext
        A                 :: AbstractMatrix{Float64}
        lanczos           :: AbstractVector{Float64}
        MC_trials_count   :: Integer
        MC_vecs_per_trial :: Integer
        SO_vecs_count     :: Integer
        memory_budget     :: Integer
end

function SLQContext(A::AbstractMatrix{Float64})
        N = size(A, 1)

        lanczos = randn(N)
        lanczos = lanczos ./ norm(lanczos)
        MC_trials_count = 200
        MC_vecs_per_trial = 10 # will get changed at runtime to never trigger selective orthogonalization.
        SO_vecs_count = max(N - MC_trials_count*MC_vecs_per_trial, 0)
        memory_budget = 30 # artificially set to be *something.* 

        SLQContext(A, lanczos, MC_trials_count, MC_vecs_per_trial, SO_vecs_count, memory_budget)
end

function slq(slq_context :: SLQContext)
        lanczos_context = LanczosContext(SO(), slq_context.A, slq_context.lanczos, slq_context.SO_vecs_count)
        lanczos(context, SO())

        # TODO: Monte Carlo
end

# need a Lanczos no orthogonalization.
function deflate_matrix!(A::AbstractMatrix{T}, Q::AbstractMatrix{T},
                         evals::AbstractVector{T}, subdiagonal::AbstractVector{T}) where T <: AbstractFloat
        evals, evecs = eigen!(SymTridiagonal(evals, subdiagonal))
        Z  = Q * evecs
        P  = I - Z*Z'
        A .= P * A * P
end
