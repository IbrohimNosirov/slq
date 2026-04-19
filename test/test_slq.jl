include("../src/slq.jl")
include("../utils/matrix_gallery.jl")

let
        evals_count = 1000
        memory_budget = 200
        evals_true = Array{Float64}(undef, evals_count)
        vec_rand = Array{Float64}(undef, evals_count)
        A = Array{Float64}(undef, evals_count, evals_count)
        make_cluster!(evals_true, Interval(0.1, 1), 1.0)
        make_kronecker_quasirand!(vec_rand)
        make_matrix!(A, vec_rand, evals_true)
        evals = Array{Float64}(undef, evals_count)
        evec_row = Array{Float64}(undef, evals_count)
        worker_count = Threads.nthreads()
        slq(A, evals, evec_row, memory_budget, worker_count)
        println(evals[1:200])
        println(evec_row[1:200])
end

# functional trace estimation test.
let

end

# spectral density test.
let
end
