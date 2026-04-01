include("../src/slq.jl")

let
        n = 100
        A = randn(n,n)
        evals = Array{Float64}(undef, n)
        evec_row = Array{Float64}(undef, n)
        memory_budget = 30
        worker_count = Threads.nthreads()
        slq(A, evals, evec_row, memory_budget, worker_count)
        println(evals)
        println(evec_row)
end

# functional trace estimation test.
let

end

# spectral density test.
let
end
