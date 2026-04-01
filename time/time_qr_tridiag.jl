using LinearAlgebra
using DelimitedFiles
using Printf

include("qr_tridiag.jl")

function time_mine(diag::Vector{Float64}, subdiag::Vector{Float64}, evec_row::Vector{Float64}, n::Int64)
        evec_row[1] = 1.0
	t = @elapsed qr_tridiag!(view(diag, 1:n), view(subdiag, 1:n-1), view(evec_row, 1:n))
        diag[1:n] .= fill(2.0, n)
        subdiag[1:n-1] .= fill(-1.0, n-1)
        evec_row[1:n] .= zeros(n)

	t
end

function time_lapack(diag::Vector{Float64}, subdiag::Vector{Float64}, n::Int64)
        T = SymTridiagonal(view(diag, 1:n), view(subdiag, 1:n-1))
        t = @elapsed evals_lapack, evecs_lapack = eigen!(T)
        diag[1:n] .= fill(2.0, n)
        subdiag[1:n-1] .= fill(-1.0, n-1)

        t
end

function run_time()
        sample_count = 15
        sizes = round.(Int, exp10.(range(1, 4, length=sample_count)))
        diag = fill(2.0, sizes[end])
        subdiag = fill(-1.0, sizes[end]-1)
        evec_row = zeros(sizes[end])
	open("time_qr_tridiag.csv", "w") do io
		println(io, "n,time_mine,time_lapack,max_eval_rel_error")
		for n in sizes
			println("Running n = $n")
			t_mine = time_mine(diag, subdiag, evec_row, n)
                        t_lapack = time_lapack(diag, subdiag, n)
			@printf(io, "%d,%.6e,%.6e\n", n, t_mine, t_lapack)
			flush(io)
		end
	end
	println("Timing complete. Results written to test_qr_tridiag.csv")
end

run_time()
