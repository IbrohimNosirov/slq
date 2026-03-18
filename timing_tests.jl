using LinearAlgebra
using DelimitedFiles
using Printf

include("qr_tridiag.jl")

function make_test_tridiag(n)
	diagonal = fill(2.0, n)
	subdiagonal = fill(-1.0, n-1)
	return diagonal, subdiagonal
end

function time_qr_tridiag_evec(diag, subdiag)
	d = copy(diag)
	s = copy(subdiag)
	evec = zeros(size(d, 1))
        evec[1] = 1.0
	t = @elapsed qr_tridiag!(d, s, evec)
	t, d, evec
end

function time_lapack(diag, subdiag)
	T = SymTridiagonal(copy(diag), copy(subdiag))
	t = @elapsed evals, evecs = eigen!(T)
	t, evals, evecs[:,1]
end

function run_benchmarks()
        sizes = round.(Int, exp10.(range(1, 4, length=15)))
	open("qr_tridiagonal_benchmark.csv", "w") do io
		println(io, "n,qr_time,lapack_time,max_eval_rel_error")
		for n in sizes
			println("Running n = $n")
			diag, subdiag = make_test_tridiag(n)
			qr_time, evals_qr, _ = time_qr_tridiag_evec(diag, subdiag)
			lapack_time, evals_lapack, _ = time_lapack(diag, subdiag)
			max_err = maximum(abs.(evals_qr .- evals_lapack) ./ abs.(evals_lapack))
			@printf(io, "%d,%.6e,%.6e,%.6e\n",
					n, qr_time, lapack_time, max_err)
                        # why was the IO flushed here?
			flush(io)
		end
	end
	println("Benchmark complete. Results written to qr_tridiagonal_benchmark.csv")
end

run_benchmarks()
