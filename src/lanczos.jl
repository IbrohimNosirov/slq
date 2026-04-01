include("qr_tridiag.jl")
abstract type LanczosContext end

# TODO: finish and test subspace dim logic.
struct DeflatedSubspace
        Q_store::AbstractMatrix{Float64}
        work1  ::AbstractVector{Float64}
        work2  ::AbstractVector{Float64}
        work3  ::AbstractVector{Float64}
        work4  ::AbstractVector{Float64}
        dim    ::Int64
end

function DeflatedSubspace(problem_dim::Int64, memory_budget::Int64)
        Q_store = Array{Float64}(undef, problem_dim, memory_budget)
        work1 = Array{Float64}(undef, problem_dim)
        work2 = Array{Float64}(undef, problem_dim)
        work3 = Array{Float64}(undef, problem_dim)
        work4 = Array{Float64}(undef, problem_dim)
        dim = 1
        DeflatedSubspace(Q_store, work1, work2, work3, work4, dim)
end

struct FullOrthogonalization <: LanczosContext
        Q_store   ::Matrix{Float64}
        diag      ::Vector{Float64}
        subdiag   ::Vector{Float64}
        vec_curr  ::Vector{Float64}
        vec_prev  ::Vector{Float64}
        vec_work  ::Vector{Float64}
        step_count::Int64
end

function FullOrthogonalization(vec_curr::AbstractVector{Float64}, step_count::Int64)
        @assert sum(vec_curr .* vec_curr) ≈ 1.0 "pass a unit vec."
        problem_dim = size(vec_curr, 1)
        Q_store = Array{Float64}(undef, problem_dim, step_count)
        diag = Array{Float64}(undef, step_count)
        subdiag = Array{Float64}(undef, step_count-1)
        vec_prev = similar(vec_curr)
        vec_work = similar(vec_curr)
        FullOrthogonalization(Q_store, diag, subdiag, vec_curr, vec_prev, vec_work, step_count)
end

# Lanczos with orthogonalization at every step.
function lanczos!(A::AbstractMatrix{Float64}, c::FullOrthogonalization)
        @assert c.Q_store != nothing
        for j = 1:c.step_count
                c.vec_work .= A * c.vec_curr
                if j > 1
                        c.vec_work .-= c.subdiag[j-1] * c.vec_prev
                end
                c.diag[j] = c.vec_curr' * c.vec_work
                if j == c.step_count
                        break
                end
                c.vec_work .-= c.diag[j] * c.vec_curr
                c.subdiag[j] = sqrt(sum(c.vec_work .* c.vec_work))
                if c.subdiag[j] < 2.0 * eps(Float64)
                        break
                end
                c.Q_store[:,j] = c.vec_curr
                c.vec_work .-= @views c.Q_store[:, 1:j] * (c.Q_store[:, 1:j]' * c.vec_work)
                c.vec_work .-= @views c.Q_store[:, 1:j] * (c.Q_store[:, 1:j]' * c.vec_work)
                c.vec_prev .= c.vec_curr
                c.vec_curr .= c.vec_work / c.subdiag[j]
        end
end

struct NoOrthogonalization <: LanczosContext
        diag      ::Vector{Float64}
        subdiag   ::Vector{Float64}
        vec_curr  ::Vector{Float64}
        vec_prev  ::Vector{Float64}
        vec_work  ::Vector{Float64}
        step_count::Int64
end

function NoOrthogonalization(vec_curr::AbstractVector{Float64}, step_count::Int64)
        @assert sum(vec_curr .* vec_curr) ≈ 1.0 "pass a unit vec."
        diag = Array{Float64}(undef, step_count)
        subdiag = Array{Float64}(undef, step_count-1)
        vec_prev = similar(vec_curr)
        vec_work = similar(vec_curr)
        NoOrthogonalization(diag, subdiag, vec_curr, vec_prev, vec_work, step_count)
end

function lanczos!(A::AbstractMatrix{Float64}, c::NoOrthogonalization, s::DeflatedSubspace)
        for j = 1:c.step_count
                mul_subspace!(s, A, c.vec_curr, c.vec_work)
                if j > 1
                        c.vec_work .-= c.subdiag[j-1] * c.vec_prev
                end
                c.diag[j] = c.vec_curr' * c.vec_work
                if j == c.step_count
                        break
                end
                c.vec_work .-= c.diag[j] * c.vec_curr
                c.subdiag[j] = sqrt(sum(c.vec_work .* c.vec_work))
                if c.subdiag[j] < 2.0 * eps(Float64)
                        break
                end
                c.vec_prev .= c.vec_curr
                c.vec_curr .= c.vec_work / c.subdiag[j]
        end
end

struct SelectiveOrthogonalization <: LanczosContext
        ritz_errors ::Vector{Float64}
        diag        ::Vector{Float64}
        subdiag     ::Vector{Float64}
        copy_diag   ::Vector{Float64}
        copy_subdiag::Vector{Float64}
        vec_curr    ::Vector{Float64}
        vec_prev    ::Vector{Float64}
        vec_work    ::Vector{Float64}
        evec_row    ::Vector{Float64}
        step_count  ::Int64
end

function SelectiveOrthogonalization(vec_curr::AbstractVector{Float64}, step_count::Int64)
        @assert sum(vec_curr .* vec_curr) ≈ 1.0 "pass a unit vec."
        problem_dim = size(vec_curr, 1)
        ritz_errors = Array{Float64}(undef, step_count)
        diag = Array{Float64}(undef, step_count)
        subdiag = Array{Float64}(undef, step_count-1)
        copy_diag = similar(diag)
        copy_subdiag = similar(subdiag)
        vec_prev = similar(vec_curr)
        vec_work = similar(vec_curr)
        evec_row = zeros(step_count)
        SelectiveOrthogonalization(ritz_errors, diag, subdiag, copy_diag, copy_subdiag, vec_curr, vec_prev, vec_work, evec_row, step_count)
end

# Lanczos with selective orthogonalization.
function lanczos!(A::AbstractMatrix{Float64}, c::SelectiveOrthogonalization, s::DeflatedSubspace)
        deflate = false
        @assert c.step_count - s.dim > 0
        for j = s.dim:c.step_count
                mul_subspace!(s, A, c.vec_curr, c.vec_work)
                if j > s.dim
                        c.vec_work .-= c.subdiag[j-1] * c.vec_prev
                end
                c.diag[j] = c.vec_curr' * c.vec_work
                if j == c.step_count
                        break
                end
                c.vec_work .-= c.diag[j] * c.vec_curr
                c.subdiag[j] = sqrt(sum(c.vec_work .* c.vec_work))
                if c.subdiag[j] < 2.0*eps(Float64)
                        break
                end
                s.Q_store[:,s.dim] = c.vec_curr
                c.vec_prev .= c.vec_curr
                c.vec_curr .= c.vec_work / c.subdiag[j]
                if j > s.dim
                        c.copy_diag[s.dim:j] .= c.diag[s.dim:j]
                        c.copy_subdiag[s.dim:j-1] .= c.subdiag[s.dim:j-1]
                        c.evec_row[j] = 1.0
                        deflate = convergence_check!(view(c.copy_diag, s.dim:j), view(c.copy_subdiag, s.dim:j-1),
                                                        view(c.subdiag, s.dim:j-1), view(c.evec_row, s.dim:j))
                        @assert c.subdiag[s.dim:j-1] != c.copy_subdiag[s.dim:j-1]
                end
                if deflate
                        diag, evecs = eigen!(SymTridiag(diag, subdiag))
                        view(s.Q, s.dim:s.dim+j) .*= evecs
                        s.dim = j
                        break
                end
        end
        return deflate
end

# Check if a lanczos estimate has converged to a Ritz value. SubArray is here because I should only be allowed to pass in views.
function convergence_check!(copy_diag::SubArray{Float64}, copy_subdiag::SubArray{Float64}, subdiag::SubArray{Float64}, evec_row::SubArray{Float64})
        qr_tridiag!(copy_diag, copy_subdiag, evec_row)
        Tj_norm = maximum(abs.(copy_diag))
        # copy_subdiag is now all zeros.
        ritz_errors = copy_subdiag
        ritz_errors .= abs.(@view(evec_row[1:end-1]) .* subdiag)
        evec_row .= zeros(size(evec_row,1))
        for e in ritz_errors
                if e < sqrt(eps(Float64)) * Tj_norm
                        return true
                end
        end
        return false
end

# TODO: check this
function mul_subspace!(s::DeflatedSubspace, A::AbstractMatrix{Float64}, source::AbstractVector{Float64}, target::AbstractVector{Float64})
        s.work1 .= A * source
        if s.dim == 1
                target = s.work1
        else
                Q = view(s.Q, 1:s.dim)
                s.work2 .= copy(source)
                s.work2 .= Q * (Q' * s.work_1)
                s.work3 .= A * (Q * (Q' * source))
                s.work4 .= copy(s.work3)
                s.work4 .= Q * (Q' * s.work4)
                target .= s.work1 - s.work2 - s.work3 + s.work4
        end
end
