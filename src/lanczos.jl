include("qr_tridiag.jl")
abstract type LanczosContext end

struct DeflatedSubspace{M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}
        storage::M
        work1  ::V
        work2  ::V
        work3  ::V
        work4  ::V
        dim    ::Int64 #dimension of the subspace; starts at 0
        budget ::Int64
end

function DeflatedSubspace(problem_dim::T, budget::T) where {T <: Int64}
        storage = Array{Float64}(undef, problem_dim, budget)
        work1   = Array{Float64}(undef, budget)
        work2   = Array{Float64}(undef, problem_dim)
        work3   = Array{Float64}(undef, problem_dim)
        work4   = Array{Float64}(undef, problem_dim)
        DeflatedSubspace(storage, work1, work2, work3, work4, 0, budget)
end

function DeflatedSubspace(s::DeflatedSubspace, dim::Int64)
        snew = DeflatedSubspace(s.storage, s.work1, s.work2, s.work3, s.work4, dim, s.budget)
        snew
end

function get_Q(s::DeflatedSubspace)
        vec_count = size(s.storage,2)
        @assert s.dim < vec_count
        @view s.storage[:, s.dim+1:vec_count]
end

function get_work_small(s::DeflatedSubspace)
        vec_count  = size(s.storage,2)
        @assert s.dim > 0
        @assert s.dim <= vec_count
        @view s.work1[1:s.dim]
end

function get_ritz_vecs(s::DeflatedSubspace)
        vec_count = size(s.storage,2)
        @assert s.dim > 0
        @assert s.dim <= vec_count
        @view s.storage[:, 1:s.dim]
end

function set_ritz_vecs(s::DeflatedSubspace, ritz_vecs::AbstractMatrix{Float64})
        ritz_count = size(ritz_vecs, 2)
        Q = get_Q(s)
        Q_count = size(Q,2)
        @assert ritz_count <= Q_count
        copyto!(view(Q, :, 1:ritz_count), ritz_vecs)
        # should be able to make with no allocations.
        snew = DeflatedSubspace(s, s.dim+ritz_count)
        snew
end

struct FullOrthogonalization{V <: AbstractVector{Float64}} <: LanczosContext
        diagonal   ::V
        subdiagonal::V
        curr       ::V
        prev       ::V
        work       ::V
        step_count ::Int64
end

function FullOrthogonalization(curr::AbstractVector{Float64}, step_count::Int64)
        problem_dim = size(curr, 1)
        @assert step_count <= problem_dim "step count can't be larger than problem dimension."
        @assert sum(curr .* curr) ≈ 1.0 "pass a unit vec."
        diagonal = Array{Float64}(undef, step_count)
        subdiagonal = Array{Float64}(undef, step_count-1)
        prev = similar(curr)
        work = similar(curr)
        FullOrthogonalization(diagonal, subdiagonal, curr, prev, work, step_count)
end

# Lanczos with orthogonalization at every step.
function lanczos!(A::AbstractMatrix{Float64}, c::FullOrthogonalization, s::DeflatedSubspace)
        Q = get_Q(s)
        for j = 1:c.step_count
                mul!(c.work, A, c.curr)
                if j > 1
                        c.work .-= c.subdiagonal[j-1] * c.prev
                end
                c.diagonal[j] = c.curr' * c.work
                if j == c.step_count
                        break
                end
                c.work .-= c.diagonal[j] * c.curr
                c.subdiagonal[j] = sqrt(sum(c.work .* c.work))
                if c.subdiagonal[j] < 2.0 * eps(Float64)
                        break
                end
                Q[:,j] = c.curr
                if j > 1
                        c.work .-= @views Q[:, 1:j-1] * (Q[:, 1:j-1]' * c.work)
                        c.work .-= @views Q[:, 1:j-1] * (Q[:, 1:j-1]' * c.work)
                end
                c.prev .= c.curr
                c.curr .= c.work / c.subdiagonal[j]
        end
end

struct NoOrthogonalization{V<:Vector{Float64}} <: LanczosContext
        diagonal   ::V
        subdiagonal::V
        curr       ::V
        prev       ::V
        work       ::V
        step_count ::Int64
end

function NoOrthogonalization(curr::AbstractVector{Float64}, step_count::Int64)
        @assert sum(curr .* curr) ≈ 1.0 "pass a unit vec."
        diagonal = Array{Float64}(undef, step_count)
        subdiagonal = Array{Float64}(undef, step_count-1)
        prev = similar(curr)
        work = similar(curr)
        NoOrthogonalization(diagonal, subdiagonal, curr, prev, work, step_count)
end

function lanczos!(A::AbstractMatrix{Float64}, c::NoOrthogonalization, s::DeflatedSubspace)
        for j = 1:c.step_count
                mul_subspace!(c.work, A, c.curr, s)
                if j > 1
                        c.work .-= c.subdiagonal[j-1] * c.prev
                end
                c.diagonal[j] = c.curr' * c.work
                if j == c.step_count
                        break
                end
                c.work .-= c.diagonal[j] * c.curr
                c.subdiagonal[j] = sqrt(sum(c.work .* c.work))
                if c.subdiagonal[j] < 2.0 * eps(Float64)
                        break
                end
                c.prev .= c.curr
                c.curr .= c.work / c.subdiagonal[j]
        end
end

struct SelectiveOrthogonalization{V<:Vector{Float64}} <: LanczosContext
        ritz_errors     ::V
        diagonal        ::V
        subdiagonal     ::V
        copy_diagonal   ::V
        copy_subdiagonal::V
        curr            ::V
        prev            ::V
        work            ::V
        evec_row        ::V
end

function SelectiveOrthogonalization(curr::AbstractVector{Float64}, budget::Int64)
        @assert sum(curr .* curr) ≈ 1.0 "pass a unit vec."
        ritz_errors = Array{Float64}(undef, budget)
        diagonal = Array{Float64}(undef, budget)
        subdiagonal = Array{Float64}(undef, budget-1)
        copy_diagonal = similar(diagonal)
        copy_subdiagonal = similar(subdiagonal)
        prev = similar(curr)
        work = similar(curr)
        evec_row = zeros(budget)
        SelectiveOrthogonalization(ritz_errors, diagonal, subdiagonal, copy_diagonal, copy_subdiagonal, curr, prev, work, evec_row)
end

function get_diagonal(c::SelectiveOrthogonalization, s::DeflatedSubspace, j::Int64)
        offset = s.dim
        @assert offset + j <= s.budget
        @view c.diagonal[offset+1:offset+j]
end

function get_subdiagonal(c::SelectiveOrthogonalization, s::DeflatedSubspace, j::Int64)
        offset = s.dim
        @assert offset + j <= s.budget
        @view c.subdiagonal[offset+1:offset+j]
end

function get_evec_row(c::SelectiveOrthogonalization, s::DeflatedSubspace, j::Int64)
        offset = s.dim
        @assert offset + j <= s.budget
        @view c.evec_row[offset+1:offset+j]
end

function get_ritz_errors(c::SelectiveOrthogonalization, s::DeflatedSubspace, j::Int64)
        offset = s.dim
        @assert offset + j <= s.budget
        @view c.ritz_errors[offset+1:offset+j]
end

# Lanczos with selective orthogonalization.
function lanczos!(A::AbstractMatrix{Float64}, c::SelectiveOrthogonalization, s::DeflatedSubspace)
        deflation = false
        step_count = s.budget - s.dim
        Q = get_Q(s)
        converged_indices = []
        @assert sum(c.curr .* c.curr) ≈ 1.0 "pass a unit vector."
        for j = 1:step_count
                diagonal = get_diagonal(c, s, j)
                evec_row = get_evec_row(c, s, j)
                mul_subspace!(c.work, A, c.curr, s)
                if j > 1
                        subdiagonal = get_subdiagonal(c, s, j-1)
                        ritz_errors = get_ritz_errors(c, s, j-1)
                        c.work .-= subdiagonal[j-1] * c.prev
                end
                diagonal[j] = c.curr' * c.work
                if j == step_count
                        break
                end
                Q[:,j] .= c.curr
                c.work .-= diagonal[j] * c.curr
                subdiagonal = get_subdiagonal(c, s, j)
                subdiagonal[j] = sqrt(sum(c.work .* c.work))
                if subdiagonal[j] < 2.0*eps(Float64)
                        break
                end
                c.prev .= c.curr
                c.curr .= c.work / subdiagonal[j]
                subdiagonal = get_subdiagonal(c, s, j-1)
                if j > 1
                        # selective orthogonalization check.
                        copy_diagonal = @view c.copy_diagonal[1:j]
                        copy_subdiagonal = @view c.copy_subdiagonal[1:j-1]
                        copyto!(copy_diagonal, diagonal)
                        copyto!(copy_subdiagonal, subdiagonal)
                        fill!(evec_row, 0.0)
                        evec_row[j] = 1.0
                        qr_tridiag!(copy_diagonal, copy_subdiagonal, evec_row)
                        Tj_norm = maximum(abs.(copy_diagonal))
                        @views ritz_errors .= abs.(evec_row[1:j-1] .* subdiagonal)
                        converged_indices = findall(<(sqrt(eps(Float64))*Tj_norm),abs.(ritz_errors))
                end
                if !isempty(converged_indices)
                        deflation_dim = size(converged_indices,1)
                        evals, evecs = eigen!(SymTridiagonal(diagonal, subdiagonal))
                        #slightly complicated; example: indices = [0,1,0,1] -> diagonal[1,2] .= diagonal[2,4]
                        @views diagonal[1:deflation_dim] .= evals[converged_indices]
                        @views evec_row[1:deflation_dim] .= evecs[1, converged_indices] # first row index.
                        @views Q[:, 1:j] .= Q[:, 1:j] * evecs[1:j]
                        @views Q[:, 1:deflation_dim] .= Q[:, converged_indices]
                        snew = DeflatedSubspace(s, s.dim + deflation_dim)
                        deflation = true
                        return snew, deflation
                end
        end
        s, deflation
end

# correctly does target = (I - QQ')A(I - QQ')*source; don't alias target and source.
function mul_subspace!(target::StridedVector{Float64}, A::AbstractMatrix{Float64}, source::StridedVector{Float64},
                        s::DeflatedSubspace)
        if s.dim == 0
                mul!(target, A, source)
        else
                Q = get_ritz_vecs(s)
                Qt = Q'
                work_small = get_work_small(s)
                # s.work3 = (I - QQ')*source
                mul!(work_small, Qt, source)
                mul!(s.work2, Q, work_small)
                # s.work3 = source - s.work2
                copyto!(s.work3, source)
                axpy!(-1.0, s.work2, s.work3)
                # s.work4 = A(I - QQ')*source
                mul!(s.work4, A, s.work3)
                # target = (I - QQ')A(I - QQ')*source
                mul!(work_small, Qt, s.work4)
                mul!(s.work2, Q, work_small)
                #target .= s.work4 .- s.work2
                copyto!(target, s.work4)
                axpy!(-1.0, s.work2, target)
        end
end
