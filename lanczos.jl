include("qr_tridiagonal.jl")

# TODO: there is a way to do this correctly -> we'll pass FullOrth and not FullOrth()
# I read in the documentation that there is a idiomatic way of making buttons.

abstract type OrthStrategy end
struct FullOrth <: OrthStrategy end
struct PRO <: OrthStrategy end
struct FirstOrth <: OrthStrategy end
struct SO <: OrthStrategy end

struct LanczosContext
	A            :: Matrix{Float64}
	Q_store      :: Union{Matrix{Float64}, Nothing}
	W_store      :: Union{Matrix{Float64}, Nothing}
	R_store      :: Matrix{Float64} 
	diagonal     :: Vector{Float64}
	subdiagonal  :: Vector{Float64}
	vec_curr     :: Vector{Float64}
        vec_prev     :: Vector{Float64}
        vec_residual :: Vector{Float64} # I think we can get rid of this(?)
        evec_row     :: Vector{Float64}
	step_count   :: Int64
	matrix_size  :: Int64
end

function LanczosContext(::FullOrth, A::AbstractMatrix, vec_curr::AbstractVector, step_count::Int64)
        @assert sum(vec_curr .* vec_curr) ≈ 1.0 "pass a unit vec."
        matrix_size = size(A, 1)
        Q_store = Array{Float64}(undef, matrix_size, step_count)
        W_store = nothing
        R_store = Array{Float64}(undef, step_count, step_count)
        diagonal = Array{Float64}(undef, step_count)
        subdiagonal = Array{Float64}(undef, step_count-1)
        vec_prev = similar(vec_curr)
        vec_residual = similar(vec_curr)
        evec_row = zeros(matrix_size)
        evec_row[1] = 1.0
        LanczosContext(A, Q_store, W_store, R_store, diagonal, subdiagonal, vec_curr, vec_prev, vec_residual, evec_row,
                        step_count, matrix_size)
end

#function LanczosContext(::PRO, A :: AbstractMatrix, vec_curr :: AbstractVector, k :: Int64)
#        @assert sum(vec_curr .* vec_curr) ≈ 1.0 "pass a unit vec."
#        matrix_size = size(A, 1)
#
#        Q_store = zeros(matrix_size, k)
#        W_store = diagonalm(0  => ones(k+1), -1  => eps(Float64)*ones(k), 1  => eps(Float64)*ones(k))
#        R_store = zeros(k, k)
#        diagonal = zeros(k)
#        subdiagonal = zeros(k-1)
#        w_distance_arr = zeros(k)
#
#        LanczosContext(A, Q_store, W_store, R_store, diagonal, subdiagonal, vec_curr, w_distance_arr, k, matrix_size)
#end

#function LanczosContext(::FirstOrth, A :: AbstractMatrix, vec_curr :: AbstractVector, k :: Int64)
#        @assert sum(vec_curr .* vec_curr) ≈ 1.0 "pass a unit vec."
#        matrix_size = size(A, 1)
#
#        Q_store = nothing
#        W_store = diagonalm(0  => ones(k+1), -1  => eps(Float64)*ones(k), 1  => eps(Float64)*ones(k))
#        R_store = zeros(k, k)
#        diagonal = zeros(k)
#        subdiagonal = zeros(k-1)
#        w_distance_arr = zeros(k)
#
#        LanczosContext(A, Q_store, W_store, R_store, diagonal, subdiagonal, vec_curr, w_distance_arr, k, matrix_size)
#end

function LanczosContext(::SO, A, vec_curr, step_count)
        @assert sum(vec_curr .* vec_curr) ≈ 1.0 "pass a unit vec."
        matrix_size = size(A, 1)
        Q_store = Array{Float64}(undef, matrix_size, step_count)
        W_store = nothing
        R_store = Array{Float64}(undef, step_count, step_count)
        diagonal = Array{Float64}(undef, step_count)
        subdiagonal = Array{Float64}(undef, step_count-1)
        vec_prev = similar(vec_curr)
        vec_residual = similar(vec_curr)
        evec_row = zeros(matrix_size)
        evec_row[1] = 1.0
        LanczosContext(A, Q_store, W_store, R_store, diagonal, subdiagonal, vec_curr, vec_prev, vec_residual, evec_row,
                        step_count, matrix_size)
end

## Dispatch to implementations
#lanczos(context::LanczosContext, ::PRO) = lanczos_pro(context)
#lanczos(context::LanczosContext, ::FirstOrth) = lanczos_first_orth(context)

# Lanczos with orthogonalization at every step.
function lanczos(::FullOrth, c :: LanczosContext)
        @assert size(c.A) == (c.matrix_size, c.matrix_size)
        @assert size(c.vec_curr, 1)  == c.matrix_size
        @assert size(c.diagonal, 1) == c.step_count
        @assert size(c.diagonal, 1) - size(c.subdiagonal, 1) == 1
        @assert c.Q_store != nothing
        @assert c.W_store == nothing

        for j = 1:c.step_count
                c.vec_residual .= c.A * c.vec_curr

                if j > 1
                        c.vec_residual .-= c.subdiagonal[j-1] * c.vec_prev
                end

                c.diagonal[j] = c.vec_curr' * c.vec_residual

                if j == c.step_count
                        break
                end

                c.vec_residual .-= c.diagonal[j] * c.vec_curr
                c.subdiagonal[j] = sqrt(sum(c.vec_residual .* c.vec_residual))

                if c.subdiagonal[j] < 2.0 * eps(Float64)
                        break
                end
      
                c.Q_store[:,j] = c.vec_curr

                c.vec_residual .-= @views c.Q_store[:, 1:j] * (c.Q_store[:, 1:j]' * c.vec_residual)
                c.vec_residual .-= @views c.Q_store[:, 1:j] * (c.Q_store[:, 1:j]' * c.vec_residual)

                c.vec_prev .= c.vec_curr
                c.vec_curr .= c.vec_residual / c.subdiagonal[j]
        end
end

# TODO: implement this for completeness's sake.
## Lanczos with selective orthogonalization (LanPRO).
#function lanczos_pro(context :: LanczosContext)
#  A = get_A(context)
#  q = get_vec_curr(context)
#  a = get_diagonal(context)
#  b = get_subdiagonal(context)
#  W = get_W_store(context)
#
#  norm_A = norm(A)
#
#  z = A * q
#  a[1] = q' * z
#  z = z - a[1]*q
#  b[1] = norm(z)
#
#  for j = 2:context.step_count
#      q_prev = q
#      q = z / b[j-1]
#      Q = get_Q_store(context, j)
#      Q[:,j-1] = q
#
#      z = A*q - b[j-1] * q_prev
#      a[j] = q' * z
#      z = z - a[j]*q
#      b[j] = norm(z)
#
#      if b[j] == 0.0
#        break
#      end
#
#      orthogonalized = false
#      for i = 2:j
#          w_tilde  = b[i]*W[j,i+1] + (a[i] - a[j])*W[j,i]
#          w_tilde += b[i-1]*W[j,i-1] - b[j-1]*W[j-1,i]
#          W[j+1,i] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]
#
#          if W[j+1,i] > sqrt(eps(Float64))
#            if orthogonalized == false
#              U = @view Q[:,1:j-1]
#              U = Matrix(qr(U).Q)
#              @assert U' * U ≈ I(j-1)
#              z -= U * (U' * z)
#              z -= U * (U' * z)
#              orthogonalized = true
#            end
#            W[j+1,i] = eps(Float64)
#            W[j,i] = eps(Float64)
#          end
#      end
#      
#      # vec_residuals
#      compute_vec_residuals!(context, j)
#      
#      # Wasserstein distance.
#      μ = compute_μ(context, j)
#      context.w_distance_arr[j] = wasserstein(μ, context.ν_distribution; p=Val(1))
#  end
#
#  context.step_count
#end

## Lanczos until the first reorthogonalization.
#function lanczos_first_orth(context :: LanczosContext)
#  A = get_A(context)
#  q = get_vec_curr(context)
#  a = get_diagonal(context)
#  b = get_subdiagonal(context)
#  W = get_W_store(context)
#
#  norm_A = norm(A)
#
#  z = A * q
#  a[1] = q' * z
#  z = z - a[1]*q
#  b[1] = norm(z)
#
#  for j = 2:context.step_count
#    q_prev = q
#    q = z / b[j-1]
#
#    z = A*q - b[j-1] * q_prev
#    a[j] = q' * z
#    z = z - a[j]*q
#    b[j] = norm(z)
#
#    if b[j] == 0
#      break
#    end
#
#    # vec_residuals
#    compute_vec_residuals!(context, j)
#    
#    # Wasserstein distance.
#    μ = compute_μ(context, j)
#    context.w_distance_arr[j] = wasserstein(μ, context.ν_distribution; p=Val(1))
#
#    for i = 2:j
#      w_tilde  = b[i]*W[j,i+1] + (a[i] - a[j])*W[j,i]
#      w_tilde += b[i-1]*W[j,i-1] - b[j-1]*W[j-1,i]
#      W[j+1,i] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]
#      if W[j+1,i] > sqrt(eps(Float64))
#        println("converged at iter ", j)
#        return j
#      end
#    end
#  end
#
#  context.step_count
#end

# Lanczos with selective orthogonalization.
function lanczos!(::SO, c::LanczosContext)
        @assert c.Q_store != nothing
        @assert c.W_store == nothing
        deflate = false
        for j = 1:c.step_count
                c.vec_residual .= c.A * c.vec_curr
                if j > 1
                        c.vec_residual .-= c.subdiagonal[j-1] * c.vec_prev
                end
                c.diagonal[j] = c.vec_curr' * c.vec_residual
                if j == c.step_count
                        break
                end
                c.vec_residual .-= c.diagonal[j] * c.vec_curr
                c.subdiagonal[j] = sqrt(sum(c.vec_residual .* c.vec_residual))
                if c.subdiagonal[j] < 2.0*eps(Float64)
                        break
                end
                c.Q_store[:,j] = c.vec_curr
                c.vec_prev .= c.vec_curr
                c.vec_curr .= c.vec_residual / c.subdiagonal[j]
                # do a QR and get vec_residual
                # TODO: this area leaks memory like crazy.
                if j > 1
                        evec_row_j = @view c.evec_row[1:j]
                        # TODO: don't allocate at build time
                        evals_j = copy(c.diagonal[1:j])
                        subdiagonal_j = copy(c.subdiagonal[1:j-1])
                        deflate = ritz_value_convergence_check(evals_j, subdiagonal_j, evec_row_j, j)
                end
                if deflate
                        println("SO triggered wohooo!")
                        break
                end
        end
end

# store the projection as an operator.
function ritz_value_convergence_check(diagonal::AbstractVector{Float64}, subdiagonal::AbstractVector{Float64},
                                        evec_row::AbstractVector{Float64}, index::Integer)
        # deflation zeros out the Lanczos tridiagonal matrix.
        # sets [start, finish] entries in the eigenvec row to zero.
        @assert size(evals, 1) == index
        @assert index - size(subdiagonal, 1) == 1
        @assert size(evec_row, 1) == index 
        # another allocation
        ritz_errors = copy(subdiagonal)
        qr_tridiagonal!(diagonal, subdiagonal, evec_row)
        println("index ", index)
        println("evals :", diagonal)
        println("evec_row :", evec_row .* evec_row)
        @assert !iszero(ritz_errors) ritz_errors
        T_j_norm = maximum(abs.(evals))
        ritz_errors .= abs.(evec_row[1:index-1] .* ritz_errors)
        for e in ritz_errors
                if e < sqrt(eps(Float64)) * T_j_norm
                        return true
                end
        end
        return false
end

# are we going by subdiagonal_n or diagonal_n?
#function compute_vec_residuals!(context :: LanczosContext, j :: Int64)
#        # b[j] |s[j]|
#
#        a = get_diagonal(context, j)
#        b = get_subdiagonal(context, j)
#        for i = 1:j
#                evec_row = qr_tridiagonal!(copy(a[1:i]), copy(b[1:i]), j)
#                R = get_R_store(context, i)
#                R .= abs.(evec_row) .* b[1:i] 
#        end
#end

# this function is presently broken.
#function compute_μ(context :: LanczosContext, j :: Int64)
#        @assert j > 0 "iter must be greater than 0."
#        evec_row = zeros(j)
#        evals    = zeros(j)
#        if j == 1
#                evec_row[1] = 1
#                evals[1] = context.diagonal[1]
#        else
#                a = get_diagonal(context, j)
#                evals = copy(a)
#                b = get_subdiagonal(context, j)
#                evec_row = qr_tridiagonal!(evals, copy(b), 1)
#        end
#        evec_row .= evec_row .* evec_row
#
##        discretemeasure(evals, evec_row)
#end

