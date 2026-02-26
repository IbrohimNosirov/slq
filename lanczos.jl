# Date: 2026-02-02
using Test
using LinearAlgebra
include("tridiagonal_qr.jl")

const MACHEPS = eps(Float64)

# use active names for functions
# variable names should have [noun]_[qualifier] e.g. cluster_max better than max_cluster

abstract type OrthStrategy end
struct FullOrth <: OrthStrategy end
struct PRO <: OrthStrategy end
struct FirstOrth <: OrthStrategy end
struct SO <: OrthStrategy end

struct LanczosContext
  A              :: Matrix{Float64}                             # input matrix.
  Q_store        :: Union{Matrix{Float64}, Nothing}             # Lanczos vectors.
  W_store        :: Union{Matrix{Float64}, Nothing}             # orthogonality tracker.
  R_store        :: Matrix{Float64}                             # Ritz residuals.
  a_vec          :: Vector{Float64}                             # main diagonal; size k.
  b_vec          :: Vector{Float64}                             # off diagonal;size k-1.
  q_vec          :: Vector{Float64}                             # starting vector.
  w_distance_arr :: Vector{Float64}                             # Wasserstein distances.
  ν_distribution :: DiscreteNonParametric                       # true distribution.
  k_lanczos      :: Int64                                       # number of iterations.
  N              :: Int64                                       # NxN matrix.
end

function LanczosContext(::FullOrth, A :: AbstractMatrix, q_vec :: AbstractVector, k :: Int64,
                                    ν :: DiscreteNonParametric)
    @assert norm(q_vec) ≈ 1.0 "pass a unit vector."
    N = size(A, 1)
    
    Q_store    = zeros(N, k)
    W_store    = nothing
    R_store    = zeros(k, k)
    a_vec      = zeros(k)
    b_vec      = zeros(k-1)
    w_distance_arr = zeros(k)
    
    LanczosContext(A, Q_store, W_store, R_store, a_vec, b_vec, q_vec, w_distance_arr, ν, k, N)
end

function LanczosContext(::PRO, A :: AbstractMatrix, q_vec :: AbstractVector, k :: Int64,
                               ν :: DiscreteNonParametric)
    @assert norm(q_vec) ≈ 1.0 "pass a unit vector."
    N = size(A, 1)
    
    Q_store = zeros(N, k)
    W_store = diagm(0  => ones(k+1),
                   -1  => eps(Float64)*ones(k),
                    1  => eps(Float64)*ones(k))
    R_store = zeros(k, k)
    a_vec = zeros(k)
    b_vec = zeros(k-1)
    w_distance_arr = zeros(k)
    
    LanczosContext(A, Q_store, W_store, R_store, a_vec, b_vec, q_vec, w_distance_arr, ν, k, N)
end

function LanczosContext(::FirstOrth, A :: AbstractMatrix, q_vec :: AbstractVector, k :: Int64,
                               ν :: DiscreteNonParametric)
    @assert norm(q_vec) ≈ 1.0 "pass a unit vector."
    N = size(A, 1)
    
    Q_store = nothing
    W_store = diagm(0  => ones(k+1),
                   -1  => eps(Float64)*ones(k),
                    1  => eps(Float64)*ones(k))
    R_store = zeros(k, k)
    a_vec = zeros(k)
    b_vec = zeros(k-1)
    w_distance_arr = zeros(k)
    
    LanczosContext(A, Q_store, W_store, R_store, a_vec, b_vec, q_vec, w_distance_arr, ν, k, N)
end

function LanczosContext(::SO, A, q_vec, k, ν)
    @assert norm(q_vec) ≈ 1.0 "pass a unit vector."
    N = size(A, 1)
    
    Q_store = zeros(N, k)
    W_store = nothing
    R_store = zeros(k, k)
    a_vec = zeros(k)
    b_vec = zeros(k-1)
    w_distance_arr = zeros(k)
    
    LanczosContext(A, Q_store, W_store, R_store, a_vec, b_vec, q_vec, w_distance_arr, ν, k, N)
end

# Dispatch to implementations
lanczos(ctx :: LanczosContext, :: FullOrth) = lanczos_full_orth(ctx)
lanczos(ctx :: LanczosContext, :: PRO) = lanczos_pro(ctx)
lanczos(ctx :: LanczosContext, :: FirstOrth) = lanczos_first_orth(ctx)
lanczos(ctx :: LanczosContext, :: SO) = lanczos_so(ctx)

# Getters for A and q
get_A(ctx :: LanczosContext) = ctx.A
get_q_vec(ctx :: LanczosContext) = ctx.q_vec

# These jth iteration getters act as guards against off-by-one (overcount)
# errors; we shouldn't be able to load the j'th result into the j+1th entry.
function get_Q_store(ctx :: LanczosContext, j :: Int64)
  # what is the difference between != and !== ? 
  @assert ctx.Q_store !== nothing "can't retrieve unallocated space."
  view(ctx.Q_store, 1:ctx.N, 1:j)
end

function get_Q_store_col(ctx :: LanczosContext, j :: Int64)
  @assert ctx.Q_store !== nothing "can't retrieve unallocated space."
  view(ctx.Q_store, 1:ctx.N, j)
end

function get_W_store(ctx :: LanczosContext, j :: Int64)
  @assert ctx.W_store !== nothing "can't retrieve unallocated space."
  view(ctx.W_store, :, 1:j)
end

function get_W_store(ctx :: LanczosContext)
  @assert ctx.W_store !== nothing "can't retrieve unallocated space."
  ctx.W_store
end

# There can only be j residuals after j steps; we store these in the jth column.
function get_R_store(ctx :: LanczosContext, j :: Int64)
  @assert ctx.R_store !== nothing "can't retrieve unallocated space."
  view(ctx.R_store, 1:j, j)
end

function get_R_store_col(ctx :: LanczosContext, j :: Int64)
  @assert ctx.R_store !== nothing "can't retrieve unallocated space."
  view(ctx.R_store, 1:j, j)
end

get_a_vec(ctx :: LanczosContext, j) = view(ctx.a_vec, 1:j)
get_a_vec(ctx :: LanczosContext) = ctx.a_vec
get_b_vec(ctx :: LanczosContext, j) = view(ctx.b_vec, 1:j-1)
get_b_vec(ctx :: LanczosContext) = ctx.b_vec
get_w_distance_arr(ctx :: LanczosContext, j) = view(ctx.w_distance_arr, 1:j)

# Computes and stores residuals
function compute_residuals!(ctx :: LanczosContext, j :: Int64)
  # b[j] |s[j]|
  a = get_a_vec(ctx, j)
  b = get_b_vec(ctx, j)
  evec_row = qr_tridiag!(copy(a), copy(b), j)

  R = get_R_store(ctx, j)
  R .= abs.(evec_row) .* ctx.b_vec[j-1]
end

function compute_μ(ctx :: LanczosContext, j :: Int64)
  @assert j > 0 "iteration must be greater than 0."
  evec_row = zeros(j)
  evals    = zeros(j)
  if j == 1
    evec_row[1] = 1
    evals[1] = ctx.a_vec[1]
  else
    a = get_a_vec(ctx, j)
    evals = copy(a)
    b = get_b_vec(ctx, j)
    evec_row = qr_tridiag!(evals, copy(b), 1) # first row of eigenvectors.
  end
  evec_row .= evec_row.^2
  evec_row  = evec_row / norm(evec_row, 1)

  discretemeasure(evals, evec_row)
end

# Indices are all wrong.
# Lanczos with full orthogonalization (orthogonalization at every step).
function lanczos_full_orth(ctx :: LanczosContext)
  A = get_A(ctx)
  q = get_q_vec(ctx)
  a = get_a_vec(ctx)
  b = get_b_vec(ctx)

  @assert ctx.Q_store !== nothing
  @assert ctx.W_store === nothing

  z = A * q
  a[1] = q' * z
  z = z - a[1]*q
  b[1] = norm(z)

  for j = 1:ctx.k_lanczos-1
      q_prev = q
      q = z / b[j]
      Q = get_Q_store(ctx, j)
      Q[:,j] = q

      z = A*q - b[j] * q_prev
      a[j+1] = q' * z
      z = z - a[j+1]*q
      b[j+1] = norm(z)

      z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
      z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
      
      # residuals
      compute_residuals!(ctx, j)
      
      # Wasserstein distance.
      μ = compute_μ(ctx, j)
      ctx.w_distance_arr[j] = wasserstein(μ, ctx.ν_distribution; p=Val(1))

      if b[j+1] == 0
        break
      end
  end
  ctx.k_lanczos
end

# Lanczos with selective orthogonalization (LanPRO).
function lanczos_pro(ctx :: LanczosContext)
  A = get_A(ctx)
  q = get_q_vec(ctx)
  a = get_a_vec(ctx)
  b = get_b_vec(ctx)
  W = get_W_store(ctx)

  norm_A = norm(A)

  z = A * q
  a[1] = q' * z
  z = z - a[1]*q
  b[1] = norm(z)

  for j = 2:ctx.k_lanczos
      q_prev = q
      q = z / b[j-1]
      Q = get_Q_store(ctx, j)
      Q[:,j-1] = q

      z = A*q - b[j-1] * q_prev
      a[j] = q' * z
      z = z - a[j]*q
      b[j] = norm(z)

      if b[j] == 0.0
        break
      end

      orthogonalized = false
      for i = 2:j
          w_tilde  = b[i]*W[j,i+1] + (a[i] - a[j])*W[j,i]
          w_tilde += b[i-1]*W[j,i-1] - b[j-1]*W[j-1,i]
          W[j+1,i] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]

          if W[j+1,i] > sqrt(eps(Float64))
            if orthogonalized == false
              U = @view Q[:,1:j-1]
              U = Matrix(qr(U).Q)
              @assert U' * U ≈ I(j-1)
              z -= U * (U' * z)
              z -= U * (U' * z)
              orthogonalized = true
            end
            W[j+1,i] = eps(Float64)
            W[j,i] = eps(Float64)
          end
      end
      
      # residuals
      compute_residuals!(ctx, j)
      
      # Wasserstein distance.
      μ = compute_μ(ctx, j)
      ctx.w_distance_arr[j] = wasserstein(μ, ctx.ν_distribution; p=Val(1))
  end

  ctx.k_lanczos
end

# Lanczos until the first reorthogonalization.
function lanczos_first_orth(ctx :: LanczosContext)
  A = get_A(ctx)
  q = get_q_vec(ctx)
  a = get_a_vec(ctx)
  b = get_b_vec(ctx)
  W = get_W_store(ctx)

  norm_A = norm(A)

  z = A * q
  a[1] = q' * z
  z = z - a[1]*q
  b[1] = norm(z)

  for j = 2:ctx.k_lanczos
    q_prev = q
    q = z / b[j-1]

    z = A*q - b[j-1] * q_prev
    a[j] = q' * z
    z = z - a[j]*q
    b[j] = norm(z)

    if b[j] == 0
      break
    end

    # residuals
    compute_residuals!(ctx, j)
    
    # Wasserstein distance.
    μ = compute_μ(ctx, j)
    ctx.w_distance_arr[j] = wasserstein(μ, ctx.ν_distribution; p=Val(1))

    for i = 2:j
      w_tilde  = b[i]*W[j,i+1] + (a[i] - a[j])*W[j,i]
      w_tilde += b[i-1]*W[j,i-1] - b[j-1]*W[j-1,i]
      W[j+1,i] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]
      if W[j+1,i] > sqrt(eps(Float64))
        println("converged at iteration ", j)
        return j
      end
    end
  end

  ctx.k_lanczos
end

# Lanczos with selective orthogonalization.
function lanczos_so(ctx :: LanczosContext)
  terminate = false

  A = get_A(ctx)
  q = get_q_vec(ctx)
  a = get_a_vec(ctx)
  b = get_b_vec(ctx)

  z = A * q
  a[1] = q' * z
  z = z - a[1]*q
  b[1] = norm(z)

  μ = compute_μ(ctx, 1)
  ctx.w_distance_arr[1] = wasserstein(μ, ctx.ν_distribution; p=Val(1))

  for j = 2:ctx.k_lanczos
    q_prev = q
    q = z / b[j-1]
    Q = get_Q_store(ctx, j)
    Q[:,j-1] = q

    z = A*q - b[j-1] * q_prev
    a[j] = q' * z
    z = z - a[j]*q
    b[j] = norm(z)

    if b[j] == 0
      break
    end

    # get residuals.
    evals  = deepcopy(get_a_vec(ctx, j))
    b_copy = deepcopy(get_b_vec(ctx, j))
    
    evec_row = qr_tridiag!(evals, b_copy, j)
    T_norm = maximum(abs.(evals))

    R = get_R_store(ctx, j)
    R[:] = abs.(evec_row) .* b[j-1]

    # LanSO check.
    for i = 1:j
      if R[i] < sqrt(eps(Float64)) * T_norm
        terminate = true
        # perform deflation
        evals    = deepcopy(get_a_vec(ctx, j))
        b_copy   = deepcopy(get_b_vec(ctx, j))
        evals, evecs = eigen!(SymTridiagonal(evals, b_copy))
        Z = Q * evecs
        P = I - Z*Z'
        A = P * A * P
        println("Good Ritz val #", i, ", ", R[i])
      end
    end

    # TODO: needs full implementation.

    # Wasserstein distance.
    μ = compute_μ(ctx, j)
    ctx.w_distance_arr[j] = wasserstein(μ, ctx.ν_distribution; p=Val(1))

    if terminate == true
      println("residuals ", get_R_store(ctx, j))
      println("Wasserstein distances ", get_w_distance_arr(ctx, j))
      println("Done")
      return j
    end
  end
  ctx.k_lanczos
end

@testset "Lanczos Implementations Data Flow" begin
  @testset "FullOrth" begin
    let
      # Common setup
      N = 100
      k = 20
      A = randn(N, N)
      A = A + A'
      q_vec = randn(N)
      q_vec = q_vec / norm(q_vec)
      
      # Create a true distribution (discrete measure)
      evals_true = sort(randn(N))
      weights_true = ones(N) ./ N
      weights_true = weights_true / sum(weights_true)
      ν = DiscreteNonParametric(evals_true, weights_true)

      ctx = LanczosContext(FullOrth(), A, q_vec, k, ν)
      j = lanczos(ctx, FullOrth())
      
      # Check a_vec and b_vec are populated
      @test any(ctx.a_vec .!= 0.0)
      @test any(ctx.b_vec .!= 0.0)
      
      # Check residuals are tracked (R_store should be populated)
      @test any(ctx.R_store .!= 0.0)
      @test size(ctx.R_store, 1) == k
      
      # Check Wasserstein distances are tracked
      @test any(ctx.w_distance_arr .!= 0.0)
      @test length(ctx.w_distance_arr) == k
      
      # Check Q_store exists and is populated
      @test ctx.Q_store !== nothing
      @test size(ctx.Q_store) == (N, k)
    end
  end

#    @testset "PRO" begin
#        let
#            # Common setup
#            N = 100
#            k = 20
#            A = Symmetric(randn(N, N))
#            q = randn(N)
#            q = q / norm(q)
#            
#            # Create a true distribution (discrete measure)
#            evals_true = sort(randn(N))
#            weights_true = rand(N)
#            weights_true = weights_true / sum(weights_true)
#            ν = DiscreteNonParametric(evals_true, weights_true)
#            ctx = LanczosContext(PRO(), A, q, k, ν)
#            j = lanczos(ctx, PRO())
#            
#            # Check a_vec and b_vec are populated
#            @test any(ctx.a_vec .!= 0.0)
#            @test any(ctx.b_vec .!= 0.0)
#            
#            # Check residuals are tracked
#            @test any(ctx.R_store .!= 0.0)
#            @test size(ctx.R_store, 1) == k
#            
#            # Check Wasserstein distances are tracked
#            @test any(ctx.w_distance_arr .!= 0.0)
#            @test length(ctx.w_distance_arr) == k
#            
#            # Check W_store exists (orthogonality tracker)
#            @test ctx.W_store !== nothing
#            @test size(ctx.W_store) == (k+1, k+1)
#        end
#    end

#    @testset "FirstOrth" begin
#        let
#            # Common setup
#            N = 100
#            k = 20
#            A = Symmetric(randn(N, N))
#            q = randn(N)
#            q = q / norm(q)
#            
#            # Create a true distribution (discrete measure)
#            evals_true = sort(randn(N))
#            weights_true = rand(N)
#            weights_true = weights_true / sum(weights_true)
#            ν = DiscreteNonParametric(evals_true, weights_true)
#            ctx = LanczosContext(FirstOrth(), A, q, k, ν)
#            j = lanczos(ctx, FirstOrth())
#            
#            # Check a_vec and b_vec are populated
#            @test any(ctx.a_vec .!= 0.0)
#            @test any(ctx.b_vec .!= 0.0)
#            
#            # Check residuals are tracked
#            @test any(ctx.R_store .!= 0.0)
#            @test size(ctx.R_store, 1) == k
#            
#            # Check Wasserstein distances are tracked
#            @test any(ctx.w_distance_arr .!= 0.0)
#            @test length(ctx.w_distance_arr) == k
#            
#            # Check W_store exists (used for first orth detection)
#            @test ctx.W_store !== nothing
#            
#            # Q_store should be nothing for FirstOrth
#            @test ctx.Q_store === nothing
#        end
#    end

#    @testset "SO" begin
#        let
#            # Common setup
#            N = 100
#            k = 20
#            A = Symmetric(randn(N, N))
#            q = randn(N)
#            q = q / norm(q)
#            
#            # Create a true distribution (discrete measure)
#            evals_true = sort(randn(N))
#            weights_true = rand(N)
#            weights_true = weights_true / sum(weights_true)
#            ν = DiscreteNonParametric(evals_true, weights_true)
#            ctx = LanczosContext(SO(), A, q, k, ν)
#            j = lanczos(ctx, SO())
#            
#            # Check a_vec and b_vec are populated
#            @test any(ctx.a_vec .!= 0.0)
#            @test any(ctx.b_vec .!= 0.0)
#            
#            # Check residuals are tracked
#            @test any(ctx.R_store .!= 0.0)
#            @test size(ctx.R_store, 1) == k
#            
#            # Check Wasserstein distances are tracked
#            @test any(ctx.w_distance_arr .!= 0.0)
#            @test length(ctx.w_distance_arr) == k
#            
#            # Check Q_store exists and is populated
#            @test ctx.Q_store !== nothing
#            @test size(ctx.Q_store) == (N, k)
#        end
#    end
end
