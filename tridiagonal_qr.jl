const MACHEPS     = eps(Float64)
const MACHEPS_INV = 1.0 / eps(Float64)

# NEED AT MINIMUM 20 asserts, HAVE 5 asserts.
# I should traverse only the off-diagonal elements.

function givens_rotation(F::Float64, G::Float64)
#       DLARTG generates a plane rotation so that
#       
#          [  C  S  ]  .  [ F ]  =  [ R ]
#          [ -S  C  ]     [ G ]     [ 0 ]
#       
#       where C**2 + S**2 = 1.

        C = Ref{Float64}()
        S = Ref{Float64}()
        R = Ref{Float64}()

        ccall((:dlartg_, Base.liblapack_name), Cvoid,
              (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}), F, G, C, S, R)

        C[], S[]
end

# this works perfectly
function eigen_small!(A::SubArray{Float64}, B::SubArray{Float64}, C::SubArray{Float64})
#       DLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
#               [  A   B  ]
#               [  B   C  ].
#       On return, RT1 is the eigenvalue of larger absolute value,
#                  RT2 is the eigenvalue of smaller absolute value,
#       and (CS1,SN1) is the unit right eigenvector for RT1, giving the decomposition
#
#               [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
#               [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ]. 

        CS1 = Ref{Float64}()
        SN1 = Ref{Float64}()
        RT1 = Ref{Float64}()
        RT2 = Ref{Float64}()

        ccall((:dlaev2_, Base.liblapack_name), Cvoid,
              (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}),
               A, B, C, A, C, CS1, SN1)

        B .= 0.0

        CS1[], SN1[]
end

function wilkinson_shift(a1::T, a2::T, b::T) where T <: AbstractFloat
  # a1 last index, a2 second to last index, b last index.
        d = (a2 - a1)/2.0
        if abs(d) < MACHEPS && abs(b) < MACHEPS
                return a1
        end
        denominator = d + sign(d)*sqrt(d*d + b*b)
        if abs(denominator) < 5.0 * MACHEPS
                return a1
        end
        shift = a1 - (b*b)/denominator

        shift
end

function apply_givens_to_evec_row!(evec_row::AbstractVector{T}, c::T, s::T, i::Integer) where T <: AbstractFloat
  tau1 = evec_row[i]
  tau2 = evec_row[i+1]
  evec_row[i]   = c*tau1 - s*tau2
  evec_row[i+1] = s*tau1 + c*tau2
end

# 2x2 matrix case
function apply_evec_to_evec_row!(evec_row::AbstractVector{T}, v1::T, v2::T, v3::T, v4::T,
                                 i::Integer) where T <: AbstractFloat
  tau1 = evec_row[i]
  tau2 = evec_row[i+1]
  evec_row[i]   = v1*tau1 - v3*tau2
  evec_row[i+1] = v2*tau1 + v4*tau2
end

function chase_bulge!(diagonal::AbstractVector{T}, subdiagonal::AbstractVector{T}, evec_row::AbstractVector{T},
                           idx_start::Integer, idx_finish::Integer) where T <: AbstractFloat
        @assert size(diagonal, 1) - size(subdiagonal, 1) == 1 "diagonal-subdiagonal dimension mismatch."

        # idx_start and idx_finish are subdiagonal indices of an unreduced block.
        # We chase the bulge through this big submatrix without any checks.
        shift = wilkinson_shift(diagonal[idx_finish+1], diagonal[idx_finish], subdiagonal[idx_finish])
        x = diagonal[idx_start] - shift
        z = subdiagonal[idx_start]

        for i = idx_start:idx_finish
                c, s = givens_rotation(x, z)
                apply_givens_to_evec_row!(evec_row, c, s, i)

                tmp1 = c*subdiagonal[i] - s*diagonal[i]
                tmp2 = c*diagonal[i+1] - s*subdiagonal[i]

                diagonal[i]    = c*(c*diagonal[i] + s*subdiagonal[i]) + s*(c*subdiagonal[i] + s*diagonal[i+1])
                diagonal[i+1]  = c*tmp2 - s*tmp1
                subdiagonal[i] = c*tmp1 + s*tmp2

                if i > idx_start
                        if abs(subdiagonal[i-1]) < 2.0 * MACHEPS * (abs(diagonal[i-1]) + abs(diagonal[i]))
                                subdiagonal[i-1] = 0.0
                        else
                                subdiagonal[i-1] = c*subdiagonal[i-1] + s*z
                        end
                end

                x = subdiagonal[i]

                if i < idx_finish
                        z = s*subdiagonal[i+1]
                        subdiagonal[i+1] = c*subdiagonal[i+1]
                end
        end

        if abs(subdiagonal[idx_finish]) < 2.0 * MACHEPS * (abs(diagonal[idx_finish])+abs(diagonal[idx_finish+1]))
                subdiagonal[idx_finish] = 0.0
        end
end

function qr_tridiag!(diagonal::AbstractVector{T}, subdiagonal::AbstractVector{T}, evec_row::AbstractVector{T},
                                                        index::Integer) where T <: AbstractFloat 
        diagonal_n = size(diagonal, 1)
        subdiagonal_n = size(subdiagonal, 1)
        @assert diagonal_n - subdiagonal_n == 1 "diagonal-subdiagonal mismatch."
        @assert diagonal_n == size(evec_row, 1) "evec_row diagonal mismatch."

        max_iter = 30*diagonal_n
        evec_row[index] = 1.0

        # Key detail: there are diagonal_n number of eval,
        # but we iterate until all subdiagonal_n entries
        # go to zero.
        
        converged = false

	for i = 1:max_iter
		@assert i != max_iter "hit max iteration."

                # if all subdiagonal entries are zero, converge.
                if iszero(subdiagonal)
                        converged = true
                        break
                end

		# sweep off-diagonal looking for unreduced blocks.
                j = 1
                idx_start  = 0
                idx_finish = 0
                unreduced_block_found = false

                # logic is right, but needs cleaning
                while j <= subdiagonal_n
                        idx_start  = 0
                        idx_finish = 0

                        while j <= subdiagonal_n
                                if subdiagonal[j] != 0.0
                                        idx_start = j
                                        break
                                end
                                j += 1
                        end

                        while j < subdiagonal_n
                                if subdiagonal[j+1] == 0.0
                                        idx_finish = j
                                        break
                                elseif j+1 == subdiagonal_n
                                        j += 1
                                        idx_finish = j
                                        break
                                end
                                j += 1
                        end

                        if idx_start != 0 && idx_finish != 0
                                unreduced_block_found = true
                        end

                        if idx_finish == idx_start && unreduced_block_found
                                CS, SN = eigen_small!(view(diagonal,    idx_start:idx_finish),
                                                      view(subdiagonal, idx_start:idx_finish),
                                                      view(diagonal,    idx_start+1:idx_finish+1))
                                apply_evec_to_evec_row!(evec_row, CS, SN, SN, CS, idx_start)
                                unreduced_block_found = false
                                j += 1
                        end

                        if idx_finish > idx_start && unreduced_block_found
                                chase_bulge!(diagonal, subdiagonal, evec_row, idx_start, idx_finish)
                                unreduced_block_found = false
                                j += 1
                        end
                end
	end

        p = sortperm(diagonal)
        diagonal .= diagonal[p]
        evec_row .= evec_row[p]
end
