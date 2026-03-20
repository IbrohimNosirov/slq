# from LAPACK and Parlett, 30 iterations per eval.
const eval_itr_max = 30 

# TODO: Have the option to choose between QR and QL
# TODO: add guards against very large inputs.

function qr_tridiagonal!(diagonal::AbstractVector{Float64}, subdiagonal::AbstractVector{Float64},
                        evec_row::AbstractVector{Float64})
        @assert length(findall(!iszero, evec_row)) == 1 length(findall(!iszero, evec_row)), evec_row
        diagonal_n = size(diagonal, 1)
        subdiagonal_n = size(subdiagonal, 1)
        @assert diagonal_n - subdiagonal_n == 1
        @assert diagonal_n == size(evec_row, 1)
        itr_max = eval_itr_max * diagonal_n
        # There are diagonal_n number of evals, but we iterate until all
        # subdiagonal_n entries go to zero.
        for i = 1:itr_max
                if iszero(subdiagonal)
                        break
                end
                # sweep subdiagonal looking for unreduced blocks.
                j = 1; idx_begin = 0; idx_end = 0
                while j <= subdiagonal_n
                        if subdiagonal[j] == 0.0
                                j += 1
                                continue
                        end
                        idx_begin = j
                        while j < subdiagonal_n
                                if subdiagonal[j+1] == 0.0
                                        break
                                else
                                        j += 1
                                end
                        end
                        idx_end = j
                        @assert idx_begin > 0 "invalid matrix index"
                        @assert idx_end >= idx_begin (idx_begin, idx_end)
                        if idx_end == idx_begin
                                cs, sn = eigen_small!(view(diagonal,    idx_begin:idx_end),
                                                      view(subdiagonal, idx_begin:idx_end),
                                                      view(diagonal,    idx_begin+1:idx_end+1))
                                apply_evec_to_evec_row!(evec_row, cs, sn, sn, cs, idx_begin)
                        else
                                # add QL here.
                                #chase_bulge_ql!(diagonal, subdiagonal, evec_row, idx_begin, idx_end)
                                chase_bulge_qr!(diagonal, subdiagonal, evec_row, idx_begin, idx_end)
                        end
                        j += 1
                end
        end
        p = sortperm(diagonal)
        diagonal .= diagonal[p]
        evec_row .= evec_row[p]
end

function chase_bulge_qr!(diagonal::AbstractVector{T}, subdiagonal::AbstractVector{T}, evec_row::AbstractVector{T},
                           idx_begin::Integer, idx_end::Integer) where T <: AbstractFloat
        @assert size(diagonal, 1) - size(subdiagonal, 1) == 1 "diagonal-subdiagonal dimension mismatch."
        shift = wilkinson_shift(diagonal[idx_end+1], diagonal[idx_end], subdiagonal[idx_end])
        x = diagonal[idx_begin] - shift
        z = subdiagonal[idx_begin]
        # idx_begin and idx_end are subdiagonal indices of an unreduced block.
        for i = idx_begin:idx_end
                c, s = givens_rotation(x, z)
                apply_givens_to_evec_row!(evec_row, c, s, i)
                tmp1 = c*subdiagonal[i] - s*diagonal[i]
                tmp2 = c*diagonal[i+1] - s*subdiagonal[i]
                diagonal[i] = c*(c*diagonal[i] + s*subdiagonal[i]) + s*(c*subdiagonal[i] + s*diagonal[i+1])
                diagonal[i+1] = c*tmp2 - s*tmp1
                subdiagonal[i] = c*tmp1 + s*tmp2
                if i > idx_begin
                        subdiagonal[i-1] = c*subdiagonal[i-1] + s*z
                end
                x = subdiagonal[i]
                if i < idx_end
                        z = s*subdiagonal[i+1]
                        subdiagonal[i+1] = c*subdiagonal[i+1]
                end
        end
        for i = idx_begin:idx_end
                if abs(subdiagonal[i]) < 2.0 * eps(Float64) * (abs(diagonal[i]) + abs(diagonal[i+1]))
                        subdiagonal[i] = 0.0
                end
        end
end

#= DLARTG generates a plane rotation so that
  [ c  s].[r]=[r]
  [-s  c] [g] [0]
where c**2 + s**2 = 1. =#
function givens_rotation(f::Float64, g::Float64)
        c = Ref{Float64}()
        s = Ref{Float64}()
        r = Ref{Float64}()
        ccall((:dlartg_64_, Base.liblapack_name), Cvoid,
              (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}), f, g, c, s, r)
        c[], s[]
end

# DLAEV2 is an LAPACK function that computes the eigendecomposition of a 2-by-2 symmetric matrix.
# We overwrite diagonal1 and diagonal2 with eigenvalues.
function eigen_small!(diagonal1::SubArray{Float64}, subdiagonal::SubArray{Float64}, diagonal2::SubArray{Float64})
        cs = Ref{Float64}()
        sn = Ref{Float64}()
        ccall((:dlaev2_64_, Base.liblapack_name), Cvoid,
              (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}),
               diagonal1, subdiagonal, diagonal2, diagonal1, diagonal2, cs, sn)
        subdiagonal .= 0.0
        cs[], sn[]
end

function wilkinson_shift(diagonal1::T, diagonal2::T, subdiagonal::T) where T <: AbstractFloat
        # a1 last index, a2 second to last index, b last index.
        denominator = (diagonal2 - diagonal1)/2.0
        if abs(denominator) < eps(Float64) && abs(subdiagonal) < eps(Float64)
                return diagonal1
        end
        denominator += sign(denominator)*sqrt(denominator*denominator + subdiagonal*subdiagonal)
        if abs(denominator) < 2.0 * eps(Float64)
                return diagonal1
        end
        shift = diagonal1 - (subdiagonal*subdiagonal)/denominator
        shift
end

function apply_givens_to_evec_row!(evec_row::AbstractVector{T}, cs::T, sn::T, i::Integer) where T <: AbstractFloat
        tau1 = evec_row[i]
        evec_row[i]   =  cs*tau1 + sn*evec_row[i+1]
        evec_row[i+1] = -sn*tau1 + cs*evec_row[i+1]
end

# 2x2 matrix case
function apply_evec_to_evec_row!(evec_row::AbstractVector{T}, v1::T, v2::T, v3::T, v4::T,
                                 i::Integer) where T <: AbstractFloat
        tau1 = evec_row[i]
        evec_row[i]   =  v1*tau1 + v3*evec_row[i+1]
        evec_row[i+1] = -v2*tau1 + v4*evec_row[i+1]
end
