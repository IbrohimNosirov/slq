# from LAPACK and Parlett, 30 iterations per eval.
const eval_itr_max = 30 

# TODO: Need better assert coverage.
# TODO: Have the option to choose between QR and QL
# TODO: add guards against very large inputs.

# TODO: should change this to pass in evec_row with the right idx modification.
function qr_tridiag!(diag::AbstractVector{Float64}, subdiag::AbstractVector{Float64}, evec_row::AbstractVector{Float64})
        diag_n = size(diag, 1)
        subdiag_n = size(subdiag, 1)
        @assert diag_n - subdiag_n == 1
        @assert diag_n == size(evec_row, 1)
        itr_max = eval_itr_max * diag_n
        # There are diag_n number of evals, but we iterate until all
        # subdiag_n entries go to zero.
	for i = 1:itr_max
                if iszero(subdiag)
                        break
                end
                # sweep subdiag looking for unreduced blocks.
                j = 1; idx_begin = 0; idx_end = 0
                while j <= subdiag_n
                        if subdiag[j] == 0.0
                                j += 1
                                continue
                        end
                        idx_begin = j
                        while j < subdiag_n
                                if subdiag[j+1] == 0.0
                                        break
                                else
                                        j += 1
                                end
                        end
                        idx_end = j
                        @assert idx_begin > 0 "invalid matrix index"
                        @assert idx_end >= idx_begin (idx_begin, idx_end)
                        if idx_end == idx_begin
                                cs, sn = eigen_small!(view(diag,    idx_begin:idx_end),
                                                      view(subdiag, idx_begin:idx_end),
                                                      view(diag,    idx_begin+1:idx_end+1))
                                apply_evec_to_evec_row!(evec_row, cs, sn, sn, cs, idx_begin)
                        else
                                # add QL here.
                                #chase_bulge_ql!(diag, subdiag, evec_row, idx_begin, idx_end)
                                chase_bulge_qr!(diag, subdiag, evec_row, idx_begin, idx_end)
                        end
                        j += 1
                end
	end
        p = sortperm(diag)
        diag .= diag[p]
        evec_row .= evec_row[p]
end

function chase_bulge_qr!(diag::AbstractVector{T}, subdiag::AbstractVector{T}, evec_row::AbstractVector{T},
                           idx_begin::Integer, idx_end::Integer) where T <: AbstractFloat
        @assert size(diag, 1) - size(subdiag, 1) == 1 "diag-subdiag dimension mismatch."
        shift = wilkinson_shift(diag[idx_end+1], diag[idx_end], subdiag[idx_end])
        x = diag[idx_begin] - shift
        z = subdiag[idx_begin]
        # idx_begin and idx_end are subdiag indices of an unreduced block.
        for i = idx_begin:idx_end
                c, s = givens_rotation(x, z)
                apply_givens_to_evec_row!(evec_row, c, s, i)
                tmp1 = c*subdiag[i] - s*diag[i]
                tmp2 = c*diag[i+1] - s*subdiag[i]
                diag[i] = c*(c*diag[i] + s*subdiag[i]) + s*(c*subdiag[i] + s*diag[i+1])
                diag[i+1] = c*tmp2 - s*tmp1
                subdiag[i] = c*tmp1 + s*tmp2
                if i > idx_begin
                        subdiag[i-1] = c*subdiag[i-1] + s*z
                end
                x = subdiag[i]
                if i < idx_end
                        z = s*subdiag[i+1]
                        subdiag[i+1] = c*subdiag[i+1]
                end
        end
        for i = idx_begin:idx_end
                if abs(subdiag[i]) < 2.0 * eps(Float64) * (abs(diag[i]) + abs(diag[i+1]))
                        subdiag[i] = 0.0
                end
        end
end

#= DLARTG generates a plane rotation so that
  [  c  s  ]  .  [ r ]  =  [ r ]
  [ -s  c  ]     [ g ]     [ 0 ]
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
# We overwrite diag1 and diag2 with eigenvalues.
function eigen_small!(diag1::SubArray{Float64}, subdiag::SubArray{Float64}, diag2::SubArray{Float64})
        cs = Ref{Float64}()
        sn = Ref{Float64}()
        ccall((:dlaev2_64_, Base.liblapack_name), Cvoid,
              (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}),
               diag1, subdiag, diag2, diag1, diag2, cs, sn)
        subdiag .= 0.0
        cs[], sn[]
end

function wilkinson_shift(diag1::T, diag2::T, subdiag::T) where T <: AbstractFloat
        # a1 last index, a2 second to last index, b last index.
        denominator = (diag2 - diag1)/2.0
        if abs(denominator) < eps(Float64) && abs(subdiag) < eps(Float64)
                return diag1
        end
        denominator += sign(denominator)*sqrt(denominator*denominator + subdiag*subdiag)
        if abs(denominator) < 2.0 * eps(Float64)
                return diag1
        end
        shift = diag1 - (subdiag*subdiag)/denominator
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
