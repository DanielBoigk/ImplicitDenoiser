# rotate spatial dims (1,2) by 90° * k
function rot90_spatial(K, k::Int)
    k = mod(k, 4)
    k == 0 && return K
    k == 1 && return reverse(permutedims(K, (2,1,3,4)), dims=1)
    k == 2 && return reverse(reverse(K, dims=1), dims=2)
    k == 3 && return reverse(permutedims(K, (2,1,3,4)), dims=2)
end

# reflection (horizontal mirror)
reflect_spatial(K) = reverse(K, dims=2)

function D4_symmetrize(K::AbstractArray{<:Real,4})
    @assert size(K,1) == size(K,2)

    acc = zero(K)
    for k in 0:3
        acc .+= rot90_spatial(K, k)
        acc .+= rot90_spatial(reflect_spatial(K), k)
    end
    acc ./ 8
end


function D4_symmetrize!(K::AbstractArray{<:Real,4})
    @assert size(K,1) == size(K,2)

    acc = zero(K)   # one buffer (same device as K)

    for k in 0:3
        acc .+= rot90_spatial(K, k)
        acc .+= rot90_spatial(reflect_spatial(K), k)
    end

    K .= acc ./ 8
    return K
end