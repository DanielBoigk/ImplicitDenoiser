function normalize_image(img)
    out= 2 .* (Float32.(img) .- 0.5)
    return out
end

function denormalize_image(img)
    out = (0.5 .* img) .+ 0.5
    return out
end

function forward_sample(x0, t, ᾱ)
    αbar = ᾱ(t)
    ε = randn(size(x0))
    xt = sqrt(αbar) .* x0 .+ sqrt(1 - αbar) .* ε
    return xt, ε
end


β(t) = βmin + (βmax-βmin)*t/T
ᾱ(t) = exp(-βmin*t - (βmax-βmin)/(2*T) * t^2)


# rotate spatial dims (1,2) by 90° * k
function rot90_spatial(K, k::Int)
    k = mod(k, 4)
    k == 0 && return K
    k == 1 && return reverse(permutedims(K, (2,1)), dims=1)
    k == 2 && return reverse(reverse(K, dims=1), dims=2)
    k == 3 && return reverse(permutedims(K, (2,1)), dims=2)
end

# reflection (horizontal mirror)
reflect_spatial(K) = reverse(K, dims=2)

#=
function load_images_to_array(path, t, num, forward, size)
    x = ones(Float32, size, size, 2, num)
    y = zeros(Float32, size, size, 1, num)
    for i in 1:num
        img = Float32.(load("$path$i.jpg"))
        
        # Equivariant training data augmentation
        if rand(Bool)
            img = reflect_spatial(img)
        end
        if rand(Bool)
            img = rot90_spatial(img, rand(1:3)) # Optimized to select a random rotation
        end

        xt, ϵ = forward(img, t[i])
        x[:, :, 1, i] = xt
        
        # FIX: Multiply and assign back to the second channel
        x[:, :, 2, i] .*= t[i] 
        
        y[:, :, 1, i] = ϵ
    end
    return x, y
end

function load_images(batchsize, path, num, forward,size,tmax=1)
    t_array = tmax .* rand(Float32, num)
    x_img,y_img = load_images_to_array(path,t_array,num, forward, size)
    DataLoader(mapobs(gdev, (x_img, y_img)); batchsize, shuffle = true)
end

=#