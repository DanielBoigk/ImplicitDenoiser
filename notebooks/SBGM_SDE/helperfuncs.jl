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

forward(x,t) = forward_sample(x,t,ᾱ)

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

function sinusoidal_embedding(t::Float32, embedding_dim::Int, max_positions::Int=10000)
    half_dim = embedding_dim ÷ 2
    emb_scale = log(Float32(max_positions)) / (half_dim - 1)
    emb = exp.(-emb_scale .* Float32.(0:half_dim-1))
    emb = t .* emb  # shape: (half_dim,)
    emb = vcat(sin.(emb), cos.(emb))  # shape: (embedding_dim,)
    return Float32.(emb)
end

function load_images_to_array(path, t, num, forward, img_size, emb_size)
    x = ones(Float32, img_size, img_size, 2+emb_size, num)
    y = zeros(Float32, img_size, img_size, 1, num)
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
        
        emb = reshape(sinusoidal_embedding(t[i], emb_size), (1,1,emb_size,1))
        x[:, :, 3:end, i] .= emb
        y[:, :, 1, i] = ϵ
    end
    return x, y
end

function load_images(batchsize, path, num, forward,img_size,emb_size,dev, tmax=1)
    t_array = tmax .* rand(Float32, num)
    x_img,y_img = load_images_to_array(path,t_array,num, forward, img_size, emb_dim)
    DataLoader(mapobs(dev, (x_img, y_img)); batchsize=batchsize, shuffle = true)
end
