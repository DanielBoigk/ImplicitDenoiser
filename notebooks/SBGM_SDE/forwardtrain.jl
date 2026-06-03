using Pkg
Pkg.activate("../../")
#Pkg.instantiate()



im_path = "../../../Images/64/"

using Lux, LuxCUDA, MLUtils, NNlib
using Optimisers, Random, Statistics
using Zygote
#using DiffEqFlux, OrdinaryDiffEq
using FFTW
using Images, JLD2
using ComponentArrays
using Plots
using Dates

using LinearAlgebra
using Random

# Hyperparameters for Variance Preserving (VP) SDE
const βmin = 0.1
const βmax = 20.0
T = 1
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

model= Chain(
    Conv((17, 17), 2 => 256; pad = 8),
    SkipConnection(
        Chain(
            Conv((1, 1), 256 => 512, gelu),
            SkipConnection(
                Chain(
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                ),
                +
            ),
            Conv((1, 1), 512 => 256, gelu),
            Conv((1, 1), 256 => 256, gelu),
        ),
        +
    ),
    Conv((1, 1), 256 => 128, gelu),
    Conv((1, 1), 128 => 64, gelu),
    Conv((1, 1), 64 => 1)
)

const cdev = cpu_device()
const gdev = gpu_device()
dev = gdev

rng = Xoshiro()

ps, st = Lux.setup(rng, model)
ps = ps |> ComponentArray{Float32}|> dev  
#@load "ps_latestvn.jld2" ps
#@load "st_latestvn.jld2" st

ps = ps |> ComponentArray |> dev
st = st |> dev
opt = Optimisers.NAdam(3.3e-4)
state = Optimisers.setup(opt,ps)


forward(img, t) = forward_sample(normalize_image(img), t, ᾱ)

function loss_function(model, ps, st, (x, y_true))
    y_pred, st = model(x, ps, st)
    loss_mse= MSELoss()
    mse_loss = loss_mse(y_pred, y_true)
    return mse_loss, st
end

function inner_loop(model, ps, st, data, state)
    #train_state = Lux.Training.TrainState(model, ps, st, opt)
    #Training.single_train_step!(AutoZygote(), loss_function, (x_dev, y_dev), train_state)
    (loss, st), back = Zygote.pullback(ps -> loss_function(model, ps, st, data), ps)
    grads = back((one(loss), nothing))[1]

    state, ps = Optimisers.update(state, ps, grads)
    return ps, st, state, loss
end

function train!(
    model,
    ps,
    st,
    opt,
    im_path,
    forward;
    epochs = 2,
    batchsize = 64,
    num_images = 7000,
    log_every = 10,
    size = 128
)
    state = Optimisers.setup(opt, ps)
    losses = Float32[]
    for epoch in 1:epochs
    t = now(); println("Starting Training for epoch $epoch ...   $t\n")

        epoch_loss = 0.0f0
        dataloader = dataloader = load_images(batchsize, im_path, num_images, forward, size)
        num_batches = length(dataloader)
        x_sample, y_sample = first(dataloader)
        println("Input shape: ", size(x_sample))    # Expected: (64, 64, 2, 64)
        println("Target shape: ", size(y_sample))   # Expected: (64, 64, 1, 64)
        for (i, data) in enumerate(dataloader)
            ps, st, state, loss = inner_loop(model, ps, st, data, state)
            epoch_loss += loss
            push!(losses, loss)
            #print("| $i: Loss: $loss Depth: $r ")
            print("-")
            if i % log_every == 0
                t = now()
                print(
                    " Batch $i/$num_batches | " *
                    "Loss = $(round(loss; sigdigits=4))" *
                    " Time: $t \n"
                )
            end
        end

        avg_epoch_loss = epoch_loss / num_batches
        t = now()

        @save "snapshots/ps$(avg_epoch_loss)_$t.jld2" ps
        @save "ps_latestvn.jld2" ps
        @save "snapshots/st$(avg_epoch_loss)_$t.jld2" st
        @save "st_latestvn.jld2" st
        print(
            "\n=== Epoch $epoch finished | " *
            "avg loss = $(round(avg_epoch_loss; sigdigits=5)) | "*
            " Time: $t" *
            " ===\n"
        )
    end
    return ps, st, losses
end

ps, st, losses = train!(
    model,
    ps,
    st,
    opt,
    im_path,
    forward;
    size = 64,
    epochs = 2
)