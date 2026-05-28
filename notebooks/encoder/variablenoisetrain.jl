using Pkg
Pkg.activate("../../")
Pkg.instantiate()

im_path = "../../../Images/64/"

using Lux, LuxCUDA, MLUtils
using Optimisers, Random, Statistics
using Zygote
using DiffEqFlux, OrdinaryDiffEq
using FFTW
#using Reactant
using Images, JLD2
using ComponentArrays
using DeepEquilibriumNetworks
using Plots
using Dates

#Reactant.set_default_backend("cpu")
#const device = reactant_device()
const cdev = cpu_device()
const gdev = gpu_device()
dev = gdev

include("model.jl")

rng = Xoshiro()
#ps, st = Lux.setup(rng, model)
#@load "ps0.023593083_2026-04-20T18:11:11.406.jld2" ps
#@load "st0.023593083_2026-04-20T18:11:11.406.jld2" st
@load "ps_latestvn.jld2" ps
@load "st_latestvn.jld2" st
#@load "st4.jld2" st
#@load "ps4.jld2" ps

ps = ps |> ComponentArray |> dev
st = st |> dev

opt = Optimisers.NAdam(1e-4)
state = Optimisers.setup(opt,ps)
#train_state = Lux.Training.TrainState(model, ps, st, opt)

function loss_function(model, ps, st, (x, y_true))
    y, st = model(x, ps, st)
    y_pred = y[1]
    #y_pred = model(x, ps, st)[1][1]
    loss_mse= MSELoss()
    mse_loss = loss_mse(y_pred, y_true)
    #sptrl_loss = loss_mse(dct(dct(y_pred,1),2),dct(dct(y_pred,1),2))
    return mse_loss, st
    #return mes_loss + sptrl_loss, st
end

# This is to test the model if it works:
#=
x = randn(Float32,128,128,1,4)
y_true = randn(Float32,128,128,1,4)
x_dev = x |> dev
y_dev = y_true |> dev
y, _ = model(x_dev, ps, st)
=#

function add_gaussian_noise(img, σ = 1e-3 )
    out = img .+ σ .* randn(eltype(img), size(img))
    if rand(Bool)
        @. out = min(max(out, 0.0),1.0) 
    end
    out
end

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

function load_images_to_array(path,num,corrupt_func,size)
    x = zeros(Float32,size,size,1,num)
    y = zeros(Float32,size,size,1,num)
    for i in 1:num
        img = Float32.(load("$path$i.jpg"))
        # equivariant training:
        if rand(Bool)
            img = reflect_spatial(img)
        end
        if rand(Bool)
            img = rot90_spatial(img,1)
        end
        if rand(Bool)
            img = rot90_spatial(img,2)
        end

        x[:,:,1,i] = corrupt_func(img)
        y[:,:,1,i] = img
    end
    return x,y
end

cfunc(img) = add_gaussian_noise(Float32.(img),2e-1)

function load_images(batchsize, path, num, corrupt_func,size)
    x_img,y_img = load_images_to_array(path,num, corrupt_func, size)
    DataLoader(mapobs(gdev, (x_img, y_img)); batchsize, shuffle = true)
end



function inner_loop(model_conv, tspan, ps, st, data, state)
    model = NeuralODE(model_conv, tspan, Tsit5(); save_everystep = false, sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()), reltol = 1e-3, abstol = 1e-4, save_start = false)
    #train_state = Lux.Training.TrainState(model, ps, st, opt)
    #Training.single_train_step!(AutoZygote(), loss_function, (x_dev, y_dev), train_state)
    (loss, st), back = Zygote.pullback(ps -> loss_function(model, ps, st, data), ps)
    grads = back((one(loss), nothing))[1]

    state, ps = Optimisers.update(state, ps, grads)
    return ps, st, state, loss
end


# 📚 Training Parameters (use the values you set previously)
epochs = 1
losses = Float32[]
#num_batches = length(dataloader)
#println("Starting Training for $epochs epochs...")


function train!(
    model,
    ps,
    st,
    opt,
    im_path;
    epochs = 2,
    batchsize = 64,
    num_images = 7000,
    log_every = 10,
    size = 128
)
    state = Optimisers.setup(opt, ps)
    losses = Float32[]
    for epoch in 1:epochs
    println("Starting Training for epoch $epoch ...")

        epoch_loss = 0.0f0
        dataloader = load_images(batchsize, im_path, num_images, cfunc, size)
        num_batches = length(dataloader)

        for (i, data) in enumerate(dataloader)
            r = abs(randn(Float32))
            tspan = (0.0f0, 1.5f0 + r) 
            ps, st, state, loss = inner_loop(model, tspan, ps, st, data, state)
            epoch_loss += loss
            push!(losses, loss)
            print("-")
            if i % log_every == 0
                t = now()
                print(
                    "Batch $i/$num_batches | " *
                    "Loss = $(round(loss; sigdigits=4))" *
                    "Time: $t \n" 
                )
            end
        end

        avg_epoch_loss = epoch_loss / num_batches
        t = now()

        @save "ps$(avg_epoch_loss)_$t.jld2" ps
        @save "ps_latestvn.jld2" ps
        @save "st$(avg_epoch_loss)_$t.jld2" st
        @save "st_latestvn.jld2" st
        println(
            "\n=== Epoch $epoch finished | " *
            "avg loss = $(round(avg_epoch_loss; sigdigits=5)) | "*
            " Time: $t" *
            " ==="
        )
    end
    return ps, st, losses
end

ps, st, losses = train!(
    model_conv,
    ps,
    st,
    opt,
    im_path;
    size = 64
)