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
#using Distributions

#Reactant.set_default_backend("cpu")
#const device = reactant_device()
const cdev = cpu_device()
const gdev = gpu_device()
dev = gdev

include("model.jl")
rng = Xoshiro()


#model = NeuralODE(model_conv, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()), reltol = 1e-3, abstol = 1e-4, save_start = false)
#ps, st = Lux.setup(rng, model)
#ps = ps |> ComponentArray{Float32}|> dev  
@load "ps_latestvn.jld2" ps
@load "st_latestvn.jld2" st

ps = ps |> ComponentArray |> dev
st = st |> dev

opt = Optimisers.NAdam(3.3e-4)
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
x = randn(T,128,128,1,4)
y_true = randn(T,128,128,1,4)
x_dev = x |> dev
y_dev = y_true |> dev
y, _ = model(x_dev, ps, st)
=#

function add_gaussian_noise(img, σ = 1e-3 )
    out = img .+ σ .* randn(eltype(img), size(img))
    @. out = max(out, 0.0)
    #if rand(Bool)
        @. out = min(out, 1.0)
    #end
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

function load_images_to_array(path, num, corrupt_func, size)
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

function spectral_corrupt_image(img::AbstractMatrix{T};
    σ_spatial = T(1e-2), σ_freq = T(6e-5), iter = 3, λ = T(1e-3), s = T(1.0),
) where T <: AbstractFloat
    out = copy(img)
    n, m = size(out)
    k = repeat(T.(0:n-1), 1, m)
    ℓ = repeat(T.(0:m-1)', n, 1)
    freqsq = k.^2 + ℓ.^2
    amplitude = freqsq .+ one(T)
    for i in 1:iter
        out .+= σ_spatial .* randn(n,m) # Spatial corruption
        C = dct(dct(out, 1), 2) # DCT projection
        C .+= σ_freq .* randn(n,m) .* amplitude # Frequency corruption
        C .*= exp.(-λ .* (freqsq.^s)) # Linear Blur
        out .= idct(idct(C, 1), 2) # Back projection
    end
    @. out = clamp(out, zero(T), one(T))              # clamp is cleaner than two passes
    return out
end

#cfunc(img) = add_gaussian_noise(Float32.(img), abs(randn(Float32))*0.1f0 )

cfunc(img) = spectral_corrupt_image( Float32.(img), σ_spatial = Float32(abs(randn(Float32))*5e-3), σ_freq = Float32(abs(randn(Float32))*3e-5), iter = rand(1:10) , λ = Float32(abs(randn(Float32))*5e-2))

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
    t = now(); println("Starting Training for epoch $epoch ...   $t\n")

        epoch_loss = 0.0f0
        dataloader = dataloader = load_images(batchsize, im_path, num_images, cfunc, size)
        num_batches = length(dataloader)

        for (i, data) in enumerate(dataloader)
            r = 0.5f0 * abs(randn(Float32))
            tspan = (0.0f0, 1.0f0 + r)
            ps, st, state, loss = inner_loop(model, tspan, ps, st, data, state)
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
    model_conv,
    ps,
    st,
    opt,
    im_path;
    size = 64,
    epochs = 2
)

opt2 = Optimisers.NAdam(1e-4)

ps, st, losses = train!(
    model_conv,
    ps,
    st,
    opt2,
    im_path;
    size = 64,
    epochs = 2
)
#=
opt3 = Optimisers.NAdam(3.33e-5)

ps, st, losses = train!(
    model_conv,
    ps,
    st,
    opt3,
    im_path;
    size = 64,
    epochs = 15
)
=#