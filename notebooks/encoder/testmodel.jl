using Lux, LuxCUDA, MLUtils
using Optimisers, Random, Statistics
using Zygote
using DiffEqFlux, OrdinaryDiffEq
using Images, JLD2
using ComponentArrays
using DeepEquilibriumNetworks
using Plots
using Dates

const cdev = cpu_device()
const gdev = gpu_device()
dev = gdev

include("model.jl")

rng = Xoshiro(0)
ps, st = Lux.setup(rng, model)

ps = ps |> ComponentArray |> dev
st = st |> dev

opt = Optimisers.NAdam(1e-4)
state = Optimisers.setup(opt,ps)
train_state = Lux.Training.TrainState(model, ps, st, opt)

function loss_function(model, ps, st, (x, y_true))
    y_pred = model(x, ps, st)[1][1]
    loss_mse= MSELoss()
    mse_loss = loss_mse(y_pred, y_true)
    #sptrl_loss = loss_mse(dct(dct(y_pred,1),2),dct(dct(y_pred,1),2))
    return mse_loss, st
    #return mes_loss + sptrl_loss, st
end

x = randn(Float32,128,128,1,4)
y_true = randn(Float32,128,128,1,4)
x_dev = x |> dev
y_dev = y_true |> dev
y, _ = model(x_dev, ps, st)