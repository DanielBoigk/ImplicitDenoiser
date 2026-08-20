using Pkg
Pkg.activate("../")
using Lux, Reactant, Enzyme, MLUtils, NNlib
using Optimisers, Random, Statistics, Images
using LinearAlgebra, Images, JLD2, ComponentArrays
using Dates, Plots, UnicodePlots

batch_size = 128
in_channels = 3
out_channels = 2
dim = 64

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev
rng = Xoshiro()
opt = Optimisers.Adam(2.0f-4)

include("model.jl") 

train!(data, train_state) = Training.single_train_step!(
    AutoEnzyme(),
    MSELoss(),
    data,
    train_state;
    return_gradients=Val(false),
)

function fill_inscribed_circle!(m::Array{Float32,4}, value::Float32=1.0f0)
    dim, dim2, _, batch_size = size(m)
    @assert dim == dim2 "First two dimensions must be equal (square slice)"

    center = (dim - 1) / 2f0        # 0-based center, works for even/odd dim
    radius = dim / 2f0
    r2 = radius^2

    for j in 1:dim, i in 1:dim
        # distance from center in 0-based coords
        di = (i - 1) - center
        dj = (j - 1) - center
        if di^2 + dj^2 <= r2
            @inbounds for b in 1:batch_size
                m[i, j, 1, b] = value
            end
        end
    end
    return m
end

if test_model
    x_trial = randn(Float32, dim, dim, in_channels, batch_size) |> dev
    # This is the mask channel
    m_trial = zeros(Float32, dim, dim, 1, batch_size)
    fill_inscribed_circle!(m_trial)
    m_trial = m_trial |> dev
    y_trial = randn(Float32, dim, dim, out_channels, batch_size) |> dev
    data_trial = ((x_trial, m_trial), y_trial)

    model_compiled = @compile model((x_trial, m_trial), ps, st)
    y_pred, st = model_compiled((x_trial, m_trial), ps, st)
    println("Model successfully compiled!")
    train!(data_trial, train_state)
    println("Train step successfully compiled!")
end
