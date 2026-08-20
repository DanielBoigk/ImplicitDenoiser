using Pkg
Pkg.activate("../../../")
using Lux, LuxCore, Reactant, Enzyme, MLUtils, NNlib
using Optimisers, Random, Statistics, Images
using LinearAlgebra, Images, JLD2, ComponentArrays
using Dates, Plots, UnicodePlots

include("loadImgnet.jl")   # defines `imgs` :: (64, 64, 100000) Float32, in [0, 1]
include("../model.jl")     # defines the attention U-Net (see unet_tinyimagenet64)

batch_size = 128
dim = 64

# Width of the internal sinusoidal embedding of the noise level. This is now
# purely an architectural hyperparameter of the U-Net (embedded internally
# and broadcast to every pixel) — unlike the old flat-CNN model, it no longer
# needs to match anything about the input tensor's channel count.
emb_dim = 32

load_model = true
test_model = true

model = unet_tinyimagenet64(; embedding_dims=emb_dim)

const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev
rng = Xoshiro()
opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.NAdam(2.0f-4))

if load_model
    @load "ps_latestvn.jld2" ps_cpu
    @load "st_latestvn.jld2" st_cpu
    ps = ps_cpu |> dev
    st = st_cpu |> dev
else
    ps, st = Lux.setup(rng, model) |> dev
end

train_state = Training.TrainState(model, ps, st, opt)

train!(data, train_state) = Training.single_train_step!(
    AutoEnzyme(),
    MSELoss(),
    data,
    train_state;
    return_gradients=Val(false),
)

if test_model
    # Model input is a (noisy_images, noise_variances) tuple, not a single
    # concatenated tensor: images keep their 1 (grayscale) channel, and the
    # per-sample scalar noise level is embedded internally by the U-Net.
    x_trial = randn(Float32, dim, dim, 1, batch_size) |> dev
    m_trial = ones(Float32, dim, dim, 1, 1) |> dev
    t_trial = rand(Float32, 1, 1, 1, batch_size) |> dev
    y_trial = randn(Float32, dim, dim, 1, batch_size) |> dev
    data_trial = ((x_trial, t_trial), y_trial)

    model_compiled = @compile model((x_trial, m_trial, t_trial), ps, st)
    y_pred, st = model_compiled((x_trial, m_trial, t_trial), ps, st)
    println("Model successfully compiled!")
    train!(data_trial, train_state)
    println("Train step successfully compiled!")
end


# Hyperparameters for the Variance Preserving (VP) SDE
const βmin = 0.1
const βmax = 20.0
T = 1

function normalize_image(img)
    return 2 .* (Float32.(img) .- 0.5)
end

function denormalize_image(img)
    return (0.5 .* img) .+ 0.5
end

function forward_sample(x0, t, ᾱ)
    αbar = ᾱ(t)
    ε = randn(Float32, size(x0))
    xt = sqrt(αbar) .* x0 .+ sqrt(1 - αbar) .* ε
    return xt, ε
end

β(t) = βmin + (βmax - βmin) * t / T
ᾱ(t) = exp(-βmin * t - (βmax - βmin) / (2 * T) * t^2)

forward(x, t) = forward_sample(x, t, ᾱ)

# rotate spatial dims (1,2) by 90° * k
function rot90_spatial(K, k::Int)
    k = mod(k, 4)
    k == 0 && return K
    k == 1 && return reverse(permutedims(K, (2, 1)), dims=1)
    k == 2 && return reverse(reverse(K, dims=1), dims=2)
    k == 3 && return reverse(permutedims(K, (2, 1)), dims=2)
end

# reflection (horizontal mirror)
reflect_spatial(K) = reverse(K, dims=2)

struct NoisyImageDataset
    imgs::Array{Float32,3}   # (H, W, N) — raw images on CPU, in [0, 1]
    t_max::Float32
end

Base.length(d::NoisyImageDataset) = size(d.imgs, 3)

function Base.getindex(d::NoisyImageDataset, i::Int)
    img = d.imgs[:, :, i]

    # Equivariant training data augmentation
    if size(img, 1) == size(img, 2)
        rand(Bool) && (img = reflect_spatial(img))
        img = rot90_spatial(img, rand(0:3))
    end

    img = normalize_image(img)

    t = d.t_max * rand(Float32)
    xt, ε = forward(img, t)

    H, W = size(img)
    ximg = reshape(Float32.(xt), H, W, 1)
    t_arr = fill(Float32(t), 1, 1, 1)
    y = reshape(Float32.(ε), H, W, 1)
    return (ximg, t_arr), y
end

# getindex over a range → stack individual samples into a batch
function Base.getindex(d::NoisyImageDataset, idxs::AbstractVector{Int})
    samples = [d[i] for i in idxs]
    ximgs = cat((s[1][1] for s in samples)...; dims=4)
    ts = cat((s[1][2] for s in samples)...; dims=4)
    ys = cat((s[2] for s in samples)...; dims=4)
    return (ximgs, ts), ys
end

function create_dataloader(imgs, T, dev, batch_size)
    dataset = NoisyImageDataset(imgs, Float32(T))
    DataLoader(dataset; batchsize=batch_size, shuffle=true, partial=false, collate=true, buffer=false) |> dev
end
data_args = (imgs, T, dev, batch_size)
dataloader = create_dataloader(data_args...)

function train_epoch!(dataloader, train_state, epoch_idx::Int, print_intermediate::Bool=false)
    start_time = now()
    println("\n" * "="^50)
    println("🚀 Epoch $epoch_idx Started at: $(Dates.format(start_time, "yyyy-mm-dd HH:MM:SS"))")
    println("="^50)

    batch_losses = Float32[]
    num_batches = length(dataloader)

    for (i, data) in enumerate(dataloader)
        _, loss, _, train_state = train!(data, train_state)

        current_loss = Float32(loss)
        push!(batch_losses, current_loss)
        if print_intermediate
            if i % 10 == 0 || i == num_batches
                println("⏳ Batch $i / $num_batches | Loss: $(round(current_loss, sigdigits=5))")
            end
        end
    end

    end_time = now()
    duration = end_time - start_time
    avg_loss = mean(batch_losses)

    println("-"^50)
    println("✅ Epoch $epoch_idx Finished at: $(Dates.format(end_time, "yyyy-mm-dd HH:MM:SS"))")
    println("⏱️  Duration: $(duration)")
    println("📊 Average Loss: $(round(avg_loss, sigdigits=5))")
    println("="^50 * "\n")

    return train_state, batch_losses
end

function save_checkpoint(train_state; snapshot::Bool=false, avg_loss=nothing)
    ps_cpu = train_state.parameters |> cdev
    st_cpu = train_state.states |> cdev
    @save "ps_latestvn.jld2" ps_cpu
    @save "st_latestvn.jld2" st_cpu
    if snapshot
        t = now()
        @save "snapshots/ps$(avg_loss)_$t.jld2" ps_cpu
        @save "snapshots/st$(avg_loss)_$t.jld2" st_cpu
    end
end

# Plot the batch losses directly to the console
function plot_epoch(losses, epoch)
    println("📈 Loss Curve (Epoch $epoch):")
    plt = lineplot(
        losses,
        title="Training Loss",
        xlabel="Batch",
        ylabel="MSE",
        border=:dotted,
        width=60,
        height=15,
    )
    display(plt)
end

"""
    train_many_epochs!(train_state, n_epochs; base_lr, decay, warmup_epochs, start_epoch)

Runs `n_epochs` of training: a linear LR warmup from 0 up to `base_lr` over
the first `warmup_epochs` epochs (guards against early instability from the
attention block — self-attention logits are unnormalized at init and can
produce a few outsized gradients before the block settles down), then
geometric decay (`lr *= decay` every epoch) after that. Checkpoints to
`ps_latestvn.jld2` / `st_latestvn.jld2` (plus a timestamped snapshot) after
every epoch. Safe to interrupt (e.g. Ctrl-C in the REPL) between epochs and
re-run — pass `start_epoch` to pick the LR schedule back up where it left off.
"""
function train_many_epochs!(
    train_state, n_epochs::Int;
    base_lr::Float32=2.0f-4, decay::Float32=0.985f0, warmup_epochs::Int=3, start_epoch::Int=1,
)
    epoch = start_epoch
    for _ in 1:n_epochs
        learn_rate = if epoch <= warmup_epochs
            base_lr * epoch / warmup_epochs
        else
            base_lr * decay^(epoch - warmup_epochs)
        end
        train_state = Optimisers.adjust!(train_state, Float32(learn_rate))
        train_state, losses = train_epoch!(dataloader, train_state, epoch)
        plot_epoch(losses, epoch)
        save_checkpoint(train_state; snapshot=true, avg_loss=mean(losses))

        epoch += 1
        global dataloader = create_dataloader(data_args...)
    end
    return train_state
end

# --- Suggested training schedule --------------------------------------------
# This U-Net is ~2.0M parameters; at batch_size=64 over the 100k-image
# TinyImageNet set that's ~1562 steps/epoch. NAdam + grad-norm clipping,
# 3-epoch warmup up to a peak LR of 2e-4, then 0.985 decay/epoch — 150 epochs
# (~234k steps) is a reasonable first full run, ending around lr≈2e-5.
# Loss alone is a weak proxy for sample quality here, so periodically render
# a few images with sample.jl (e.g. every ~10 epochs) to actually look at
# progress. Extend further (larger n_epochs / start_epoch) if still improving.

# Sanity check first — one (gently warmed-up) epoch to confirm the loss is
# actually going down before committing to the full run:
#train_state = train_many_epochs!(train_state, 6)

# Then the rest of the schedule:
train_state = train_many_epochs!(train_state, 150; start_epoch=1)
