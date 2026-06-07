using Pkg
Pkg.activate("../../")
using Lux, Reactant, Enzyme, MLUtils, NNlib
using Optimisers, Random, Statistics, Images
using LinearAlgebra, Images, JLD2, ComponentArrays
using Dates, Plots, UnicodePlots

using MLDatasets
imgs_color, _ = CIFAR10(split=:train)[:]
imgs = reshape(mean(Float32.(imgs_color), dims=3),(32,32,50000))

batch_size = 64
dim = 32 

spatial_conv_layer = Conv((17, 17), 2 => 256; pad = 8)
struct ConvFirstTwo{C}<: Lux.AbstractLuxWrapperLayer{:conv}
    conv::C
    length::Int
end
# 2. Explicitly define how state is generated
function Lux.initialstates(rng::AbstractRNG, m::ConvFirstTwo)
    return (conv = Lux.initialstates(rng, m.conv),)
end

# 3. Explicitly define how parameters are generated
function Lux.initialparameters(rng::AbstractRNG, m::ConvFirstTwo)
    return (conv = Lux.initialparameters(rng, m.conv),)
end
function (m::ConvFirstTwo)(x, ps, st) 
    # Use standard slicing that maps cleanly to XLA operations
    x1 = x[:, :, 1:2, :]
    x2 = x[:, :, 3:end, :] # 'end' is perfectly fine and statically resolved by XLA
    y1, st_conv = m.conv(x1, ps.conv, st.conv)
    return cat(y1, x2; dims=3), (conv = st_conv,)
end

first_conv = ConvFirstTwo(spatial_conv_layer, 34)

emb_dim = 32
load_model = true
test_model = false

model= Chain(
    #Conv((17, 17), 2 => 256; pad = 8),
    first_conv,
    #last_conv,
    SkipConnection(
        Chain(
            Conv((1, 1), 256+emb_dim => 512, gelu),
            SkipConnection(
                Chain(
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                ),
                +
            ),
            Conv((1, 1), 512 => 256, gelu),
            Conv((1, 1), 256 => 256+emb_dim, gelu),
        ),
        +
    ),
    Conv((1, 1), 256+emb_dim => 128, gelu),
    Conv((1, 1), 128 => 64, gelu),
    Conv((1, 1), 64 => 1)
)


const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev
rng = Xoshiro()
opt = Optimisers.NAdam(1e-3)


if load_model
    @load "ps_latestvn.jld2" ps_cpu 
    @load "st_latestvn.jld2" st_cpu
    ps = ps_cpu |> dev
    st = st_cpu |> dev
else
    ps, st = Lux.setup(rng, model)|> dev  
end
state = Optimisers.setup(opt,ps)


train_state = Training.TrainState(model, ps, st, opt)

train!(data, train_state) = Training.single_train_step!(
                AutoEnzyme(),
                MSELoss(),
                data,
                train_state;
                return_gradients=Val(false),
            )

if test_model
    x_trial = randn(Float32,dim,dim,2+emb_dim,batch_size) |> dev
    y_trial = randn(Float32,dim,dim,1,batch_size) |> dev
    data_trial = (x_trial,y_trial)
    
    model_compiled = @compile model(x_trial, ps, st)
    y_pred, st = model_compiled(x_trial, ps, st)
    println("Model successfully compiled!")
    train!(data_trial, train_state)
    println("Train step successfully compiled!")
end

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

function load_images(imgs::AbstractArray, t_max, forward,emb_dim)
    x_dim, y_dim ,num_imgs = size(imgs)
    x = ones(Float32, x_dim, y_dim, 2+emb_dim, num_imgs)
    y = zeros(Float32, x_dim,y_dim, 1, num_imgs)
    t = t_max .* rand(Float32, num_imgs)
    # Equivariant training data augmentation
    if x_dim == y_dim
        for i in 1:num_imgs
            img = Float32.(imgs[:,:,i])
            if rand(Bool)
                img = reflect_spatial(img)
            end
            img = rot90_spatial(img, rand(0:3))
            xt, ϵ = forward(img, t[i])
            x[:, :, 1, i] = xt
            emb = reshape(sinusoidal_embedding(t[i], emb_dim), (1,1,emb_dim,1))
            x[:, :, 3:end, i] .= emb
            y[:, :, 1, i] = ϵ
        end
    else
        for i in 1:num_imgs
            img = Float32.(imgs[:,:,i])
            xt, ϵ = forward(img, t[i])
            x[:, :, 1, i] = xt
            emb = reshape(sinusoidal_embedding(t[i], emb_dim), (1,1,emb_dim,1))
            x[:, :, 3:end, i] .= emb
            y[:, :, 1, i] = ϵ
        end
    end
    return x,y
end
function create_dataloader(imgs, T, forward, emb_dim, dev, batch_size)        
    data = load_images(imgs, T, forward, emb_dim)
    DataLoader(mapobs(dev, data); batchsize=batch_size, shuffle = true, partial=false)
end
data_args = (imgs, T, forward, emb_dim, dev, batch_size)
dataloader = create_dataloader(data_args...)

function train_epoch!(dataloader, train_state, epoch_idx::Int, print_intermediate::Bool = false)
    start_time = now()
    println("\n" * "="^50)
    println("🚀 Epoch $epoch_idx Started at: $(Dates.format(start_time, "yyyy-mm-dd HH:MM:SS"))")
    println("="^50)

    batch_losses = Float32[]
    num_batches = length(dataloader)

    for (i, data) in enumerate(dataloader)
        # train! edits the training state and returns the current loss
        _, loss, _, train_state = train!(data, train_state)
        
        # Move loss to CPU to store/print safely
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

epoch = 1
train_state, losses = train_epoch!(dataloader, train_state, epoch)

#Plot the batch losses directly to the console
function plot_epoch(losses,epoch)
    println("📈 Loss Curve (Epoch $epoch):")
    plt = lineplot(
        losses, 
        title = "Training Loss", 
        xlabel = "Batch", 
        ylabel = "MSE", 
        border = :dotted,
        width = 60,
        height = 15
    )
    display(plt)
end
plot_epoch(losses,epoch)
ps_cpu = train_state.parameters |> cdev
st_cpu = train_state.states |> cdev
@save "ps_latestvn.jld2" ps_cpu        
@save "st_latestvn.jld2" st_cpu

using Optimisers: adjust!
learn_rate = 1.0e-3
decay = 0.98

learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)
learn_rate *= decay; train_state = adjust!(train_state, learn_rate); epoch +=1; dataloader = create_dataloader(data_args...);train_state, losses = train_epoch!(dataloader, train_state, epoch)




















ps_cpu = train_state.parameters |> cdev
st_cpu = train_state.states |> cdev
@save "ps_latestvn.jld2" ps_cpu        
@save "st_latestvn.jld2" st_cpu