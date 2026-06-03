
const xdev = reactant_device(; force=true)
const cdev = cpu_device()
dev = xdev
rng = Xoshiro()
opt = Optimisers.NAdam(3.3e-4)


if load_model
    @load "ps_latestvn.jld2" ps
    @load "st_latestvn.jld2" st
else
    ps, st = Lux.setup(rng, model)|> dev  
end
state = Optimisers.setup(opt,ps)

x_trial = randn(Float32,128,128,1+emb_dim,10) |> dev
y_trial = randn(Float32,128,128,1,10) |> dev
data_trial = (x_trial,y_trial)

model_compiled = @compile model(x_trial, ps, st)
y_pred, st = model_compiled(x_trial, ps, st)

train_state = Training.TrainState(model, ps, st, opt)

train!(data, train_state) = Training.single_train_step!(
                AutoEnzyme(),
                MSELoss(),
                data,
                train_state;
                return_gradients=Val(false),
            )
