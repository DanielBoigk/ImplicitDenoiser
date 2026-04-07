encoder = Chain(
        
        Conv((5, 5), 1 => 8, gelu; pad = 2),
        Conv((5, 5), 8 => 16, gelu; pad = 2),
        SkipConnection(Chain(
            Conv((1, 1), 16 => 64, tanh),
            Conv((1, 1), 64 => 32, tanh),
            Conv((1, 1), 32 => 16),
        ), +),
    ) 


model_inner = Chain(
        model_first_conv,
        Conv((5, 5), 16 => 32; pad = 2),
        Conv((5, 5), 32 => 64; pad = 2),
        SkipConnection(Chain(
            Conv((1, 1), 64 => 128, gelu),
            Conv((1, 1), 128 => 128, gelu),
            Conv((1, 1), 128 => 64, gelu),
        ), +),
        Conv((1, 1), 64 => 32, tanh),
        Conv((1, 1), 32 => 16),
    ) 

sdeq = NeuralODE(model_inner, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
    reltol = 1e-5, abstol = 1e-6, save_start = false)



model = Chain(
    encoder,
    sdeq,
    readout,
)