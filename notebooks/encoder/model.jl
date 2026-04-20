model_first_conv = Chain(
        
        Conv((5, 5), 1 => 8, tanh; pad = 2),
        Conv((5, 5), 8 => 16; pad = 2),
        SkipConnection(Chain(
            Conv((1, 1), 16 => 64, gelu),
            Conv((1, 1), 64 => 32, gelu),
            Conv((1, 1), 32 => 16),
        ), +),
    ) 
#GroupNorm(64, 4),
model_conv = Chain(
        model_first_conv,
        GroupNorm(16, 4),
        Conv((5, 5), 16 => 32, tanh; pad = 2),
        Conv((5, 5), 32 => 64; pad = 2),
        SkipConnection(Chain(
            Conv((1, 1), 64 => 128, gelu),
            Conv((1, 1), 128 => 128, gelu),
            Conv((1, 1), 128 => 64, gelu),
        ), +),
        GroupNorm(64, 4),
        Conv((1, 1), 64 => 64, gelu),
        Conv((1, 1), 64 => 32, gelu),
        Conv((1, 1), 32 => 1),
    ) 

model = NeuralODE(model_conv, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()), reltol = 1e-3, abstol = 1e-4, save_start = false)

#model = SkipDeepEquilibriumNetwork(model_conv, Tsit5(); save_everystep = false, sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()), reltol = 1e-5, abstol = 1e-6, save_start = false)