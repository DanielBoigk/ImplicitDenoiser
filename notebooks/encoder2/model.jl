model_first_conv = Chain(
        
        Conv((5, 5), 1 => 8; pad = 2),
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
        #GroupNorm(16, 4),
        Conv((5, 5), 16 => 32; pad = 2),
        Conv((5, 5), 32 => 64; pad = 2),
        SkipConnection(Chain(
            Conv((1, 1), 64 => 256, gelu),
            Conv((1, 1), 256 => 256, gelu),
            Conv((1, 1), 256 => 256, gelu),
            Conv((1, 1), 256 => 64, gelu),
        ), +),
        #GroupNorm(64, 4),
        Conv((1, 1), 64 => 64, gelu),
        Conv((1, 1), 64 => 32, gelu),
        Conv((1, 1), 32 => 1),
)