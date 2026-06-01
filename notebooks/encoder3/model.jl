model_conv = Chain(
    Conv((17, 17), 1 => 64; pad = 8),
    SkipConnection(
        Chain(
            Conv((1, 1), 64 => 256, gelu),
            SkipConnection(
                Chain(
                    Conv((1, 1), 256 => 256, gelu),
                    Conv((1, 1), 256 => 256, gelu),
                    Conv((1, 1), 256 => 256, gelu),
                ),
                +
            ),
            Conv((1, 1), 256 => 128, gelu),
            Conv((1, 1), 128 => 64, gelu),
        ),
        +
    ),
    Conv((1, 1), 64 => 64, gelu),
    Conv((1, 1), 64 => 64, gelu),
    Conv((1, 1), 64 => 1)
)   