model_conv = Chain(
    Conv((17, 17), 1 => 45; pad = 8),
    SkipConnection(
        Chain(
            Conv((1, 1), 45 => 256, gelu),
            SkipConnection(
                Chain(
                    Conv((1, 1), 256 => 256, gelu),
                    Conv((1, 1), 256 => 256, gelu),
                    Conv((1, 1), 256 => 256, gelu),
                ),
                +
            ),
            Conv((1, 1), 256 => 128, gelu),
            Conv((1, 1), 128 => 45, gelu),
        ),
        +
    ),
    Conv((1, 1), 45 => 45, gelu),
    Conv((1, 1), 45 => 45, gelu),
    Conv((1, 1), 45 => 1)
)   