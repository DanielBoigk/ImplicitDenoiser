model= Chain(
    Conv((17, 17), 2 => 256; pad = 8),
    SkipConnection(
        Chain(
            Conv((1, 1), 256 => 512, gelu),
            SkipConnection(
                Chain(
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                    Conv((1, 1), 512 => 512, gelu),
                ),
                +
            ),
            Conv((1, 1), 512 => 256, gelu),
            Conv((1, 1), 256 => 256, gelu),
        ),
        +
    ),
    Conv((1, 1), 256 => 128, gelu),
    Conv((1, 1), 128 => 64, gelu),
    Conv((1, 1), 64 => 1)
)