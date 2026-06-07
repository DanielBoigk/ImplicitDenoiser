spatial_conv_layer = Conv((17, 17), 2 => 256; pad = 8)

struct ConvFirst{C}<: Lux.AbstractLuxWrapperLayer{:conv}
    conv::C
    split::Int
end

function (m::ConvFirst)(x, ps, st) 
    x1 = x[:,:,1:m.split,:]
    x2  = x[:,:,m.split+1:end,:]

    y1, st_conv = m.conv(x1, ps, st)

    return cat(y1, x2; dims=3),
           (conv = st_conv,)
end
struct ConvLast{C}<: Lux.AbstractLuxWrapperLayer{:conv}
    conv::C
    split::Int
end
Lux.initialstate(rng::AbstractRNG, m::ConvLast) = (conv = Lux.initialstate(rng, m.conv),)
function (m::ConvLast)(x, ps, st) 
    x1 = x[:,:,1:m.split,:]
    x2  = x[:,:,m.split+1:end,:]

    y2, st_conv = m.conv(x2, ps, st)

    return cat(x1, y2; dims=3),
           (conv = st_conv,)
end

struct ConvFirstTwo{C}<: Lux.AbstractLuxWrapperLayer{:conv}
    conv::C
end
Lux.initialstate(rng::AbstractRNG, m::ConvFirstTwo) = (conv = Lux.initialstate(rng, m.conv),)
# Add these definitions right after your struct declarations



function (m::ConvFirstTwo)(x, ps, st) 
    x1 = x[:,:,1:2,:]
    x2  = x[:,:,3:end,:]

    y1, st_conv = m.conv(x1, ps, st)

    return cat(y1, x2; dims=3),
           (conv = st_conv,)
end
first_conv = ConvFirst(spatial_conv_layer, 2)

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