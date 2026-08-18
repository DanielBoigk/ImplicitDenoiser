using Images
using FileIO
using ProgressMeter

function load_tinyimagenet(root)
    files = String[]

    for (dir, _, fs) in walkdir(root)
        for f in fs
            endswith(f, ".JPEG") && push!(files, joinpath(dir, f))
        end
    end

    N = length(files)

    X = Array{Float32}(undef, 64, 64, 1, N)

    @show N

    @showprogress for (i, file) in enumerate(files)
        img = load(file)

        gray = Gray.(img)

        X[:, :, 1, i] .= Float32.(channelview(gray))
    end

    return X, files
end

imgs, _ = load_tinyimagenet("../../../../ImageNet")
imgs = reshape(imgs, (64,64,100000))