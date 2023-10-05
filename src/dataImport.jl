using Plots
using Colors
using Parameters

"""
This datatype contains all the information of a single MKG measurement.
It contains two fields:
- map: A (in our case 36x1000) matrix [nChannels x samples] containing the average MKG field map.
- cursors: A vector containing the cursor positions (QRST ... positions) in samples.
"""
@with_kw mutable struct MKG{T}
    map::Array{T}
    cursors::Vector{Int}
end

# This function is specific to our data and should not be used in general.
# The user should load their own data and create the MKG type from it.
function loadMKG(folder)
    cIn = Vector{UInt16}(undef, 3)
    read!(joinpath(folder, "cursors.dat"), cIn)
    cIn = [334; Int.(cIn)]

    mapIn = Array{Float64}(undef, 36, 1000)
    read!(joinpath(folder, "outputarr.dat"), mapIn)

    return MKG(map = mapIn, cursors = cIn)
end

# Plotting function for the MKG type using the Plots.jl package.
# Use plot(mkg) to plot the MKG map.
@recipe function f(mkg::MKG{Float64})
    nChannels = size(mkg.map, 1)
    # colorMap = distinguishable_colors(nChannels, [RGB(1, 1, 1), RGB(0, 0, 0)], dropseed = true)
    # colorMap = range(HSL(colorant"red"), stop=HSL(colorant"green"), length=nChannels)
    # colorMap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=nChannels)
    colorMap = colormap("RdBu", nChannels; mid=0.5, logscale=false)

    title := "MKG with $nChannels channels"
    for i = 1:nChannels
        @series begin
            # subplot := 1
            yguide := ""
            label := ""
            seriescolor := colorMap[i]
            mkg.map[i,:]
        end
    end
    @series begin
        label := ""
        seriescolor := :black
        seriestype := :vline
        linewidth := 2

        mkg.cursors
    end
end

@userplot MKGcontour

@recipe function g(mkgArgs::MKGcontour)
    mkg = mkgArgs.args[1]
    frame = mkgArgs.args[2]

    x = 1:6
    Z = reverse(reshape(mkg.map[:, frame], 6, 6), dims = 2)
    @series begin
        seriestype := contour
        x, x, Z
    end    
end