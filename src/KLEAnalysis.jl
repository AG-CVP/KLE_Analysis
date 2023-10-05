using Statistics

# KLE identical to matlab
function KLE(mkg::MKG, reference::MKG; startIndex::Int = 1, segmentLength::Int = size(reference.map, 2), startCursor::Int = -1)
    @assert size(mkg.map) == size(reference.map) "The maps have incompatible sizes."
    @assert 1 <= startIndex <= size(reference.map, 2) "Start index is not in map."

    if startCursor > 0
    mkgStart = mkg.cursors[startCursor]
    referenceStart = reference.cursors[startCursor]
    else
        mkgStart = referenceStart = startIndex
    end
    ratio = (mkg.map[:, mkgStart:mkgStart+segmentLength-1] ./ reference.map[:, referenceStart:referenceStart+segmentLength-1])
    # @show ratio
    ratio = log.(ratio)
    # @show ratio
    tosum = ratio .* mkg.map[:, mkgStart:mkgStart+segmentLength-1]
    # @show tosum
    result = sum(tosum, dims = 1)
    return vec(result)
end

"""
Preprocess the MKG map by normalizing it to the range [0,1].
This function modifies the MKG object inplace. 
"""
function preprocess!(mkg::MKG; og = true)
    nChannels, len = size(mkg.map)
    ε = 1 / nChannels
    if og 
        mkg.map = mapslices(x -> ((x .- minimum(x))) .+ ε * (maximum(x) - minimum(x)), mkg.map, dims = 1)
        mkg.map = mapslices(x -> x ./ sum(x), mkg.map, dims = 1)
        return mkg
    end
    # Normalize
    mkg.map = mapslices(x -> x .- minimum(x), mkg.map, dims = 1)
    # mkg.map = mapslices(x -> x .+ (maximum(x) * 1/nChannels), mkg.map, dims = 1)
    mkg.map .+= ε
    mkg.map = mapslices(x -> x ./ maximum(x), mkg.map, dims = 1)
    # mkg.map = mapslices(x -> x .+ ((maximum(x) > 0 ? maximum(x) : 1) * 1/nChannels), mkg.map, dims = 1)
    mkg.map = mapslices(x -> x ./ sum(x), mkg.map, dims = 1)
    return mkg
end

"""
Preprocess the MKG map by normalizing it to the range [0,1].
This function modifies the MKG object inplace. 
"""
function preprocess(mkg::MKG; og = true)
    mkg = MKG(copy(mkg.map), copy(mkg.cursors))
    return preprocess!(mkg; og = og)
end

function getReferenceMap(mkgs, normalizationInds = (2,3), og = true)
    if og
        data_collector = zeros(size(mkgs[1].map))
        cursor_collector = zeros(size(mkgs[1].cursors))
        for mkg in mkgs
            data_collector .+= mkg.map
            cursor_collector .+= mkg.cursors
        end
        return MKG(data_collector ./ length(mkgs), round.(Int, cursor_collector ./ length(mkgs)))
    end
    # Area around R peak is used for normalization
    # nCoeffs = map(x -> mean(mean(abs, x.map[:, x.cursors[normalizationInds[1]]:x.cursors[normalizationInds[2]]], dims = 1)), mkgs)
    nCoeffs = 1 # As we already preprocessed the data, we can assume that area around R peak has the value 1. 
    # Reference is the mean of all maps normalized by their QRS region 
    referenceMap = mean(getfield.(mkgs, :map) ./ nCoeffs)

    # Mean Cursors are considered
    referenceCursor = round.(Int, mean(getfield.(mkgs, :cursors)))
    return MKG(referenceMap, referenceCursor)
end

"""
Calculate the discrimination index (DI) for a given set of values (prediction) and labels (ground truth).
"""
function DI(values, labels)
    labs = unique(labels)
    @assert length(labs) == 2 "There need to be exactly 2 labels."
    inA = labs[1] .== labels
    A = values[inA]
    B = values[.!inA]

    return abs(mean(A) .- mean(B)) ./ std(vcat(A, B))
end

"""
Normalization of the MKG by correcting differences in HR using weno4 Interpolation.
Results in equidistant segments (QRS, STT, ...)
"""
function normHR(mkg::MKG; qrsCursor = 1, qrsPosition = 400, goalLength = 1000)
    isSignal = vec(any(mkg.map .!= 0, dims = 1)) 
    newCursors = round.(Int, (mkg.cursors .- (1:1000)[isSignal][1]) .* (length(isSignal) / sum(isSignal)))
    toShift = newCursors[qrsCursor] - qrsPosition
    newCursors .-= toShift

    newX = range((1:length(isSignal))[isSignal][1], (1:length(isSignal))[isSignal][end], length = goalLength)
    newMap = mapslices(channel -> circshift(interpolate_weno4(newX, 1:1000, channel), -toShift), mkg.map, dims = 2)
    
    newMKG = MKG(newMap, newCursors)
    return newMKG
end

"""
Normalization of segments of the MKG defined by its cursors (start_c and end_c) by using weno4 Interpolation to a fixed goal length.
"""
function norm_segment(mkg::MKG; start_c = 2, end_c = 3, goal_len = 100)
    x = mkg.cursors[start_c]:mkg.cursors[end_c]
    new_x = range(mkg.cursors[start_c], mkg.cursors[end_c], length = goal_len)
    newMap = mapslices(channel -> interpolate_weno4(new_x, x, channel), mkg.map[:, x], dims = 2)
    return newMap
end


function KLE_segment(mkg::MKG, reference::MKG; start_c = 2, end_c = 3, goal_len = 100)
    @assert size(mkg.map) == size(reference.map) "The maps have incompatible sizes."
    @assert 1 <= start_c <= size(reference.map, 2) "Start index is not in map."

    mkg_seg = norm_segment(mkg; start_c = start_c, end_c = end_c, goal_len = goal_len)
    ref_seg = norm_segment(reference; start_c = start_c, end_c = end_c, goal_len = goal_len)
    ratio = (mkg_seg ./ ref_seg)
    ratio = log.(ratio)
    tosum = ratio .* mkg_seg
    result = sum(tosum, dims = 1)
    return vec(result)
end

function segment_length(mkg::MKG, start_c = 2, end_c = 3)
    len = mkg.cursors[end_c] - mkg.cursors[start_c]
    return len
end

function plot_KLE_segment(df, reference, source, segment, normalization = false; args...)
    dfRes = subset(df, [:Reference, :Segment, :Source] => (ref, seg, src) -> ((ref .== reference) .& (seg .== segment) .& (src .== source)), view = true)
    dfResNorm = subset(dfRes, :Group => group -> group .== "Norm")
    dfResMyo = subset(dfRes, :Group => group -> group .== "Myo")

    if normalization
        p1 = plot(vec(mean(hcat(dfResNorm.KLEvalues_norm...), dims = 2)), lab = "Healthy", legend = :topleft; args...)
        plot!(vec(mean(hcat(dfResMyo.KLEvalues_norm...), dims = 2)), lab = "Myocarditis"; args...)
    else
        p1 = plot(vec(mean(hcat(dfResNorm.KLEvalues...), dims = 2)), lab = "Healthy", legend = :topleft; args...)
        plot!(vec(mean(hcat(dfResMyo.KLEvalues...), dims = 2)), lab = "Myocarditis"; args...)
    end
    return p1
end

import LinearAlgebra.normalize!
function normalize!(x::MKG; frame = 333)
    x.map .-= minimum(x.map[:, frame])
    x.map ./= maximum(x.map[:, frame])
    return x
end