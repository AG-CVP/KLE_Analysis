using Statistics
using DataFrames
using StatsPlots
using Distributions
using LinearAlgebra
using MultivariateStats
using Random
using WENO4
using Plots
using MLBase

include("dataImport.jl")
include("KLEAnalysis.jl")

# Load all data in a DF
(root, ids, files) = first(walkdir("./data/SystemII"))
ids = ids[1:30] # exclude reference map
(rootSystemIMyo, SystemIMyoIds, files) = first(walkdir("./data/SystemI_data/myo"))
(rootSystemINorm, SystemINormIds, files) = first(walkdir("./data/SystemI_data/normal"))
dfMKG = DataFrame(ID = String[], Source = String[], Group = String[], MKG = MKG[], normalMKG = MKG[])
## SystemII
for id in ids
    mkg = loadMKG(joinpath(root, id))
    push!(dfMKG, (ID = id, Source = "SystemII", Group = (id[1] == '6' ? "Myo" : "Norm"), MKG = mkg, normalMKG = preprocess(mkg)))
end

## SystemI
for id in SystemIMyoIds
    mkg = loadMKG(joinpath(rootSystemIMyo, id))
    push!(dfMKG, (ID = id, Source = "SystemI", Group = "Myo", MKG = mkg, normalMKG = preprocess(mkg)))
end

for id in SystemINormIds
    mkg = loadMKG(joinpath(rootSystemINorm, id))
    push!(dfMKG, (ID = id, Source = "SystemI", Group = "Norm", MKG = mkg, normalMKG = preprocess(mkg)))
end

# reference_SystemII = preprocess!(loadMKG("./data/reference/SystemII"))
reference_SystemII = preprocess!(getReferenceMap(filter([:Group, :Source] => (g, s) -> g == "Norm" && s == "SystemII", dfMKG).MKG))
reference_SystemI = preprocess!(getReferenceMap(filter([:Group, :Source] => (g, s) -> g == "Norm" && s == "SystemI", dfMKG).MKG))
reference_Joined = preprocess!(getReferenceMap(filter(:Group => g -> g == "Norm", dfMKG).MKG))

# Precalculate KLE in all different combinations 
dfKLE = DataFrame(ID = String[], Source = String[], Group = String[], Reference = String[], Segment = String[], KLEvalues = [], KLEvalues_norm = [])
## Full
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "SystemII", Segment = "FULL", KLEvalues = map(x -> KLE(x, reference_SystemII), dfMKG.normalMKG), KLEvalues_norm = NaN))
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "SystemI", Segment = "FULL", KLEvalues = map(x -> KLE(x, reference_SystemI), dfMKG.normalMKG), KLEvalues_norm = NaN))
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "Joined", Segment = "FULL", KLEvalues = map(x -> KLE(x, reference_Joined), dfMKG.normalMKG), KLEvalues_norm = NaN))
## QRS
qrs_length = 100
@show qrs_length
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "SystemII", Segment = "QRS",
 KLEvalues = map(x -> KLE(x, reference_SystemII, startCursor = 2, segmentLength = qrs_length), dfMKG.normalMKG),
 KLEvalues_norm = map(x -> KLE_segment(x, reference_SystemII; start_c = 2, end_c = 3, goal_len = 20), dfMKG.normalMKG)))
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "SystemI", Segment = "QRS", 
KLEvalues = map(x -> KLE(x, reference_SystemI, startCursor = 2, segmentLength = qrs_length), dfMKG.normalMKG),
KLEvalues_norm = map(x -> KLE_segment(x, reference_SystemI; start_c = 2, end_c = 3, goal_len = 20), dfMKG.normalMKG)))
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "Joined", Segment = "QRS", 
KLEvalues = map(x -> KLE(x, reference_Joined, startCursor = 2, segmentLength = qrs_length), dfMKG.normalMKG),
KLEvalues_norm = map(x -> KLE_segment(x, reference_Joined; start_c = 2, end_c = 3, goal_len = 20), dfMKG.normalMKG)))
## STT
stt_length = 250
@show stt_length
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "SystemII", Segment = "STT", 
KLEvalues = map(x -> KLE(x, reference_SystemII, startCursor = 3, segmentLength = stt_length), dfMKG.normalMKG),
KLEvalues_norm = map(x -> KLE_segment(x, reference_SystemII; start_c = 3, end_c = 4, goal_len = 40), dfMKG.normalMKG)))
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "SystemI", Segment = "STT", 
KLEvalues = map(x -> KLE(x, reference_SystemI, startCursor = 3, segmentLength = stt_length), dfMKG.normalMKG),
KLEvalues_norm = map(x -> KLE_segment(x, reference_SystemI; start_c = 3, end_c = 4, goal_len = 40), dfMKG.normalMKG)))
append!(dfKLE, DataFrame(ID = dfMKG.ID, Source = dfMKG.Source, Group = dfMKG.Group, Reference = "Joined", Segment = "STT", 
KLEvalues = map(x -> KLE(x, reference_Joined, startCursor = 3, segmentLength = stt_length), dfMKG.normalMKG),
KLEvalues_norm = map(x -> KLE_segment(x, reference_Joined; start_c = 3, end_c = 4, goal_len = 40), dfMKG.normalMKG)))

# Statistical Analysis using CV and LDA on STT / QRS and Full
## CV
function generateReport(df; segment = "STT", reference_of = "SystemII", evaluate_on = "SystemII", normalization = false)
    if evaluate_on == "Joined"
        if segment == "QRSSTT"
            dfResR = subset(df, [:Reference, :Segment] => (ref, seg) -> ((ref .== reference_of) .& (seg .!= "FULL")), view = true)
            dfResg = groupby(dfResR, [:ID, :Group, :Source])
            dfRes = combine(dfResg, :KLEvalues => ((x) -> [vcat(x...)]) => :KLEvalues, :KLEvalues_norm => ((x) -> [vcat(x...)]) => :KLEvalues_norm)
        else
            dfRes = subset(df, [:Reference, :Segment] => (ref, seg) -> ((ref .== reference_of) .& (seg .== segment)), view = true)
        end
    else
        if segment == "QRSSTT"
            dfResR = subset(df, [:Reference, :Segment, :Source] => (ref, seg, src) -> ((ref .== reference_of) .& (seg .!= "FULL") .& (src .== evaluate_on)), view = true)
            dfResg = groupby(dfResR, [:ID, :Group, :Source])
            dfRes = combine(dfResg, :KLEvalues => ((x) -> [vcat(x...)]) => :KLEvalues, :KLEvalues_norm => ((x) -> [vcat(x...)]) => :KLEvalues_norm)
        else
            dfRes = subset(df, [:Reference, :Source, :Segment] => (ref, src, seg) -> ((ref .== reference_of) .& (src .== evaluate_on) .& (seg .== segment)), view = true)
        end
    end


    if reference_of == "SystemII"
        reference = reference_SystemII
    elseif reference_of == "SystemI"
        reference = reference_SystemI
    else
        reference = reference_Joined
    end

    if normalization
        getX = (dfx) -> hcat(dfx.KLEvalues_norm...)
    else
        getX = (dfx) -> hcat(dfx.KLEvalues...)
    end

    function estfun(inds; β = 0.99)
        C(X, β) = β .* cov(X) + (1 - β)I
        # @show size(dfRes)
        train = view(dfRes, inds, :)
        dfMyo = subset(train, :Group => g -> g .== "Myo", view = true)
        dfNorm = subset(train, :Group => g -> g .== "Norm", view = true)
        Xn = getX(dfNorm)
        Xp = getX(dfMyo)
        # @show size(Xp)
        Cp = C(Xp', β)
        Cn = C(Xn', β)
        μp = vec(mean(Xp, dims = 2))
        μn = vec(mean(Xn, dims = 2))
        LDA = ldacov(Cp, Cn, μp, μn)
        return LDA
    end

    dfEval = DataFrame(Sensitivity = [], Specificity = [], Accuracy = [], ErrorRate = [], Source = String[])
    function evalfun(model, inds)
        test = view(dfRes, inds, :)
        X = getX(test)
        res = predict(model, X)
        allCorrect = res .== (test.Group .== "Myo")
        SystemIIInds = test.Source .== "SystemII"
        SystemIInds = .!SystemIIInds
        push!(dfEval, (statMetrics((test.Group .== "Myo"), allCorrect)..., "ALL"))
        push!(dfEval, (statMetrics((test.Group.=="Myo")[SystemIIInds], allCorrect[SystemIIInds])..., "SystemII"))
        push!(dfEval, (statMetrics((test.Group.=="Myo")[SystemIInds], allCorrect[SystemIInds])..., "SystemI"))
        # return DataFrame(Correct = correct, Sens = Sens, Spec = Spec)
        return 1.0
    end
    # scores = cross_validate(estfun, evalfun, nrow(dfRes), Kfold(nrow(dfRes), 5))
    # scores = cross_validate(estfun, evalfun, nrow(dfRes), StratifiedKfold(dfRes.Group, 5))
    scores = cross_validate(estfun, evalfun, nrow(dfRes), RandomSub(nrow(dfRes), round(Int, 0.9nrow(dfRes)), 200))
    # scores = cross_validate(estfun, evalfun, nrow(dfRes), LOOCV(nrow(dfRes)))
    
    if evaluate_on == "Joined"
        df0 = describe(filter(:Source => s -> s == "ALL", dfEval), :mean, :std, :min, :max, :nmissing)#[1:4, 1:6]
        insertcols!(df0, 2, :Source => "ALL")
        df1 = describe(filter(:Source => s -> s == "SystemII", dfEval), :mean, :std, :min, :max, :nmissing)#[1:4, 1:6]
        insertcols!(df1, 2, :Source => "SystemII")
        df2 = describe(filter(:Source => s -> s == "SystemI", dfEval), :mean, :std, :min, :max, :nmissing)#[1:4, 1:6]
        insertcols!(df2, 2, :Source => "SystemI")
        return vcat(df0, df1, df2)
    else
        filter!(:Source => s -> s == "ALL", dfEval)
        # return describe(dfEval)[:, 1:5]
        return describe(dfEval, :mean, :std, :min, :max, :nmissing)
    end
end

# Optional: use different kind of Entropy / normalize HR
function statMetrics(correct_classified, prediction)
    TP = sum(prediction .* correct_classified)
    FP = sum(.!prediction .* correct_classified)
    TN = sum(prediction .* .!correct_classified)
    FN = sum(.!prediction .* .!correct_classified)
    # @info TP, TN, FP, FN 
    N = (TP + TN + FN + FP)
    if N == 0
        return (missing, missing, missing, missing)
    end
    Sens = FN == 0 ? (TP == 0 ? missing : 1) : TP / (TP + FN)
    Spec = FP == 0 ? (TN == 0 ? missing : 1) : TN / (TN + FP)
    # Acc = 0.5(Sens + Spec) #(TN == 0 & TP == 0) ? 1 : ((TP + TN) / N)
    Acc = (TP + TN) / N
    Err = (FP + FN) / N

    return (Sens, Spec, Acc, Err)
end


getMKGLength(mkg::MKG) = sum(sum(mkg.map, dims = 1) .!= 0)

