using CSV, DataFrames

export RVData, CompositeRVData, get_times, get_instnames, get_rvs, get_rverrs, read_rvs, read_radvel_file, get_view, get_λs

"""
Stores the times, RVs, and RV errors for a single spectrograph.

# Fields
- `t::Vector{Float64}` The times of the observations in BJD.
- `rv::Vector{Float64}` The RVs in m/s.
- `rverr::Vector{Float64}` The RV errors in m/s.
- `instname::Union{String, Nothing}` The name of the spectrograph.
- `λ::Union{Int, Nothing}` The wavelength of this dataset which may be used in any modeling.
Construct an RVData object.
"""
mutable struct RVData
    t::Vector{Float64}
    rv::Vector{Float64}
    rverr::Vector{Float64}
    instname::Union{String, Nothing}
    λ::Union{Int, Nothing}
    """
        RVData(t::Vector{Float64}, rv::Vector{Float64}, rverr::Vector{Float64}; instname=nothing, λ=nothing)
    Construct an RVData object. If using the `CompositeRVData` interface, `instname` will be set automatically.
    """
    function RVData(t::Vector{Float64}, rv::Vector{Float64}, rverr::Vector{Float64}; instname::Union{String, Nothing}=nothing, λ::Union{Int, Nothing}=nothing)
        return new(t, rv, rverr, instname, λ)
    end
end

"""
Wrapper dictionary-like type to store RVs from multiple spectrographs.
"""
struct CompositeRVData
    dict::Dict{String, RVData}
    indices::Dict{String, Vector{Int}}
end

"""
    CompositeRVData()
Construct an empty CompositeRVData object.
"""
CompositeRVData() = CompositeRVData(Dict{String, RVData}(), Dict{String, Vector{Int}}())


Base.length(data::CompositeRVData) = length(data.dict)
Base.keys(data::CompositeRVData) = keys(data.dict)
Base.values(data::CompositeRVData) = values(data.dict)
Base.getindex(data::CompositeRVData, k) = getindex(data.dict, k)
function Base.setindex!(data::CompositeRVData, v, k)
    v.instname = k
    setindex!(data.dict, v, k)
    update_indices!(data)
end
function Base.delete!(data::CompositeRVData, k)
    delete!(data.dict, k)
    update_indices!(data)
end
Base.iterate(data::CompositeRVData) = iterate(data.dict)

function num_rvs(data::CompositeRVData)
    n = 0
    for d ∈ values(data)
        n += length(d.t)
    end
    return n
end

"""
    get_times(data::CompositeRVData; do_sort=true)
Generates a single vector of times of all observations, sorted in time if `do_sort=true`.
"""
function get_times(data::CompositeRVData; do_sort=true)
    t = Float64[]
    for d ∈ values(data)
        t = vcat(t, d.t)
    end
    if do_sort
        sort!(t)
    end
    return t
end

"""
    get_instnames(data::CompositeRVData; do_sort=true)
Generates a single vector of instrument names for all observations, sorted in time if `do_sort=true`.
"""
function get_instnames(data::CompositeRVData; do_sort=true)
    instnames = String[]
    for instname ∈ keys(data)
        instnames = vcat(instnames, fill(instname, length(data[instname].t)))
    end
    if do_sort
        t = get_times(data, do_sort=false)
        ss = sortperm(t)
        instnames .= instnames[ss]
    end
    return instnames
end

"""
    get_λs(data::CompositeRVData; do_sort=true)
Generates a single vector of wavelengths of all observations, sorted in time if `do_sort=true`.
"""
function get_λs(data::CompositeRVData; do_sort=true)
    λs = Int[]
    for instname ∈ keys(data)
        λs = vcat(λs, fill(data[instname].λ, length(data[instname].t)))
    end
    if do_sort
        t = get_times(data, do_sort=false)
        ss = sortperm(t)
        λs .= λs[ss]
    end
    return λs
end


"""
    get_rvs(data::CompositeRVData; do_sort=true)
Generates a single vector of RVs of all observations, sorted in time if `do_sort=true`.
"""
function get_rvs(data::CompositeRVData; do_sort=true)
    rvs = Float64[]
    for d ∈ values(data)
        rvs = vcat(rvs, d.rv)
    end
    if do_sort
        t = get_times(data, do_sort=false)
        ss = sortperm(t)
        rvs .= rvs[ss]
    end
    return rvs
end

"""
    get_rverrs(data::CompositeRVData; do_sort=true)
Generates a single vector of RV errors of all observations, sorted in time if `do_sort=true`.
"""
function get_rverrs(data::CompositeRVData; do_sort=true)
    rverrs = Float64[]
    for d ∈ values(data)
        rverrs = vcat(rverrs, d.rverr)
    end
    if do_sort
        t = get_times(data, do_sort=false)
        ss = sortperm(t)
        rverrs .= rverrs[ss]
    end
    return rverrs
end


function update_indices!(data::CompositeRVData)
    t = get_times(data)
    instnames = get_instnames(data)
    for d ∈ values(data)
        data.indices[d.instname] = findall(instnames .== d.instname)
    end
end


function get_view(data::CompositeRVData, instnames)
    data_out = CompositeRVData()
    for instname ∈ instnames
        data_out[instname] = data[instname]
    end
    return data_out
end



function read_rvs(fname::String; delim=",", instname=nothing, λ=nothing)
    df = CSV.read(fname, DataFrame, delim=delim)
    t = df.time
    rvs = df.mnvel
    rverrs = df.errvel
    d = RVData(t, rvs, rverrs, instname=instname, λ=λ)
    return d
end

"""
    read_radvel_file(fname::String, λs=nothing)
Reads a RadVel-like formatted .csv file with columns `time,mnvel,errvel,tel` into a `CompositeRVData` object.
"""
function read_radvel_file(fname::String, λs=nothing)
    df = CSV.read(fname, DataFrame, delim=",")
    tels = df.tel
    tels_unq = unique(df.tel)
    data = CompositeRVData()
    for tel ∈ tels_unq
        inds = findall(tels .== tel)
        if !isnothing(λs)
            data[tel] = RVData(df.time[inds], df.mnvel[inds], df.errvel[inds], instname=string(tel), λ=λs[tel])
        else
            data[tel] = RVData(df.time[inds], df.mnvel[inds], df.errvel[inds], instname=string(tel))
        end
    end
    return data
end

function write_radvel_file(fname::String, t::AbstractVector{<:Real}, rv::AbstractVector{<:Real}, rverr::AbstractVector{<:Real}, tels::AbstractVector{<:AbstractString})
    ss = sortperm(t)
    t .= t[ss]
    rv .= rv[ss]
    rverr .= rverr[ss]
    tels .= tels[ss]
    open(fname, "w") do io
        writedlm(io, [t rv rverr tels], ',')
    end
end

function write_radvel_file(fname::String, tvecs::Tuple, rvvecs::Tuple, rverrvecs::Tuple, telnames::Vector{String})
    t = Float64[]
    rv = Float64[]
    rverr = Float64[]
    tels = String[]
    for i=1:length(tvecs)
        t = vcat(t, tvecs[i])
        rv = vcat(rv, rvvecs[i])
        rverr = vcat(rverr, rverrvecs[i])
        tels = vcat(tels, fill(telnames[i], length(tvecs[i])))
    end
    ss = sortperm(t)
    t .= t[ss]
    rv .= rv[ss]
    rverr .= rverr[ss]
    tels .= tels[ss]
    open(fname, "w") do io
        writedlm(io, [t rv rverr tels], ',')
    end
end

function write_radvel_file(data::CompositeRVData, fname)
    t = get_times(data)
    rvs = get_times(data)
    rverrs = get_times(data)
    tels = get_instnames(data)
    open(fname, "w") do io
        writedlm(io, [t rvs rverrs tels])
    end
end