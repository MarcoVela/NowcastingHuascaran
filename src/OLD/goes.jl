using AWSS3
using DelimitedFiles
using Dates
using ProgressMeter
using Distributed

"""
    get_products(satellite::String) -> Vector{String}

Returns the product names of the satellite
"""
function get_products(satellite)::Vector{String}
    config = AWSS3.AWSConfig(nothing, "us-east-1", "")
    q = Dict{String, String}()
    q["delimiter"] = "/"
    q["prefix"] = ""
    objects = AWSS3.S3.list_objects_v2("noaa-$satellite", q; aws_config=config)
    prefixes::Vector = objects["CommonPrefixes"]
    [x.vals[1][begin:end-1] for x in prefixes]
end

"""
    getfilekeys(bucket::String, q::Dict{String,String};region::String)

Returns up to 1000 objects from a bucket in the specified region, q is a dict where additional parameters can be set.
### Keyword arguments

* `region` String representing the region of the bucket

### Example
    q = Dict("prefix" => "")
    response = getfilekeys("mybucket", q; region="us-east-1")

This returns up to 1000 objects from the root directory of mybucket
"""
function getfilekeys(bucket, q::Dict{String, String}; region)
    config = AWSS3.AWSConfig(nothing, region, "")
    AWSS3.S3.list_objects_v2(bucket, q; aws_config=config)
end

"""
    getfilekeys(bucket::String, prefix::String;region::String) -> Vector{String}

Returns all the keys of the objects from a bucket in the specified region under the prefix specified.
### Keyword arguments

* `region` String representing the region of the bucket

### Example

    response = getfilekeys("mybucket", "mydir/"; region="us-east-1")

This returns the keys of the objects of mybucket whose keys start with "mydir/".
"""
function getfilekeys(bucket, prefix; region="us-east-1")
    q = Dict{String, String}()
    q["prefix"] = prefix
    keys = Vector{String}()
    flag::Bool = true
    while flag
        result = getfilekeys(bucket, q; region)
        !haskey(result, "Contents") && break
        sizehint!(keys, length(keys) + parse(Int, result["KeyCount"]))
        for obj in result["Contents"]
            push!(keys, obj["Key"])
        end
        flag = haskey(result, "NextContinuationToken")
        pop!(q, "continuation-token", "")
        if flag
            q["continuation-token"] = result["NextContinuationToken"]
        end
    end
    keys
end


function show_products(io=stdout)
    satellites = ["goes16","goes17"]
    cols = [[x, "---", get_products(x)...] for x in satellites]
    products = hcat(cols...)
    writedlm(io, products)
end



function endofday(date)
    DateTime(year(date), month(date), day(date), 23, 59, 59, 999)
end

function endofyear(date)
    DateTime(year(date), 12, 31, 23, 59, 59, 999)
end

function startofday(date)
    DateTime(year(date), month(date), day(date))
end

function startofyear(date)
    DateTime(year(date))
end


function startofnextday(date)
    startofday(date + Day(1))
end

function startofnextyear(date)
    startofyear(date + Year(1))
end


function list_satellite_data_keys(satellite, product, start, finish::DateTime)
    data_keys = Vector{String}()
    # Get yearly keys
    while year(start) < year(finish)
        append!(data_keys, list_satellite_data_keys(satellite, product, start, endofyear(start)))
        start = startofnextyear(start)
    end
    # Get daily keys
    @sync while dayofyear(start) < dayofyear(finish)
        @async append!(data_keys, list_satellite_data_keys(satellite, product, $start, endofday($start)))
        start = startofnextday(start)
    end
    append!(data_keys, list_satellite_data_keys(satellite, product, start, Time(finish)))
end

function list_satellite_data_keys(satellite, product, start, finish::Time)
    prefix = "$product/" * Dates.format(start, "yyyy/$(lpad(dayofyear(start), 3, "0"))/")
    file_keys = getfilekeys("noaa-$satellite", prefix)
    finishtimestring = Dates.format(finish, "HHMMSSsss")[begin:end-2]
    starttimestring = Dates.format(start, "HHMMSSsss")[begin:end-2]

    filter!(key -> begin
        _, _, _, startofscan, _, _ = split(key[findlast('/', key)+1:end], '_')
        starttimestring <= startofscan[9:end] <= finishtimestring
    end, file_keys)
end

function notin(arr1::Vector{T}, arr2::Vector{T})::Vector{Int64} where T
    i = 1
    j = 1
    res = Vector(1:length(arr1))

    while (i <= length(arr1)) && (j <= length(arr2))
        if arr1[i] == arr2[j]
            res[i] = 0
            i += 1
        elseif arr1[i] > arr2[j]
            j += 1
        else
            i += 1
        end
    end
    filter!(!=(0), res)
end


"""
    download_satellite_data([satellite = "goes16"], product::AbstractString, start::DateTime, finish = endofday(start); path=tempdir(), show_progress=true, ntasks=20) -> Dict{String,String}

Downloads files of the given NOAA satellite product between start and finish dates into the specified path and returns a dictionary consisting of local path and key pairs.

### Keyword arguments

* `path` String setting the path where files will be downloaded
* `show_progress` Whether to show a progress bar or not
* `ntasks` number of concurrent tasks for download, increasing this number could result in a "too many files open" error

### Example

    start = now(UTC) - Day(7)
    finish = now(UTC)
    files = download_satellite_data("goes16", "GLM-L2-LCFA", start, finish)

This downloads the last 7 days files of the product GLM-L2-LCFA from the GOES 16 satellite, storing them in a temporary directory.
"""
function download_satellite_data(satellite::AbstractString, product::AbstractString, start::DateTime, finish::DateTime = endofday(start); path=tempdir(), show_progress=true, ntasks=20)
    keys = list_satellite_data_keys(satellite, product, start, finish)
    first_slash_idx = findfirst('/', first(keys))
    paths = unique(dirname.(joinpath.(path, [k[first_slash_idx+1:end] for k in keys])))
    for p in paths
        if !ispath(p)
            mkpath(p)
        end
    end
    files_in_path = sort!(basename.(vcat(readdir.(paths)...)))
    indices = notin(sort!(basename.(keys)), files_in_path)
    keys = keys[indices]
    #filter!(key -> !(basename(key) ∈ files_in_path), keys)
    length(keys) == 0 && return Dict{String,String}()
    replace_path = "https://noaa-$satellite.s3.amazonaws.com/$(first(keys)[begin:first_slash_idx])"
    paths = download_many(["https://noaa-$satellite.s3.amazonaws.com/$key" for key in keys], path, Val(show_progress); ntasks=ntasks, replace_path=replace_path)
    paths
end

download_satellite_data(product::AbstractString, start::DateTime, args...; kwargs...) = download_satellite_data("goes16", product, start, args...; kwargs...)

@inline function download_many(urls, path, ::Val{false}; ntasks, replace_path="")::Vector{String}
    asyncmap(url -> download(url, joinpath(path, replace(url, replace_path => ""))), urls; ntasks=ntasks)
end

@inline function download_many(urls, path, ::Val{true}; ntasks, replace_path="")::Vector{String}
    length(urls) == 0 && return String[]
    p = Progress(length(urls); desc="Downloading files: ", barglyphs=BarGlyphs('|','█', ['▁' ,'▃' ,'▅' ,'▆', '▇'],' ','|',), showspeed=true)
    channel = RemoteChannel(()->Channel{String}(length(urls)), 1)
    fetch(@sync begin
        @async while let 
            filename = take!(channel); 
            filename != "" && next!(p; showvalues = [(:current_file, filename)]); 
            filename != "" 
            end
        end

        @async begin 
            results = asyncmap(url -> begin
                #save_path = joinpath(path, basename(url))
                save_path = joinpath(path, replace(url, replace_path => ""))
                downloadedpath = download(url, save_path)
                put!(channel, downloadedpath)
                downloadedpath
            end, urls; ntasks=ntasks) 
            put!(channel, "")
            results
        end
    end)
end





