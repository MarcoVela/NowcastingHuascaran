using AWSS3
using DelimitedFiles
using Dates

function get_products(satellite)
    config = AWSS3.AWSConfig(nothing, "us-east-1", "")
    q = Dict{String, String}()
    q["delimiter"] = "/"
    q["prefix"] = ""
    objects = AWSS3.S3.list_objects_v2("noaa-$satellite", q; aws_config=config)
    prefixes = get(objects, "CommonPrefixes", [])
    [x.vals[1][begin:end-1] for x in prefixes]
end

function getfilekeys(bucket, q::Dict{String, String}; region)
    config = AWSS3.AWSConfig(nothing, region, "")
    AWSS3.S3.list_objects_v2(bucket, q; aws_config=config)
end

function getfilekeys(bucket, prefix; region="us-east-1")
    q = Dict{String, String}()
    q["prefix"] = prefix
    keys = Vector{String}()
    flag::Bool = true
    while flag
        result = getfilekeys(bucket, q; region)
        @assert haskey(result, "Contents")
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

function show_products()
    satellites = ["goes16" "goes17"]
    writedlm(stdout, satellites, "\t\t")
    println("-"^30)
    products = hcat(get_products.(satellites)...)
    writedlm(stdout, products)
end


function download_hour(satellite, product, date = now(UTC); path=tempdir())
    prefix = "$product/" * Dates.format(date, "yyyy/$(lpad(dayofyear(date), 3, "0"))/HH/")
    file_keys = getfilekeys(satellite, prefix)
    Dict(file_keys .=> [download("https://noaa-$satellite.s3.amazonaws.com/$key", joinpath(path, basename(key))) for key in file_keys])
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

function download_satellite_data(satellite, product, start, finish::DateTime = endofday(start); path=tempdir(), cb::Function=(_...) -> nothing)
    result = Dict{String, String}()
    # Download yearly data
    while year(start) < year(finish)
        partialresult = download_satellite_data(satellite, product, start, endofyear(start); path, cb)
        merge!(result, partialresult)
        start = startofnextyear(start)
    end

    # Download dayly data
    while dayofyear(start) < dayofyear(finish)
        partialresult = download_satellite_data(satellite, product, start, endofday(start); path, cb)
        merge!(result, partialresult)
        start = startofnextday(start)
    end

    # Download from the same day but different times
    partialresult = download_satellite_data(satellite, product, start, Time(finish); path, cb)
    merge!(result, partialresult)

    return result
end



function download_satellite_data(satellite, product, start, finish::Time; path=tempdir(), cb::Function=(_...) -> nothing)
    prefix = "$product/" * Dates.format(start, "yyyy/$(lpad(dayofyear(start), 3, "0"))/")
    file_keys = getfilekeys("noaa-$satellite", prefix)
    outputdict = Dict{String, String}()

    finishtimestring = Dates.format(finish, "HHMMSSsss")[begin:end-2]
    starttimestring = Dates.format(start, "HHMMSSsss")[begin:end-2]
    filter!(key -> begin
        last_delim_index = findlast('/', key)
        fname = key[last_delim_index+1:end]
        _, _, _, startofscan, _, _ = split(fname, '_')
        if starttimestring <= startofscan[9:end] <= finishtimestring
            save_path = joinpath(path, basename(key))
            !isfile(save_path)
        else
            false
        end
    end, file_keys)

    mappedres = asyncmap(key -> begin
        last_delim_index = findlast('/', key)
        fname = key[last_delim_index+1:end]
        _, _, _, startofscan, _, _ = split(fname, '_')
        save_path = joinpath(path, basename(key))
        downloadedpath = download("https://noaa-$satellite.s3.amazonaws.com/$key", save_path)
        cb(save_path)
        (save_path, downloadedpath)
    end, file_keys; ntasks=10)
    for (saved, downloaded) in mappedres
        outputdict[saved] = downloaded
    end
    outputdict
end



