using NCDatasets
import Serialization
using ProgressMeter
using Dates

struct Flash
    latitude::Vector{Float32}
    longitude::Vector{Float32}
    quality::Vector{Int16}
    energy::Vector{Int16}
    area::Vector{Int16}
    time_start::DateTime
    time_end::DateTime
    fname::String
end

function dataset_to_flash(ds)
    Flash(
        ds["flash_lat"].var[:],
        ds["flash_lon"].var[:],
        ds["flash_quality_flag"].var[:],
        ds["flash_energy"].var[:],
        ds["flash_area"].var[:],
        DateTime(ds.attrib["time_coverage_start"][begin:end-1], dateformat"yyyy-mm-ddTHH:MM:SS.sss"),
        DateTime(ds.attrib["time_coverage_end"][begin:end-1], dateformat"yyyy-mm-ddTHH:MM:SS.sss"),
        basename(NCDatasets.path(ds))
    )
end

function readdirall(path)
    names = String[]
    for (root, _, files) in walkdir(path)
      append!(names, joinpath.(root, files))
    end
    names
  end
  
function read_folder_consolidate(indir)
    fs = readdirall(indir)
    sort!(fs)
    datasets = Vector{Flash}(undef, length(fs))
    for i = 1:length(fs)
        ds = NCDataset(fs[i])
        datasets[i] = dataset_to_flash(ds)
        close(ds)
    end
    datasets
end


function main(indir, outdir)
    dirs = readdir(indir; join=true)
    @showprogress for dir in dirs
        datasets = read_folder_consolidate(dir)
        outpath = joinpath(outdir, basename(dir) * ".jls")
        Serialization.serialize(outpath, datasets)
    end
end
