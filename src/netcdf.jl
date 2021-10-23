using NetCDF
using Dates

struct GLM_L2_LCFA
    flash_id::Vector{Int16}
    flash_energy::Vector{Int16}
    flash_lat::Vector{Float32}
    flash_lon::Vector{Float32}
    flash_quality_flag::Vector{Int16}
    flash_count::Int32

    group_id::Vector{Int32}
    group_energy::Vector{Int16}
    group_lat::Vector{Float32}
    group_lon::Vector{Float32}
    group_quality_flag::Vector{Int16}
    group_parent_flash_id::Vector{Int16}
    group_count::Int32

    product_time::DateTime
end


function readtostruct(ncfile::NcFile)::GLM_L2_LCFA
    k = (Symbol.(Base.keys(ncfile.vars))...,)
    v = (NetCDF.readvar.(Base.values(ncfile.vars))...,)
    t = NamedTuple{k}(v)
    GLM_L2_LCFA(t.flash_id, t.flash_energy, t.flash_lat, t.flash_lon, t.flash_quality_flag, t.flash_count[],
                t.group_id, t.group_energy, t.group_lat, t.group_lon, t.group_quality_flag, t.group_parent_flash_id, t.group_count[],
                Millisecond(t.product_time[] * 1000) + DateTime(2000, 1, 1, 12))
end


function collect_datasets(directory)
    structs = readtostruct.(NetCDF.open.(joinpath.(directory, readdir(directory))))
    flash_data = (flash_id=Int16[], flash_energy=Int16[], flash_lat=Float32[], flash_lon=Float32[], flash_quality_flag=Int16[])
    group_data = (group_id=Int32[], group_energy=Int16[], group_lat=Float32[], group_lon=Float32[], group_quality_flag=Int16[], group_parent_flash_id=Int16[])
    n_flash = sum(f.flash_count for f in structs)
    n_group = sum(f.group_count for f in structs)
    for k in keys(flash_data)
        v = getfield(flash_data, k)
        sizehint!(v, n_flash)
        for s in structs
            append!(v, getfield(s, k))
        end
    end
    for k in keys(group_data)
        v = getfield(group_data, k)
        sizehint!(v, n_group)
        for s in structs
            append!(v, getfield(s, k))
        end
    end
    (flash_data, group_data)
end
