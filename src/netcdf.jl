using NetCDF
using Dates
using Geodesy

struct GLM_L2_LCFA
    flash_id::Vector{Int16}
    flash_energy::Vector{Int16}
    flash_coord::Vector{LLA{Float64}}
    flash_quality_flag::Vector{Int16}
    flash_count::Int32

    group_id::Vector{Int32}
    group_energy::Vector{Int16}
    group_coord::Vector{LLA{Float64}}
    group_quality_flag::Vector{Int16}
    group_parent_flash_id::Vector{Int16}
    group_count::Int32

    product_time::DateTime
end


function readtostruct(ncfile::NcFile)::GLM_L2_LCFA
    initial_date_GOES = DateTime(2000, 1, 1, 12)
    k = (Symbol.(Base.keys(ncfile.vars))...,)
    v = (NetCDF.readvar.(Base.values(ncfile.vars))...,)
    t = NamedTuple{k}(v)
    flash_coords = @. LLA(Float64(t.flash_lat), Float64(t.flash_lon))
    group_coords = @. LLA(Float64(t.group_lat), Float64(t.group_lon))
    GLM_L2_LCFA(t.flash_id, t.flash_energy, flash_coords, t.flash_quality_flag, t.flash_count[],
                t.group_id, t.group_energy, group_coords, t.group_quality_flag, t.group_parent_flash_id, t.group_count[],
                Millisecond(t.product_time[] * 1000) + initial_date_GOES)
end

function collect_tostruct(directory; ntasks=20)
    paths = readdir(directory, join=true)
    asyncmap((readtostruct ∘ NetCDF.open), paths; ntasks=ntasks)
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
