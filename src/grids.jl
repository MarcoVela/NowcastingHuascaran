using Geodesy

struct GeoGrid{T, U}
    topleftcorner::UTM{T}
    spacialresolution::Int64
    grid::Matrix{U}
end

#=function accumulate!(grid, point::UTM)
    scale_factor = 1e3 * grid.spacialresolution
    dx::Int64 = (point.x - grid.topleftcorner.x) ÷ scale_factor
    dy::Int64 = (grid.topleftcorner.y - point.y) ÷ scale_factor
    n = size(grid.grid, 1)
    !((0 <= dx < n) && (0 <= dy < n)) && return 0
    grid.grid[dy + 1, dx + 1] += one(eltype(grid.grid))
    return 1
end=#


function accumulate!(grid, points)
    scale_factor = 1e3 * grid.spacialresolution
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    n = size(grid.grid, 1)

    dx = @. Base.unsafe_trunc(Int64, (xs - grid.topleftcorner.x) / scale_factor) + 1
    dy = @. n - Base.unsafe_trunc(Int64, (grid.topleftcorner.y - ys) / scale_factor) + 1

    mask = @. (0 < dx <= n) && (0 < dy <= n)
    dx = dx[mask]
    dy = dy[mask]

    n = length(dx)
    @inbounds for i = 1:n
        grid.grid[dy[i], dx[i]] += one(eltype(grid.grid))
    end
    sum(mask)
end




