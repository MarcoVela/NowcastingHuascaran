
function readdirall(path)
  names = String[]
  for (root, _, files) in walkdir(path)
    append!(names, joinpath.(root, files))
  end
  names
end

function Base.cumsum(A::AbstractArray, dims::NTuple)
  out = copy(A)
  for i in dims
      cumsum!(out, out; dims=i)
  end
  return out
end



function integralimage(A::Matrix)
  n,m = size(A)
  out = zeros(eltype(A), n+1, m+1)
  out[2:end, 2:end] = A
  cumsum(out, (1,2))
end


function sumgrid(grid, k)
  summed = integralimage(grid)
  w₁, h₁ = size(grid)
  w₂, h₂ = w₁÷k, h₁÷k
  reescaled = zeros(eltype(grid), w₂, h₂)
  for x in 1:w₂, y in 1:h₂
      reescaled[x, y] = summed[begin+x*k,     begin+y*k] + 
                        summed[begin+(x-1)*k, begin+(y-1)*k] - 
                        summed[begin+x*k,     begin+(y-1)*k] - 
                        summed[begin+(x-1)*k, begin+y*k]
  end
  reescaled
end

