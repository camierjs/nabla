-- Copyright 2016 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
import "regent"
local c = regentlib.c

sqrt = terralib.intrinsic("llvm.sqrt.f64", double -> double)

-- Import max/min for Terra
max = regentlib.fmax
min = regentlib.fmin

terra abs(a : double) : double
  if a < 0 then
    return -a
  else
    return a
  end
end

terra vec2.metamethods.__add(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x + b.x, y = a.y + b.y }
end

terra vec2.metamethods.__sub(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x - b.x, y = a.y - b.y }
end

vec2.metamethods.__mul = terralib.overloadedfunction(
  "__mul", {
    terra(a : double, b : vec2) : vec2
      return vec2 { x = a * b.x, y = a * b.y }
    end,
    terra(a : vec2, b : double) : vec2
      return vec2 { x = a.x * b, y = a.y * b }
    end
})

terra dot(a : vec2, b : vec2) : double
  return a.x*b.x + a.y*b.y
end

terra cross(a : vec2, b : vec2) : double
  return a.x*b.y - a.y*b.x
end

terra length(a : vec2) : double
  return sqrt(dot(a, a))
end

terra rotateCCW(a : vec2) : vec2
  return vec2 { x = -a.y, y = a.x }
end

terra project(a : vec2, b : vec2)
  return a - b*dot(a, b)
end
