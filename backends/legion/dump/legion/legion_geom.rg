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
