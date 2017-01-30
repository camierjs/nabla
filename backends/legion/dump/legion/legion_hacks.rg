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

-- Hack: This exists to make the compiler recompute the bitmasks for
-- each pointer. This needs to happen here (rather than at
-- initialization time) because we subverted the type system in the
-- construction of the mesh pieces.
task init_pointers(rz : region(zone), rpp : region(point), rpg : region(point),
                   rs : region(side(rz, rpp, rpg, rs)))
where
  reads writes(rs.{mapsp1, mapsp2})
do
  c.printf("\t[32m[init_pointers][m\n");
  for s in rs do
    s.mapsp1 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp1)
    regentlib.assert(not isnull(s.mapsp1), "dynamic_cast failed on mapsp1")
    s.mapsp2 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp2)
    regentlib.assert(not isnull(s.mapsp2), "dynamic_cast failed on mapsp2")
  end
end
