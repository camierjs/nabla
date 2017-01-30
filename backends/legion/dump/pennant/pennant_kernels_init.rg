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

-- #####################################
-- ## Initialization
-- #################


task init_mesh_zones(rz : region(zone))
where
  writes(rz.{zx, zarea, zvol})
do
  c.printf("\t[32m[init_mesh_zones][m\n");
  for z in rz do
    z.zx = vec2 {x = 0.0, y = 0.0}
    z.zarea = 0.0
    z.zvol = 0.0
  end
end


task init_side_fracs(rz : region(zone), rpp : region(point), rpg : region(point),
                     rs : region(side(rz, rpp, rpg, rs)))
where
  reads(rz.zarea, rs.{mapsz, sarea}),
  writes(rs.smf)
do
  c.printf("\t[32m[init_side_fracs][m\n");
  for s in rs do
    var z = s.mapsz
    s.smf = s.sarea / z.zarea
  end
end


task init_hydro(rz : region(zone), rinit : double, einit : double,
                rinitsub : double, einitsub : double,
                subregion_x0 : double, subregion_x1 : double,
                subregion_y0 : double, subregion_y1 : double)
where
  reads(rz.{zx, zvol}),
  writes(rz.{zr, ze, zwrate, zm, zetot})
do
  c.printf("\t[32m[init_hydro][m\n");
  for z in rz do
    var zr = rinit
    var ze = einit

    var eps = 1e-12
    if z.zx.x > subregion_x0 - eps and
      z.zx.x < subregion_x1 + eps and
      z.zx.y > subregion_y0 - eps and
      z.zx.y < subregion_y1 + eps
    then
      zr = rinitsub
      ze = einitsub
    end

    var zm = zr * z.zvol

    z.zr = zr
    z.ze = ze
    z.zwrate = 0.0
    z.zm = zm
    z.zetot = ze * zm
  end
end


task init_radial_velocity(rp : region(point), vel : double)
where
  reads(rp.px),
  writes(rp.pu)
do
  c.printf("\t[32m[init_radial_velocity][m\n");
  for p in rp do
    if vel == 0.0 then
      p.pu = {x = 0.0, y = 0.0}
    else
      var pmag = length(p.px)
      p.pu = (vel / pmag)*p.px
    end
  end
end
