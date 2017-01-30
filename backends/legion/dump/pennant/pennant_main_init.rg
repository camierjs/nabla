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


-- ##########################
-- ## simulate
-- ##########################     
task initialize(rz_all : region(zone), rz_all_p : partition(disjoint, rz_all),
                rp_all : region(point),
                rp_all_private : region(point),
                rp_all_private_p : partition(disjoint, rp_all_private),
                rp_all_ghost : region(point),
                rp_all_ghost_p : partition(aliased, rp_all_ghost),
                rp_all_shared_p : partition(disjoint, rp_all_ghost),
                rs_all : region(side(wild, wild, wild, wild)),
                rs_all_p : partition(disjoint, rs_all),
                conf : config)
where
  reads writes(rz_all, rp_all_private, rp_all_ghost, rs_all),
  rp_all_private * rp_all_ghost
do
  c.printf("\t[32m[initialize][m\n");
  var einit = conf.einit
  var einitsub = conf.einitsub
  var rinit = conf.rinit
  var rinitsub = conf.rinitsub
  var subregion = conf.subregion
  var uinitradial = conf.uinitradial

  var enable = true

  for i = 0, conf.npieces do
    init_pointers(rz_all_p[i],
                  rp_all_private_p[i],
                  rp_all_ghost_p[i],
                  rs_all_p[i])
  end

  for i = 0, conf.npieces do
    init_mesh_zones(rz_all_p[i])
  end

  for i = 0, conf.npieces do
    calc_centers_full(rz_all_p[i],
                      rp_all_private_p[i],
                      rp_all_ghost_p[i],
                      rs_all_p[i],
                      enable)
  end

  for i = 0, conf.npieces do
    calc_volumes_full(rz_all_p[i],
                      rp_all_private_p[i],
                      rp_all_ghost_p[i],
                      rs_all_p[i],
                      enable)
  end

  for i = 0, conf.npieces do
    init_side_fracs(rz_all_p[i],
                    rp_all_private_p[i],
                    rp_all_ghost_p[i],
                    rs_all_p[i])
  end

  for i = 0, conf.npieces do
    init_hydro(rz_all_p[i],
               rinit, einit, rinitsub, einitsub,
               subregion[0], subregion[1], subregion[2], subregion[3])
  end

  for i = 0, conf.npieces do
    init_radial_velocity(rp_all_private_p[i], uinitradial)
    init_radial_velocity(rp_all_shared_p[i], uinitradial)
  end

  if conf.warmup then
    -- Do one iteration to warm up the runtime.
    var conf_warmup = conf
    conf_warmup.cstop = 1
    conf_warmup.enable = false
    simulate(rz_all, rz_all_p,
             rp_all,
             rp_all_private, rp_all_private_p,
             rp_all_ghost, rp_all_ghost_p, rp_all_shared_p,
             rs_all, rs_all_p,
             conf_warmup)
  end
end

