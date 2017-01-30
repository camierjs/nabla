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
task simulate(rz_all : region(zone), rz_all_p : partition(disjoint, rz_all),
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
  c.printf("\t[32m[simulate][m\n");
  var alfa = conf.alfa
  var cfl = conf.cfl
  var cflv = conf.cflv
  var cstop = conf.cstop
  var dtfac = conf.dtfac
  var dtinit = conf.dtinit
  var dtmax = conf.dtmax
  var gamma = conf.gamma
  var q1 = conf.q1
  var q2 = conf.q2
  var qgamma = conf.qgamma
  var ssmin = conf.ssmin
  var tstop = conf.tstop
  var uinitradial = conf.uinitradial
  var vfix = {x = 0.0, y = 0.0}

  var enable = conf.enable

  var interval = 10
  var start_time = c.legion_get_current_time_in_micros()/1.e6
  var last_time = start_time

  var time = 0.0
  var cycle : int64 = 0
  var dt = dtmax
  var dthydro = dtmax
  while continue_simulation(cycle, cstop, time, tstop) do
    c.legion_runtime_begin_trace(__runtime(), __context(), 0)

    __demand(__parallel)
    for i = 0, conf.npieces do
      init_step_points(rp_all_private_p[i], enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      init_step_points(rp_all_shared_p[i], enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      init_step_zones(rz_all_p[i], enable)
    end

    dt = calc_global_dt(dt, dtfac, dtinit, dtmax, dthydro, time, tstop, cycle)

    if cycle > 0 and cycle %% interval == 0 then
      var current_time = c.legion_get_current_time_in_micros()/1.e6
      c.printf("cycle %%4ld    sim time %%.3e    dt %%.3e    time %%.3e (per iteration) %%.3e (total)\n",
               cycle, time, dt, (current_time - last_time)/interval, current_time - start_time)
      last_time = current_time
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_half(rp_all_private_p[i], dt, enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_half(rp_all_shared_p[i], dt, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_centers(rz_all_p[i],
                   rp_all_private_p[i],
                   rp_all_ghost_p[i],
                   rs_all_p[i],
                   enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_volumes(rz_all_p[i],
                   rp_all_private_p[i],
                   rp_all_ghost_p[i],
                   rs_all_p[i],
                   enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_char_len(rz_all_p[i],
                    rp_all_private_p[i],
                    rp_all_ghost_p[i],
                    rs_all_p[i],
                    enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_rho_half(rz_all_p[i], enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      sum_point_mass(rz_all_p[i],
                     rp_all_private_p[i],
                     rp_all_ghost_p[i],
                     rs_all_p[i],
                     enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_state_at_half(rz_all_p[i], gamma, ssmin, dt, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_force_pgas_tts(rz_all_p[i],
                          rp_all_private_p[i],
                          rp_all_ghost_p[i],
                          rs_all_p[i],
                          alfa, ssmin,
                          enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_zone_center_velocity(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_corner_divergence(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_qcn_force(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        gamma, q1, q2,
        enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_force(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_vel_diff(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        q1, q2,
        enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      sum_point_force(rz_all_p[i],
                      rp_all_private_p[i],
                      rp_all_ghost_p[i],
                      rs_all_p[i],
                      enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      apply_boundary_conditions(rp_all_private_p[i], enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      apply_boundary_conditions(rp_all_shared_p[i], enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_full(rp_all_private_p[i], dt, enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_full(rp_all_shared_p[i], dt, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_centers_full(rz_all_p[i],
                        rp_all_private_p[i],
                        rp_all_ghost_p[i],
                        rs_all_p[i],
                        enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_volumes_full(rz_all_p[i],
                        rp_all_private_p[i],
                        rp_all_ghost_p[i],
                        rs_all_p[i],
                        enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_work(rz_all_p[i],
                rp_all_private_p[i],
                rp_all_ghost_p[i],
                rs_all_p[i],
                dt, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_work_rate_energy_rho_full(rz_all_p[i], dt, enable)
    end

    dthydro = dtmax
    __demand(__parallel)
    for i = 0, conf.npieces do
      dthydro min= calc_dt_hydro(rz_all_p[i], dt, dtmax, cfl, cflv, enable)
    end

    cycle += 1
    time += dt

    c.legion_runtime_end_trace(__runtime(), __context(), 0)
  end
end
