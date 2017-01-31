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
-- ## test
-- ##########################     
task test()
  c.printf("\n[31mRunning test (t=%%.1f):[m\n", c.legion_get_current_time_in_micros()/1.e6)

  var conf : config = read_config()

  var rz_all = region(ispace(ptr, conf.nz), zone)
  var rp_all = region(ispace(ptr, conf.np), point)
  var rs_all = region(ispace(ptr, conf.ns), side(wild, wild, wild, wild))

  new(ptr(zone, rz_all), conf.nz)
  new(ptr(point, rp_all), conf.np)
  new(ptr(side(wild, wild, wild, wild), rs_all), conf.ns)

  var colorings : mesh_colorings

  regentlib.assert(conf.seq_init or conf.par_init,
                   "enable one of sequential or parallel initialization")

  if conf.seq_init then
    -- Hack: This had better run on the same node...
    -- c.printf("\n[31;1mNO unwrap(read_input_sequential[m\n")
    colorings = unwrap(read_input_sequential(rz_all, rp_all, rs_all, conf))
  end

  if conf.par_init then
    if conf.seq_init then
      c.legion_coloring_destroy(colorings.rz_all_c)
      c.legion_coloring_destroy(colorings.rz_spans_c)
      c.legion_coloring_destroy(colorings.rp_all_c)
      c.legion_coloring_destroy(colorings.rp_all_private_c)
      c.legion_coloring_destroy(colorings.rp_all_ghost_c)
      c.legion_coloring_destroy(colorings.rp_all_shared_c)
      c.legion_coloring_destroy(colorings.rp_spans_c)
      c.legion_coloring_destroy(colorings.rs_all_c)
      c.legion_coloring_destroy(colorings.rs_spans_c)
    end
    colorings = read_partitions(conf)
  end

  -- Partition zones into disjoint pieces.
  var rz_all_p = partition(disjoint, rz_all, colorings.rz_all_c)

  -- Partition points into private and ghost regions.
  var rp_all_p = partition(disjoint, rp_all, colorings.rp_all_c)
  var rp_all_private = rp_all_p[0]
  var rp_all_ghost = rp_all_p[1]

  -- Partition private points into disjoint pieces by zone.
  var rp_all_private_p = partition(
    disjoint, rp_all_private, colorings.rp_all_private_c)

  -- Partition ghost points into aliased pieces by zone.
  var rp_all_ghost_p = partition(
    aliased, rp_all_ghost, colorings.rp_all_ghost_c)

  -- Partition ghost points into disjoint pieces, breaking ties
  -- between zones so that each point goes into one region only.
  var rp_all_shared_p = partition(
    disjoint, rp_all_ghost, colorings.rp_all_shared_c)

  -- Partition sides into disjoint pieces by zone.
  var rs_all_p = partition(disjoint, rs_all, colorings.rs_all_c)

  if conf.par_init then
    __demand(__parallel)
    for i = 0, conf.npieces do
      initialize_topology(conf, i, rz_all_p[i],
                          rp_all_private_p[i],
                          rp_all_shared_p[i],
                          rp_all_ghost_p[i],
                          rs_all_p[i])
    end
  end

  c.printf("[31mInitializing (t=%%.1f)...[m\n", c.legion_get_current_time_in_micros()/1.e6)
  initialize(rz_all, rz_all_p,
             rp_all,
             rp_all_private, rp_all_private_p,
             rp_all_ghost, rp_all_ghost_p, rp_all_shared_p,
             rs_all, rs_all_p,
             conf)
  -- Hack: Force main task to wait for initialization to finish.
  do
    var _ = 0
    for i = 0, conf.npieces do
      _ += dummy(rz_all_p[i])
    end
    wait_for(_)
  end

  c.printf("[31mStarting simulation (t=%%.1f)...[m\n", c.legion_get_current_time_in_micros()/1.e6)
  var start_time = c.legion_get_current_time_in_micros()/1.e6
  simulate(rz_all, rz_all_p,
           rp_all,
           rp_all_private, rp_all_private_p,
           rp_all_ghost, rp_all_ghost_p, rp_all_shared_p,
           rs_all, rs_all_p,
           conf)
  -- Hack: Force main task to wait for simulation to finish.
  do
    var _ = 0
    for i = 0, conf.npieces do
      _ += dummy(rz_all_p[i])
    end
    wait_for(_)
  end
  var stop_time = c.legion_get_current_time_in_micros()/1.e6
  c.printf("[31mElapsed time = %%.6e[m\n", stop_time - start_time)

  if conf.seq_init then
    c.printf("[31mWarning: Skipping sequential validation[m\n")
    --validate_output_sequential(rz_all, rp_all, rs_all, conf)
  else
    c.printf("[31mWarning: Skipping sequential validation[m\n")
  end

  -- write_output(conf, rz_all, rp_all, rs_all)
end

task toplevel()
  c.printf("\n[32m[toplevel][m");
  test()
end
    
c.printf("\n[32m[cpennant.register_mappers][m");
cpennant.register_mappers()
        
c.printf("\n[32m[regentlib.start][m");
regentlib.start(toplevel)
