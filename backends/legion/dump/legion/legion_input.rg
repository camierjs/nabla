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
local cmath = terralib.includec("math.h")

-- HW values, pennant: 24/17/34
-- HW pnnnt: 24/18/39
terra read_input(runtime : c.legion_runtime_t,
                 ctx : c.legion_context_t,
                 rz_physical : c.legion_physical_region_t[24],
                 rz_fields : c.legion_field_id_t[24],
                 rp_physical : c.legion_physical_region_t[18],
                 rp_fields : c.legion_field_id_t[18],
                 rs_physical : c.legion_physical_region_t[39],
                 rs_fields : c.legion_field_id_t[39],
                 conf : config)
  c.printf("\n[33m[read_input][m");

  var color_words : c.size_t = cmath.ceil(conf.npieces/64.0)

  -- Allocate buffers for the mesh generator
  var pointpos_x_size : c.size_t = conf.np
  var pointpos_y_size : c.size_t = conf.np
  var pointcolors_size : c.size_t = conf.np
  var pointmcolors_size : c.size_t = conf.np * color_words
  var pointspancolors_size : c.size_t = conf.np
  var zonestart_size : c.size_t = conf.nz
  var zonesize_size : c.size_t = conf.nz
  var zonepoints_size : c.size_t = conf.nz * conf.maxznump
  var zonecolors_size : c.size_t = conf.nz
  var zonespancolors_size : c.size_t = conf.nz

  var pointpos_x : &double = [&double](c.malloc(pointpos_x_size*sizeof(double)))
  var pointpos_y : &double = [&double](c.malloc(pointpos_y_size*sizeof(double)))
  var pointcolors : &int64 = [&int64](c.malloc(pointcolors_size*sizeof(int64)))
  var pointmcolors : &uint64 = [&uint64](c.malloc(pointmcolors_size*sizeof(uint64)))
  var pointspancolors : &int64 = [&int64](c.malloc(pointspancolors_size*sizeof(int64)))
  var zonestart : &int64 = [&int64](c.malloc(zonestart_size*sizeof(int64)))
  var zonesize : &int64 = [&int64](c.malloc(zonesize_size*sizeof(int64)))
  var zonepoints : &int64 = [&int64](c.malloc(zonepoints_size*sizeof(int64)))
  var zonecolors : &int64 = [&int64](c.malloc(zonecolors_size*sizeof(int64)))
  var zonespancolors : &int64 = [&int64](c.malloc(zonespancolors_size*sizeof(int64)))

  regentlib.assert(pointpos_x ~= nil, "pointpos_x nil")
  regentlib.assert(pointpos_y ~= nil, "pointpos_y nil")
  regentlib.assert(pointcolors ~= nil, "pointcolors nil")
  regentlib.assert(pointmcolors ~= nil, "pointmcolors nil")
  regentlib.assert(pointspancolors ~= nil, "pointspancolors nil")
  regentlib.assert(zonestart ~= nil, "zonestart nil")
  regentlib.assert(zonesize ~= nil, "zonesize nil")
  regentlib.assert(zonepoints ~= nil, "zonepoints nil")
  regentlib.assert(zonecolors ~= nil, "zonecolors nil")
  regentlib.assert(zonespancolors ~= nil, "zonespancolors nil")

  var nspans_zones : int64 = 0
  var nspans_points : int64 = 0

  -- Call the mesh generator
  cpennant.generate_mesh_raw(
    conf.np,
    conf.nz,
    conf.nzx,
    conf.nzy,
    conf.lenx,
    conf.leny,
    conf.numpcx,
    conf.numpcy,
    conf.npieces,
    conf.meshtype,
    conf.compact,
    conf.stripsize,
    conf.spansize,
    pointpos_x, &pointpos_x_size,
    pointpos_y, &pointpos_y_size,
    pointcolors, &pointcolors_size,
    pointmcolors, &pointmcolors_size,
    pointspancolors, &pointspancolors_size,
    zonestart, &zonestart_size,
    zonesize, &zonesize_size,
    zonepoints, &zonepoints_size,
    zonecolors, &zonecolors_size,
    zonespancolors, &zonespancolors_size,
    &nspans_zones,
    &nspans_points)

  -- Write mesh data into regions
  do
    var rz_znump = c.legion_physical_region_get_field_accessor_array(
      rz_physical[23], rz_fields[23])

    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      regentlib.assert(zonesize[i] < 255, "zone has more than 255 sides")
      @[&uint8](c.legion_accessor_array_ref(rz_znump, p)) = uint8(zonesize[i])
    end

    c.legion_accessor_array_destroy(rz_znump)
  end

  do
    var rp_px_x = c.legion_physical_region_get_field_accessor_array(
      rp_physical[0], rp_fields[0])
    var rp_px_y = c.legion_physical_region_get_field_accessor_array(
      rp_physical[1], rp_fields[1])
    var rp_has_bcx = c.legion_physical_region_get_field_accessor_array(
      rp_physical[16], rp_fields[16]) -- 15
    var rp_has_bcy = c.legion_physical_region_get_field_accessor_array(
      rp_physical[17], rp_fields[17]) -- 16

    var eps : double = 1e-12
    for i = 0, conf.np do
      var p = c.legion_ptr_t { value = i }
      @[&double](c.legion_accessor_array_ref(rp_px_x, p)) = pointpos_x[i]
      @[&double](c.legion_accessor_array_ref(rp_px_y, p)) = pointpos_y[i]

      @[&bool](c.legion_accessor_array_ref(rp_has_bcx, p)) = (
        (conf.bcx_n > 0 and cmath.fabs(pointpos_x[i] - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(pointpos_x[i] - conf.bcx[1]) < eps))
      @[&bool](c.legion_accessor_array_ref(rp_has_bcy, p)) = (
        (conf.bcy_n > 0 and cmath.fabs(pointpos_y[i] - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(pointpos_y[i] - conf.bcy[1]) < eps))
    end

    c.legion_accessor_array_destroy(rp_px_x)
    c.legion_accessor_array_destroy(rp_px_y)
    c.legion_accessor_array_destroy(rp_has_bcx)
    c.legion_accessor_array_destroy(rp_has_bcy)
  end

  do
    var rs_mapsz = c.legion_physical_region_get_field_accessor_array(
      rs_physical[0], rs_fields[0])
    var rs_mapsp1_ptr = c.legion_physical_region_get_field_accessor_array(
      rs_physical[1], rs_fields[1])
    var rs_mapsp1_index = c.legion_physical_region_get_field_accessor_array(
      rs_physical[2], rs_fields[2])
    var rs_mapsp2_ptr = c.legion_physical_region_get_field_accessor_array(
      rs_physical[3], rs_fields[3])
    var rs_mapsp2_index = c.legion_physical_region_get_field_accessor_array(
      rs_physical[4], rs_fields[4])
    var rs_mapss3 = c.legion_physical_region_get_field_accessor_array(
      rs_physical[5], rs_fields[5])
    var rs_mapss4 = c.legion_physical_region_get_field_accessor_array(
      rs_physical[6], rs_fields[6])

    var sstart = 0
    for iz = 0, conf.nz do
      var zsize = zonesize[iz]
      var zstart = zonestart[iz]
      for izs = 0, zsize do
        var izs3 = (izs + zsize - 1)%%zsize
        var izs4 = (izs + 1)%%zsize

        var p = c.legion_ptr_t { value = sstart + izs }
        @[&c.legion_ptr_t](c.legion_accessor_array_ref(rs_mapsz, p)) = c.legion_ptr_t { value = iz }
        @[&c.legion_ptr_t](c.legion_accessor_array_ref(rs_mapsp1_ptr, p)) = c.legion_ptr_t { value = zonepoints[zstart + izs] }
        @[&uint8](c.legion_accessor_array_ref(rs_mapsp1_index, p)) = 0
        @[&c.legion_ptr_t](c.legion_accessor_array_ref(rs_mapsp2_ptr, p)) = c.legion_ptr_t { value = zonepoints[zstart + izs4] }
        @[&uint8](c.legion_accessor_array_ref(rs_mapsp2_index, p)) = 0
        @[&c.legion_ptr_t](c.legion_accessor_array_ref(rs_mapss3, p)) = c.legion_ptr_t { value = sstart + izs3 }
        @[&c.legion_ptr_t](c.legion_accessor_array_ref(rs_mapss4, p)) = c.legion_ptr_t { value = sstart + izs4 }
      end
      sstart = sstart + zsize
    end

    c.legion_accessor_array_destroy(rs_mapsz)
    c.legion_accessor_array_destroy(rs_mapsp1_ptr)
    c.legion_accessor_array_destroy(rs_mapsp1_index)
    c.legion_accessor_array_destroy(rs_mapsp2_ptr)
    c.legion_accessor_array_destroy(rs_mapsp2_index)
    c.legion_accessor_array_destroy(rs_mapss3)
    c.legion_accessor_array_destroy(rs_mapss4)
  end

  -- Create colorings
  var result : mesh_colorings
  result.rz_all_c = c.legion_coloring_create()
  result.rz_spans_c = c.legion_coloring_create()
  result.rp_all_c = c.legion_coloring_create()
  result.rp_all_private_c = c.legion_coloring_create()
  result.rp_all_ghost_c = c.legion_coloring_create()
  result.rp_all_shared_c = c.legion_coloring_create()
  result.rp_spans_c = c.legion_coloring_create()
  result.rs_all_c = c.legion_coloring_create()
  result.rs_spans_c = c.legion_coloring_create()
  result.nspans_zones = nspans_zones
  result.nspans_points = nspans_points

  compute_coloring(
    conf.npieces, conf.nz, result.rz_all_c, zonecolors, nil, nil)
  compute_coloring(
    nspans_zones, conf.nz, result.rz_spans_c, zonespancolors, nil, nil)

  compute_coloring(
    2, conf.np, result.rp_all_c, pointcolors, nil, filter_ismulticolor)
  compute_coloring(
    conf.npieces, conf.np, result.rp_all_private_c, pointcolors, nil, nil)

  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rp_all_ghost_c, i)
  end
  for i = 0, conf.np do
    for color = 0, conf.npieces do
      var word = i + color/64
      var bit = color %% 64

      if (pointmcolors[word] and (1 << bit)) ~= 0 then
        c.legion_coloring_add_point(
          result.rp_all_ghost_c,
          color,
          c.legion_ptr_t { value = i })
      end
    end
  end

  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rp_all_shared_c, i)
  end
  for i = 0, conf.np do
    var done = false
    for color = 0, conf.npieces do
      var word = i + color/64
      var bit = color %% 64

      if not  done and (pointmcolors[word] and (1 << bit)) ~= 0 then
        c.legion_coloring_add_point(
          result.rp_all_shared_c,
          color,
          c.legion_ptr_t { value = i })
        done = true
      end
    end
  end

  compute_coloring(
    nspans_points, conf.np, result.rp_spans_c, pointspancolors, nil, nil)

  compute_coloring(
    conf.npieces, conf.nz, result.rs_all_c, zonecolors, zonesize, nil)
  compute_coloring(
    nspans_zones, conf.nz, result.rs_spans_c, zonespancolors, zonesize, nil)

  -- Free buffers
  c.free(pointpos_x)
  c.free(pointpos_y)
  c.free(pointcolors)
  c.free(pointmcolors)
  c.free(pointspancolors)
  c.free(zonestart)
  c.free(zonesize)
  c.free(zonepoints)
  c.free(zonecolors)
  c.free(zonespancolors)

  return result
end
     

c.printf("\n[33m[read_input:compile()][m");
read_input:compile()
-- c.printf("[33;1m[read_input:sequential()] SKIPPING read_input!![m\n");

-- ##########################
-- ## read_input_sequential
-- ##########################     
task read_input_sequential(rz_all : region(zone),
                           rp_all : region(point),
                           rs_all : region(side(wild, wild, wild, wild)),
                           conf : config)
where reads writes(rz_all, rp_all, rs_all) do
    -- c.printf("[33;1m[NO read_input][m\n");
    return read_input(
      __runtime(), __context(),
      __physical(rz_all), __fields(rz_all),
      __physical(rp_all), __fields(rp_all),
      __physical(rs_all), __fields(rs_all),
      conf)
end
