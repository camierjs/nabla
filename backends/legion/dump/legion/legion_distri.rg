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


-- #####################################
-- ## Distributed Mesh Generator
-- #################

terra ptr_t(x : int64)
  return c.legion_ptr_t { value = x }
end

-- Indexing scheme for ghost points:

terra grid_p(conf : config)
  var npx, npy = conf.nzx + 1, conf.nzy + 1
  return npx, npy
end

terra get_num_ghost(conf : config)
  var npx, npy = grid_p(conf)
  var num_ghost = (conf.numpcy - 1) * npx + (conf.numpcx - 1) * npy - (conf.numpcx - 1)*(conf.numpcy - 1)
  return num_ghost
end

terra all_ghost_p(conf : config)
  var num_ghost = get_num_ghost(conf)
  var first_ghost : int64 = 0
  var last_ghost = num_ghost -- exclusive
  return first_ghost, last_ghost
end

terra all_private_p(conf : config)
  var num_ghost = get_num_ghost(conf)
  var first_private = num_ghost
  var last_private = conf.np -- exclusive
  return first_private, last_private
end

terra block_zx(conf : config, pcx : int64)
  var first_zx = conf.nzx * pcx / conf.numpcx
  var last_zx = conf.nzx * (pcx + 1) / conf.numpcx -- exclusive
  var stride_zx = last_zx - first_zx
  return first_zx, last_zx, stride_zx
end

terra block_zy(conf : config, pcy : int64)
  var first_zy = conf.nzy * pcy / conf.numpcy
  var last_zy = conf.nzy * (pcy + 1) / conf.numpcy -- exclusive
  var stride_zy = last_zy - first_zy
  return first_zy, last_zy, stride_zy
end

terra block_z(conf : config, pcx : int64, pcy : int64)
  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)
  var first_z = first_zy * conf.nzx + first_zx * stride_zy
  var last_z = first_z + stride_zy*stride_zx -- exclusive
  return first_z, last_z
end

terra block_px(conf : config, pcx : int64)
  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_px = first_zx - pcx + [int64](pcx ~= 0)
  var last_px = last_zx - pcx + [int64](pcx == conf.numpcx - 1) -- exclusive
  var stride_px = last_px - first_px
  return first_px, last_px, stride_px
end

terra block_py(conf : config, pcy : int64)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)
  var first_py = first_zy - pcy + [int64](pcy ~= 0)
  var last_py = last_zy - pcy + [int64](pcy == conf.numpcy - 1) -- exclusive
  var stride_py = last_py - first_py
  return first_py, last_py, stride_py
end

terra block_p(conf : config, pcx : int64, pcy : int64)
  var npx, npy = grid_p(conf)
  var first_private, last_private = all_private_p(conf)
  var first_py, last_py, stride_py = block_py(conf, pcy)
  var first_px, last_px, stride_px = block_px(conf, pcx)
  var first_p = first_private + first_py * (npx - conf.numpcx + 1) + first_px * stride_py
  var last_p = first_p + stride_py*stride_px -- exclusive
  return first_p, last_p
end

-- Ghost nodes are counted starting at the right face and moving down
-- to the bottom and then bottom-right. This is identical to a point
-- numbering where points are sorted first by number of colors (ghosts
-- first) and then by first color.
terra ghost_first_p(conf : config, pcx : int64, pcy : int64)
  var npx, npy = conf.nzx + 1, conf.nzy + 1

  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

  -- Count previous vertical segments
  var prev_vertical_rows = (conf.numpcx - 1) * (first_zy - pcy + [int64](pcy > 0))
  var prev_vertical_row = pcx * (stride_zy - [int64](pcy > 0 and pcy < conf.numpcy - 1))

  -- Count previous horizontal segments
  var prev_horizontal_rows = pcy * npx
  var prev_horizontal_row = [int64](pcy < conf.numpcy - 1) * (first_zx + [int64](pcx > 0))

  return prev_vertical_rows + prev_vertical_row + prev_horizontal_rows + prev_horizontal_row
end

-- Corners:
terra ghost_bottom_right_p(conf : config, pcx : int64, pcy : int64)
  if pcx < conf.numpcx - 1 and pcy < conf.numpcy - 1 then
    var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
    var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

    var first_p = ghost_first_p(conf, pcx, pcy)
    var corner_p = first_p + stride_zy + stride_zx - [int64](pcy > 0) - [int64](pcx > 0)
    return corner_p, corner_p + 1
  end
  return 0, 0
end

terra ghost_top_left_p(conf : config, pcx : int64, pcy : int64)
  if pcx > 0 and pcy > 0 then
    return ghost_bottom_right_p(conf, pcx-1, pcy-1)
  end
  return 0, 0
end


terra ghost_top_right_p(conf : config, pcx : int64, pcy : int64)
  if pcy > 0 then
    return ghost_bottom_right_p(conf, pcx, pcy-1)
  end
  return 0, 0
end

terra ghost_bottom_left_p(conf : config, pcx : int64, pcy : int64)
  if pcx > 0 then
    return ghost_bottom_right_p(conf, pcx-1, pcy)
  end
  return 0, 0
end

-- Faces:
terra ghost_bottom_p(conf : config, pcx : int64, pcy : int64)
  if pcy < conf.numpcy - 1 then
    var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
    var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

    var first_p = ghost_first_p(conf, pcx, pcy)
    var first_face_p = first_p + [int64](pcx < conf.numpcx - 1) * (stride_zy - [int64](pcy > 0))
    var last_face_p = first_face_p + stride_zx - 1 + [int64](pcx == 0) + [int64](pcx == conf.numpcx - 1) -- exclusive
    return first_face_p, last_face_p
  end
  return 0, 0
end

terra ghost_top_p(conf : config, pcx : int64, pcy : int64)
  if pcy > 0 then
    return ghost_bottom_p(conf, pcx, pcy-1)
  end
  return 0, 0
end

terra ghost_right_p(conf : config, pcx : int64, pcy : int64)
  if pcx < conf.numpcx - 1 then
    var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
    var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

    var first_p = ghost_first_p(conf, pcx, pcy)
    var first_face_p = first_p
    var last_face_p = first_face_p + stride_zy - 1 + [int64](pcy == 0) + [int64](pcy == conf.numpcy - 1) -- exclusive
    return first_face_p, last_face_p
  end
  return 0, 0
end

terra ghost_left_p(conf : config, pcx : int64, pcy : int64)
  if pcx > 0 then
    return ghost_right_p(conf, pcx-1, pcy)
  end
  return 0, 0
end

terra read_partitions(conf : config) : mesh_colorings
  c.printf("[33m[read_partitions][m\n");
  regentlib.assert(conf.npieces > 0, "npieces must be > 0")
  regentlib.assert(conf.compact, "parallel initialization requires compact")
  regentlib.assert(
    conf.meshtype == MESH_RECT,
    "parallel initialization only works on rectangular meshes")
  var znump = 4

  -- Create colorings.
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

  -- Zones and sides: private partitions.
  var max_stride_zx = (conf.nzx + conf.numpcx - 1) / conf.numpcx
  var max_stride_zy = (conf.nzy + conf.numpcy - 1) / conf.numpcy
  var zspansize = conf.spansize/znump
  result.nspans_zones = (max_stride_zx*max_stride_zy + zspansize - 1) / zspansize

  for pcy = 0, conf.numpcy do
    for pcx = 0, conf.numpcx do
      var piece = pcy * conf.numpcx + pcx
      var first_z, last_z = block_z(conf, pcx, pcy)

      c.legion_coloring_add_range(
        result.rz_all_c, piece,
        ptr_t(first_z), ptr_t(last_z - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rs_all_c, piece,
        ptr_t(first_z * znump), ptr_t(last_z * znump - 1)) -- inclusive

      var span = 0
      for z = first_z, last_z, zspansize do
        c.legion_coloring_add_range(
          result.rz_spans_c, span,
          ptr_t(z), ptr_t(min(z + zspansize, last_z) - 1)) -- inclusive
        c.legion_coloring_add_range(
          result.rs_spans_c, span,
          ptr_t(z * znump), ptr_t(min(z + zspansize, last_z) * znump - 1)) -- inclusive
        span = span + 1
      end
      regentlib.assert(span <= result.nspans_zones, "zone span overflow")
    end
  end

  -- Points: top level partition (private vs ghost).
  var first_ghost, last_ghost = all_ghost_p(conf)
  var first_private, last_private = all_private_p(conf)

  var all_private_color, all_ghost_color = 0, 1
  c.legion_coloring_add_range(
    result.rp_all_c, all_ghost_color,
    ptr_t(first_ghost), ptr_t(last_ghost - 1)) -- inclusive
  c.legion_coloring_add_range(
    result.rp_all_c, all_private_color,
    ptr_t(first_private), ptr_t(last_private - 1)) -- inclusive

  -- This math is hard, so just keep track of these as we go along.
  result.nspans_points = 0

  -- Points: private partition.
  for pcy = 0, conf.numpcy do
    for pcx = 0, conf.numpcx do
      var piece = pcy * conf.numpcx + pcx
      var first_p, last_p = block_p(conf, pcx, pcy)

      c.legion_coloring_add_range(
        result.rp_all_private_c, piece,
        ptr_t(first_p), ptr_t(last_p - 1)) -- inclusive

      var span = 0
      for p = first_p, last_p, conf.spansize do
        c.legion_coloring_add_range(
          result.rp_spans_c, span,
          ptr_t(p), ptr_t(min(p + conf.spansize, last_p) - 1)) -- inclusive
        span = span + 1
      end
      result.nspans_points = max(result.nspans_points, span)
    end
  end

  -- Points: ghost and shared partitions.
  for pcy = 0, conf.numpcy do
    for pcx = 0, conf.numpcx do
      var piece = pcy * conf.numpcx + pcx

      var top_left = ghost_top_left_p(conf, pcx, pcy)
      var top_right = ghost_top_right_p(conf, pcx, pcy)
      var bottom_left = ghost_bottom_left_p(conf, pcx, pcy)
      var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)
      var top = ghost_top_p(conf, pcx, pcy)
      var bottom = ghost_bottom_p(conf, pcx, pcy)
      var left = ghost_left_p(conf, pcx, pcy)
      var right = ghost_right_p(conf, pcx, pcy)

      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(top_left._0), ptr_t(top_left._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(top._0), ptr_t(top._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(top_right._0), ptr_t(top_right._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(left._0), ptr_t(left._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(right._0), ptr_t(right._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(bottom_left._0), ptr_t(bottom_left._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(bottom._0), ptr_t(bottom._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(bottom_right._0), ptr_t(bottom_right._1 - 1)) -- inclusive

      c.legion_coloring_add_range(
        result.rp_all_shared_c, piece, ptr_t(right._0), ptr_t(right._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_shared_c, piece, ptr_t(bottom._0), ptr_t(bottom._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_shared_c, piece, ptr_t(bottom_right._0), ptr_t(bottom_right._1 - 1)) -- inclusive

      var first_p : int64, last_p : int64 = -1, -1
      if right._0 < right._1 then
        first_p = right._0
        last_p = right._1
      end
      if bottom._0 < bottom._1 then
        if first_p < 0 then first_p = bottom._0 end
        last_p = bottom._1
      end
      if bottom_right._0 < bottom_right._1 then
        if first_p < 0 then first_p = bottom_right._0 end
        last_p = bottom_right._1
      end

      var span = 0
      for p = first_p, last_p, conf.spansize do
        c.legion_coloring_add_range(
          result.rp_spans_c, span,
          ptr_t(p), ptr_t(min(p + conf.spansize, last_p) - 1)) -- inclusive
        span = span + 1
      end
      result.nspans_points = max(result.nspans_points, span)
    end
  end

  return result
end
c.printf("[33m[read_partitions:compile()][m\n");
read_partitions:compile()

terra get_zone_position(conf : config, pcx : int64, pcy : int64, z : int64)
  c.printf("[33m[get_zone_position][m\n");
  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)
  var first_z, last_z = block_z(conf, pcx, pcy)

  var zstripsize = conf.stripsize
  if zstripsize <= 0 then
    zstripsize = stride_zx
  end

  var z_strip_num = (z - first_z) / (zstripsize * stride_zy)
  var z_strip_elt = (z - first_z) %% (zstripsize * stride_zy)
  var leftover = stride_zx - zstripsize * z_strip_num
  if leftover == 0 then leftover = zstripsize end
  var z_strip_width = min(zstripsize, leftover)
  regentlib.assert(z_strip_width > 0, "zero width strip")
  var z_strip_x = z_strip_elt %% z_strip_width
  var z_strip_y = z_strip_elt / z_strip_width
  var z_x = z_strip_num * zstripsize + z_strip_x
  var z_y = z_strip_y
  return z_x, z_y
end


task initialize_spans(conf : config,
                      piece : int64,
                      rz_spans : region(span),
                      rp_spans_private : region(span),
                      rp_spans_shared : region(span),
                      rs_spans : region(span))
where
  reads writes(rz_spans, rp_spans_private, rp_spans_shared, rs_spans),
  rz_spans * rp_spans_private, rz_spans * rp_spans_shared, rz_spans * rs_spans,
  rp_spans_private * rp_spans_shared, rp_spans_private * rs_spans,
  rp_spans_shared * rs_spans
do
  c.printf("[33m[initialize_spans][m\n");
  -- Unfortunately, this duplicates a lot of functionality in read_partitions.

  regentlib.assert(conf.compact, "parallel initialization requires compact")
  regentlib.assert(
    conf.meshtype == MESH_RECT,
    "parallel initialization only works on rectangular meshes")
  var znump = 4
  var zspansize = conf.spansize/znump

  var pcx, pcy = piece %% conf.numpcx, piece / conf.numpcx

  -- Zones and sides.
  do
    var { first_zx = _0, last_zx = _1, stride_zx = _2 } = block_zx(conf, pcx)
    var { first_zy = _0, last_zy = _1, stride_zy = _2 } = block_zy(conf, pcy)
    var { first_z = _0, last_z = _1} = block_z(conf, pcx, pcy)

    var num_external = 0
    var span_i = 0
    for z = first_z, last_z, zspansize do
      var external = true
      if conf.internal then
        if conf.interior then
          var interior_zx = stride_zx - [int64](pcx > 0) - [int64](pcx < conf.numpcx - 1)
          var interior_zy = stride_zy - [int64](pcy > 0) - [int64](pcy < conf.numpcy - 1)
          var interior = interior_zx * interior_zy
          external = min(z + zspansize, last_z) - first_z > interior
        else
          var { z0_x = _0, z0_y = _1 } = get_zone_position(conf, pcx, pcy, z)
          var { zn_x = _0, zn_y = _1 } = get_zone_position(conf, pcx, pcy, min(z + zspansize, last_z) - 1)

          var external = (zn_y > z0_y and conf.numpcx > 1) or
            (z0_x == 0 and pcx > 0) or (z0_x == stride_zx - 1 and pcx < conf.numpcx - 1) or
            (zn_x == 0 and pcx > 0) or (zn_x == stride_zx - 1 and pcx < conf.numpcx - 1) or
            (z0_y == 0 and pcy > 0) or (z0_y == stride_zy - 1 and pcy < conf.numpcy - 1) or
            (zn_y == 0 and pcy > 0) or (zn_y == stride_zy - 1 and pcy < conf.numpcy - 1)
        end
      end
      num_external += int(external)

      var zs = unsafe_cast(ptr(span, rz_spans), piece * conf.nspans_zones + span_i)
      var ss = unsafe_cast(ptr(span, rs_spans), piece * conf.nspans_zones + span_i)

      var z_span = {
        start = z,
        stop = min(z + zspansize, last_z), -- exclusive
        internal = not external,
      }
      var s_span = {
        start = z * znump,
        stop = min(z + zspansize, last_z) * znump, -- exclusive
        internal = not external,
      }
      if conf.seq_init then regentlib.assert(zs.start == z_span.start and zs.stop == z_span.stop, "bad value: zone span") end
      if conf.seq_init then regentlib.assert(ss.start == s_span.start and ss.stop == s_span.stop, "bad value: side span") end

      @zs = z_span
      @ss = s_span
      span_i = span_i + 1
    end
    regentlib.assert(span_i <= conf.nspans_zones, "zone span overflow")
    c.printf("Spans: total %%ld external %%ld percent external %%f\n",
             span_i, num_external, double(num_external*100)/span_i)
  end

  -- Points: private spans.
  do
    var piece = pcy * conf.numpcx + pcx
    var { first_p = _0, last_p = _1 } = block_p(conf, pcx, pcy)

    var span_i = 0
    for p = first_p, last_p, conf.spansize do
      var ps = unsafe_cast(ptr(span, rp_spans_private), piece * conf.nspans_points + span_i)

      var p_span = { start = p, stop = min(p + conf.spansize, last_p), internal = false } -- exclusive
      if conf.seq_init then regentlib.assert(ps.start == p_span.start and ps.stop == p_span.stop, "bad value: private point span") end

      @ps = p_span

      span_i = span_i + 1
    end
  end

  -- Points: shared spans.
  do
    var piece = pcy * conf.numpcx + pcx

    var right = ghost_right_p(conf, pcx, pcy)
    var bottom = ghost_bottom_p(conf, pcx, pcy)
    var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)

    var first_p, last_p = -1, -1
    if right._0 < right._1 then
      first_p = right._0
      last_p = right._1
    end
    if bottom._0 < bottom._1 then
      if first_p < 0 then first_p = bottom._0 end
      last_p = bottom._1
    end
    if bottom_right._0 < bottom_right._1 then
      if first_p < 0 then first_p = bottom_right._0 end
      last_p = bottom_right._1
    end

    var span_i = 0
    for p = first_p, last_p, conf.spansize do
      var ps = unsafe_cast(ptr(span, rp_spans_shared), piece * conf.nspans_points + span_i)

      var p_span = { start = p, stop = min(p + conf.spansize, last_p), internal = false } -- exclusive
      if conf.seq_init then regentlib.assert(ps.start == p_span.start and ps.stop == p_span.stop, "bad value: private point span") end

      @ps = p_span

      span_i = span_i + 1
    end
  end
end

task initialize_topology(conf : config,
                         piece : int64,
                         rz : region(zone),
                         rpp : region(point),
                         rps : region(point),
                         rpg : region(point),
                         rs : region(side(rz, rpp, rpg, rs)))
where reads writes(rz.znump,
                   rpp.{px, has_bcx, has_bcy},
                   rps.{px, has_bcx, has_bcy},
                   rs.{mapsz, mapsp1, mapsp2, mapss3, mapss4}),
  reads(rpg.{px0}) -- Hack: Work around runtime bug with no-acccess regions.
do
  c.printf("[33m[initialize_topology][m\n");
  regentlib.assert(
    conf.meshtype == MESH_RECT,
    "distributed initialization only works on rectangular meshes")
  var znump = 4

  var pcx, pcy = piece %% conf.numpcx, piece / conf.numpcx

  -- Initialize zones.
  fill(rz.znump, znump)

  -- Initialize points: private.
  var dx = conf.lenx / double(conf.nzx)
  var dy = conf.leny / double(conf.nzy)
  var eps = 1e-12
  do
    var {first_zx = _0, last_zx = _1, stride_zx = _2} = block_zx(conf, pcx)
    var {first_zy = _0, last_zy = _1, stride_zy = _2} = block_zy(conf, pcy)
    var {first_p = _0, last_p = _1} = block_p(conf, pcx, pcy)

    var p_ = first_p
    for y = first_zy + [int64](pcy > 0), last_zy + [int64](pcy == conf.numpcy - 1) do
      for x = first_zx + [int64](pcx > 0), last_zx + [int64](pcx == conf.numpcx - 1) do
        var p = dynamic_cast(ptr(point, rpp), [ptr](p_))
        regentlib.assert(not isnull(p), "bad pointer")

        var px = { x = dx*x, y = dy*y }
        if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
        if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

        var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
          (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
        var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
          (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
        if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
        if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

        p.px = px
        p.has_bcx = has_bcx
        p.has_bcy = has_bcy

        p_ += 1
      end
    end
    regentlib.assert(p_ == last_p, "point underflow")
  end

  -- Initialize points: shared.
  do
    var {first_zx = _0, last_zx = _1, stride_zx = _2} = block_zx(conf, pcx)
    var {first_zy = _0, last_zy = _1, stride_zy = _2} = block_zy(conf, pcy)

    var right = ghost_right_p(conf, pcx, pcy)
    var bottom = ghost_bottom_p(conf, pcx, pcy)
    var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)

    for p_ = right._0, right._1 do
      var p = dynamic_cast(ptr(point, rps), [ptr](p_))
      regentlib.assert(not isnull(p), "bad pointer")

      var x, y = last_zx, first_zy + (p_ - right._0 + [int64](pcy > 0))

      var px = { x = dx*x, y = dy*y }
      if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
      if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

      var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
      var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
      if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
      if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

      p.px = px
      p.has_bcx = has_bcx
      p.has_bcy = has_bcy
    end

    for p_ = bottom._0, bottom._1 do
      var p = dynamic_cast(ptr(point, rps), [ptr](p_))
      regentlib.assert(not isnull(p), "bad pointer")

      var x, y = first_zx + (p_ - bottom._0 + [int64](pcx > 0)), last_zy

      var px = { x = dx*x, y = dy*y }
      if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
      if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

      var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
      var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
      if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
      if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

      p.px = px
      p.has_bcx = has_bcx
      p.has_bcy = has_bcy
    end

    for p_ = bottom_right._0, bottom_right._1 do
      var p = dynamic_cast(ptr(point, rps), [ptr](p_))
      regentlib.assert(not isnull(p), "bad pointer")

      var x, y = last_zx, last_zy

      var px = { x = dx*x, y = dy*y }
      if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
      if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

      var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
      var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
      if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
      if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

      p.px = px
      p.has_bcx = has_bcx
      p.has_bcy = has_bcy
    end
  end

  -- Initialize sides.
  do
    var {first_zx = _0, last_zx = _1, stride_zx = _2} = block_zx(conf, pcx)
    var {first_zy = _0, last_zy = _1, stride_zy = _2} = block_zy(conf, pcy)
    var {first_z = _0, last_z = _1} = block_z(conf, pcx, pcy)
    var {first_p = _0, last_p = _1} = block_p(conf, pcx, pcy)

    var zstripsize = conf.stripsize
    if zstripsize <= 0 then
      zstripsize = stride_zx
    end

    var z_ = first_z
    var passes = 1 + [int64](conf.interior)
    for pass = 0, passes do
      for x0 = 0, stride_zx, conf.stripsize do
        for y = 0, stride_zy do
          for x = x0, min(x0 + conf.stripsize, stride_zx) do
            if not conf.interior or
              (pass == 0) ~=
                 ((y == 0 and pcy > 0) or (x == 0 and pcx > 0) or
                  (y == stride_zy - 1 and pcy < conf.numpcy - 1) or
                  (x == stride_zx - 1 and pcx < conf.numpcx - 1))
            then
              var z = dynamic_cast(ptr(zone, rz), [ptr](z_))
              regentlib.assert(not isnull(z), "bad pointer")

              var top_left = ghost_top_left_p(conf, pcx, pcy)
              var top_right = ghost_top_right_p(conf, pcx, pcy)
              var bottom_left = ghost_bottom_left_p(conf, pcx, pcy)
              var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)
              var top = ghost_top_p(conf, pcx, pcy)
              var bottom = ghost_bottom_p(conf, pcx, pcy)
              var left = ghost_left_p(conf, pcx, pcy)
              var right = ghost_right_p(conf, pcx, pcy)

              var inner_x = x - [int64](pcx > 0)
              var inner_y = y - [int64](pcy > 0)
              var inner_stride_x = stride_zx - 1 + [int64](pcx == 0) + [int64](pcx == conf.numpcx - 1)
              var pp : ptr(point, rpp, rpg)[4]
              do
                var p_ : int64 = -1
                if y == 0 and x == 0 and pcy > 0 and pcx > 0 then
                  p_ = top_left._0
                elseif y == 0 and pcy > 0 then
                  p_ = top._0 + inner_x
                elseif x == 0 and pcx > 0 then
                  p_ = left._0 + inner_y
                else -- private
                  p_ = first_p + inner_y * inner_stride_x + inner_x
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[0] = p
              end
              do
                var p_ : int64 = -1
                if y == 0 and x == stride_zx - 1 and pcy > 0 and pcx < conf.numpcx - 1 then
                  p_ = top_right._0
                elseif y == 0 and pcy > 0 then
                  p_ = top._0 + (inner_x + 1)
                elseif x == stride_zx - 1 and pcx < conf.numpcx - 1 then
                  p_ = right._0 + inner_y
                else -- private
                  p_ = first_p + inner_y * inner_stride_x + (inner_x + 1)
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[1] = p
              end
              do
                var p_ : int64 = -1
                if y == stride_zy - 1 and x == stride_zx - 1 and pcy < conf.numpcy - 1 and pcx < conf.numpcx - 1 then
                  p_ = bottom_right._0
                elseif y == stride_zy - 1 and pcy < conf.numpcy - 1 then
                  p_ = bottom._0 + (inner_x + 1)
                elseif x == stride_zx - 1 and pcx < conf.numpcx - 1 then
                  p_ = right._0 + (inner_y + 1)
                else -- private
                  p_ = first_p + (inner_y + 1) * inner_stride_x + (inner_x + 1)
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[2] = p
              end
              do
                var p_ : int64 = -1
                if y == stride_zy - 1 and x == 0 and pcy < conf.numpcy - 1 and pcx > 0 then
                  p_ = bottom_left._0
                elseif y == stride_zy - 1 and pcy < conf.numpcy - 1 then
                  p_ = bottom._0 + inner_x
                elseif x == 0 and pcx > 0 then
                  p_ = left._0 + (inner_y + 1)
                else -- private
                  p_ = first_p + (inner_y + 1) * inner_stride_x + inner_x
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[3] = p
              end

              var ss : ptr(side(rz, rpp, rpg, rs), rs)[4]
              for i = 0, znump do
                var s_ = z_ * znump + i
                var s = dynamic_cast(ptr(side(rz, rpp, rpg, rs), rs), [ptr](s_))
                regentlib.assert(not isnull(s), "bad pointer")
                ss[i] = s
              end

              for i = 0, znump do
                var prev_i = (i + znump - 1) %% znump
                var next_i = (i + 1) %% znump

                var s = ss[i]
                if conf.seq_init then regentlib.assert(s.mapsz == z, "bad value: mapsz") end
                s.mapsz = z

                if conf.seq_init then regentlib.assert(s.mapsp1 == pp[i], "bad value: mapsp1") end
                if conf.seq_init then regentlib.assert(s.mapsp2 == pp[next_i], "bad value: mapsp2") end
                s.mapsp1 = pp[i]
                s.mapsp2 = pp[next_i]

                if conf.seq_init then regentlib.assert(s.mapss3 == ss[prev_i], "bad value: mapss3") end
                if conf.seq_init then regentlib.assert(s.mapss4 == ss[next_i], "bad value: mapss4") end
                s.mapss3 = ss[prev_i]
                s.mapss4 = ss[next_i]
              end

              z_ += 1
            end
          end
        end
      end
    end
    regentlib.assert(z_ == last_z, "zone underflow")
  end
end
