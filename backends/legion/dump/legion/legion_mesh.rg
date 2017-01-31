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
-- ## Mesh Generator
-- #################

c.printf("\n[33m[mesh_colorings][m");
struct mesh_colorings {
  rz_all_c : c.legion_coloring_t,
  rz_spans_c : c.legion_coloring_t,
  rp_all_c : c.legion_coloring_t,
  rp_all_private_c : c.legion_coloring_t,
  rp_all_ghost_c : c.legion_coloring_t,
  rp_all_shared_c : c.legion_coloring_t,
  rp_spans_c : c.legion_coloring_t,
  rs_all_c : c.legion_coloring_t,
  rs_spans_c : c.legion_coloring_t,
  nspans_zones : int64,
  nspans_points : int64,
}


terra filter_none(i : int64, cs : &int64) return cs[i] end
terra filter_ismulticolor(i : int64, cs : &int64)
  return int64(cs[i] == cpennant.MULTICOLOR)
end


terra compute_coloring(ncolors : int64, nitems : int64,
                       coloring : c.legion_coloring_t,
                       colors : &int64, sizes : &int64,
                       filter : {int64, &int64} -> int64)
  c.printf("\n[33m[compute_coloring][m");
  if filter == nil then
    filter = filter_none
  end

  for i = 0, ncolors do
    c.legion_coloring_ensure_color(coloring, i)
  end
  do
    if nitems > 0 then
      var i_start = 0
      var i_end = 0
      var i_color = filter(0, colors)
      for i = 0, nitems do
        var i_size = 1
        if sizes ~= nil then
          i_size = sizes[i]
        end

        var color = filter(i, colors)
        if i_color ~= color then
          if i_color >= 0 then
            c.legion_coloring_add_range(
              coloring, i_color,
              c.legion_ptr_t { value = i_start },
              c.legion_ptr_t { value = i_end - 1 })
          end
          i_start = i_end
          i_color = color
        end
        i_end = i_end + i_size
      end
      if i_color >= 0 then
        c.legion_coloring_add_range(
          coloring, i_color,
          c.legion_ptr_t { value = i_start },
          c.legion_ptr_t { value = i_end - 1 })
      end
    end
  end
end
