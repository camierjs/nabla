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

struct vec2 {
  x : double,
  y : double,
}

-- #####################################
-- ## Data Structures
-- #################

config_fields_meshgen = terralib.newlist({
  -- Mesh generator variables.
  {field = "meshtype", type = int64, default_value = 0},
  {field = "nzx", type = int64, default_value = 0},
  {field = "nzy", type = int64, default_value = 0},
  {field = "numpcx", type = int64, default_value = 0},
  {field = "numpcy", type = int64, default_value = 0},
  {field = "lenx", type = double, default_value = 0.0},
  {field = "leny", type = double, default_value = 0.0},
})

c.printf("[33m[legion_data] Mesh variables[m\n");
config_fields_mesh = terralib.newlist({
  -- Mesh variables.
  {field = "nz", type = int64, default_value = 0},
  {field = "np", type = int64, default_value = 0},
  {field = "ns", type = int64, default_value = 0},
  {field = "maxznump", type = int64, default_value = 0},
})

c.printf("[33m[legion_data] Command-line parameters[m\n");
config_fields_cmd = terralib.newlist({
  -- Command-line parameters.
  {field = "npieces", type = int64, default_value = 1},
  {field = "par_init", type = bool, default_value = false},
  {field = "seq_init", type = bool, default_value = true},
  {field = "print_ts", type = bool, default_value = false},
  {field = "enable", type = bool, default_value = true},
  {field = "warmup", type = bool, default_value = false},
  {field = "compact", type = bool, default_value = true},
  {field = "internal", type = bool, default_value = true},
  {field = "interior", type = bool, default_value = true},
  {field = "stripsize", type = int64, default_value = 128},
  {field = "spansize", type = int64, default_value = 2048},
  {field = "nspans_zones", type = int64, default_value = 0},
  {field = "nspans_points", type = int64, default_value = 0},
                                    })

c.printf("[33m[legion_data] config_fields_all[m\n");
config_fields_all = terralib.newlist()
-- done in pennant_data: config_fields_all:insertall(config_fields_input)
config_fields_all:insertall(config_fields_meshgen)
config_fields_all:insertall(config_fields_mesh)
config_fields_all:insertall(config_fields_cmd)

c.printf("[33m[legion_data] span[m\n");
fspace span {
  start : int64,
  stop  : int64,
  internal : bool,
}
