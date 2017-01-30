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

c.printf("[33m[pennant_common] Configuration variables.[m\n");
config_fields_input = terralib.newlist({
  -- Configuration variables.
  {field = "alfa", type = double, default_value = 0.5},
  {field = "bcx", type = double[2], default_value = `array(0.0, 0.0), linked_field = "bcx_n"},
  {field = "bcx_n", type = int64, default_value = 0, is_linked_field = true},
  {field = "bcy", type = double[2], default_value = `array(0.0, 0.0), linked_field = "bcy_n"},
  {field = "bcy_n", type = int64, default_value = 0, is_linked_field = true},
  {field = "cfl", type = double, default_value = 0.6},
  {field = "cflv", type = double, default_value = 0.1},
  {field = "chunksize", type = int64, default_value = 99999999},
  {field = "cstop", type = int64, default_value = 999999},
  {field = "dtfac", type = double, default_value = 1.2},
  {field = "dtinit", type = double, default_value = 1.0e99},
  {field = "dtmax", type = double, default_value = 1.0e99},
  {field = "dtreport", type = double, default_value = 10},
  {field = "einit", type = double, default_value = 0.0},
  {field = "einitsub", type = double, default_value = 0.0},
  {field = "gamma", type = double, default_value = 5.0 / 3.0},
  {field = "meshscale", type = double, default_value = 1.0},
  {field = "q1", type = double, default_value = 0.0},
  {field = "q2", type = double, default_value = 2.0},
  {field = "qgamma", type = double, default_value = 5.0 / 3.0},
  {field = "rinit", type = double, default_value = 1.0},
  {field = "rinitsub", type = double, default_value = 1.0},
  {field = "ssmin", type = double, default_value = 0.0},
  {field = "subregion", type = double[4], default_value = `arrayof(double, 0, 0, 0, 0), linked_field = "subregion_n"},
  {field = "subregion_n", type = int64, default_value = 0, is_linked_field = true},
  {field = "tstop", type = double, default_value = 1.0e99},
  {field = "uinitradial", type = double, default_value = 0.0},
  {field = "meshparams", type = double[4], default_value = `arrayof(double, 0, 0, 0, 0), linked_field = "meshparams_n"},
  {field = "meshparams_n", type = int64, default_value = 0, is_linked_field = true},
})

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

c.printf("[33m[pennant_common] Mesh variables[m\n");
config_fields_mesh = terralib.newlist({
  -- Mesh variables.
  {field = "nz", type = int64, default_value = 0},
  {field = "np", type = int64, default_value = 0},
  {field = "ns", type = int64, default_value = 0},
  {field = "maxznump", type = int64, default_value = 0},
})

c.printf("[33m[pennant_common] Command-line parameters[m\n");
config_fields_cmd = terralib.newlist({
  -- Command-line parameters.
  {field = "npieces", type = int64, default_value = 1},
  {field = "par_init", type = bool, default_value = true},
  {field = "seq_init", type = bool, default_value = false},
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


c.printf("[33m[pennant_common] config_fields_all[m\n");
config_fields_all = terralib.newlist()
config_fields_all:insertall(config_fields_input)
config_fields_all:insertall(config_fields_meshgen)
config_fields_all:insertall(config_fields_mesh)
config_fields_all:insertall(config_fields_cmd)

c.printf("[33m[pennant_common] config[m\n");
config = terralib.types.newstruct("config")
config.entries:insertall(config_fields_all)

c.printf("[33m[pennant_common] fspace zone[m\n");
fspace zone {
  zxp :    vec2,         -- zone center coordinates, middle of cycle
  zx :     vec2,         -- zone center coordinates, end of cycle
  zareap : double,       -- zone area, middle of cycle
  zarea :  double,       -- zone area, end of cycle
  zvol0 :  double,       -- zone volume, start of cycle
  zvolp :  double,       -- zone volume, middle of cycle
  zvol :   double,       -- zone volume, end of cycle
  zdl :    double,       -- zone characteristic length
  zm :     double,       -- zone mass
  zrp :    double,       -- zone density, middle of cycle
  zr :     double,       -- zone density, end of cycle
  ze :     double,       -- zone specific energy
  zetot :  double,       -- zone total energy
  zw :     double,       -- zone work
  zwrate : double,       -- zone work rate
  zp :     double,       -- zone pressure
  zss :    double,       -- zone sound speed
  zdu :    double,       -- zone delta velocity (???)

  -- Temporaries for QCS
  zuc :    vec2,         -- zone center velocity
  z0tmp :  double,       -- temporary for qcs_vel_diff

  -- Placed at end to avoid messing up alignment
  znump :  uint8,        -- number of points in zone
}

c.printf("[33m[pennant_common] fspace point[m\n");
fspace point {
  px0 : vec2,            -- point coordinates, start of cycle
  pxp : vec2,            -- point coordinates, middle of cycle
  px :  vec2,            -- point coordinates, end of cycle
  pu0 : vec2,            -- point velocity, start of cycle
  pu :  vec2,            -- point velocity, end of cycle
  pap : vec2,            -- point acceleration, middle of cycle -- FIXME: dead
  pf :  vec2,            -- point force
  pmaswt : double,       -- point mass

  -- Used for computing boundary conditions
  has_bcx : bool,
  has_bcy : bool,
}

c.printf("[33m[pennant_common] fspace side[m\n");
fspace side(rz : region(zone),
            rpp : region(point),
            rpg : region(point),
            rs : region(side(rz, rpp, rpg, rs))) {
  mapsz :  ptr(zone, rz),                      -- maps: side -> zone
  mapsp1 : ptr(point, rpp, rpg),               -- maps: side -> points 1 and 2
  mapsp2 : ptr(point, rpp, rpg),
  mapss3 : ptr(side(rz, rpp, rpg, rs), rs),    -- maps: side -> previous side
  mapss4 : ptr(side(rz, rpp, rpg, rs), rs),    -- maps: side -> next side

  sareap : double,       -- side area, middle of cycle
  sarea :  double,       -- side area, end of cycle
  svolp :  double,       -- side volume, middle of cycle -- FIXME: dead field
  svol :   double,       -- side volume, end of cycle    -- FIXME: dead field
  ssurfp : vec2,         -- side surface vector, middle of cycle -- FIXME: dead
  smf :    double,       -- side mass fraction
  sfp :    vec2,         -- side force, pgas
  sft :    vec2,         -- side force, tts
  sfq :    vec2,         -- side force, qcs

  -- In addition to storing their own state, sides also store the
  -- state of edges and corners. This can be done because there is a
  -- 1-1 correspondence between sides and edges/corners. Technically,
  -- edges can be shared between zones, but the computations on edges
  -- are minimal, and are not actually used for sharing information,
  -- so duplicating computations on edges is inexpensive.

  -- Edge variables
  exp :    vec2,         -- edge center coordinates, middle of cycle
  ex :     vec2,         -- edge center coordinates, end of cycle
  elen :   double,       -- edge length, end of cycle

  -- Corner variables (temporaries for QCS)
  carea :  double,       -- corner area
  cevol :  double,       -- corner evol
  cdu :    double,       -- corner delta velocity
  cdiv :   double,       -- ??????????
  ccos :   double,       -- corner cosine
  cqe1 :   vec2,         -- ??????????
  cqe2 :   vec2,         -- ??????????
}

c.printf("[33m[pennant_common] span[m\n");
fspace span {
  start : int64,
  stop  : int64,
  internal : bool,
}
