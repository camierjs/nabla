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
local c = regentlib.c
local cmath = terralib.includec("math.h")
local cstring = terralib.includec("string.h")

MESH_PIE = 0
MESH_RECT = 1
MESH_HEX = 2

--
-- Configuration
--

terra get_mesh_config(conf : &config)
  -- Calculate mesh size (nzx, nzy) and dimensions (lenx, leny).
  --c.printf("\n[33m[get_mesh_config] Calculate mesh size (nzx, nzy) and dimensions (lenx, leny)[m");
  conf.nzx = conf.meshparams[0]
  if conf.meshparams_n >= 2 then
    conf.nzy = conf.meshparams[1]
  else
    conf.nzy = conf.nzx
  end
  if conf.meshtype ~= MESH_PIE then
    if conf.meshparams_n >= 3 then
      conf.lenx = conf.meshparams[2]
    else
      conf.lenx = 1.0
    end
  else
    -- convention:  x = theta, y = r
    if conf.meshparams_n >= 3 then
      conf.lenx = conf.meshparams[2] * cmath.M_PI / 180.0
    else
      conf.lenx = 90.0 * cmath.M_PI / 180.0
    end
  end
  if conf.meshparams_n >= 4 then
    conf.leny = conf.meshparams[3]
  else
    conf.leny = 1.0
  end

  if conf.nzx <= 0 or conf.nzy <= 0 or conf.lenx <= 0. or conf.leny <= 0. then
    c.printf("Error: meshparams values must be positive\n")
    c.abort()
  end
  if conf.meshtype == MESH_PIE and conf.lenx >= 2. * cmath.M_PI then
    c.printf("Error: meshparams theta must be < 360\n")
    c.abort()
  end

  -- Calculate numbers of mesh elements (nz, np and ns).
  conf.nz = conf.nzx * conf.nzy
  conf.np = (conf.nzx + 1) * (conf.nzy + 1)
  if conf.meshtype ~= MESH_HEX then
    conf.maxznump = 4
  else
    conf.maxznump = 6
  end
  conf.ns = conf.nz * conf.maxznump
  c.printf("\n");
end

 terra get_submesh_config(conf : &config)
  -- Calculate numbers of submeshes.
   --c.printf("\n[33m[get_submesh_config] Calculate numbers of submeshes.[m");
  var nx : double, ny : double = conf.nzx, conf.nzy
  var swapflag = nx > ny
  if swapflag then nx, ny = ny, nx end
  var n = sqrt(conf.npieces * nx / ny)
  var n1 : int64 = max(cmath.floor(n + 1e-12), 1)
  while conf.npieces %% n1 ~= 0 do n1 = n1 - 1 end
  var n2 : int64 = cmath.ceil(n - 1e-12)
  while conf.npieces %% n2 ~= 0 do n2 = n2 + 1 end
  var longside1 = max(nx / n1, ny / (conf.npieces/n1))
  var longside2 = max(nx / n2, ny / (conf.npieces/n2))
  if longside1 <= longside2 then
    conf.numpcx = n1
  else
    conf.numpcx = n2
  end
  conf.numpcy = conf.npieces / conf.numpcx
  if swapflag then conf.numpcx, conf.numpcy = conf.numpcy, conf.numpcx end
end
                     

--c.printf("\n[33mmax_items=1024[m");
do
local max_items = 1024
local max_item_len = 1024
local fixed_string = int8[max_item_len]

function get_type_specifier(t, as_input)
  --c.printf("[33m[get_type_specifier][m\n");
  if t == fixed_string then
    if as_input then
      return "%%" .. max_item_len .. "s", 1
    else
      return "%%s"
    end
  elseif t == int64 then
    return "%%lld", 1
  elseif t == bool then
    return "%%d", 1
  elseif t == double then
    if as_input then
      return "%%lf", 1
    else
      return "%%.2e", 1
    end
  elseif t:isarray() then
    local elt_type_specifier = get_type_specifier(t.type, as_input)
    local type_specifier = ""
    for i = 1, t.N do
      if i > 1 then
        type_specifier = type_specifier .. " "
      end
      type_specifier = type_specifier .. elt_type_specifier
    end
    return type_specifier, t.N
  else
    assert(false)
  end
end

local function explode_array(t, value)
  if t == fixed_string then
    return terralib.newlist({`@value})
  elseif t:isarray() then
    local values = terralib.newlist()
    for i = 0, t.N - 1 do
      values:insert(`&((@value)[i]))
    end
    return values
  else
    return terralib.newlist({value})
  end
end

local extract = terralib.memoize(function(t)
 local str_specifier = get_type_specifier(fixed_string, true)
  local type_specifier, size = get_type_specifier(t, true)

  return terra(items : &(fixed_string), nitems : int64, key : rawstring, result : &t)
    var item_key : fixed_string
    for i = 0, nitems do
      var matched = c.sscanf(&(items[i][0]), [str_specifier], item_key)
      if matched >= 1 and cstring.strncmp(key, item_key, max_item_len) == 0 then
        var matched = c.sscanf(
          &(items[i][0]), [str_specifier .. " " .. type_specifier],
          item_key, [explode_array(t, result)])
        if matched >= 1 then
          return matched - 1
        end
      end
    end
  end
end)

terra read_config()
  --c.printf("\n[33m[read_config][m");
  var input_filename = get_positional_arg()
  if input_filename == nil then
    c.printf("Usage: ./pennant <filename>\n")
    c.abort()
  end

  c.printf("Reading \"%%s\"...\n", input_filename)
  var input_file = c.fopen(input_filename, "r")
  if input_file == nil then
    c.printf("Error: Failed to open \"%%s\"\n", input_filename)
    c.abort()
  end

  var items : fixed_string[max_items]

  var nitems = 0
  for i = 0, max_items do
    if c.fgets(items[i], max_item_len, input_file) == nil then
      nitems = i + 1
      break
    end
  end

  if c.fclose(input_file) ~= 0 then
    c.printf("Error: Failed to close \"%%s\"\n", input_filename)
    c.abort()
  end

  var conf : config

  -- Set defaults.
  [config_fields_all:map(function(field)
       return quote conf.[field.field] = [field.default_value] end
     end)]

  -- Read parameters from command line.
  var npieces = get_optional_arg("-npieces")
  if npieces ~= nil then
    conf.npieces = c.atoll(npieces)
  end
  if conf.npieces <= 0 then
    c.printf("Error: npieces (%%lld) must be >= 0\n", conf.npieces)
    c.abort()
  end

  var numpcx = get_optional_arg("-numpcx")
  if numpcx ~= nil then
    conf.numpcx = c.atoll(numpcx)
  end
  var numpcy = get_optional_arg("-numpcy")
  if numpcy ~= nil then
    conf.numpcy = c.atoll(numpcy)
  end
  if (conf.numpcx > 0 or conf.numpcy > 0) and
    conf.numpcx * conf.numpcy ~= conf.npieces
  then
    c.printf("Error: numpcx (%%lld) * numpcy (%%lld) must be == npieces (%%lld)\n",
             conf.numpcx, conf.numpcy, conf.npieces)
    c.abort()
  end

  var par_init = get_optional_arg("-par_init")
  if par_init ~= nil then
    conf.par_init = [bool](c.atoll(par_init))
  end

  var seq_init = get_optional_arg("-seq_init")
  if seq_init ~= nil then
    conf.seq_init = [bool](c.atoll(seq_init))
  end

  var warmup = get_optional_arg("-warmup")
  if warmup ~= nil then
    conf.warmup = [bool](c.atoll(warmup))
  end

  var compact = get_optional_arg("-compact")
  if compact ~= nil then
    conf.compact = [bool](c.atoll(compact))
  end

  var internal = get_optional_arg("-internal")
  if internal ~= nil then
    conf.internal = [bool](c.atoll(internal))
  end

  var interior = get_optional_arg("-interior")
  if interior ~= nil then
    conf.interior = [bool](c.atoll(interior))
  end

  var stripsize = get_optional_arg("-stripsize")
  if stripsize ~= nil then
    conf.stripsize = c.atoll(stripsize)
  end

  var spansize = get_optional_arg("-spansize")
  if spansize ~= nil then
    conf.spansize = c.atoll(spansize)
  end

  var print_ts = get_optional_arg("-print_ts")
  if print_ts ~= nil then
    conf.print_ts = [bool](c.atoll(print_ts))
  end

  -- Read parameters from input file.
  [config_fields_input:map(function(field)
       if field.is_linked_field then
         return quote end
       else
         if field.linked_field then
           return quote
             conf.[field.linked_field] = [extract(field.type)](items, nitems, field.field, &(conf.[field.field]))
           end
         else
           return quote
             [extract(field.type)](items, nitems, field.field, &(conf.[field.field]))
           end
         end
       end
     end)]

  -- Configure and run mesh generator.
  --c.printf("\n[33m[read_config] Configure and run mesh generator.[m");
  var meshtype : fixed_string
  if [extract(fixed_string)](items, nitems, "meshtype", &meshtype) < 1 then
    c.printf("Error: Missing meshtype\n")
    c.abort()
  end
  if cstring.strncmp(meshtype, "pie", max_item_len) == 0 then
    conf.meshtype = MESH_PIE
  elseif cstring.strncmp(meshtype, "rect", max_item_len) == 0 then
    conf.meshtype = MESH_RECT
  elseif cstring.strncmp(meshtype, "hex", max_item_len) == 0 then
    conf.meshtype = MESH_HEX
  else
    c.printf("Error: Invalid meshtype \"%%s\"\n", meshtype)
    c.abort()
  end

  c.printf("Config meshtype = \"%%s\"\n", meshtype)

  get_mesh_config(&conf)
  if conf.numpcx <= 0 or conf.numpcy <= 0 then
    get_submesh_config(&conf)
  end

  [config_fields_all:map(function(field)
       return quote c.printf(
         ["Config " .. field.field .. " = " .. get_type_specifier(field.type, false) .. "\n"],
         [explode_array(field.type, `&(conf.[field.field])):map(
            function(value)
              return `@value
            end)])
       end
     end)]

  -- report mesh size in bytes
  --c.printf("\n[33m[read_config] report mesh size in bytes.[m");
  do
    var zone_size = terralib.sizeof(zone)
    var point_size = terralib.sizeof(point)
    var side_size = [ terralib.sizeof(side(wild,wild,wild,wild)) ]
    c.printf("Mesh memory usage:\n")
    c.printf("  Zones  : %%9lld * %%4d bytes = %%11lld bytes\n", conf.nz, zone_size, conf.nz * zone_size)
    c.printf("  Points : %%9lld * %%4d bytes = %%11lld bytes\n", conf.np, point_size, conf.np * point_size)
    c.printf("  Sides  : %%9lld * %%4d bytes = %%11lld bytes\n", conf.ns, side_size, conf.ns * side_size)
    var total = ((conf.nz * zone_size) + (conf.np * point_size) + (conf.ns * side_size))
    c.printf("  Total                             %%11lld bytes\n", total)
  end

  return conf
end
end
