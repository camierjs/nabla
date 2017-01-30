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
local cstring = terralib.includec("string.h")

--
-- Validation
--

c.printf("[33m[Validation][m\n");
do
local solution_filename_maxlen = 1024
terra validate_output(runtime : c.legion_runtime_t,
                      ctx : c.legion_context_t,
                      rz_physical : c.legion_physical_region_t[24],
                      rz_fields : c.legion_field_id_t[24],
                      rp_physical : c.legion_physical_region_t[17],
                      rp_fields : c.legion_field_id_t[17],
                      rs_physical : c.legion_physical_region_t[34],
                      rs_fields : c.legion_field_id_t[34],
                      conf : config)
  c.printf("[33m[validate_output][m\n");
  c.printf("Running validate_output (t=%%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6)

  var solution_zr : &double = [&double](c.malloc(conf.nz*sizeof(double)))
  var solution_ze : &double = [&double](c.malloc(conf.nz*sizeof(double)))
  var solution_zp : &double = [&double](c.malloc(conf.nz*sizeof(double)))

  regentlib.assert(solution_zr ~= nil, "solution_zr nil")
  regentlib.assert(solution_ze ~= nil, "solution_ze nil")
  regentlib.assert(solution_zp ~= nil, "solution_zp nil")

  var input_filename = get_positional_arg()
  regentlib.assert(input_filename ~= nil, "input_filename nil")

  var solution_filename : int8[solution_filename_maxlen]
  do
    var sep = cstring.strrchr(input_filename, (".")[0])
    if sep == nil then
      c.printf("Error: Failed to find file extention in \"%%s\"\n", input_filename)
      c.abort()
    end
    var len : int64 = [int64](sep - input_filename)
    regentlib.assert(len + 8 < solution_filename_maxlen, "solution_filename exceeds maximum length")
    cstring.strncpy(solution_filename, input_filename, len)
    cstring.strncpy(solution_filename + len, ".xy.std", 8)
  end

  c.printf("Reading \"%%s\"...\n", solution_filename)
  var solution_file = c.fopen(solution_filename, "r")
  if solution_file == nil then
    c.printf("Warning: Failed to open \"%%s\"\n", solution_filename)
    c.printf("Warning: Skipping validation step\n")
    return
  end

  c.fscanf(solution_file, " # zr")
  for i = 0, conf.nz do
    var iz : int64
    var zr : double
    var count = c.fscanf(
      solution_file,
      [" " .. get_type_specifier(int64, true) .. " " .. get_type_specifier(double, true)],
      &iz, &zr)
    if count ~= 2 then
      c.printf("Error: malformed file, expected 2 and got %%d\n", count)
      c.abort()
    end
    solution_zr[i] = zr
  end

  c.fscanf(solution_file, " # ze")
  for i = 0, conf.nz do
    var iz : int64
    var ze : double
    var count = c.fscanf(
      solution_file,
      [" " .. get_type_specifier(int64, true) .. " " .. get_type_specifier(double, true)],
      &iz, &ze)
    if count ~= 2 then
      c.printf("Error: malformed file, expected 2 and got %%d\n", count)
      c.abort()
    end
    solution_ze[i] = ze
  end

  c.fscanf(solution_file, " # zp")
  for i = 0, conf.nz do
    var iz : int64
    var zp : double
    var count = c.fscanf(
      solution_file,
      [" " .. get_type_specifier(int64, true) .. " " .. get_type_specifier(double, true)],
      &iz, &zp)
    if count ~= 2 then
      c.printf("Error: malformed file, expected 2 and got %%d\n", count)
      c.abort()
    end
    solution_zp[i] = zp
  end

  var absolute_eps = 1.0e-8
  var absolute_eps_text = get_optional_arg("-absolute")
  if absolute_eps_text ~= nil then
    absolute_eps = c.atof(absolute_eps_text)
  end

  var relative_eps = 1.0e-8
  var relative_eps_text = get_optional_arg("-relative")
  if relative_eps_text ~= nil then
    relative_eps = c.atof(relative_eps_text)
  end

  -- FIXME: This is kind of silly, but some of the really small values
  -- (around 1e-17) have fairly large relative error (1e-3), tripping
  -- up the validator. For now, stop complaining about those cases if
  -- the absolute error is small.
  var relative_absolute_eps = 1.0e-17
  var relative_absolute_eps_text = get_optional_arg("-relative_absolute")
  if relative_absolute_eps_text ~= nil then
    relative_absolute_eps = c.atof(relative_absolute_eps_text)
  end

  do
    var rz_zr = c.legion_physical_region_get_field_accessor_array(
      rz_physical[12], rz_fields[12])
    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      var ck = @[&double](c.legion_accessor_array_ref(rz_zr, p))
      var sol = solution_zr[i]
      if cmath.fabs(ck - sol) > absolute_eps or
        (cmath.fabs(ck - sol) / sol > relative_eps and
           cmath.fabs(ck - sol) > relative_absolute_eps)
      then
        c.printf("Error: zr value out of bounds at %%d, expected %%.12e and got %%.12e\n",
                 i, sol, ck)
        c.printf("absolute %%.12e relative %%.12e\n",
                 cmath.fabs(ck - sol),
                 cmath.fabs(ck - sol) / sol)
        c.abort()
      end
    end
    c.legion_accessor_array_destroy(rz_zr)
  end

  do
    var rz_ze = c.legion_physical_region_get_field_accessor_array(
      rz_physical[13], rz_fields[13])
    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      var ck = @[&double](c.legion_accessor_array_ref(rz_ze, p))
      var sol = solution_ze[i]
      if cmath.fabs(ck - sol) > absolute_eps or
        (cmath.fabs(ck - sol) / sol > relative_eps and
           cmath.fabs(ck - sol) > relative_absolute_eps)
      then
        c.printf("Error: ze value out of bounds at %%d, expected %%.8e and got %%.8e\n",
                 i, sol, ck)
        c.printf("absolute %%.12e relative %%.12e\n",
                 cmath.fabs(ck - sol),
                 cmath.fabs(ck - sol) / sol)
        c.abort()
      end
    end
    c.legion_accessor_array_destroy(rz_ze)
  end

  do
    var rz_zp = c.legion_physical_region_get_field_accessor_array(
      rz_physical[17], rz_fields[17])
    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      var ck = @[&double](c.legion_accessor_array_ref(rz_zp, p))
      var sol = solution_zp[i]
      if cmath.fabs(ck - sol) > absolute_eps or
        (cmath.fabs(ck - sol) / sol > relative_eps and
           cmath.fabs(ck - sol) > relative_absolute_eps)
      then
        c.printf("Error: zp value out of bounds at %%d, expected %%.8e and got %%.8e\n",
                 i, sol, ck)
        c.printf("absolute %%.12e relative %%.12e\n",
                 cmath.fabs(ck - sol),
                 cmath.fabs(ck - sol) / sol)
        c.abort()
      end
    end
    c.legion_accessor_array_destroy(rz_zp)
  end

  c.printf("Successfully validate output\n")

  c.free(solution_zr)
  c.free(solution_ze)
  c.free(solution_zp)
end
c.printf("[33m[validate_output:compile][m\n");
validate_output:compile()
end

  
-- ##########################
-- ## validate_output_sequential
-- ##########################     
task validate_output_sequential(rz_all : region(zone),
                                rp_all : region(point),
                                rs_all : region(side(wild, wild, wild, wild)),
                                conf : config)
where reads(rz_all, rp_all, rs_all) do
  validate_output(
    __runtime(), __context(),
    __physical(rz_all), __fields(rz_all),
    __physical(rp_all), __fields(rp_all),
    __physical(rs_all), __fields(rs_all),
    conf)
end
