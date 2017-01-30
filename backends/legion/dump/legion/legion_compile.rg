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

-- Compile and link pennant.cc
do
  c.printf("[33m[pennant_common] Compile and link pennant.cc[m\n");
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = "/tmp/runtime/"
  local legion_dir = runtime_dir .. "legion/"
  local mapper_dir = runtime_dir .. "mappers/"
  local realm_dir = runtime_dir .. "realm/"
  local pennant_cc = root_dir .. "legion.cc"
  if os.getenv('SAVEOBJ') == '1' then
    pennant_so = root_dir .. "libpennant.so"
  else
    --pennant_so = os.tmpname() .. ".so" -- root_dir .. "pennant.so"
    pennant_so = root_dir .. "pennant.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -std=c++0x -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                " -I " .. mapper_dir .. " " .. " -I " .. legion_dir .. " " ..
                " -I " .. realm_dir .. " " ..  pennant_cc .. " -o " .. pennant_so)
  c.printf("[33m[pennant_common] cmd=%%s[m\n",cmd);
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. pennant_cc)
    assert(false)
  end
  c.printf("[33m[pennant_common] done, linking[m\n");
  terralib.linklibrary(pennant_so)
  c.printf("[33m[pennant_common] done, includec[m\n");
  cpennant = terralib.includec("legion.h", {"-I", root_dir, "-I", runtime_dir,
                                             "-I", mapper_dir, "-I", legion_dir,
                                             "-I", realm_dir})
  c.printf("[33m[pennant_common] done\n[m");
end
