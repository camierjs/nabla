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
local cstring = terralib.includec("string.h")

--
-- Command Line Processing
--

terra get_positional_arg()
  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if args.argv[i][0] == ('-')[0] then
      i = i + 1
    else
      return args.argv[i]
    end
    i = i + 1
  end
  return nil
end

terra get_optional_arg(key : rawstring)
  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if cstring.strcmp(args.argv[i], key) == 0 then
      if i + 1 < args.argc then
        return args.argv[i + 1]
      else
        return nil
      end
    elseif args.argv[i][0] == ('-')[0] then
      i = i + 1
    end
    i = i + 1
  end
  return nil
end
