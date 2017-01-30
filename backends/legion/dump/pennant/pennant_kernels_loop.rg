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
-- ## Main simulation loop
-- #################

-- Save off point variable values from previous cycle.
task init_step_points(rp : region(point),
                      enable : bool)
where
  writes(rp.{pmaswt, pf})
do
  c.printf("\t[32m[init_step_points][m\n");
  if not enable then return end
  -- Initialize fields used in reductions.
  __demand(__vectorize)
  for p in rp do
    p.pmaswt = 0.0
  end
  __demand(__vectorize)
  for p in rp do
    p.pf.x = 0.0
  end
  __demand(__vectorize)
  for p in rp do
    p.pf.y = 0.0
  end
end

--
-- 1. Advance mesh to center of time step.
--
task adv_pos_half(rp : region(point), dt : double,
                  enable : bool)
where
  reads(rp.{px, pu}),
  writes(rp.{px0, pxp, pu0})
do
  c.printf("\t[32m[adv_pos_half][m\n");
  if not enable then return end
  var dth = 0.5 * dt
  -- Copy state variables from previous time step and update position.
  __demand(__vectorize)
  for p in rp do
    var px0_x = p.px.x
    var pu0_x = p.pu.x
    p.px0.x = px0_x
    p.pu0.x = pu0_x
    p.pxp.x = px0_x + dth*pu0_x
  end
  __demand(__vectorize)
  for p in rp do
    var px0_y = p.px.y
    var pu0_y = p.pu.y
    p.px0.y = px0_y
    p.pu0.y = pu0_y
    p.pxp.y = px0_y + dth*pu0_y
  end
end

-- Save off zone variable value from previous cycle.
task init_step_zones(rz : region(zone), enable : bool)
where
  reads(rz.zvol),
  writes(rz.zvol0)
do
  c.printf("\t[32m[init_step_zones][m\n");
  if not enable then return end

  -- Copy state variables from previous time step.
  __demand(__vectorize)
  for z in rz do
    z.zvol0 = z.zvol
  end
end

--
-- 1a. Compute new mesh geometry.
--

-- Compute centers of zones and edges.
task calc_centers(rz : region(zone), rpp : region(point), rpg : region(point),
                  rs : region(side(rz, rpp, rpg, rs)),
                  enable : bool)
where
  reads(rz.znump, rpp.pxp, rpg.pxp, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zxp, rs.exp)
do
  c.printf("\t[32m[calc_centers][m\n");
  if not enable then return end

  var zxp = vec2 { x = 0.0, y = 0.0 }
  var nside = 1
  for s in rs do
    var z = s.mapsz
    var p1 = s.mapsp1
    var p2 = s.mapsp2
    var e = s
    var p1_pxp = p1.pxp
    e.exp = 0.5*(p1_pxp + p2.pxp)
    zxp += p1_pxp
    if nside == z.znump then
      z.zxp = (1/double(z.znump)) * zxp
      zxp = vec2 { x = 0.0, y = 0.0 }
      nside = 0
    end
    nside += 1
  end
end

-- Compute volumes of zones and sides.
-- Compute edge lengths.
task calc_volumes(rz : region(zone), rpp : region(point), rpg : region(point),
                  rs : region(side(rz, rpp, rpg, rs)),
                  enable : bool)
where
  reads(rz.{zxp, znump}, rpp.pxp, rpg.pxp, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zareap, zvolp}, rs.{sareap, elen})
do
  c.printf("\t[32m[calc_volumes][m\n");
  if not enable then return end

  var zareap = 0.0
  var zvolp = 0.0
  var nside = 1
  for s in rs do
    var z = s.mapsz
    var p1 = s.mapsp1
    var p2 = s.mapsp2

    var p1_pxp = p1.pxp
    var p2_pxp = p2.pxp
    var sa = 0.5 * cross(p2_pxp - p1_pxp, z.zxp - p1_pxp)
    var sv = sa * (p1_pxp.x + p2_pxp.x + z.zxp.x)
    s.sareap = sa
    -- s.svolp = sv
    s.elen = length(p2_pxp - p1_pxp)
    zareap += sa
    zvolp += sv
    if nside == z.znump then
      z.zareap = zareap
      z.zvolp = (1.0 / 3.0) * zvolp
      zareap = 0.0
      zvolp = 0.0
      nside = 0
    end
    nside += 1

    regentlib.assert(sv > 0.0, "sv negative")
  end
end

-- Compute zone characteristic lengths.
task calc_char_len(rz : region(zone), rpp : region(point), rpg : region(point),
                   rs : region(side(rz, rpp, rpg, rs)),
                   enable : bool)
where
  reads(rz.znump, rs.{mapsz, sareap, elen}),
  writes(rz.zdl)
do
  c.printf("\t[32m[calc_char_len][m\n");
  if not enable then return end

  var zdl = 1e99
  var nside = 1
  for s in rs do
    var z = s.mapsz
    var e = s

    var area = s.sareap
    var base = e.elen
    var fac = 0.0
    if z.znump == 3 then
      fac = 3.0
    else
      fac = 4.0
    end
    var sdl = fac * area / base
    zdl = min(zdl, sdl)

    if nside == z.znump then
      z.zdl = zdl
      zdl = 1e99
      nside = 0
    end
    nside += 1
  end
end

--
-- 2. Compute point masses.
--

-- Compute zone densities.
task calc_rho_half(rz : region(zone), enable : bool)
where
  reads(rz.{zvolp, zm}),
  writes(rz.zrp)
do
  c.printf("\t[32m[calc_rho_half][m\n");
  if not enable then return end

  __demand(__vectorize)
  for z in rz do
    z.zrp = z.zm / z.zvolp
  end
end

-- Reduce masses into points.
task sum_point_mass(rz : region(zone), rpp : region(point), rpg : region(point),
                    rs : region(side(rz, rpp, rpg, rs)),
                    enable : bool)
where
  reads(rz.{zareap, zrp}, rs.{mapsz, mapsp1, mapss3, smf}),
  reads writes(rpp.pmaswt),
  reduces+(rpg.pmaswt)
do
  c.printf("\t[32m[sum_point_mass][m\n");
  if not enable then return end

  for s in rs do
    var z = s.mapsz
    var p1 = s.mapsp1
    var s3 = s.mapss3

    var m = z.zrp * z.zareap * 0.5 * (s.smf + s3.smf)
    p1.pmaswt += m
  end
end

--
-- 3. Compute material state (half-advanced).
--

task calc_state_at_half(rz : region(zone),
                        gamma : double, ssmin : double, dt : double,
                        enable : bool)
where
  reads(rz.{zvol0, zvolp, zm, zr, ze, zwrate}),
  writes(rz.{zp, zss})
do
  c.printf("\t[32m[calc_state_at_half][m\n");
  if not enable then return end

  var gm1 = gamma - 1.0
  var ss2 = max(ssmin * ssmin, 1e-99)
  var dth = 0.5 * dt

  for z in rz do
    var rx = z.zr
    var ex = max(z.ze, 0.0)
    var px = gm1 * rx * ex
    var prex = gm1 * ex
    var perx = gm1 * rx
    var csqd = max(ss2, prex + perx * px / (rx * rx))
    var z0per = perx
    var zss = sqrt(csqd)
    z.zss = zss

    var zminv = 1.0 / z.zm
    var dv = (z.zvolp - z.zvol0) * zminv
    var bulk = z.zr * zss * zss
    var denom = 1.0 + 0.5 * z0per * dv
    var src = z.zwrate * dth * zminv
    z.zp = px + (z0per * src - z.zr * bulk * dv) / denom
  end
end

--
-- 4. Compute forces.
--

-- Compute PolyGas and TTS forces.
task calc_force_pgas_tts(rz : region(zone), rpp : region(point),
                         rpg : region(point),
                         rs : region(side(rz, rpp, rpg, rs)),
                         alfa : double, ssmin : double,
                         enable : bool)
where
  reads(rz.{zxp, zareap, zrp, zss, zp}, rs.{mapsz, sareap, smf, exp}),
  writes(rs.{sfp, sft})
do
  c.printf("\t[32m[calc_force_pgas_tts][m\n");
  if not enable then return end

  for s in rs do
    var z = s.mapsz

    -- Compute surface vectors of sides.
    var ssurfp = rotateCCW(s.exp - z.zxp)

    -- Compute PolyGas forces.
    var sfx = (-z.zp)*ssurfp
    s.sfp = sfx

    -- Compute TTS forces.
    var svfacinv = z.zareap / s.sareap
    var srho = z.zrp * s.smf * svfacinv
    var sstmp = max(z.zss, ssmin)
    sstmp = alfa * sstmp * sstmp
    var sdp = sstmp * (srho - z.zrp)
    var sqq = (-sdp)*ssurfp
    s.sft = sfx + sqq
  end
end

task qcs_zone_center_velocity(rz : region(zone), rpp : region(point), rpg : region(point),
                              rs : region(side(rz, rpp, rpg, rs)),
                              enable : bool)
where
  reads(rz.znump, rpp.pu, rpg.pu, rs.{mapsz, mapsp1}),
  writes(rz.zuc)
do
 c.printf("\t[32m[qcs_zone_center_velocity][m\n");
 if not enable then return end

  var zuc = vec2 { x = 0.0, y = 0.0 }
  var nside = 1
  for s in rs do
    var z = s.mapsz
    var p1 = s.mapsp1

    zuc += (1.0 / double(z.znump))*p1.pu

    if nside == z.znump then
      z.zuc = zuc
      zuc = vec2 { x = 0.0, y = 0.0 }
      nside = 0
    end
    nside += 1
  end
end

task qcs_corner_divergence(rz : region(zone), rpp : region(point), rpg : region(point),
                           rs : region(side(rz, rpp, rpg, rs)),
                           enable : bool)
where
  reads(rz.{zxp, zuc}, rpp.{pxp, pu}, rpg.{pxp, pu},
        rs.{mapsz, mapsp1, mapsp2, mapss3, exp, elen}),
  writes(rs.{carea, ccos, cdiv, cevol, cdu})
do
  c.printf("\t[32m[qcs_corner_divergence][m\n");
  if not enable then return end

  for s2 in rs do
    var c = s2
    var s = s2.mapss3
    var z = s.mapsz
    var p = s.mapsp2
    var p1 = s.mapsp1
    var p2 = s2.mapsp2
    var e1 = s
    var e2 = s2

    -- velocities and positions
    -- point p
    var up0 = p.pu
    var xp0 = p.pxp
    -- edge e2
    var up1 = 0.5*(p.pu + p2.pu)
    var xp1 = e2.exp
    -- zone center z
    var up2 = z.zuc
    var xp2 = z.zxp
    -- edge e1
    var up3 = 0.5*(p1.pu + p.pu)
    var xp3 = e1.exp

    -- compute 2d cartesian volume of corner
    var cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1)
    c.carea = cvolume

    -- compute cosine angle
    var v1 = xp3 - xp0
    var v2 = xp1 - xp0
    var de1 = e1.elen
    var de2 = e2.elen
    var minelen = min(de1, de2)
    if minelen < 1e-12 then
      c.ccos = 0.0
    else
      c.ccos = 4.0 * dot(v1, v2) / (de1 * de2)
    end

    -- compute divergence of corner
    var cdiv = (cross(up2 - up0, xp3 - xp1) -
                cross(up3 - up1, xp2 - xp0)) / (2.0 * cvolume)
    c.cdiv = cdiv

    -- compute evolution factor
    var dxx1 = 0.5*(((xp1 + xp2) - xp0) - xp3)
    var dxx2 = 0.5*(((xp2 + xp3) - xp0) - xp1)
    var dx1 = length(dxx1)
    var dx2 = length(dxx2)

    -- average corner-centered velocity
    var duav = 0.25*(((up0 + up1) + up2) + up3)

    var test1 = abs(dot(dxx1, duav) * dx2)
    var test2 = abs(dot(dxx2, duav) * dx1)
    var num = 0.0
    var den = 0.0
    if test1 > test2 then
      num = dx1
      den = dx2
    else
      num = dx2
      den = dx1
    end
    var r = num / den
    var evol = min(sqrt(4.0 * cvolume * r), 2.0 * minelen)

    -- compute delta velocity
    var dv1 = length(((up1 + up2) - up0) - up3)
    var dv2 = length(((up2 + up3) - up0) - up1)
    var du = max(dv1, dv2)

    if cdiv < 0.0 then
      c.cevol = evol
      c.cdu = du
    else
      c.cevol = 0.0
      c.cdu = 0.0
    end
  end
end

task qcs_qcn_force(rz : region(zone), rpp : region(point), rpg : region(point),
                   rs : region(side(rz, rpp, rpg, rs)),
                   gamma : double, q1 : double, q2 : double,
                   enable : bool)
where
  reads(rz.{zrp, zss}, rpp.pu, rpg.pu,
        rs.{mapsz, mapsp1, mapsp2, mapss3, elen, cdiv, cdu, cevol}),
  writes(rs.{cqe1, cqe2})
do
  c.printf("\t[32m[qcs_qcn_force][m\n");
  if not enable then return end

  var gammap1 = gamma + 1.0

  for s4 in rs do
    var c = s4
    var z = c.mapsz

    var ztmp2 = q2 * 0.25 * gammap1 * c.cdu
    var ztmp1 = q1 * z.zss
    var zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1)
    var rmu = zkur * z.zrp * c.cevol
    if c.cdiv > 0.0 then
      rmu = 0.0
    end

    var s = c.mapss3
    var p = s.mapsp2
    var p1 = s.mapsp1
    var e1 = s
    var p2 = s4.mapsp2
    var e2 = s4

    c.cqe1 = rmu / e1.elen*(p.pu - p1.pu)
    c.cqe2 = rmu / e2.elen*(p2.pu - p.pu)
  end
end

task qcs_force(rz : region(zone), rpp : region(point), rpg : region(point),
               rs : region(side(rz, rpp, rpg, rs)),
               enable : bool)
where
  reads(rs.{mapss4, elen, carea, ccos, cqe1, cqe2}),
  writes(rs.sfq)
do
  c.printf("\t[32m[qcs_force][m\n");
  if not enable then return end

  for s in rs do
    var c1 = s
    var c2 = s.mapss4
    var e = s
    var el = e.elen

    var c1sin2 = 1.0 - c1.ccos * c1.ccos
    var c1w = 0.0
    var c1cos = 0.0
    if c1sin2 >= 1e-4 then
      c1w = c1.carea / c1sin2
      c1cos = c1.ccos
    end

    var c2sin2 = 1.0 - c2.ccos * c2.ccos
    var c2w = 0.0
    var c2cos = 0.0
    if c2sin2 >= 1e-4 then
      c2w = c2.carea / c2sin2
      c2cos = c2.ccos
    end

    s.sfq = (1.0 / el)*(c1w*(c1.cqe2 + c1cos*c1.cqe1) +
                          c2w*(c2.cqe1 + c2cos*c2.cqe2))
  end
end

task qcs_vel_diff(rz : region(zone), rpp : region(point), rpg : region(point),
                  rs : region(side(rz, rpp, rpg, rs)),
                  q1 : double, q2 : double,
                  enable : bool)
where
  reads(rz.{zss, z0tmp}, rpp.{pxp, pu}, rpg.{pxp, pu},
        rs.{mapsp1, mapsp2, mapsz, elen}),
  writes(rz.{zdu, z0tmp})
do
  c.printf("\t[32m[qcs_vel_diff][m\n");
  if not enable then return end

  for z in rz do
    z.z0tmp = 0.0
  end

  for s in rs do
    var p1 = s.mapsp1
    var p2 = s.mapsp2
    var z = s.mapsz
    var e = s

    var dx = p2.pxp - p1.pxp
    var du = p2.pu - p1.pu
    var lenx = e.elen
    var dux = dot(du, dx)
    if lenx > 0.0 then
      dux = abs(dux) / lenx
    else
      dux = 0.0
    end
    z.z0tmp = max(z.z0tmp, dux)
  end

  for z in rz do
    z.zdu = q1 * z.zss + 2.0 * q2 * z.z0tmp
  end
end

local function accessor_generic_get_base_pointer(field_type)
  return terra(accessor : c.legion_accessor_generic_t)

    var base_pointer : &opaque = nil
    var stride : c.size_t = terralib.sizeof(field_type)
    var ok = c.legion_accessor_generic_get_soa_parameters(
      accessor, &base_pointer, &stride)

    regentlib.assert(ok, "failed to get base pointer")
    regentlib.assert(stride == terralib.sizeof(field_type),
                     "stride does not match expected value")

    return [&field_type](base_pointer)
  end
end

-- Reduce forces into points.
task sum_point_force(rz : region(zone), rpp : region(point), rpg : region(point),
                     rs : region(side(rz, rpp, rpg, rs)),
                     enable : bool)
where
  reads(rz.znump, rs.{mapsz, mapsp1, mapss3, sfq, sft}),
  reads writes(rpp.pf),
  reduces+(rpg.pf.{x, y})
do
  c.printf("\t[32m[sum_point_force][m\n");
  if not enable then return end

  for s in rs do
    var p1 = s.mapsp1
    var s3 = s.mapss3

    var f = (s.sfq + s.sft) - (s3.sfq + s3.sft)
    p1.pf.x += f.x
    p1.pf.y += f.y
  end
end

--
-- 4a. Apply boundary conditions.
--

task apply_boundary_conditions(rp : region(point),
                               enable : bool)
where
  reads(rp.{has_bcx, has_bcy}),
  reads writes(rp.{pu0, pf})
do
  c.printf("\t[32m[apply_boundary_conditions][m\n");
  if not enable then return end

  var vfixx = {x = 1.0, y = 0.0}
  var vfixy = {x = 0.0, y = 1.0}
  for p in rp do
    if p.has_bcx then
      p.pu0 = project(p.pu0, vfixx)
      p.pf = project(p.pf, vfixx)
    end
    if p.has_bcy then
      p.pu0 = project(p.pu0, vfixy)
      p.pf = project(p.pf, vfixy)
    end
  end
end

--
-- 5. Compute accelerations.
--

-- Fused into adv_pos_full.

--
-- 6. Advance mesh to end of time step.
--

task adv_pos_full(rp : region(point), dt : double,
                  enable : bool)
where
  reads(rp.{px0, pu0, pf, pmaswt}),
  writes(rp.{px, pu})
do
  c.printf("\t[32m[adv_pos_full][m\n");
  if not enable then return end

  var fuzz = 1e-99
  var dth = 0.5 * dt
  __demand(__vectorize)
  for p in rp do
    var fac = 1.0 / max(p.pmaswt, fuzz)
    var pap_x = fac*p.pf.x
    var pap_y = fac*p.pf.y

    var pu_x = p.pu0.x + dt*(pap_x)
    p.pu.x = pu_x
    p.px.x = p.px0.x + dth*(pu_x + p.pu0.x)

    var pu_y = p.pu0.y + dt*(pap_y)
    p.pu.y = pu_y
    p.px.y = p.px0.y + dth*(pu_y + p.pu0.y)
  end
end

--
-- 6a. Compute new mesh geometry.
--

-- FIXME: This is a duplicate of calc_centers but with different
-- code. Struct slicing ought to make it possible to use the same code
-- in both cases.
task calc_centers_full(rz : region(zone), rpp : region(point), rpg : region(point),
                       rs : region(side(rz, rpp, rpg, rs)),
                       enable : bool)
where
  reads(rz.znump, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zx, rs.ex)
do
  c.printf("\t[32m[calc_centers_full][m\n");
  if not enable then return end

  var zx = vec2 { x = 0.0, y = 0.0 }
  var nside = 1
  for s in rs do
    var z = s.mapsz
    var p1 = s.mapsp1
    var p2 = s.mapsp2
    var e = s

    var p1_px = p1.px
    e.ex = 0.5*(p1_px + p2.px)

    zx += p1_px

    if nside == z.znump then
      z.zx = (1/double(z.znump)) * zx
      zx = vec2 { x = 0.0, y = 0.0 }
      nside = 0
    end
    nside += 1
  end
end

-- FIXME: This is a duplicate of calc_volumes but with different
-- code. Struct slicing ought to make it possible to use the same code
-- in both cases.
task calc_volumes_full(rz : region(zone), rpp : region(point), rpg : region(point),
                       rs : region(side(rz, rpp, rpg, rs)),
                       enable : bool)
where
  reads(rz.{zx, znump}, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zarea, zvol}, rs.{sarea})
do
  c.printf("\t[32m[calc_volumes_full][m\n");
  if not enable then return end

  var zarea = 0.0
  var zvol = 0.0
  var nside = 1
  for s in rs do
    var z = s.mapsz
    var p1 = s.mapsp1
    var p2 = s.mapsp2

    var p1_px = p1.px
    var p2_px = p2.px
    var sa = 0.5 * cross(p2_px - p1_px, z.zx - p1_px)
    var sv = sa * (p1_px.x + p2_px.x + z.zx.x)
    s.sarea = sa
    -- s.svol = sv

    zarea += sa
    zvol += sv

    if nside == z.znump then
      z.zarea = zarea
      z.zvol = (1.0 / 3.0) * zvol
      zarea = 0.0
      zvol = 0.0
      nside = 0
    end
    nside += 1

    regentlib.assert(sv > 0.0, "sv negative")
  end
end

--
-- 7. Compute work
--

task calc_work(rz : region(zone), rpp : region(point), rpg : region(point),
               rs : region(side(rz, rpp, rpg, rs)),
               dt : double,
               enable : bool)
where
  reads(rz.{zetot, znump}, rpp.{pxp, pu0, pu}, rpg.{pxp, pu0, pu},
        rs.{mapsz, mapsp1, mapsp2, sfp, sfq}),
  writes(rz.{zw, zetot})
do
  c.printf("\t[32m[calc_work][m\n");
  if not enable then return end

  var zdwork = 0.0
  var nside = 1
  for s in rs do
    var z = s.mapsz
    var p1 = s.mapsp1
    var p2 = s.mapsp2

    var sftot = s.sfp + s.sfq
    var sd1 = dot(sftot, p1.pu0 + p1.pu)
    var sd2 = dot(-1.0*sftot, p2.pu0 + p2.pu)
    var dwork = -0.5 * dt * (sd1 * p1.pxp.x + sd2 * p2.pxp.x)

    zdwork += dwork

    if nside == z.znump then
      z.zetot += zdwork
      z.zw = zdwork
      zdwork = 0.0
      nside = 0
    end
    nside += 1
  end
end

--
-- 7a. Compute work rate.
-- 8. Update state variables.
--

task calc_work_rate_energy_rho_full(rz : region(zone), dt : double,
                                    enable : bool)
where
  reads(rz.{zvol0, zvol, zm, zw, zp, zetot}),
  writes(rz.{zwrate, ze, zr})
do
  c.printf("\t[32m[calc_work_rate_energy_rho_full][m\n");
  if not enable then return end

  var dtiny = 1.0 / dt
  var fuzz = 1e-99

  __demand(__vectorize)
  for z in rz do
    var dvol = z.zvol - z.zvol0
    z.zwrate = (z.zw + z.zp * dvol) * dtiny

    z.ze = z.zetot / (z.zm + fuzz)

    z.zr = z.zm / z.zvol
  end
end

--
-- 9. Compute timstep for next cycle.
--

--[[
task calc_dt_courant(rz : region(zone), dtmax : double, cfl : double) : double
where
  reads(rz.{zdl, zss, zdu})
do
  c.printf("\t[32m[calc_dt_courant][m\n");
  var fuzz = 1e-99
  var dtnew = dtmax
  for z in rz do
    var cdu = max(z.zdu, max(z.zss, fuzz))
    var zdthyd = z.zdl * cfl / cdu

    dtnew min= zdthyd
  end

  return dtnew
end

task calc_dt_volume(rz : region(zone), dtlast : double, cflv : double) : double
where
  reads(rz.{zvol0, zvol})
do
  c.printf("\t[32m[calc_dt_volume][m\n");
  var dvovmax = 1e-99
  for z in rz do
    var zdvov = abs((z.zvol - z.zvol0) / z.zvol0)
    dvovmax max= zdvov
  end
  return dtlast * cflv / dvovmax
end
]]

task calc_dt_hydro(rz : region(zone), dtlast : double, dtmax : double,
                   cfl : double, cflv : double, enable : bool) : double
where
  reads(rz.{zdl, zvol0, zvol, zss, zdu})
do
  c.printf("\t[32m[calc_dt_hydro][m\n");
  var dthydro = dtmax

  if not enable then return dthydro end

  -- dthydro min= min(calc_dt_courant(rz, dtmax, cfl),
  --                  calc_dt_volume(rz, dtlast, cflv))

  -- Hack: manually inline calc_dt_courant
  do
    var fuzz = 1e-99
    var dtnew = dtmax
    for z in rz do
      var cdu = max(z.zdu, max(z.zss, fuzz))
      var zdthyd = z.zdl * cfl / cdu

      dtnew min= zdthyd
    end

    dthydro min= dtnew
  end

  -- Hack: manually inline calc_dt_volume
  do
    var dvovmax = 1e-99
    for z in rz do
      var zdvov = abs((z.zvol - z.zvol0) / z.zvol0)
      dvovmax max= zdvov
    end
    dthydro min= dtlast * cflv / dvovmax
  end

  return dthydro
end

task calc_global_dt(dt : double, dtfac : double, dtinit : double,
                    dtmax : double, dthydro : double,
                    time : double, tstop : double, cycle : int64) : double
  c.printf("\t[32m[calc_global_dt][m\n");
  var dtlast = dt

  dt = dtmax

  if cycle == 0 then
    dt = min(dt, dtinit)
  else
    var dtrecover = dtfac * dtlast
    dt = min(dt, dtrecover)
  end

  dt = min(dt, tstop - time)
  dt = min(dt, dthydro)

  return dt
end

task continue_simulation(cycle : int64, cstop : int64,
                         time : double, tstop : double)
  return (cycle < cstop and time < tstop)
end
