///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
// This NABLA port is an implementation of the NDSPMHD software
// Computes an SPH estimate of the curl of a vector quantity
// This version computes the curl for all particles and therefore only does each pairwise interaction once

/*
∀ cells void get_curl(Integer icurltype,
                    Integer npart,
                    Real3* x,
                    Real pmass,
                    Real rho,
                    Real hh,
                    Real3* Bvec,
                    Real3* curlB,
                    Real3* curlBgradh){
  // initialise quantities
  curlB = 0.0;
  Real dr = 0.0;
  Real weight = 1./hfact**ndim;
  Real h1 = 1./hh;
  if (present(curlBgradh)) curlBgradh = 0.0;
  foreach i particle{
    Real hi1 = h1;
    Real hi21 = hi1*hi1;
    Real hfacwabi = hi1**ndim;
    Real pmassi = pmass(i);
    Real rho21i = 1./rho**2;
    Real rho21gradhi = rho21i*gradh;
    Bi = Bvec;
    curlBi = 0.0;
    curlBgradhi = 0.0;
    // for each particle in the current cell, loop over its neighbours
    foreach j particles{
      Real3 dx = x[i]-x[j];
      Real hj1 = h1[j]; // 1./hj
      Real rij2 = dot_product(dx,dx);
      Real q2i = rij2*hi21;
      Real q2j = rij2*hj1*hj1;
      // do interaction if r/h < compact support size
      // don't calculate interactions between ghost particles
      if ((q2i < radkern2) || (q2j < radkern2)){
        hfacwabj = hj1**ndim;
        rij = sqrt(rij2);
        dr.x = dx.x/(rij + tiny(rij));  // unit vector;
        // interpolate from kernel table          
        // (use either average h or average kernel gradient)
        // (using hi)
        interpolate_kernel_curl(q2i,grkerni,grgrkerni);
        // (using hj)
        interpolate_kernel_curl(q2j,grkernj,grgrkernj);
        // calculate curl of Bvec (NB dB is 3-dimensional, dr is also but zero in parts)
        if (icurltype==2){ // symmetric curl for vector potential current
          Real grkerni = grkerni*hfacwabi*hi1;
          Real grkernj = grkernj*hfacwabj*hj1*gradh(j);
          cross_product3D(Bi(:),dr,curlBtermi);
          cross_product3D(Bvec(:,j),dr,curlBtermj);
          curlBterm = (curlBtermi*rho21gradhi*grkerni + curlBtermj/rho**2*grkernj);
          curlBi = curlBi + pmass*curlBterm;
          curlB = curlB - pmassi*curlBterm;
        } elseif (icurltype==3){ // (dB using weights) -- multiply by weight below
          Real grkerni = grkerni*hi1;
          Real grkernj = grkernj*hj1;
          dB = Bi - Bvec;
          cross_product3D(dB,dr,curlBterm);
          curlBi = curlBi + curlBterm*grkerni;
          curlB[j] = curlB[j] + curlBterm*grkernj;
        } elseif (icurltype==4){ // (dB, m_j/rho_j**2) -- multiply by rho(i) below
          Real grkerni = grkerni*hfacwabi*hi1;
          Real grkernj = grkernj*hfacwabj*hj1;
          dB.x = Bi.x - Bvec.x[j];
          cross_product3D(dB,dr,curlBterm);
          curlBi = curlBi + pmass[j]/rho[j]**2*curlBterm*grkerni;
          curlB[j] = curlB[j] + pmassi*rho21i*curlBterm*grkernj;
        } else {                 // default curl (dB, m_j/rho_i with gradh) -- divide by rho(i) below
          Real grkerni = grkerni*hfacwabi*hi1;
          Real grkernj = grkernj*hfacwabj*hj1;
          dB.x = Bi.x - Bvec.x[j];
          cross_product3D(dB,dr,curlBterm);
          curlBi = curlBi + pmass[j]*curlBterm*grkerni;
          curlB[j] = curlB[j] + pmassi*curlBterm*grkernj;
          if (present(curlBgradh)){
            dgradwdhi = -(ndim+1.)*hi1*grkerni - rij*hi1**3*grgrkerni*hfacwabi;
            dgradwdhj = -(ndim+1.)*hj1*grkernj - rij*hj1**3*grgrkernj*hfacwabj;
            curlBgradhi = curlBgradhi + pmass[j]*curlBterm*dgradwdhi;
            curlBgradh[j] = curlBgradh[j] + pmassi*curlBterm*dgradwdhj;
          }
        }
      }
    }
    curlB += curlBi;
    if (present(curlBgradh))
      curlBgradh += curlBgradhi;
  }
}
*/
∀ particles void finishCurlB(void){
  if (icurltype==4) curlB = rho*curlB;
  if (icurltype==3) curlB = weight*curlB; //*gradh(i)
  if (icurltype==2) curlB = -rho*curlB;    
  if (icurltype!=4 && icurltype!=3 && icurltype!=2){
    curlB = curlB*gradh/rho;
    if (present(curlBgradh))
      curlBgradh *= gradh;
  }
}
