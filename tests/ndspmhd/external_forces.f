///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
// * Computes external (body) forces on a particle given its co-ordinates

void external_forces(const Integer iexternal_force,
                     const Real* xpart,
                     Real* fext,
                     const Integer ndim,
                     const Integer ndimV,
                     const Real* vpart,
                     const Real hpart,
                     const Real spsound,
                     const Integer itypei){
  *fext = 0.0;
  if (iexternal_force==1) fext = - xpart; // toy star force (linear in co-ordinates)
}


// toy star force (linear in co-ordinates)
∀ particles void external_forces_toy_star(void) {//if (option_toy_star){
  fext = - xpart;
}


// 1/r^2 force from central point mass
∀ particles void external_forces_inv_r2(void) {//if (option_inv_r2){
  Real rr2= xpart⋅xpart;
  Real rr = √(rr2);
  Real drr2 = 1.0/rr2;
  Real dr = xpart/rr;
  fext = - dr*drr2;
 }


// gravitational force
∀ particles void external_forces_gravitational(void) {//if (option_gravitational_force){
  fext = 0.0;
}


// boundary layer for acoustic-shock problem
∀ particles void external_forces_gravitational_boundary(void) {//if (option_gravitational_force){
  fext = 0.0;
  fext.y = Alayercs*dwidthlayer*1./(cosh((xpart.x-xlayer)*dwidthlayer))²;
 }


// 2D cartesian shearing box
∀ particles void external_2D_cartesian_shearing_box(void) {//if (option_2D_cartesian_shearing_box){
  fext.x =  2.0*domegadr*Omega2*xpart.x + 2.0*Omega0*vpart.z;
  fext.y =  0.0;
  fext.z = -2.0*Omega0*vpart.x;
 }


// effective potential for equilibrium torus (Stone, Pringle, Begelman)
//  centripedal force balances pressure gradient and gravity of central point mass
∀ particles void external_forces_equilibrium_torus(void) {//if (option_equilibrium_torus){
  //Real rcyl2 = DOT_PRODUCT(xpart(1:2),xpart(1:2));
  Real rcyl = SQRT(rcyl2);
  Real rsph,v2onr;
  if (ndim==3)
    rsph = sqrt(rcyl2 + xpart(3)*xpart(3));
  else
    rsph = rcyl;
  v2onr = 1./(Rtorus)*(-Rtorus*rcyl/rsph**3 + Rtorus**2/(rcyl2*rcyl));
  fext = v2onr*xpart/rcyl;
  // for 3D need to add vertical component of gravity
  if (ndim == 3) fext.z = -xpart.z/rsph**3;
}


// sinusoidal potential as in Dobbs, Bonnell etc.
∀ particles void external_forces_sinusoidal_potential(void) {//if (option_sinusoidal_potential){
  Real sink = 0.25*pi;
  fext.x = -Asin*sink*sin(sink*(xpart.x + Bsin));
}

// gravity
∀ particles void external_forces_gravity(void){// if (option_gravity){
  if (ndim >= 2){
    fext.y = -0.1;
    if (ibound.y == 0){
      q2i = abs(xmax.y-xpart.y)/hpart;
      if (q2i < 4.0){
        betai = 0.02*spsound**2/xpart.y;
        fext.y = fext.y - betai*gradwkernbound(q2i);
      }
      q2i = abs(xpart.y-xmin.y)/hpart;
      if (q2i < 4.0){
        betai = 0.02*spsound**2/xpart.y;
        fext.y = fext.y + betai*gradwkernbound(q2i);
      }
    }
  }
  if (ndim==2) fext.y = -0.5;
}


// potential for relaxation into Kelvin-Helmholtz initial conditions
∀ particles void external_forces_kelvin_helmholtz(void) {//if (option_kelvin_helmholtz){
  Real denszero = 1.0;
  Real densmedium = 2.0;
  Real densmid = 0.5*(denszero - densmedium);
  Real yi = xpart.y - xmin.y;
  if (yi > 0.75){
    expterm = exp(-(yi-0.75)/smoothl);
    dens    = denszero - densmid*expterm;
    ddensdy = densmid/smoothl*expterm;
  }else if (yi > 0.5){
    expterm = exp(-(0.75-yi)/smoothl);
    dens    = densmedium + densmid*expterm;
    ddensdy = densmid/smoothl*expterm;
  }else if (yi > 0.25){
    expterm = exp((-yi + 0.25)/smoothl);
    dens    = densmedium + densmid*expterm;
    ddensdy = -densmid/smoothl*expterm;
  } else{
    expterm = exp((yi - 0.25)/smoothl);
    dens    = denszero - densmid*expterm;
    ddensdy = -densmid/smoothl*expterm;
  }
  fext.y = polyk*gamma*dens**(gamma-2.)*ddensdy;
}


// this is for the 2D cartesian shearing box for SI
∀ particles void external_forces_2D_cartesian_shearing_box(void) {//if (option_2D_cartesian_shearing_box){
  if (itypei == itypegas)
    fext(1) = 2.*domegadr*Omega2*xpart(1) + 2.*Omega0*(vpart(3)+eta);
  else if (itypei ==itypedust)
    fext(1) = 2.*domegadr*Omega2*xpart(1) + 2.*Omega0*vpart(3);
  else{
    info()<<"external_forces SI: unexpected type of particles";
    //fatal;
  }
  fext.y = 0.0;
  fext.z = -2.0*Omega0*vpart.x;
}


// vertical linear force to simulate the vertical motion of particles in a disc
∀ particles void external_forces_vertical_motion_in_disc(void) {//if (option_vertical_motion_in_disc){
  fext.x = 0.0;
  fext.y = - xpart.y; //2D pb
}


// vertical cubic force to benchmark the settling of dust particles
∀ particles void external_forces_vertical_cubic_dust_particles(void) {//if (option_vertical_cubic_dust_particles){
  fext.x = 0.0;
  fext.y = - xpart.y**3; //2D pb 
}


// vertical square root force to benchmark the settling of dust particles
∀ particles void external_forces_vertical_square_root_dust_particles(void) {//if (option_vertical_square_root_dust_particles){
  fext.x = 0.0;
  if ( xpart.y >= 0.0)
    fext(2) = - xpart(2)**0.5; // 2D pb
  else
    fext(2) = (-xpart(2))**0.5; //2D pb
}


// default
∀ particles void external_forces_default(void) {//if (option_default){
  fext=0.0;
}


Real gradwkernbound(Real q2){
  Real q = sqrt(q2);
  if (q < 2.0/3.0) return 2./3.;
  if (q < 1.0) return 2.0*q - 1.5*q2;
  if (q < 2.0) return 0.5*(2.-q)**2;
  return 0.0;
}


Real pequil(Integer iexternal_force, Real3 xpart, Real densi){
  if (iexternal_force==9) return -0.5*densi*(xpart.y-0.5);
  if (iexternal_force==8) return -0.1*densi*xpart.y;
  return 0.0;
}


// Calculate potential energy for above forces
// (called from evwrite in adding up total energy)

void external_potentials(const Integer iexternal_force,
                         const Real3 xpart,
                         Real* epot,
                         Integer ndim){
  const Real Asin = 100.0;
  const Real Bsin = 2.0;
  
  if (iexternal_force==1) // toy star force (x^2 potential)
    return epot = 0.5*DOT_PRODUCT(xpart,xpart);
  
  if (iexternal_force==2) // 1/r^2 force(1/r potential)
    return epot = -1./SQRT(DOT_PRODUCT(xpart,xpart));
  
  if (iexternal_force==3) // potential from n point masses
    return epot = 0.0;
  
  if (iexternal_force==5)
    return epot = domegadr*Omega2*xpart(1)*xpart(1);
  
  if (iexternal_force==7){
    Real sink = 0.25*pi;
    return epot = Asin*COS(sink*(xpart(1) + Bsin));
  }
  return epot = 0.0;
}
