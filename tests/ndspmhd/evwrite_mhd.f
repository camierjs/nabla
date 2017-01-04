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
// * Calculate conserved quantities etc and write to .ev file
void evwrite(const Real t, Real* etot, Real* momtot){
  ekin = 0.0;
  etherm = 0.0;
  emag = 0.0;
  emagp = 0.0;
  etot = 0.0;
 if (igravity!=00) epot = potengrav;
 else epot = 0.0;
 /*∀*/ mom = 0.0;
 momtot = 0.0;
 /*∀*/ ang = 0.0;
 ekiny = 0.0;
 totmassdust = 0.0;
 totmassgas  = 0.0;
 
 //  mhd parameters
 if (imhd != 0){
   betamhdav = 0.0;
   betamhdmin = huge(betamhdmin);
   divBmax = 0.0;
   divBav = 0.0;
   divBtot = 0.0;
   FdotBmax = 0.0;
   FdotBav = 0.0;
   force_err_max = 0.0;
   force_err_av = 0.0;
   omegamhdav = 0.0;
   omegamhdmax = 0.0;
   fracdivBok = 0.0;
   /*∀*/ fluxtot = 0.0;
   fluxtotmag = 0.0;
   crosshel = 0.0;   
 }
}


// should really recalculate the thermal energy from the total energy here
// (otherwise uu is from the half time step and same with Bfield)
/*∀ particles void recalculateThermalEnergyFromTotalEnergy(void){
  mom += pmass*vel;
  if (ndim==3){
    cross_product3D(x,vel,ang);
    ang +=  pmass*ang;
  }else if (ndim==2){
    ang.z = ang.z + pmass*(x.x*vel.y - x.y*vel.x);
  }
  ekin = ekin + ½*pmass*(vel⋅vel);
  if (idust==1){
    Real dustfraci  = dustfrac;
    Real dtgi  = dustfraci/(1.0 - dustfraci);
    Real dterm = (1.0 - dustfraci);
    Real ekindeltav = ½*pmassi*dtgi*dterm²*(deltav⋅deltav);
    ekin = ekin + ekindeltav;
    ekiny = ekiny + ekindeltav;
    etherm = etherm + pmassi*uu*dterm;
    totmassgas  = totmassgas  + pmassi*dterm;
    totmassdust = totmassdust + pmassi*dtgi*dterm;
  } else{
    if (ndim==2) ekiny = ekiny + ½*pmassi*vel*vel;
    etherm = etherm + pmassi*uu;
  }
  // potential energy from external forces
  external_potentials(iexternal_force,x,epoti,ndim);
  epot = epot + pmassi*epoti;

  // mhd parameters
  if (imhd!=0){
    Real3 Bi = Bfield;
    Real3 Brhoi = Bi/rhoi;
    Real B2i = Bi⋅Bi;
    Real Bmagi = SQRT(B2i);
    Real forcemagi = SQRT(force⋅force);
    Real divBi = abs(divB(i));
    emag = emag + ½*pmassi*B2i/rhoi;
    emagp = emagp + ½*pmassi*(Bi⋅Bi)/rhoi;
  }
  
  // Plasma beta minimum/maximum/average
  if (B2i < tiny(B2i))
    betamhdi = 0.0;
  else 
    betamhdi = pr(i)/(0.5*B2i);
  betamhdav = betamhdav + betamhdi;
  if (betamhdi < betamhdmin) betamhdmin = betamhdi;

  // Maximum divergence of B
  if (divBi > divBmax) divBmax = divBi;
  divBav = divBav + divBi;

  // volume integral of div B (int B.dS)
  divBtot = divBtot + pmassi*divBi/rhoi;

  // Max component of magnetic force in the direction of B (should be zero)
  fmagabs = SQRT(fmag(:,i)⋅fmag(:,i));
  if (fmagabs.GT.1.e-8 .and. Bmagi.gt.1.e-8)
    fdotBi = ABS(fmag(:,i)⋅Bi(:))/(fmagabs*Bmagi);
  else
    FdotBi = 0.;
  fdotBav = fdotBav + fdotBi;
  if (fdotBi > fdotBmax) fdotBmax = fdotBi;

  
  // Compute total error in the force due to the B(div B) term
  // only slight worry with this is that fmag is calculated in rates, whilst
  // B has been evolved a bit further since then. A possible solution is to
  // evaluate these quantities just after the call to rates.
  if (forcemagi > 1.e-8 && Bmagi > 1e-8)
    force_erri = ABS(fmag(:,i)⋅Bi(:))/(forcemagi*Bmagi);
  else
    force_erri = 0.0;
  force_err_av = force_err_av + force_erri;
  if (force_erri > force_err_max) force_err_max = force_erri;

  // |div B| x smoothing length / |B| (see e.g. Cerqueira and Gouveia del Pino 1999) 
  // this quantity should be less than ~0.01.
  if (Bmagi < 1e-8)
    omegamhdi = 0.0;
  else
    omegamhdi = divBi*hh(i)/Bmagi;
  if (omegamhdi < omegtol) fracdivBok = fracdivBok + 1.0;
  if (omegamhdi > omegamhdmax) omegamhdmax = omegamhdi;
  omegamhdav = omegamhdav + omegamhdi;

  // Conserved magnetic flux (int B dV)
  pmassi = pmass;
  fluxtot += pmassi*Brhoi;

  // Conserved Cross Helicity (int v.B dV)
  crosshel += pmassi*(veli⋅Brhoi);
}


void writeEvFile(void){
  etot = ekin + emag + epot;
  if (iprterm >= 0 || iprterm < -1) etot = etot + etherm;
  momtot = sqrt(mom⋅mom);
  call minmaxave(rho(1:npart),rhomin,rhomax,rhomean,npart);
  angtot = sqrt(ang⋅ang);

  //write line to .ev file  
  if (imhd != 0){
    fluxtotmag = SQRT(DOT_PRODUCT(fluxtot,fluxtot));
    betamhdav = betamhdav/FLOAT(npart);
    fracdivBok = 100.*fracdivBok/FLOAT(npart);
    omegamhdav = omegamhdav/FLOAT(npart);
    divBav = divBav/FLOAT(npart);
    fdotBav = fdotBav/FLOAT(npart);
    force_err_av = force_err_av/FLOAT(npart);
    info()<<"t="<<t<<" emag ="<<emag<<" etot = "<<etot
          <<" ekin = "<<ekin<<" etherm = "<<etherm;
    // write(ievfile,30) t,ekin,etherm,emag,epot,etot,momtot,angtot,rhomax,rhomean,dt, emagp,crosshel,betamhdmin,betamhdav, divBav,divBmax,divBtot, fdotBav,FdotBmax,force_err_av,force_err_max, omegamhdav,omegamhdmax,fracdivBok
  }else{
    //!alphatstarav = alphatstarav/float(npart)
    //!betatstarav = betatstarav/float(npart)
    //!! print*,'t=',t,' emag =',emag,' etot = ',etot, 'ekin = ',ekin,' etherm = ',etherm
    if (idust == 1){
                      //write(ievfile,40) t,ekin,etherm,emag,epot,etot,momtot,angtot,rhomax,rhomean,dt,ekiny, totmassgas,totmassdust    
    }else{
      //write(ievfile,40) t,ekin,etherm,emag,epot,etot,momtot,angtot,rhomax,rhomean,dt,ekiny
    }
  }
}
*/
