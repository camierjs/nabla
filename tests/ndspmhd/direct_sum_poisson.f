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
//----------------------------------------------------------------------------
// Calculates the 1D solution to any Poisson equation
//
// \nabla^2 \phi = \eta 
//
// by a direct summation over the particles. 
//
// Use this to check the accuracy of the tree code
//
// Input: 
//
//   x(ndim,ntot)  : co-ordinates of the particles
//   source(ntot)  : quantity to be summed over 
//                   for an arbitrary quantity \eta, this is given by
//                   source = particle mass * eta / (4.*pi*density)
//               ie. source = particle mass for gravity
//
// Output:
//
//   phitot          : total potential phi
//   gradphi(ndim,ntot) : gradient of the potential (for gravity this = force)
//----------------------------------------------------------------------------
∀ particles void direct_sum_poisson1D(const Real3* x,
                                      const Real* source,
                                      Real* phitot,
                                      Real3* gradphi,
                                      const Integer ntot){
  //reset forces initially
  phi = 0.0;
  gradphi=0.0;;
  // calculate gravitational force by direct summation
  Real sourcei=source[i];
  ∀ /*j*/ particle{
    if (this==j) continue;
    dx = x[i]-x[j];
    gradphi+=source[j]*dx;
  }
  phitot=0.0;
  }


∀ /*∀ i*/ particles void direct_sum_poisson2D(const Real3* x,
                                              const Real* source,
                                              Real* phitot,
                                              Real3* gradphi,
                                              const Integer ntot){
  //reset forces initially
  phi = 0.0;
  gradphi=0.0;;
  // calculate gravitational force by direct summation
  Real sourcei=source[i];
  ∀ /*j*/ particle{
    if (this!=j){
      Real3 dx = x[i]-x[j];
      Real3 rij2 = dot(dx,dx);
      Real3 term = dx/rij2;
      gradphi[i]+=source[j]*term;
      gradphi[j]+=sourcei*term;
    }
  }
  phitot=0.0;
}


∀ /*∀ i*/ particles void direct_sum_poisson3D(const Real3* x,
                                              const Real* source,
                                              Real* phitot,
                                              Real3* gradphi,
                                              const Integer ntot){
  //reset forces initially
  phi = 0.0;
  // calculate gravitational force by direct summation
  ∀ /*j*/ particle{
    if (this!=j){
      Real3 dx = x[i]-x[j];
      Real3 rij2 = √(dot(dx,dx) + psoft²);
      Real rij = √rij2;
      Real3 term = dx/(rij*rij2);
      phi[i]-=source[j]/rij;
      phi[j]-=sourcei/rij;
      gradphi[i]-=source[j]*term;
      gradphi[j]+=source[i]*term;
    }
  }
  phitot=0.0;
}

∀ particles void direct_sum_poisson3D(void){
  phitot += ½*source*phi;
}
