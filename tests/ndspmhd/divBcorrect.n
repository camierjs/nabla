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

// * Interface subroutine for the div B correction by the projection method

Real dpi(void){
  if (ndim==1) return   pi; // 1D
  if (ndim==2) return ½*pi; // 2D
  if (ndim==3) return ¼*pi; // 3D
  //fatal();
}


∀ particles void divBcorrect_StandardProjectionMethod_ScalarPotential(void){
  // get neighbours and calculate density if required
  iterate_density();
  // calculate div B source term for poisson equation
  get_divB();
  divB = rho*divBonrho;
  // specify the source term for the Poisson equation
  source = pmass*divBonrho*dpi;
  // calculate the correction to the magnetic field
  direct_sum_poisson(x,source,phi,gradphi,npart);
  // correct the magnetic field
  Bfield -= gradphi;
  if (imhd>=11) Bevol = Bfield;
  else Bevol = Bfield/rho;
  
}


∀ particles void divBcorrect_CurrentProjectionMethod_VectorPotential(void){
  // get neighbours and calculate density if required
  iterate_density();
  // calculate div B source term for poisson equation
  get_curl();
  // specify the source term for the Poisson equation
  sourcevec = pmass*curlB/rho*dpi;
  // calculate the correction to the magnetic field
  direct_sum_poisson_vec(x,sourcevec,curlA,npts);
  // correct the magnetic field
  Bfield = curlA;
  if (imhd>=11) Bevol = Bfield;
  else Bevol = Bfield/rho;
}
