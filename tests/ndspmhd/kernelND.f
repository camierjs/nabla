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



//----------------------------------------------------------------------
// function to interpolate linearly from kernel tables
// returns kernel and derivative given q^2 = (r_a-r_b)^2/h^2
// must then divide returned w, grad w by h^ndim, h^ndim+1 respectively
//----------------------------------------------------------------------
void interpolate_kernelND(const Real q2, Real* w, Real* gradw){
  Integer index,index1;
  Real dxx,dwdx,dgrwdx;
  //--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
  //--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
  //--calculate slope for w, gradw and interpolate for each
  dwdx =  (wij(index1)-wij(index))*ddq2table;
  w = (wij(index)+ dwdx*dxx);
  dgrwdx =  (grwij(index1)-grwij(index))*ddq2table;
  gradw = (grwij(index)+ dgrwdx*dxx);
}


//----------------------------------------------------------------------
// function to interpolate linearly from drag kernel tables
//----------------------------------------------------------------------
void  interpolate_kernelNDdrag(const Real q2, Real* w){
  Integer index,index1;
  Real dxx,dwdx;
  //--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
  //--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
  //--calculate slope for w and interpolate for each
  dwdx =  (wijdrag(index1)-wijdrag(index))*ddq2table;
  w = (wijdrag(index)+ dwdx*dxx);
}


//----------------------------------------------------------------------
// same but for kernel *and* modified kernel in anticlumping term
//----------------------------------------------------------------------
void interpolate_kernelNDs(const Real q2,
                         Real* w, Real* gradw, Real* gradwalt, Real* gradgradwalt){
  Integer index,index1;
  Real dxx,dwdx,dgrwdx,dgrwaltdx,dgrgrwaltdx;
//--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
//--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
//--calculate slope for w, gradw, waniso, gradwaniso
//  and interpolate for each
  w = wij(index);
  dwdx =  (wij(index1)-w)*ddq2table;
  w = w + dwdx*dxx;

  gradw = grwij(index);
  dgrwdx =  (grwij(index1)-gradw)*ddq2table;
  gradw = gradw + dgrwdx*dxx;
//--interpolate for alternative kernel and derivative
// walt = wijalt(index)
// dwaltdx =  (wijalt(index1)-walt)*ddq2table
// walt = walt + dwaltdx*dxx

  gradwalt = grwijalt(index);
  dgrwaltdx =  (grwijalt(index1)-gradwalt)*ddq2table;
  gradwalt = gradwalt + dgrwaltdx*dxx;

  gradgradwalt = grgrwijalt(index);
  dgrgrwaltdx =  (grgrwijalt(index1)-gradgradwalt)*ddq2table;
  gradgradwalt = gradgradwalt + dgrgrwaltdx*dxx;
}


//----------------------------------------------------------------------
// same but for kernel *and* modified kernel in anticlumping term
//----------------------------------------------------------------------
void interpolate_kernelNDs2(const Real q2, Real* w, Real* walt, Real* gradw, Real* gradwalt){
  Integer index,index1;
  Real dxx,dwdx,dwaltdx,dgrwdx,dgrwaltdx;
//--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
//--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
//--calculate slope for w, gradw, waniso, gradwaniso
//  and interpolate for each
  w = wij(index);
  dwdx =  (wij(index1)-w)*ddq2table;
  w = w + dwdx*dxx;

  gradw = grwij(index);
  dgrwdx =  (grwij(index1)-gradw)*ddq2table;
  gradw = gradw + dgrwdx*dxx;
//--interpolate for alternative kernel and derivative
  walt = wijalt(index);
  dwaltdx =  (wijalt(index1)-walt)*ddq2table;
  walt = walt + dwaltdx*dxx;

  gradwalt = grwijalt(index);
  dgrwaltdx =  (grwijalt(index1)-gradwalt)*ddq2table;
  gradwalt = gradwalt + dgrwaltdx*dxx;
}


//----------------------------------------------------------------------
// function to interpolate linearly from kernel tables
// returns kernel and derivative given q^2 = (r_a-r_b)^2/h^2
// must then divide returned w, grad w by h^ndim, h^ndim+1 respectively
//----------------------------------------------------------------------
void interpolate_softening(const Real q2, Real* phi, Real* force, Real* gradw){
  Integer index,index1;
  Real dxx,dphidx,dfdx,dgrwdx;
//--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
//--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
//--calculate slope for phi, force
  dphidx =  (potensoft(index1)-potensoft(index))*ddq2table;
  phi = (potensoft(index)+ dphidx*dxx);

  dfdx =  (fsoft(index1)-fsoft(index))*ddq2table;
  force = (fsoft(index)+ dfdx*dxx);

  dgrwdx = (grwij(index1)-grwij(index))*ddq2table;
  gradw = (grwij(index)+ dgrwdx*dxx);
}


//----------------------------------------------------------------------
// function to interpolate linearly from kernel tables
// returns kernel and derivative and dphidh given q^2 = (r_a-r_b)^2/h^2
// must then divide returned w, grad w by h^ndim, h^ndim+1 respectively
//----------------------------------------------------------------------
void interpolate_kernelND_soft(const Real q2, Real* w, Real* gradw, Real* dphidhi){
  Integer index,index1;
  Real dxx,dwdx,dgrwdx,dpotdx;
//--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
//--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
//--linear interpolation
  dwdx =  (wij(index1)-wij(index))*ddq2table;
  w = (wij(index)+ dwdx*dxx);

  dgrwdx =  (grwij(index1)-grwij(index))*ddq2table;
  gradw = (grwij(index)+ dgrwdx*dxx);

  dpotdx =  (dphidh(index1)-dphidh(index))*ddq2table;
  dphidhi = (dphidh(index) + dpotdx*dxx);
}


//----------------------------------------------------------------------
// function to interpolate linearly from kernel tables
// returns kernel and second derivative given q^2 = (r_a-r_b)^2/h^2
// (required in the new densityiterate routine)
// must then divide returned w, grad grad w by h^ndim, h^ndim+2 respectively
//----------------------------------------------------------------------
void interpolate_kernelNDs_dens(const Real q2,
                              Real* w, Real* gradw, Real* gradgradw, Real* walt, Real* gradwalt){
  Integer index,index1;
  Real dxx,dwdx,dgrwdx,dgrgrwdx,dwaltdx,dgrwaltdx;
//--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
//--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
//--calculate slope for w, gradw, gradgradw and interpolate for each
  dwdx =  (wij(index1)-wij(index))*ddq2table;
  w = (wij(index)+ dwdx*dxx);

  dgrwdx =  (grwij(index1)-grwij(index))*ddq2table;
  gradw = (grwij(index)+ dgrwdx*dxx);

  dgrgrwdx =  (grgrwij(index1)-grgrwij(index))*ddq2table;
  gradgradw = (grgrwij(index)+ dgrgrwdx*dxx);
//--interpolate for alternative kernel and derivative
  walt = wijalt(index);
  dwaltdx =  (wijalt(index1)-walt)*ddq2table;
  walt = walt + dwaltdx*dxx;

  gradwalt = grwijalt(index);
  dgrwaltdx =  (grwijalt(index1)-gradwalt)*ddq2table;
  gradwalt = gradwalt + dgrwaltdx*dxx;
}


//----------------------------------------------------------------------
// kernels used in calculating the curl in get_curl.f90
//----------------------------------------------------------------------
void interpolate_kernelND_curl(const Real q2, Real* gradwalt, Real* gradgradwalt){
  Integer index,index1;
  Real dxx,dgrwaltdx,dgrgrwaltdx;
//--find nearest index in kernel table
  index = (int)(q2*ddq2table);
  index1 = index + 1;
  if (index > ikern || index < 0) index = ikern;
  if (index1 > ikern || index1 < 0) index1 = ikern;
//--find increment from index point to actual value of q2
  dxx = q2 - index*dq2table;
//--calculate slope for w, gradw, waniso, gradwaniso
//  and interpolate for each
  gradwalt = grwijalt(index);
  dgrwaltdx =  (grwijalt(index1)-gradwalt)*ddq2table;
  gradwalt = gradwalt + dgrwaltdx*dxx;

  gradgradwalt = grgrwijalt(index);
  dgrgrwaltdx =  (grgrwijalt(index1)-gradgradwalt)*ddq2table;
  gradgradwalt = gradgradwalt + dgrgrwaltdx*dxx;
}
