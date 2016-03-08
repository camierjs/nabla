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
#ifndef _LIB_STD_GATHER_H_
#define _LIB_STD_GATHER_H_


// ****************************************************************************
// * That is for now for Lambda
// ****************************************************************************
inline real rgatherk(const int a, const real *data){
  return *(data+a);
}
inline real3 rgather3k(const int a, const real3 *data){
  const double *p=(double*)data;
  return real3(p[3*a+0],p[3*a+1],p[3*a+2]);
}
inline real3x3 rgather3x3k(const int a, const real3x3 *data){
  const real3 *p=(real3*)data;
  return real3x3(p[3*a+0],p[3*a+1],p[3*a+2]);
}

inline real rGatherAndZeroNegOnes(const int a, const real *data){
  if (a>=0) return *(data+a);
  return 0.0;
}
inline real3 rGatherAndZeroNegOnes(const int a, const real3 *data){
  if (a>=0) return *(data+a);
  return 0.0;
}
inline real3x3 rGatherAndZeroNegOnes(const int a,const  real3x3 *data){
  if (a>=0) return *(data+a);
  return real3x3(0.0);
}
inline real3 rGatherAndZeroNegOnes(const int a, const int corner, const real3 *data){
  const int i=3*8*a+3*corner;
  const double *p=(double*)data;
  if (a>=0) return real3(p[i+0],p[i+1],p[i+2]);
  return 0.0;
}


// ****************************************************************************
// * The rest for Kokkos
// ****************************************************************************


// ****************************************************************************
// * Gather: (X is the data @ offset x)       a            b       c   d
// * data:   |....|....|....|....|....|....|..A.|....|....|B...|...C|..D.|....|      
// * gather: |ABCD|
// ****************************************************************************
inline void gatherk_load(const int a, real *data, real *gthr){
  *gthr=*(data+a);
}
inline void gatherk(const int a, real *data, real *gthr){
  gatherk_load(a,data,gthr);
}

inline real gatherk_and_zero_neg_ones(const int a, real *data){
  if (a>=0) return *(data+a);
  return 0.0;
}
inline real gatherk_and_zero_neg_ones(const int a, const real *data){
  if (a>=0) return *(data+a);
  return 0.0;
}

inline void gatherFromNode_k(const int a, real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,data);
}
inline void gatherFromFace_k(const int a, real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,data);
}

inline real3 gather3k_and_zero_neg_ones(const int a, real3 *data){
  if (a>=0) return *(data+a);
  return 0.0;
}
inline void gatherFromFaces_3k(const int a, real3 *data, real3 *gthr){
  *gthr=gather3k_and_zero_neg_ones(a,data);
}

inline real3x3 gather3x3k_and_zero_neg_ones(const int a, real3x3 *data){
  if (a>=0) return *(data+a);
  return real3x3(0.0);
}
inline void gatherFromFaces_3x3k(const int a, real3x3 *data, real3x3 *gthr){
  *gthr=gather3x3k_and_zero_neg_ones(a,data);
}

// ****************************************************************************
// * Gather avec des real3
// ****************************************************************************
inline void gather3ki(const int a, real3 *data, real3 *gthr, int i){
  double *p=(double *)data;
  double value=p[3*a+i];
  if (i==0) (*gthr).x=value;
  if (i==1) (*gthr).y=value;
  if (i==2) (*gthr).z=value;
}
inline void gather3k(const int a, real3 *data, real3 *gthr){
  gather3ki(a, data, gthr, 0);
  gather3ki(a, data, gthr, 1);
  gather3ki(a, data, gthr, 2);
}

// ****************************************************************************
// * Gather avec des real3x3
// ****************************************************************************
inline void gather3x3ki(const int a, real3x3 *data, real3x3 *gthr, int i){
  real3 *p=(real3 *)data;
  real3 value=p[3*a+i];
  if (i==0) (*gthr).x=value;
  if (i==1) (*gthr).y=value;
  if (i==2) (*gthr).z=value;
}
inline void gather3x3k(const int a, real3x3 *data, real3x3 *gthr){
  gather3x3ki(a, data, gthr, 0);
  gather3x3ki(a, data, gthr, 1);
  gather3x3ki(a, data, gthr, 2);
}

// ****************************************************************************
// * Gather avec des real3[nodes(#8)]
// ****************************************************************************
inline void gatherFromNode_3kiArray8(const int a, const int corner,
                                     real3 *data, real3 *gthr, int i){
  //debug()<<"gather3ki, i="<<i;
  double *p=(double *)data;
  double value=(a<0)?0.0:p[3*8*a+3*corner+i];
  if (i==0) (*gthr).x=value;
  if (i==1) (*gthr).y=value;
  if (i==2) (*gthr).z=value;
}
inline void gatherFromNode_3kArray8(const int a, const int corner,
                                    real3 *data, real3 *gthr){
  //debug()<<"gather3k";
  gatherFromNode_3kiArray8(a,corner, data, gthr, 0);
  gatherFromNode_3kiArray8(a,corner, data, gthr, 1);
  gatherFromNode_3kiArray8(a,corner, data, gthr, 2);
  //debug()<<"gather3k done";
}

#endif //  _LIB_STD_GATHER_H_

