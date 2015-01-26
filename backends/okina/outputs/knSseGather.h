///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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
#ifndef _KN_SSE_GATHER_H_
#define _KN_SSE_GATHER_H_


// ******************************************************************************
// * Gather: (X is the data @ offset x)
//*                              a       b
// * data:   |..|..|..|..|..|..|.A|..|..|B.|..|..|..|      
// * gather: |AB|
// ******************************************************************************
inline void gatherk_load(const int a, const int b,
                         real *data, real *gthr){
  double *p=(double*)data;
  (*gthr)=real(p[2*WARP_BASE(a)+WARP_OFFSET(a)],
               p[2*WARP_BASE(b)+WARP_OFFSET(b)]);
}



inline void gatherk(const int a, const int b,
                    real *data, real *gthr){
  gatherk_load(a,b,data,gthr);
}


inline __m128d gatherk_and_zero_neg_ones(const int a, const int b,
                                         real *data){
  double *p=(double*)data;
  double dbl_a=a<0?0.0:p[2*WARP_BASE(a)+WARP_OFFSET(a)];
  double dbl_b=b<0?0.0:p[2*WARP_BASE(b)+WARP_OFFSET(b)];
  return real(dbl_a, dbl_b);
}

inline void gatherFromNode_k(const int a, const int b,
                             real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,b,data);
}


/******************************************************************************
 * Gather avec des real3
 ******************************************************************************/
inline void gather3ki(const int a, const int b,
                      real3 *data, real3 *gthr,
                      int i){
  double *p=(double*)data;
  const real ba=real(p[2*(3*WARP_BASE(a)+i)+WARP_OFFSET(a)],
                     p[2*(3*WARP_BASE(b)+i)+WARP_OFFSET(b)]);
  if (i==0) gthr->x=ba;
  if (i==1) gthr->y=ba;
  if (i==2) gthr->z=ba;
}

inline void gather3k(const int a, const int b,
                     real3 *data, real3 *gthr){
  gather3ki(a,b, data, gthr,0);
  gather3ki(a,b, data, gthr,1);
  gather3ki(a,b, data, gthr,2);
}


/******************************************************************************
 * Gather avec des real3[nodes(#8)]
 ******************************************************************************/
inline void gatherFromNode_3kiArray8(const int a, const int a_corner,
                                     const int b, const int b_corner,
                                     real3 *data, real3 *gthr, int i){
  double *p=(double*)data;
  const real ba=real((a>=0)?p[2*(3*8*WARP_BASE(a)+3*a_corner+i)+WARP_OFFSET(a)]:0.0,
                     (b>=0)?p[2*(3*8*WARP_BASE(b)+3*b_corner+i)+WARP_OFFSET(b)]:0.0);
  if (i==0) gthr->x=ba;
  if (i==1) gthr->y=ba;
  if (i==2) gthr->z=ba;
}

inline void gatherFromNode_3kArray8(const int a, const int a_corner,
                                    const int b, const int b_corner,
                                    real3 *data, real3 *gthr){
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner, data, gthr,0);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner, data, gthr,1);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner, data, gthr,2);
}


#endif //  _KN_SSE_GATHER_H_
