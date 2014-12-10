// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
