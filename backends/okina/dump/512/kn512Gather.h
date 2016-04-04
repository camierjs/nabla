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
#ifndef _KN_512_GATHER_H_
#define _KN_512_GATHER_H_

std::ostream& operator<<(std::ostream &os, const __m512d v);

/******************************************************************************
 * Gather: (X is the data @ offset x)       a            b       c   d
 * data:   |....|....|....|....|....|....|..A.|....|....|B...|...C|..D.|....|      
 * gather: |ABCD|
 ******************************************************************************/
inline void gatherk(const int a, const int b, const int c, const int d,
                    const int e, const int f, const int g, const int h,
                    const double* addr, real *gather){
  const __m256i index= _mm256_set_epi32(h,g,f,e,d,c,b,a);
  *gather = _mm512_i32gather_pd(index,addr,8);
}

inline real inlined_gatherk(const int a, const int b, const int c, const int d,
                            const int e, const int f, const int g, const int h,
                            const double* addr){
  const __m256i index= _mm256_set_epi32(h,g,f,e,d,c,b,a);
  return _mm512_i32gather_pd(index,addr,8);
}


// *****************************************************************************
// *****************************************************************************
inline __m512d masked_gather_pd(const int a, const int b,
                                const int c, const int d,
                                const int e, const int f,
                                const int g, const int h,
                                const double *base, const int s){
  // base: address used to reference the loaded FP elements
  // the vector of double-precision FP values copied to the destination
  // when the corresponding element of the double-precision FP mask is '0'
  const __m512d def_vals =_mm512_setzero_pd();
  // the vector of dword indices used to reference the loaded FP elements.
  const __m256i vindex = _mm256_set_epi32(h,g,f,e,d,c,b,a);
  // the vector of FP elements used as a vector mask;
  // only the most significant bit of each data element is used as a mask.
  const __mmask8 vmask = _mm512_cmp_pd_mask(_mm512_set_pd(h,g,f,e,d,c,b,a),
                                            _mm512_setzero_pd(),
                                            _MM_CMPINT_GE);
  // 32-bit scale used to address the loaded FP elements.
  const int scale = s;
  // Gathers 4 packed double-precision floating point values from memory referenced by the given base address,
  // dword indices and scale, and using the given double-precision FP mask values. 
  const __m512d mdcbat = _mm512_mask_i32gather_pd(def_vals, vmask, vindex, base, scale);
  return mdcbat;
}



inline __m512d gatherk_and_zero_neg_ones(const int a, const int b,
                                         const int c, const int d,
                                         const int e, const int f,
                                         const int g, const int h,
                                         real *data){
  return masked_gather_pd(a,b,c,d,e,f,g,h,(double*)data,8);
}

inline void gatherFromNode_k(const int a, const int b,
                             const int c, const int d,
                             const int e, const int f,
                             const int g, const int h,
                             real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,b,c,d,e,f,g,h,data);
}

/******************************************************************************
 * Gather avec des real3
 ******************************************************************************/
inline void gather3ki(const int a, const int b,
                      const int c, const int d,
                      const int e, const int f,
                      const int g, const int h,
                      const real3* data, real3 *gthr,
                      int i){
  const double *base =(double*)data;
  const int ia=8*(3*WARP_BASE(a)+i)+WARP_OFFSET(a);
  const int ib=8*(3*WARP_BASE(b)+i)+WARP_OFFSET(b);
  const int ic=8*(3*WARP_BASE(c)+i)+WARP_OFFSET(c);
  const int id=8*(3*WARP_BASE(d)+i)+WARP_OFFSET(d);
  const int ie=8*(3*WARP_BASE(e)+i)+WARP_OFFSET(e);
  const int jf=8*(3*WARP_BASE(f)+i)+WARP_OFFSET(f);
  const int ig=8*(3*WARP_BASE(g)+i)+WARP_OFFSET(g);
  const int ih=8*(3*WARP_BASE(h)+i)+WARP_OFFSET(h);
  const __m512d hgfedcba=inlined_gatherk(ia,ib,ic,id,ie,jf,ig,ih,base);
  if (i==0) gthr->x=hgfedcba;
  if (i==1) gthr->y=hgfedcba;
  if (i==2) gthr->z=hgfedcba;
}

inline void gather3k(const int a,
                     const int b,
                     const int c,
                     const int d,
                     const int e,
                     const int f,
                     const int g,
                     const int h,
                     real3* addr,
                     real3* gthr){
  gather3ki(a,b,c,d,e,f,g,h,addr,gthr,0);
  gather3ki(a,b,c,d,e,f,g,h,addr,gthr,1);
  gather3ki(a,b,c,d,e,f,g,h,addr,gthr,2);  
  }

// ******************************************************************************
// *
// ******************************************************************************
inline void gatherFromNode_3kiArray8(const int a, const int a_corner,
                                     const int b, const int b_corner,
                                     const int c, const int c_corner,
                                     const int d, const int d_corner,
                                     const int e, const int e_corner,
                                     const int f, const int f_corner,
                                     const int g, const int g_corner,
                                     const int h, const int h_corner,
                                     real3 *data, real3 *gthr, int i){
  const double *base=(double *)data; 
  const int ia=a<0?a:8*(3*8*WARP_BASE(a)+3*a_corner+i)+WARP_OFFSET(a);
  const int ib=b<0?b:8*(3*8*WARP_BASE(b)+3*b_corner+i)+WARP_OFFSET(b);
  const int ic=c<0?c:8*(3*8*WARP_BASE(c)+3*c_corner+i)+WARP_OFFSET(c);
  const int id=d<0?d:8*(3*8*WARP_BASE(d)+3*d_corner+i)+WARP_OFFSET(d);
  const int ie=e<0?e:8*(3*8*WARP_BASE(e)+3*e_corner+i)+WARP_OFFSET(e);
  const int jf=f<0?f:8*(3*8*WARP_BASE(f)+3*f_corner+i)+WARP_OFFSET(f);
  const int ig=g<0?g:8*(3*8*WARP_BASE(g)+3*g_corner+i)+WARP_OFFSET(g);
  const int ih=h<0?h:8*(3*8*WARP_BASE(h)+3*h_corner+i)+WARP_OFFSET(h);
  const __m512d hgfedcba=masked_gather_pd(ia,ib,ic,id,ie,jf,ig,ih,base,8);
  if (i==0) gthr->x=hgfedcba;
  if (i==1) gthr->y=hgfedcba;
  if (i==2) gthr->z=hgfedcba;
}

inline void gatherFromNode_3kArray8(const int a, const int a_corner,
                                    const int b, const int b_corner,
                                    const int c, const int c_corner,
                                    const int d, const int d_corner,
                                    const int e, const int e_corner,
                                    const int f, const int f_corner,
                                    const int g, const int g_corner,
                                    const int h, const int h_corner,
                                    real3 *data, real3 *gthr){
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,
                           e,e_corner,f,f_corner,g,g_corner,h,h_corner,data,gthr,0);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,
                           e,e_corner,f,f_corner,g,g_corner,h,h_corner,data,gthr,1);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,
                           e,e_corner,f,f_corner,g,g_corner,h,h_corner,data,gthr,2);
}

#endif //  _KN_512_GATHER_H_
