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
#ifndef _KN_AVX_GATHER_H_
#define _KN_AVX_GATHER_H_
std::ostream& operator<<(std::ostream &os, const __m256d v);
//std::ostream& operator<<(std::ostream &os, const Real &a);

/******************************************************************************
 * Gather: (X is the data @ offset x)       a            b       c   d
 * data:   |....|....|....|....|....|....|..A.|....|....|B...|...C|..D.|....|      
 * gather: |ABCD|
 ******************************************************************************/
inline void gatherk_load(const int a, const int b,
                         const int c, const int d,
                         real *data, real *gthr){
  double *p=(double*)data;
  __m128d bData=_mm_movedup_pd(_mm_load_sd(&p[b])); 
  __m256d ba=_mm256_castpd128_pd256(_mm_loadl_pd(bData,&p[a]));
  __m128d dData=_mm_movedup_pd(_mm_load_sd(&p[d]));
  __m128d dc=_mm_loadl_pd(dData,&p[c]);
  __m256d dcba=_mm256_insertf128_pd(ba,dc,0x01);
  (*gthr)=dcba;
}


inline void gatherk_bcast(const int a, const int b,
                          const int c, const int d,
                          real *data, real *gthr){
  double *p=(double*)data;
  __m256d aData=_mm256_broadcast_sd(&p[a]);
  __m256d bData=_mm256_broadcast_sd(&p[b]);
  __m256d cData=_mm256_broadcast_sd(&p[c]);
  __m256d dData=_mm256_broadcast_sd(&p[d]);
  __m256d ba=_mm256_blend_pd(aData,bData,0xA);
  __m256d dc=_mm256_blend_pd(cData,dData,0xA);
  __m256d dcba=_mm256_blend_pd(ba,dc,0xC);
  (*gthr)=dcba;
}


inline void gatherk(const int a, const int b,
                    const int c, const int d,
                    real *data, real *gthr){
  gatherk_load(a,b,c,d,data,gthr);
  //gatherk_bcast(a,b,c,d,data,gthr);
}


inline __m256d gatherk_and_zero_neg_ones(const int a, const int b,
                                         const int c, const int d,
                                         real *data){
  double *p=(double*)data;
  const __m256d zero256=_mm256_set1_pd(0.0);
  const __m128d bData=_mm_movedup_pd(_mm_load_sd(&p[(b<0)?0:b])); 
  const __m256d ba=_mm256_castpd128_pd256(_mm_loadl_pd(bData,&p[(a<0)?0:a]));
  const __m128d dData=_mm_movedup_pd(_mm_load_sd(&p[(d<0)?0:d]));
  const __m128d dc=_mm_loadl_pd(dData,&p[(c<0)?0:c]);
  const __m256d dcba=_mm256_insertf128_pd(ba,dc,0x01);
  //const __m256d dcbat=opTernary(_mm256_cmp_pd(_mm256_set_pd(d,c,b,a), zero256, _CMP_GE_OQ), dcba, zero256);
  const __m256d mask = _mm256_cmp_pd(_mm256_set_pd(d,c,b,a), zero256, _CMP_GE_OQ);
  const __m256d dcbat=opTernary(mask,dcba,zero256);
  //info()<<"a="<<a<<", b="<<b<<", c="<<c<<", d="<<d<<", dcba="<<dcba<<", dcbat="<<dcbat;
  //debug()<<"_mm256_set_pd(d,c,b,a)="<<_mm256_set_pd(d,c,b,a);
  return dcbat;
}

inline void gatherFromNode_k(const int a, const int b,
                             const int c, const int d,
                             real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,b,c,d,data);
}


// ******************************************************************************
// * Gather avec des real3
// ******************************************************************************
inline void gather3ki(const int a, const int b,
                      const int c, const int d,
                      real3 *data, real3 *gthr,
                      const int i){
  const double *p=(double *)data;
  const __m256d aData=_mm256_broadcast_sd(&(p[4*(3*WARP_BASE(a)+i)+WARP_OFFSET(a)]));
  const __m256d bData=_mm256_broadcast_sd(&(p[4*(3*WARP_BASE(b)+i)+WARP_OFFSET(b)]));
  const __m256d cData=_mm256_broadcast_sd(&(p[4*(3*WARP_BASE(c)+i)+WARP_OFFSET(c)]));
  const __m256d dData=_mm256_broadcast_sd(&(p[4*(3*WARP_BASE(d)+i)+WARP_OFFSET(d)]));
  const __m256d ba=_mm256_blend_pd(aData,bData,0xA);
  const __m256d dc=_mm256_blend_pd(cData,dData,0xA);
  const __m256d dcba=_mm256_blend_pd(ba,dc,0xC);
  if (i==0) (*gthr).x=dcba;
  if (i==1) (*gthr).y=dcba;
  if (i==2) (*gthr).z=dcba;
}

inline void gather3k(const int a, const int b,
                     const int c, const int d,
                     real3 *data, real3 *gthr){
  gather3ki(a,b,c,d, data, gthr,0);
  gather3ki(a,b,c,d, data, gthr,1);
  gather3ki(a,b,c,d, data, gthr,2);
}

// ******************************************************************************
// *
// ******************************************************************************
inline void gatherFromNode_3kiArray8(const int a, const int a_corner,
                                     const int b, const int b_corner,
                                     const int c, const int c_corner,
                                     const int d, const int d_corner,
                                     real3 *data, real3 *gthr, int i){
  const __m256d zero256=_mm256_set1_pd(0.0);
  double *p=(double *)data;
  __m256d aData=a<0?zero256:_mm256_broadcast_sd(&(p[4*(3*8*WARP_BASE(a)+3*a_corner+i)+WARP_OFFSET(a)]));
  __m256d bData=b<0?zero256:_mm256_broadcast_sd(&(p[4*(3*8*WARP_BASE(b)+3*b_corner+i)+WARP_OFFSET(b)]));
  __m256d cData=c<0?zero256:_mm256_broadcast_sd(&(p[4*(3*8*WARP_BASE(c)+3*c_corner+i)+WARP_OFFSET(c)]));
  __m256d dData=d<0?zero256:_mm256_broadcast_sd(&(p[4*(3*8*WARP_BASE(d)+3*d_corner+i)+WARP_OFFSET(d)]));
  __m256d ba=_mm256_blend_pd(aData,bData,0xA);
  __m256d dc=_mm256_blend_pd(cData,dData,0xA);
  __m256d dcba=_mm256_blend_pd(ba,dc,0xC);
  if (i==0) (*gthr).x=dcba;
  if (i==1) (*gthr).y=dcba;
  if (i==2) (*gthr).z=dcba;
}

inline void gatherFromNode_3kArray8(const int a, const int a_corner,
                                    const int b, const int b_corner,
                                    const int c, const int c_corner,
                                    const int d, const int d_corner,
                                    real3 *data, real3 *gthr){
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,data,gthr,0);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,data,gthr,1);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,data,gthr,2);
}

#endif //  _KN_AVX_GATHER_H_
