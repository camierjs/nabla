#ifndef _KN_AVX2_GATHER_H_
#define _KN_AVX2_GATHER_H_

std::ostream& operator<<(std::ostream &os, const __m256d v);

// *****************************************************************************
// * Gather: (X is the data @ offset x)       a            b       c   d
// * data:   |....|....|....|....|....|....|..A.|....|....|B...|...C|..D.|....|      
// * gather: |ABCD|
// *****************************************************************************
inline void gatherk(const int a, const int b,
                    const int c, const int d,
                    const real *data,
                    real *gather){
  const __m128i index= _mm_set_epi32(d,c,b,a);
  *gather = _mm256_i32gather_pd((double*)data, index, _MM_SCALE_8);
}

inline __m256d returned_gatherk(const int a, const int b,
                                const int c, const int d,
                                const double *base){
  const __m128i vindex= _mm_set_epi32(d,c,b,a);
  return _mm256_i32gather_pd(base, vindex, _MM_SCALE_8);
}


// *****************************************************************************
// * We are going to use this one:
// * extern __m256d _mm256_mask_i32gather_pd(__m256d def_vals,
// * double const * base, __m128i vindex __m256d vmask, const int scale);
// *****************************************************************************
inline __m256d masked_gather_pd(const int a, const int b,
                                const int c, const int d,
                                const double *base, const int s){
  // base: address used to reference the loaded FP elements
  // the vector of double-precision FP values copied to the destination
  // when the corresponding element of the double-precision FP mask is '0'
  const __m256d def_vals =_mm256_setzero_pd();
  // the vector of dword indices used to reference the loaded FP elements.
  const __m128i vindex = _mm_set_epi32(d,c,b,a);
  // the vector of FP elements used as a vector mask;
  // only the most significant bit of each data element is used as a mask.
  const __m256d vmask = _mm256_cmp_pd(_mm256_set_pd(d,c,b,a), _mm256_setzero_pd(), _CMP_GE_OQ);
  // 32-bit scale used to address the loaded FP elements.
  const int scale = s;
  // Gathers 4 packed double-precision floating point values from memory referenced by the given base address,
  // dword indices and scale, and using the given double-precision FP mask values. 
  const __m256d mdcbat = _mm256_mask_i32gather_pd(def_vals, base, vindex, vmask, scale);
  return mdcbat;
}

inline __m256d gatherk_and_zero_neg_ones(const int a, const int b,
                                         const int c, const int d,
                                         real *data){
  return masked_gather_pd(a,b,c,d,(double *)data,_MM_SCALE_8);
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
  const double *base=(double *)data;
  const int ia = ((3*WARP_BASE(a)+i)<<2)+WARP_OFFSET(a);
  const int ib = ((3*WARP_BASE(b)+i)<<2)+WARP_OFFSET(b);
  const int ic = ((3*WARP_BASE(c)+i)<<2)+WARP_OFFSET(c);
  const int id = ((3*WARP_BASE(d)+i)<<2)+WARP_OFFSET(d);
  const __m256d dcbag = returned_gatherk(ia,ib,ic,id,base);
  if (i==0) gthr->x=dcbag;
  if (i==1) gthr->y=dcbag;
  if (i==2) gthr->z=dcbag;
}

inline void gather3k(const int a, const int b,
                     const int c, const int d,
                     real3 *data, real3 *gthr){
  gather3ki(a,b,c,d,data,gthr,0);
  gather3ki(a,b,c,d,data,gthr,1);
  gather3ki(a,b,c,d,data,gthr,2);
}


// ******************************************************************************
// *
// ******************************************************************************
inline void gatherFromNode_3kiArray8(const int a, const int a_corner,
                                     const int b, const int b_corner,
                                     const int c, const int c_corner,
                                     const int d, const int d_corner,
                                     real3 *data, real3 *gthr, int i){  
  const double *base=(double *)data;
  const int ia = a<0?a:(((3*WARP_BASE(a)<<3)+3*a_corner+i)<<2)+WARP_OFFSET(a);
  const int ib = b<0?b:(((3*WARP_BASE(b)<<3)+3*b_corner+i)<<2)+WARP_OFFSET(b);
  const int ic = c<0?c:(((3*WARP_BASE(c)<<3)+3*c_corner+i)<<2)+WARP_OFFSET(c);
  const int id = d<0?d:(((3*WARP_BASE(d)<<3)+3*d_corner+i)<<2)+WARP_OFFSET(d);
  const __m256d dcbag = masked_gather_pd(ia,ib,ic,id,base,_MM_SCALE_8);
  if (i==0) gthr->x=dcbag;
  if (i==1) gthr->y=dcbag;
  if (i==2) gthr->z=dcbag;
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

#endif //  _KN_AVX2_GATHER_H_
