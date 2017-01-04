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
#ifndef _KN_AVX_REAL_H_
#define _KN_AVX_REAL_H_
  
// ****************************************************************************
// * real
// ****************************************************************************
struct __attribute__ ((aligned(32))) real {
 protected:
  __attribute__ ((aligned(32))) __m256d vec;
 public:
  // Constructors
  inline real(): vec(_mm256_setzero_pd()){}
  inline real(int i):vec(_mm256_set1_pd((double)i)){}
  inline real(Integer i):vec(_mm256_set_pd(i[3],i[2],i[1],i[0])){}
  inline real(long i):vec(_mm256_set1_pd((double)i)){}
  inline real(double d):vec(_mm256_set1_pd(d)){}
  inline real(__m256d x):vec(x){}
  inline real(double *x):vec(_mm256_load_pd(x)){}
  inline real(double d0, double d1, double d2, double d3):vec(_mm256_set_pd(d3,d2,d1,d0)){}

  // Convertors
  inline operator __m256d() const { return vec; }
  
  // Arithmetics
  friend inline real operator +(const real &a, const real &b) { return _mm256_add_pd(a,b); }
  friend inline real operator -(const real &a, const real &b) { return _mm256_sub_pd(a,b); }
  friend inline real operator *(const real &a, const real &b) { return _mm256_mul_pd(a,b); }
  // CLANG has a built-in candidate:
  // operator/(__attribute__((__vector_size__(4 * sizeof(double)))) double,
  //           __attribute__((__vector_size__(4 * sizeof(double)))) double)
#ifndef __clang_major__
  friend inline real operator /(const real &a, const real &b) { return _mm256_div_pd(a,b); }
#endif

  
  inline real& operator +=(const real &a) { return *this = _mm256_add_pd(vec,a); }
  inline real& operator -=(const real &a) { return *this = _mm256_sub_pd(vec,a); }
  inline real& operator *=(const real &a) { return *this = _mm256_mul_pd(vec,a); }
  inline real& operator /=(const real &a) { return *this = _mm256_div_pd(vec,a); }
  
  inline real operator -() const { return _mm256_xor_pd (_mm256_set1_pd(-0.0), *this); }

  // Mixed vector-scalar operations
  inline real& operator *=(const double &f) { return *this = _mm256_mul_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator /=(const double &f) { return *this = _mm256_div_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator +=(const double &f) { return *this = _mm256_add_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator +=(double &f) { return *this = _mm256_add_pd(vec,_mm256_set1_pd(f)); }
  inline real& operator -=(const double &f) { return *this = _mm256_sub_pd(vec,_mm256_set1_pd(f)); }
  
  friend inline real operator +(const real &a, const double &f) { return _mm256_add_pd(a, _mm256_set1_pd(f)); }
  friend inline real operator -(const real &a, const double &f) { return _mm256_sub_pd(a, _mm256_set1_pd(f)); } 
  friend inline real operator *(const real &a, const double &f) { return _mm256_mul_pd(a, _mm256_set1_pd(f)); } 
  friend inline real operator /(const real &a, const double &f) { return _mm256_div_pd(a, _mm256_set1_pd(f)); }

  friend inline real operator +(const double &f, const real &a) { return _mm256_add_pd(_mm256_set1_pd(f),a); }
  friend inline real operator -(const double &f, const real &a) { return _mm256_sub_pd(_mm256_set1_pd(f),a); } 
  friend inline real operator *(const double &f, const real &a) { return _mm256_mul_pd(_mm256_set1_pd(f),a); } 
  friend inline real operator /(const double &f, const real &a) { return _mm256_div_pd(_mm256_set1_pd(f),a); }

  friend inline real sqrt(const real &a) { return _mm256_sqrt_pd(a); }
  friend inline real ceil(const real &a)   { return _mm256_round_pd((a), _MM_FROUND_CEIL); }
  friend inline real floor(const real &a)  { return _mm256_round_pd((a), _MM_FROUND_FLOOR); }
  friend inline real trunc(const real &a)  { return _mm256_round_pd((a), _MM_FROUND_TO_ZERO); }
  
  friend inline real min(const real &r, const real &s){ return _mm256_min_pd(r,s);}
  friend inline real max(const real &r, const real &s){ return _mm256_max_pd(r,s);}

  /* Round */
  //friend real round(const real &a)  { return _mm256_svml_round_pd(a); }
  
  friend inline real cube_root(const real &a){
    return real(::cbrt(a[0]),::cbrt(a[1]),::cbrt(a[2]),::cbrt(a[3]));
    //return _mm256_cbrt_pd(a);
  }
  
  friend inline real norm(real a){
    return real(::fabs(a[0]),::fabs(a[1]),::fabs(a[2]),::fabs(a[3]));
  }

  /* Compares: Mask is returned  */
  friend inline real cmp_eq(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_EQ_OS); }
  friend inline real cmp_lt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
  friend inline real cmp_le(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }
  friend inline real cmp_gt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_GT_OS); }
  friend inline real cmp_ge(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_GE_OS); }
  friend inline real cmp_neq(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NEQ_US); }
  friend inline real cmp_nlt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NLT_US); }
  friend inline real cmp_nle(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NLE_US); }
  friend inline real cmp_ngt(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NGT_US); }
  friend inline real cmp_nge(const real &a, const real &b)  { return _mm256_cmp_pd(a, b, _CMP_NGE_US); }

  friend inline real operator<(const real &a, const real& b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
  friend inline real operator<(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_LT_OS); }


  friend inline real operator>(const real &a, real& r) { return _mm256_cmp_pd(a, r, _CMP_GT_OS); }
  friend inline real operator>(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_GT_OS); }
  friend inline real operator>(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_GT_OS); }

  friend inline real operator>=(const real &a, real& r) { return _mm256_cmp_pd(a, r, _CMP_GE_OS); }
  friend inline real operator>=(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_GE_OS); }
  
  friend inline real operator<=(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_LE_OS); }
  friend inline real operator<=(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_LE_OS); }

  friend inline real operator==(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_EQ_OQ); }
  friend inline real operator==(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_EQ_OQ); }
  
  friend inline real operator!=(const real &a, const real& r) { return _mm256_cmp_pd(a, r, _CMP_NEQ_UQ); }
  friend inline real operator!=(const real &a, double d) { return _mm256_cmp_pd(a, _mm256_set1_pd(d), _CMP_NEQ_UQ); }

  /*friend real unglitch(const real &a)  {
    const union{
      unsigned long long i[4];
      __m256d m;
    } __f64vec4_abs_mask = { 0xffffffffffff0000ull,
                             0xffffffffffff0000ull,
                             0xffffffffffff0000ull,
                             0xffffffffffff0000ull};
    return _mm256_and_pd(a, __f64vec4_abs_mask.m);
    }*/
  inline const double& operator[](int i) const  {
    double *d = (double*)&vec;
    return *(d+i);
  }
  inline double& operator[](int i) {
    double *d = (double*)&vec;
    return *(d+i);
  }
};

inline double ReduceMinToDouble(real r){
  double mnx[2];
  const double *dtv = (double*) &r;
  mnx[0]=dtv[0]>dtv[1]?dtv[1]:dtv[0];
  mnx[1]=dtv[2]>dtv[3]?dtv[3]:dtv[2];
  return mnx[0]>mnx[1]?mnx[1]:mnx[0];
}

inline double ReduceMaxToDouble(real r){
  double mnx[2];
  const double *dtv = (double*) &r;
  mnx[0]=dtv[0]>dtv[1]?dtv[0]:dtv[1];
  mnx[1]=dtv[2]>dtv[3]?dtv[2]:dtv[3];
  return mnx[0]>mnx[1]?mnx[0]:mnx[1];
}


#endif //  _KN_AVX_REAL_H_
