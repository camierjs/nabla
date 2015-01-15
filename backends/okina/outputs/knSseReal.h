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
#ifndef _KN_SSE_REAL_H_
#define _KN_SSE_REAL_H_
 

// ****************************************************************************
// * real
// ****************************************************************************
struct __attribute__ ((aligned(16))) real {
 protected:
  __m128d vec;
 public:
  // Constructors
  inline real(): vec(_mm_setzero_pd()){}
  inline real(int i):vec(_mm_set1_pd((double)i)){}
  inline real(Integer i):vec(_mm_set_pd(i[1],i[0])){}
  inline real(long i):vec(_mm_set1_pd((double)i)){}
  inline real(double d):vec(_mm_set1_pd(d)){}
  inline real(__m128d x):vec(x){}
  inline real(double *x):vec(_mm_load_pd(x)){}
  inline real(double d0, double d1):vec(_mm_set_pd(d1,d0)){}

  // Convertors
  inline operator __m128d() const { return vec; }
  
  // Logicals
  friend inline real operator &(const real &a, const real &b) { return _mm_and_pd(a,b); }
  friend inline real operator |(const real &a, const real &b) { return _mm_or_pd(a,b); }
  friend inline real operator ^(const real &a, const real &b) { return _mm_xor_pd(a,b); }

  // Arithmetics
  friend inline real operator +(const real &a, const real &b) { return _mm_add_pd(a,b); }
  friend inline real operator -(const real &a, const real &b) { return _mm_sub_pd(a,b); }
  friend inline real operator *(const real &a, const real &b) { return _mm_mul_pd(a,b); }
  friend inline real operator /(const real &a, const real &b) { return _mm_div_pd(a,b); }

  
  inline real& operator +=(const real &a) { return *this = _mm_add_pd(vec,a); }
  inline real& operator -=(const real &a) { return *this = _mm_sub_pd(vec,a); }
  inline real& operator *=(const real &a) { return *this = _mm_mul_pd(vec,a); }
  inline real& operator /=(const real &a) { return *this = _mm_div_pd(vec,a); }
  inline real& operator &=(const real &a) { return *this = _mm_and_pd(vec,a); }
  inline real& operator |=(const real &a) { return *this = _mm_or_pd(vec,a); }
  inline real& operator ^=(const real &a) { return *this = _mm_xor_pd(vec,a); }
  
  inline real operator -() const { return _mm_xor_pd (_mm_set1_pd(-0.0), *this); }

  // Mixed vector-scalar operations
  inline real& operator *=(const double &f) { return *this = _mm_mul_pd(vec,_mm_set1_pd(f)); }
  inline real& operator /=(const double &f) { return *this = _mm_div_pd(vec,_mm_set1_pd(f)); }
  inline real& operator +=(const double &f) { return *this = _mm_add_pd(vec,_mm_set1_pd(f)); }
  inline real& operator +=(double &f) { return *this = _mm_add_pd(vec,_mm_set1_pd(f)); }
  inline real& operator -=(const double &f) { return *this = _mm_sub_pd(vec,_mm_set1_pd(f)); }
  
  friend inline real operator +(const real &a, const double &f) { return _mm_add_pd(a, _mm_set1_pd(f)); }
  friend inline real operator -(const real &a, const double &f) { return _mm_sub_pd(a, _mm_set1_pd(f)); } 
  friend inline real operator *(const real &a, const double &f) { return _mm_mul_pd(a, _mm_set1_pd(f)); } 
  friend inline real operator /(const real &a, const double &f) { return _mm_div_pd(a, _mm_set1_pd(f)); }

  friend inline real operator +(const double &f, const real &a) { return _mm_add_pd(_mm_set1_pd(f),a); }
  friend inline real operator -(const double &f, const real &a) { return _mm_sub_pd(_mm_set1_pd(f),a); } 
  friend inline real operator *(const double &f, const real &a) { return _mm_mul_pd(_mm_set1_pd(f),a); } 
  friend inline real operator /(const double &f, const real &a) { return _mm_div_pd(_mm_set1_pd(f),a); }

  friend inline real sqrt(const real &a) { return _mm_sqrt_pd(a); }
  //friend inline real ceil(const real &a)   { return _mm_round_pd((a), _MM_FROUND_CEIL); }
  //friend inline real floor(const real &a)  { return _mm_round_pd((a), _MM_FROUND_FLOOR); }
  //friend inline real trunc(const real &a)  { return _mm_round_pd((a), _MM_FROUND_TO_ZERO); }
    //friend real round(const real &a)  { return _mm_svml_round_pd(a); }

  friend inline real min(const real &r, const real &s){ return _mm_min_pd(r,s);}
  friend inline real max(const real &r, const real &s){ return _mm_max_pd(r,s);}

  
  friend inline real rcbrt(const real &a){
    return real(::cbrt(a[0]),::cbrt(a[1]));
  }
  
  friend inline real norm(const real &u){
    return real(::fabs(u[0]),::fabs(u[1]));
  }

  /* Compares: Mask is returned  */
  friend inline real cmp_eq(const real &a, const real &b)  { return _mm_cmpeq_pd(a, b); }
  friend inline real cmp_lt(const real &a, const real &b)  { return _mm_cmplt_pd(a, b); }
  friend inline real cmp_le(const real &a, const real &b)  { return _mm_cmple_pd(a, b); }
  friend inline real cmp_gt(const real &a, const real &b)  { return _mm_cmpgt_pd(a, b); }
  friend inline real cmp_ge(const real &a, const real &b)  { return _mm_cmpge_pd(a, b); }
  friend inline real cmp_neq(const real &a, const real &b)  { return _mm_cmpneq_pd(a, b); }
  friend inline real cmp_nlt(const real &a, const real &b)  { return _mm_cmpnlt_pd(a, b); }
  friend inline real cmp_nle(const real &a, const real &b)  { return _mm_cmpnle_pd(a, b); }
  friend inline real cmp_ngt(const real &a, const real &b)  { return _mm_cmpngt_pd(a, b); }
  friend inline real cmp_nge(const real &a, const real &b)  { return _mm_cmpnge_pd(a, b); }

  friend inline real operator<(const real &a, const real& b) { return _mm_cmplt_pd(a, b); }
  friend inline real operator<(const real &a, double d) { return _mm_cmplt_pd(a, _mm_set1_pd(d)); }


  friend inline real operator>(const real &a, real& r) { return _mm_cmpgt_pd(a, r); }
  friend inline real operator>(const real &a, const real& r) { return _mm_cmpgt_pd(a, r); }
  friend inline real operator>(const real &a, double d) { return _mm_cmpgt_pd(a, _mm_set1_pd(d)); }

  friend inline real operator>=(const real &a, real& r) { return _mm_cmpge_pd(a, r); }
  friend inline real operator>=(const real &a, double d) { return _mm_cmpge_pd(a, _mm_set1_pd(d)); }
  
  friend inline real operator<=(const real &a, const real& r) { return _mm_cmple_pd(a, r); }
  friend inline real operator<=(const real &a, double d) { return _mm_cmple_pd(a, _mm_set1_pd(d)); }

  friend inline real operator==(const real &a, const real& r) { return _mm_cmpeq_pd(a, r); }
  friend inline real operator==(const real &a, double d) { return _mm_cmpeq_pd(a, _mm_set1_pd(d)); }
  
  friend inline real operator!=(const real &a, const real& r) { return _mm_cmpneq_pd(a, r); }
  friend inline real operator!=(const real &a, double d) { return _mm_cmpneq_pd(a, _mm_set1_pd(d)); }

  /*friend real unglitch(const real &a)  {
    const union{
      unsigned long long i[4];
      __m128d m;
    } __f64vec4_abs_mask = { 0xffffffffffff0000ull,
                             0xffffffffffff0000ull};
    return _mm_and_pd(a, __f64vec4_abs_mask.m);
    }*/

  inline const double& operator[](int i) const  {
    double *d= (double*)&vec;
    return d[i];
  }
  
  inline double& operator[](int i) {
    double *d = (double*)&vec;
    return d[i];
  }

};

inline double ReduceMinToDouble(Real r){
  const double *dtv = (double*) &r;
  return dtv[0]>dtv[1]?dtv[1]:dtv[0];
}

inline double ReduceMaxToDouble(Real r){
  const double *dtv = (double*) &r;
  return dtv[0]>dtv[1]?dtv[0]:dtv[1];
}


#endif //  _KN_SSE_REAL_H_
