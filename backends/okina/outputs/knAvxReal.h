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
#ifndef _KN_AVX_REAL_H_
#define _KN_AVX_REAL_H_

// ****************************************************************************
// * real
// ****************************************************************************
struct __attribute__ ((aligned(32))) real {
 protected:
  __m256d vec;
 public:
  // Constructors
  inline real(): vec(_mm256_setzero_pd()){}
  inline real(int i):vec(_mm256_set1_pd((double)i)){}
  inline real(long i):vec(_mm256_set1_pd((double)i)){}
  inline real(double d):vec(_mm256_set1_pd(d)){}
  inline real(__m256d x):vec(x){}
  inline real(double *x):vec(_mm256_load_pd(x)){}
  inline real(double d0, double d1, double d2, double d3):vec(_mm256_set_pd(d3,d2,d1,d0)){}

  // Convertors
  inline operator __m256d() const { return vec; }
  
  // Logicals
  friend inline real operator &(const real &a, const real &b) { return _mm256_and_pd(a,b); }
  friend inline real operator |(const real &a, const real &b) { return _mm256_or_pd(a,b); }
  friend inline real operator ^(const real &a, const real &b) { return _mm256_xor_pd(a,b); }

  // Arithmetics
  friend inline real operator +(const real &a, const real &b) { return _mm256_add_pd(a,b); }
  friend inline real operator -(const real &a, const real &b) { return _mm256_sub_pd(a,b); }
  friend inline real operator *(const real &a, const real &b) { return _mm256_mul_pd(a,b); }
  friend inline real operator /(const real &a, const real &b) { return _mm256_div_pd(a,b); }

  
  inline real& operator +=(const real &a) { return *this = _mm256_add_pd(vec,a); }
  inline real& operator -=(const real &a) { return *this = _mm256_sub_pd(vec,a); }
  inline real& operator *=(const real &a) { return *this = _mm256_mul_pd(vec,a); }
  inline real& operator /=(const real &a) { return *this = _mm256_div_pd(vec,a); }
  inline real& operator &=(const real &a) { return *this = _mm256_and_pd(vec,a); }
  inline real& operator |=(const real &a) { return *this = _mm256_or_pd(vec,a); }
  inline real& operator ^=(const real &a) { return *this = _mm256_xor_pd(vec,a); }
  
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
  
  friend inline real rcbrt(const real &a){
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


  // SVML functions
  /*friend real acos(const real &a)    { return _mm256_acos_pd(a);    }
  friend real acosh(const real &a)   { return _mm256_acosh_pd(a);   }
  friend real asin(const real &a)    { return _mm256_asin_pd(a);    }
  friend real asinh(const real &a)   { return _mm256_asinh_pd(a);   }
  friend real atan(const real &a)    { return _mm256_atan_pd(a);    }
  friend real atan2(const real &a, const real &b) { return _mm256_atan2_pd(a, b); }
  friend real atanh(const real &a)   { return _mm256_atanh_pd(a);   }
  friend real cos(const real &a)     { return _mm256_cos_pd(a);     }
  friend real cosh(const real &a)    { return _mm256_cosh_pd(a);    }
  friend real exp(const real &a)     { return _mm256_exp_pd(a);     }
  friend real exp2(const real &a)    { return _mm256_exp2_pd(a);    }
  friend real invcbrt(const real &a) { return _mm256_invcbrt_pd(a); }
  friend real invsqrt(const real &a) { return _mm256_invsqrt_pd(a); }
  friend real log(const real &a)     { return _mm256_log_pd(a);     }
  friend real log10(const real &a)   { return _mm256_log10_pd(a);   }
  friend real log2(const real &a)    { return _mm256_log2_pd(a);    }
  friend real pow(const real &a, const real &b) { return _mm256_pow_pd(a, b); }
  friend real sin(const real &a)     { return _mm256_sin_pd(a);     }
  friend real sinh(const real &a)    { return _mm256_sinh_pd(a);    }
  friend real tan(const real &a)     { return _mm256_tan_pd(a);     }
  friend real tanh(const real &a)    { return _mm256_tanh_pd(a);    }
  friend real erf(const real &a)     { return _mm256_erf_pd(a);     }
  friend real erfc(const real &a)    { return _mm256_erfc_pd(a);    }
  friend real erfinv(const real &a)  { return _mm256_erfinv_pd(a);  }*/

  /* Absolute value */
  /*friend real abs(const real &a)  {
    static const union
    {
      int i[8];
      __m256d m;
    } __f64vec4_abs_mask = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff,
                             0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};
    return _mm256_and_pd(a, __f64vec4_abs_mask.m);
    }*/

  friend real unglitch(const real &a)  {
    const union{
      unsigned long long i[4];
      __m256d m;
    } __f64vec4_abs_mask = { 0xffffffffffff0000ull,
                             0xffffffffffff0000ull,
                             0xffffffffffff0000ull,
                             0xffffffffffff0000ull};
    return _mm256_and_pd(a, __f64vec4_abs_mask.m);
  }

   
  inline const double& operator[](int i) const  {
    double *d = (double*)&vec;
    return *(d+i);
  }
  inline double& operator[](int i) {
    double *d = (double*)&vec;
    return *(d+i);
  }

};

inline double ReduceMinToDouble(Real r){
  double mnx[2];
  const double *dtv = (double*) &r;
  mnx[0]=dtv[0]>dtv[1]?dtv[1]:dtv[0];
  mnx[1]=dtv[2]>dtv[3]?dtv[3]:dtv[2];
  return mnx[0]>mnx[1]?mnx[1]:mnx[0];
}

inline double ReduceMaxToDouble(Real r){
  double mnx[2];
  const double *dtv = (double*) &r;
  mnx[0]=dtv[0]>dtv[1]?dtv[0]:dtv[1];
  mnx[1]=dtv[2]>dtv[3]?dtv[2]:dtv[3];
  return mnx[0]>mnx[1]?mnx[0]:mnx[1];
}


#endif //  _KN_AVX_REAL_H_
