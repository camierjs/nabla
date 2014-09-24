#ifndef _KN_AVX_REAL_H_
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
  inline real(long i):vec(_mm_set1_pd((double)i)){}
  inline real(double d):vec(_mm_set1_pd(d)){}
  inline real(__m128d x):vec(x){}
  inline real(double *x):vec(_mm_load_pd(x)){}
  inline real(double d1, double d0):vec(_mm_set_pd(d0,d1)){}

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
  
  friend inline real min(const real &r, const real &s){ return _mm_min_pd(r,s);}
  friend inline real max(const real &r, const real &s){ return _mm_max_pd(r,s);}

  /* Round */
  //friend real round(const real &a)  { return _mm_svml_round_pd(a); }
  
  friend inline real rcbrt(const real &a){
    return real(::cbrt(a[0]),::cbrt(a[1]));
  }
  
  friend inline real norm(real u){ return u;}

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

  friend real unglitch(const real &a)  {
    const union{
      unsigned long long i[4];
      __m128d m;
    } __f64vec4_abs_mask = { 0xffffffffffff0000ull,
                             0xffffffffffff0000ull};
    return _mm_and_pd(a, __f64vec4_abs_mask.m);
  }

   

  /* Element Access Only, no modifications to elements */
  inline const double& operator[](int i) const  {
    /* Assert enabled only during debug /DDEBUG */
    assert((0 <= i) && (i <= 1));
    double *dp = (double*)&vec;
    return *(dp+i);
  }
  /* Element Access and Modification*/
  inline double& operator[](int i) {
    /* Assert enabled only during debug /DDEBUG */
    assert((0 <= i) && (i <= 1));
    double *dp = (double*)&vec;
    return *(dp+i);
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
