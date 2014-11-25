#ifndef _KN_MIC_REAL_H_
#define _KN_MIC_REAL_H_

#pragma pack(push,64)
//#warning 76543210 vs 01234567

// ****************************************************************************
// * real
// ****************************************************************************
struct __attribute__ ((aligned(64))) real {
 protected:
  __m512d vec;
 public:
  inline real(): vec(_mm512_setzero_pd()){}
  inline real(double d):vec(_mm512_set1_pd(d)){}
  inline real(__m512d x):vec(x){}
  inline real(double *x):vec(_mm512_load_pd(x)){}
  inline real(double d7, double d6, double d5, double d4,
              double d3, double d2, double d1, double d0):vec(_mm512_set_pd(d0,d1,d2,d3,d4,d5,d6,d7)){}

  // Conversion operator
  operator __m512d() const { return vec; }
  
    // Logicals
  friend inline real operator &(const real &a, const real &b) {
    return _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(a),_mm512_castpd_si512(b))); }
  friend inline real operator |(const real &a, const real &b) {
    return _mm512_castsi512_pd(_mm512_or_epi64(_mm512_castpd_si512(a),_mm512_castpd_si512(b))); }
  friend inline real operator ^(const real &a, const real &b) {
    return _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(a),_mm512_castpd_si512(b))); }

  // Arithmetics
  friend inline real operator +(const real &a, const real &b) { return _mm512_add_pd(a,b); }
  friend inline real operator -(const real &a, const real &b) { return _mm512_sub_pd(a,b); }
  friend inline real operator *(const real &a, const real &b) { return _mm512_mul_pd(a,b); }
  friend inline real operator /(const real &a, const real &b) { return _mm512_div_pd(a,b); }


  inline real& operator +=(const real &a) { return *this = _mm512_add_pd(vec,a); }
  inline real& operator -=(const real &a) { return *this = _mm512_sub_pd(vec,a); }
  inline real& operator *=(const real &a) { return *this = _mm512_mul_pd(vec,a); }
  inline real& operator /=(const real &a) { return *this = _mm512_div_pd(vec,a); }
  inline real& operator &=(const real &a) { return *this = _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(vec),
                                                                                                _mm512_castpd_si512(a))); }
  inline real& operator |=(const real &a) { return *this = _mm512_castsi512_pd(_mm512_or_epi64(_mm512_castpd_si512(vec),
                                                                                               _mm512_castpd_si512(a))); }
  inline real& operator ^=(const real &a) { return *this = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(vec),
                                                                                                _mm512_castpd_si512(a))); }
  
  inline real operator -() const { return real(0.0) - vec; }
  inline real operator -()       { return real(0.0) - vec; }


  // Mixed vector-scalar operations
  inline real& operator *=(const double &f) { return *this = _mm512_mul_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator /=(const double &f) { return *this = _mm512_div_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator +=(const double &f) { return *this = _mm512_add_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator +=(double &f) { return *this = _mm512_add_pd(vec,_mm512_set1_pd(f)); }
  inline real& operator -=(const double &f) { return *this = _mm512_sub_pd(vec,_mm512_set1_pd(f)); }
  
  friend inline real operator +(const real &a, const double &f) { return _mm512_add_pd(a, _mm512_set1_pd(f)); }
  friend inline real operator -(const real &a, const double &f) { return _mm512_sub_pd(a, _mm512_set1_pd(f)); } 
  friend inline real operator *(const real &a, const double &f) { return _mm512_mul_pd(a, _mm512_set1_pd(f)); } 
  friend inline real operator /(const real &a, const double &f) { return _mm512_div_pd(a, _mm512_set1_pd(f)); }

  friend inline real operator +(const double &f, const real &a) { return _mm512_add_pd(_mm512_set1_pd(f),a); }
  friend inline real operator -(const double &f, const real &a) { return _mm512_sub_pd(_mm512_set1_pd(f),a); } 
  friend inline real operator *(const double &f, const real &a) { return _mm512_mul_pd(_mm512_set1_pd(f),a); } 
  friend inline real operator /(const double &f, const real &a) { return _mm512_div_pd(_mm512_set1_pd(f),a); }

  friend inline real sqrt(const real &a) { return _mm512_sqrt_pd(a); }
  friend inline real ceil(const real &a)   { return _mm512_ceil_pd((a)); }
  friend inline real floor(const real &a)  { return _mm512_floor_pd((a)); }
  friend inline real trunc(const real &a)  { return _mm512_trunc_pd((a)); }
  
  friend inline real min(const real &r, const real &s){ return _mm512_min_pd(r,s);}
  friend inline real max(const real &r, const real &s){ return _mm512_max_pd(r,s);}

  friend inline real rcbrt(const real &a){
    return real(::cbrt(a[0]),::cbrt(a[1]),::cbrt(a[2]),::cbrt(a[3]),::cbrt(a[4]),::cbrt(a[5]),::cbrt(a[6]),::cbrt(a[7]));
    //return _mm512_cbrt_pd(a);
  }
  
  friend inline real norm(real u){ return u;}

  /* Compares: Mask is returned  */
  friend inline __mmask8 cmp_eq(const real &a, const real &b)  { return _mm512_cmpeq_pd_mask(a,b); }
  friend inline __mmask8 cmp_lt(const real &a, const real &b)  { return _mm512_cmplt_pd_mask(a,b); }
  friend inline __mmask8 cmp_le(const real &a, const real &b)  { return _mm512_cmple_pd_mask(a,b); }
  friend inline __mmask8 cmp_gt(const real &a, const real &b)  { return _mm512_cmpnle_pd_mask(a,b); }
  friend inline __mmask8 cmp_ge(const real &a, const real &b)  { return _mm512_cmpnlt_pd_mask(a,b); }
  friend inline __mmask8 cmp_neq(const real &a, const real &b)  { return _mm512_cmpneq_pd_mask(a,b); }
  friend inline __mmask8 cmp_nlt(const real &a, const real &b)  { return _mm512_cmpnlt_pd_mask(a,b); }
  friend inline __mmask8 cmp_nle(const real &a, const real &b)  { return _mm512_cmpnle_pd_mask(a,b); }
  friend inline __mmask8 cmp_ngt(const real &a, const real &b)  { return _mm512_cmp_pd_mask(a,b,_CMP_LE_OS); }
  friend inline __mmask8 cmp_nge(const real &a, const real &b)  { return _mm512_cmp_pd_mask(a,b,_CMP_LT_OS); }

  friend inline __mmask8 operator<(const real &a, const real& b) { return _mm512_cmplt_pd_mask(a,b); }
  friend inline __mmask8 operator<(const real &a, double d) { return _mm512_cmplt_pd_mask(a, _mm512_set1_pd(d)); }


  friend inline __mmask8 operator>(const real &a, real& r) { return _mm512_cmpnle_pd_mask(a,r); }
  friend inline __mmask8 operator>(const real &a, const real& r) { return _mm512_cmpnle_pd_mask(a,r); }
  friend inline __mmask8 operator>(const real &a, double d) { return _mm512_cmpnle_pd_mask(a, _mm512_set1_pd(d)); }

  friend inline __mmask8 operator>=(const real &a, real& r) { return _mm512_cmpnlt_pd_mask(a,r); }
  friend inline __mmask8 operator>=(const real &a, double d) { return _mm512_cmpnlt_pd_mask(a, _mm512_set1_pd(d)); }
  
  friend inline __mmask8 operator<=(const real &a, const real& r) { return _mm512_cmple_pd_mask(a,r); }
  friend inline __mmask8 operator<=(const real &a, double d) { return _mm512_cmple_pd_mask(a, _mm512_set1_pd(d)); }

  friend inline __mmask8 operator==(const real &a, const real& r) { return _mm512_cmpeq_pd_mask(a,r); }
  friend inline __mmask8 operator==(const real &a, double d) { return _mm512_cmpeq_pd_mask(a, _mm512_set1_pd(d)); }
  
  friend inline __mmask8 operator!=(const real &a, const real& r) { return _mm512_cmpneq_pd_mask(a,r); }
  friend inline __mmask8 operator!=(const real &a, double d) { return _mm512_cmpneq_pd_mask(a, _mm512_set1_pd(d)); }


  /* Element Access Only, no modifications to elements */
  inline const double& operator[](const int i) const  {
    /* Assert enabled only during debug /DDEBUG */
    assert((0 <= i) && (i <= 7));
    double *dp = (double*)&vec;
    return *(dp+i);
  }
  /* Element Access and Modification*/
  inline double& operator[](const int i) {
    /* Assert enabled only during debug /DDEBUG */
    assert((0 <= i) && (i <= 7));
    double *dp = (double*)&vec;
    return *(dp+i);
  }
 
};

inline double ReduceMinToDouble(real r){
  double mnx[6];
  const double *dtv = (double*) &r;
  mnx[0]=dtv[0]>dtv[1]?dtv[1]:dtv[0];
  mnx[1]=dtv[2]>dtv[3]?dtv[3]:dtv[2];
  mnx[2]=dtv[4]>dtv[5]?dtv[5]:dtv[4];
  mnx[3]=dtv[6]>dtv[7]?dtv[7]:dtv[6];
  mnx[4]=mnx[0]>mnx[1]?mnx[1]:mnx[0];
  mnx[5]=mnx[2]>mnx[3]?mnx[3]:mnx[2];
  return mnx[4]>mnx[5]?mnx[5]:mnx[4];
}

inline double ReduceMaxToDouble(real r){
  double mnx[6];
  const double *dtv = (double*) &r;
  mnx[0]=dtv[0]<dtv[1]?dtv[1]:dtv[0];
  mnx[1]=dtv[2]<dtv[3]?dtv[3]:dtv[2];
  mnx[2]=dtv[4]<dtv[5]?dtv[5]:dtv[4];
  mnx[3]=dtv[6]<dtv[7]?dtv[7]:dtv[6];
  mnx[4]=mnx[0]<mnx[1]?mnx[1]:mnx[0];
  mnx[5]=mnx[2]<mnx[3]?mnx[3]:mnx[2];
  return mnx[4]<mnx[5]?mnx[5]:mnx[4];
}

#endif // _KN_MIC_REAL_H_
