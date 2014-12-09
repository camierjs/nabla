#ifndef _KN_AVX_INTEGER_H_
#define _KN_AVX_INTEGER_H_


// ****************************************************************************
// * SIMD integer
// ****************************************************************************
struct __attribute__ ((aligned(16))) integer {
protected:
  __m128i vec;
public:
  // Constructors
  inline integer():vec(_mm_set_epi32(0,0,0,0)){}
  inline	integer(__m128i mm):vec(mm){}
  inline integer(int i):vec(_mm_set_epi32(i,i,i,i)){}
  inline integer(int i0, int i1, int i2, int i3){vec=_mm_set_epi32(i3, i2, i1, i0);}
  
  // Convertors
  inline operator __m128i() const { return vec; }
      
  /* Logical Operations */
  inline integer& operator&=(const integer &a) { return *this = (integer) _mm_and_si128(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) _mm_or_si128(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) _mm_xor_si128(vec,a); }

  inline integer& operator +=(const integer &a) { return *this = (integer)_mm_add_epi32(vec,a); }
  inline integer& operator -=(const integer &a) { return *this = (integer)_mm_sub_epi32(vec,a); }   
  
  friend inline __m256d operator==(const integer &a, const int i);

  inline const int& operator[](int i) const  {
    int *a=(int*)&vec;
    return a[i];
  }  
};
// Logicals
inline integer operator&(const integer &a, const integer &b) { return _mm_and_si128(a,b); }
inline integer operator|(const integer &a, const integer &b) { return _mm_or_si128(a,b); }
inline integer operator^(const integer &a, const integer &b) { return _mm_xor_si128(a,b); }


inline __m256d operator==(const integer &a, const int i){
  const __m256d zero = _mm256_set1_pd(0.0);
  const __m256d true_ =_mm256_cmp_pd(zero, zero, _CMP_EQ_OQ);
  int *v=(int*)&a.vec;
  int *t=(int*)&true_;
  //debug()<<"== with "<<v[3]<<" "<<v[2]<<" "<<v[1]<<" "<<v[0]<<" vs "<<i<<", t[3]="<<t[3];
  return _mm256_set_pd((v[3]==i)?t[3]:0.0,
                       (v[2]==i)?t[2]:0.0,
                       (v[1]==i)?t[1]:0.0,
                       (v[0]==i)?t[0]:0.0);
}

#endif //  _KN_AVX_INTEGER_H_
