#ifndef _KN_MIC_INTEGER_H_
#define _KN_MIC_INTEGER_H_

struct __attribute__ ((aligned(64))) integer {
protected:
  __m512i vec;
public:
  // Constructors
  inline integer():vec(_mm512_set_epi64(0,0,0,0,0,0,0,0)){}
  inline	integer(__m512i mm):vec(mm){}
  inline integer(int i):vec(_mm512_set_epi64(i,i,i,i,i,i,i,i)){}
  inline integer(int i7, int i6, int i5, int i4,
                 int i3, int i2, int i1, int i0){vec=_mm512_set_epi64(i7,i6,i5,i4,i3,i2,i1,i0);}
  
  // Convertors
  inline operator __m512i() const { return vec; }
      
  /* Logical Operations */
  inline integer& operator&=(const integer &a) { return *this = (integer) _mm512_and_si512(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) _mm512_or_si512(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) _mm512_xor_si512(vec,a); }

  inline integer& operator +=(const integer &a) { return *this = (integer)_mm512_add_epi32(vec,a); }
  inline integer& operator -=(const integer &a) { return *this = (integer)_mm512_sub_epi32(vec,a); }   
  
  friend inline __mmask8 operator==(const integer &a, const int i);


  
};
// Logicals
inline integer operator&(const integer &a, const integer &b) { return _mm512_and_si512(a,b); }
inline integer operator|(const integer &a, const integer &b) { return _mm512_or_si512(a,b); }
inline integer operator^(const integer &a, const integer &b) { return _mm512_xor_si512(a,b); }


inline __mmask8 operator==(const integer &a, const int i){
/*  const __m512d zero = _mm512_set1_pd(0.0);
  const __mmask true_ =_mm512_cmpeq_epi32_mask(zero, zero);
  int *v=(int*)&a.vec;
  int *t=(int*)&true_;
  //debug()<<"== with "<<v[3]<<" "<<v[2]<<" "<<v[1]<<" "<<v[0]<<" vs "<<i<<", t[3]="<<t[3];
  return _mm512_set_pd((v[7]==i)?t[7]:0.0,
                       (v[6]==i)?t[6]:0.0,
                       (v[5]==i)?t[5]:0.0,
                       (v[4]==i)?t[4]:0.0,
                       (v[3]==i)?t[3]:0.0,
                       (v[2]==i)?t[2]:0.0,
                       (v[1]==i)?t[1]:0.0,
                       (v[0]==i)?t[0]:0.0);*/
  return (__mmask8)0xFF;
}

#endif // _KN_MIC_INTEGER_H_
