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
#ifndef _KN_SSE_INTEGER_H_
#define _KN_SSE_INTEGER_H_


// ****************************************************************************
// * SIMD integer
// ****************************************************************************
struct __attribute__ ((aligned(8))) integer {
protected:
  __m128i vec;
public:
  // Constructors
  inline integer():vec(_mm_set_epi32(0,0,0,0)){}
  inline	integer(__m128i mm):vec(mm){}
  inline integer(int i):vec(_mm_set_epi32(0,0,i,i)){}
  inline integer(int i0, int i1){vec=_mm_set_epi32(0, 0, i1, i0);}
  
  // Convertors
  inline operator __m128i() const { return vec; }
      
  /* Logical Operations */
  inline integer& operator&=(const integer &a) { return *this = (integer) _mm_and_si128(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) _mm_or_si128(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) _mm_xor_si128(vec,a); }
  
  friend inline integer operator<(const integer &a, const integer &b)  { return _mm_cmpeq_epi32(a, b); }

  // Arithmetics
  friend inline integer operator +(const integer &a, const integer &b) { return _mm_add_epi32(a,b); }
  friend inline integer operator -(const integer &a, const integer &b) { return _mm_sub_epi32(a,b); }
  friend inline integer operator *(const integer &a, const integer &b) { return _mm_mul_epi32(a,b); }
  friend inline integer operator /(const integer &a, const integer &b) {
    return _mm_set_epi32(a[0]/b[0],a[1]/b[1],0,0);
  }
  friend inline integer operator %(const integer &a, const integer &b) {
    return _mm_set_epi32(a[0]%b[0],a[1]%b[1],0,0);
  }

  inline integer& operator +=(const integer &a) { return *this = (integer)_mm_add_epi32(vec,a); }
  inline integer& operator -=(const integer &a) { return *this = (integer)_mm_sub_epi32(vec,a); }   
  
  friend inline __m128d operator==(const integer &a, const int i);

  inline const int& operator[](int i) const  {
    int *a=(int*)&vec;
    return a[i];
  }
};

// Logicals
inline integer operator&(const integer &a, const integer &b) { return _mm_and_si128(a,b); }
inline integer operator|(const integer &a, const integer &b) { return _mm_or_si128(a,b); }
inline integer operator^(const integer &a, const integer &b) { return _mm_xor_si128(a,b); }

inline __m128d operator==(const integer &a, const int i){
  const __m128d zero = _mm_set1_pd(0.0);
  const __m128d true_ =_mm_cmpeq_pd(zero, zero);
  int *v=(int*)&a.vec;
  int *t=(int*)&true_;
  //debug()<<"== with "<<v[3]<<" "<<v[2]<<" "<<v[1]<<" "<<v[0]<<" vs "<<i<<", t[3]="<<t[3];
  return _mm_set_pd((v[1]==i)?t[1]:0.0,
                    (v[0]==i)?t[0]:0.0);
}


#endif //  _KN_SSE_INTEGER_H_
