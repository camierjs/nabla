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
  inline integer(int i0, int i1, int i2, int i3,
                 int i4, int i5, int i6, int i7){vec=_mm512_set_epi64(i7,i6,i5,i4,i3,i2,i1,i0);}
  
  // Convertors
  inline operator __m512i() const { return vec; }
      
  /* Logical Operations */
  inline integer& operator&=(const integer &a) { return *this = (integer) _mm512_and_epi64(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) _mm512_or_epi64(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) _mm512_xor_epi64(vec,a); }

  inline integer& operator +=(const integer &a) { return *this = (integer)_mm512_add_epi64(vec,a); }
  inline integer& operator -=(const integer &a) { return *this = (integer)_mm512_sub_epi64(vec,a); }   
  
  friend inline __mmask8 operator==(const integer &a, const int i);


  
};
// Logicals
inline integer operator&(const integer &a, const integer &b) { return _mm512_and_epi64(a,b); }
inline integer operator|(const integer &a, const integer &b) { return _mm512_or_epi64(a,b); }
inline integer operator^(const integer &a, const integer &b) { return _mm512_xor_epi64(a,b); }


inline __mmask8 operator==(const integer &a, const int i){
  return _mm512_cmp_epi64_mask(a.vec,_mm512_set_epi64(i,i,i,i,i,i,i,i),_MM_CMPINT_EQ);
}

#endif // _KN_MIC_INTEGER_H_
