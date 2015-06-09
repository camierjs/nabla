///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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
