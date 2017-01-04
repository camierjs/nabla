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
#ifndef _KN_512_INTEGER_H_
#define _KN_512_INTEGER_H_

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
  // Logical Operations
  inline integer& operator&=(const integer &a) { return *this = (integer) _mm512_and_epi64(vec,a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) _mm512_or_epi64(vec,a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) _mm512_xor_epi64(vec,a); }

  inline integer& operator +=(const integer &a) { return *this = (integer)_mm512_add_epi64(vec,a); }
  inline integer& operator -=(const integer &a) { return *this = (integer)_mm512_sub_epi64(vec,a); }   
  
  friend inline __mmask8 operator==(const integer &a, const int i);
  
  inline const int& operator[](const int i) const  {
    assert((0 <= i) && (i <= 7));
    int *dp = (int*)&vec;
    return *(dp+i);
  }
  inline int& operator[](const int i) {
    assert((0 <= i) && (i <= 7));
    int *dp = (int*)&vec;
    return *(dp+i);
  }
};
// Logicals
inline integer operator&(const integer &a, const integer &b) { return _mm512_and_epi64(a,b); }
inline integer operator|(const integer &a, const integer &b) { return _mm512_or_epi64(a,b); }
inline integer operator^(const integer &a, const integer &b) { return _mm512_xor_epi64(a,b); }

inline __mmask8 operator==(const integer &a, const int i){
  return _mm512_cmp_epi64_mask(a.vec,_mm512_set_epi64(i,i,i,i,i,i,i,i),_MM_CMPINT_EQ);
}

#endif //  _KN_512_INTEGER_H_
