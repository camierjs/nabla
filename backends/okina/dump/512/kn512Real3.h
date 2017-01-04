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
#ifndef _KN_512_REAL3_H_
#define _KN_512_REAL3_H_
struct __attribute__ ((aligned(64))) real3 {
 public:
  real x;
  real y;
  real z;
 public:
  // Constructors
  inline real3(): x(_mm512_setzero_pd()),y(_mm512_setzero_pd()),z(_mm512_setzero_pd()){}
  inline real3(double d): x(_mm512_set1_pd(d)),y(_mm512_set1_pd(d)),z(_mm512_set1_pd(d)){}
  inline real3(double _x,double _y,double _z): x(_mm512_set1_pd(_x)),y(_mm512_set1_pd(_y)),z(_mm512_set1_pd(_z)){}
  inline real3(real f):x(f),y(f),z(f){}
  inline real3(real _x,real _y,real _z):x(_x),y(_y),z(_z){}
  inline real3(double *_x,double *_y,double *_z): x(_mm512_load_pd(_x)),y(_mm512_load_pd(_y)),z(_mm512_load_pd(_z)){}
  // Arithmetic operators
  friend inline real3 operator+(const real3 &a,const real3& b){
    return real3(_mm512_add_pd(a.x,b.x),
                 _mm512_add_pd(a.y,b.y),
                 _mm512_add_pd(a.z,b.z));}
  friend inline real3 operator-(const real3 &a,const real3& b){
    return real3(_mm512_sub_pd(a.x,b.x),
                 _mm512_sub_pd(a.y,b.y),
                 _mm512_sub_pd(a.z,b.z));}
  friend inline real3 operator*(const real3 &a,const real3& b){
    return real3(_mm512_mul_pd(a.x,b.x),
                 _mm512_mul_pd(a.y,b.y),
                 _mm512_mul_pd(a.z,b.z));}
  friend inline real3 operator/(const real3 &a,const real3& b){
    return real3(_mm512_div_pd(a.x,b.x),
                 _mm512_div_pd(a.y,b.y),
                 _mm512_div_pd(a.z,b.z));}

  // op= operators
  inline real3& operator+=(const real3& b){
    return *this=real3(_mm512_add_pd(x,b.x),
                       _mm512_add_pd(y,b.y),
                       _mm512_add_pd(z,b.z));}
  inline real3& operator-=(const real3& b){
    return *this=real3(_mm512_sub_pd(x,b.x),
                       _mm512_sub_pd(y,b.y),
                       _mm512_sub_pd(z,b.z));}
  inline real3& operator*=(const real3& b){
    return *this=real3(_mm512_mul_pd(x,b.x),
                       _mm512_mul_pd(y,b.y),
                       _mm512_mul_pd(z,b.z));}
  inline real3& operator/=(const real3& b){
    return *this=real3(_mm512_div_pd(x,b.x),
                       _mm512_div_pd(y,b.y),
                       _mm512_div_pd(z,b.z));}

  //inline real3 operator-(){return real3(-x,-y,-z);}
  inline real3 operator-()const {return real3(-x,-y,-z);}

  // op= operators with real
  inline real3& operator+=(real f){return *this=real3(x+f,y+f,z+f);}
  inline real3& operator-=(real f){return *this=real3(x-f,y-f,z-f);}
  inline real3& operator*=(real f){return *this=real3(x*f,y*f,z*f);}
  inline real3& operator/=(real f){return *this=real3(x/f,y/f,z/f);}

  inline real3& operator+=(double f){return *this=real3(x+f,y+f,z+f);}
  inline real3& operator-=(double f){return *this=real3(x-f,y-f,z-f);}
  inline real3& operator*=(double f){return *this=real3(x*f,y*f,z*f);}
  inline real3& operator/=(double f){return *this=real3(x/f,y/f,z/f);}
  
  friend inline real dot3(real3 u,real3 v){ return real(u.x*v.x+u.y*v.y+u.z*v.z);}
  friend inline real norm(real3 u){ return real(_mm512_sqrt_pd(dot3(u,u)));}
  friend inline real3 cross(real3 u,real3 v){
    return
      real3(_mm512_sub_pd( _mm512_mul_pd(u.y,v.z),_mm512_mul_pd(u.z,v.y) ),
            _mm512_sub_pd( _mm512_mul_pd(u.z,v.x),_mm512_mul_pd(u.x,v.z) ),
            _mm512_sub_pd( _mm512_mul_pd(u.x,v.y),_mm512_mul_pd(u.y,v.x) ));
  }
};

#endif //  _KN_512_REAL3_H_
