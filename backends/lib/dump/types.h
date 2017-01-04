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
#ifndef _NABLA_LIB_TYPES_H_
#define _NABLA_LIB_TYPES_H_

// ****************************************************************************
// * real3
// ****************************************************************************
struct __attribute__ ((aligned(8))) real3 {
 public:
  __attribute__ ((aligned(8))) double x;
  __attribute__ ((aligned(8))) double y;
  __attribute__ ((aligned(8))) double z;
 public:
  // Constructors
  inline real3(){ x=0.0; y=0.0; z=0.0;}
  inline real3(double d) {x=d; y=d; z=d;}
  inline real3(double _x,double _y,double _z): x(_x), y(_y), z(_z){}
  inline real3(double *_x, double *_y, double *_z){x=*_x; y=*_y; z=*_z;}
  // Arithmetic operators
  friend inline real3 operator+(const real3 &a, const real3& b) {
    return real3((a.x+b.x), (a.y+b.y), (a.z+b.z));}
  friend inline real3 operator-(const real3 &a, const real3& b) {
    return real3((a.x-b.x), (a.y-b.y), (a.z-b.z));}
  friend inline real3 operator*(const real3 &a, const real3& b) {
    return real3((a.x*b.x), (a.y*b.y), (a.z*b.z));}
  friend inline real3 operator/(const real3 &a, const real3& b) {
    return real3((a.x/b.x), (a.y/b.y), (a.z/b.z));}
  // op= operators
  inline real3& operator+=(const real3& b) { return *this=real3((x+b.x),(y+b.y),(z+b.z));}
  inline real3& operator-=(const real3& b) { return *this=real3((x-b.x),(y-b.y),(z-b.z));}
  inline real3& operator*=(const real3& b) { return *this=real3((x*b.x),(y*b.y),(z*b.z));}
  inline real3& operator/=(const real3& b) { return *this=real3((x/b.x),(y/b.y),(z/b.z));}
  inline real3 operator-()const {return real3(0.0-x, 0.0-y, 0.0-z);}
  // op= operators with real
  inline real3& operator+=(double f){return *this=real3(x+f,y+f,z+f);}
  inline real3& operator-=(double f){return *this=real3(x-f,y-f,z-f);}
  inline real3& operator*=(double f){return *this=real3(x*f,y*f,z*f);}
  inline real3& operator/=(double f){return *this=real3(x/f,y/f,z/f);}
  friend inline real dot3(real3 u, real3 v){
    return real(u.x*v.x+u.y*v.y+u.z*v.z);
  }
  friend inline real norm(real3 u){ return square_root(dot3(u,u));}

  friend inline real3 cross(real3 u, real3 v){
    return real3(((u.y*v.z)-(u.z*v.y)), ((u.z*v.x)-(u.x*v.z)), ((u.x*v.y)-(u.y*v.x)));
  }
  inline real abs2(){return x*x+y*y+z*z;}
};

inline real norm(real u){ return ::fabs(u);}

// ****************************************************************************
// * real3x3 
// ****************************************************************************
struct __attribute__ ((aligned(8))) real3x3 {
 public:
  __attribute__ ((aligned(8))) struct real3 x;
  __attribute__ ((aligned(8))) struct real3 y;
  __attribute__ ((aligned(8))) struct real3 z;
  inline real3x3(){ x=0.0; y=0.0; z=0.0;}
  inline real3x3(double d) {x=d; y=d; z=d;}
  inline real3x3(real3 r){ x=r; y=r; z=r;}
  inline real3x3(real3 _x, real3 _y, real3 _z) {x=_x; y=_y; z=_z;}
  // Arithmetic operators
  friend inline real3x3 operator+(const real3x3 &a, const real3x3& b) {
    return real3x3((a.x+b.x), (a.y+b.y), (a.z+b.z));}
  friend inline real3x3 operator-(const real3x3 &a, const real3x3& b) {
    return real3x3((a.x-b.x), (a.y-b.y), (a.z-b.z));}
  friend inline real3x3 operator*(const real3x3 &a, const real3x3& b) {
    return real3x3((a.x*b.x), (a.y*b.y), (a.z*b.z));}
  friend inline real3x3 operator/(const real3x3 &a, const real3x3& b) {
    return real3x3((a.x/b.x), (a.y/b.y), (a.z/b.z));}

  inline real3x3& operator+=(const real3x3& b) { return *this=real3x3((x+b.x),(y+b.y),(z+b.z));}
  inline real3x3& operator-=(const real3x3& b) { return *this=real3x3((x-b.x),(y-b.y),(z-b.z));}
  inline real3x3& operator*=(const real3x3& b) { return *this=real3x3((x*b.x),(y*b.y),(z*b.z));}
  inline real3x3& operator/=(const real3x3& b) { return *this=real3x3((x/b.x),(y/b.y),(z/b.z));}
  inline real3x3 operator-()const {return real3x3(0.0-x, 0.0-y, 0.0-z);}

  inline real3x3& operator*=(const real& d) { return *this=real3x3(x*d,y*d,z*d);}
  inline real3x3& operator/=(const real& d) { return *this=real3x3(x/d,y/d,z/d);}
  
  friend inline real3x3 operator*(real3x3 t, double d) { return real3x3(t.x*d,t.y*d,t.z*d);}
  
  friend inline real3 opProdTensVec(real3x3 t,real3 v){
    return real3(dot3(t.x,v),dot3(t.y,v),dot3(t.z,v));
  }
};
inline real3x3 opProdTens(real3 a,real3 b){
  return real3x3(a.x*b,a.y*b,a.z*b);
}
inline real matrixDeterminant(real3x3 m) {
  return (  m.x.x*(m.y.y*m.z.z-m.y.z*m.z.y)
          + m.x.y*(m.y.z*m.z.x-m.y.x*m.z.z)
          + m.x.z*(m.y.x*m.z.y-m.y.y*m.z.x));
}
inline real3x3 inverseMatrix(real3x3 m,real d){
  Real3x3 inv(real3(m.y.y*m.z.z-m.y.z*m.z.y,
                    -m.x.y*m.z.z+m.x.z*m.z.y,
                    m.x.y*m.y.z-m.x.z*m.y.y),
              real3(m.z.x*m.y.z-m.y.x*m.z.z,
                    -m.z.x*m.x.z+m.x.x*m.z.z,
                    m.y.x*m.x.z-m.x.x*m.y.z),
              real3(-m.z.x*m.y.y+m.y.x*m.z.y,
                    m.z.x*m.x.y-m.x.x*m.z.y,
                    -m.y.x*m.x.y+m.x.x*m.y.y));
  inv/=d;
  return inv;
}
inline real3x3 matrix3x3Id(){
  return real3x3(real3(1.0, 0.0, 0.0),
                 real3(0.0, 1.0, 0.0),
                 real3(0.0, 0.0, 1.0));
}
inline real3x3 opMatrixProduct(const real3x3 &t1,
                               const real3x3 &t2) {
  real3x3 temp;
  temp.x.x = t1.x.x*t2.x.x+t1.x.y*t2.y.x+t1.x.z*t2.z.x;
  temp.y.x = t1.y.x*t2.x.x+t1.y.y*t2.y.x+t1.y.z*t2.z.x;
  temp.z.x = t1.z.x*t2.x.x+t1.z.y*t2.y.x+t1.z.z*t2.z.x;
  temp.x.y = t1.x.x*t2.x.y+t1.x.y*t2.y.y+t1.x.z*t2.z.y;
  temp.y.y = t1.y.x*t2.x.y+t1.y.y*t2.y.y+t1.y.z*t2.z.y;
  temp.z.y = t1.z.x*t2.x.y+t1.z.y*t2.y.y+t1.z.z*t2.z.y;
  temp.x.z = t1.x.x*t2.x.z+t1.x.y*t2.y.z+t1.x.z*t2.z.z;
  temp.y.z = t1.y.x*t2.x.z+t1.y.y*t2.y.z+t1.y.z*t2.z.z;
  temp.z.z = t1.z.x*t2.x.z+t1.z.y*t2.y.z+t1.z.z*t2.z.z;
  return temp;
}

#endif //  _NABLA_LIB_TYPES_H_
