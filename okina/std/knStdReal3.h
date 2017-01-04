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
#ifndef _KN_STD_REAL3_H_
#define _KN_STD_REAL3_H_

#pragma pack(push,8)

// ****************************************************************************
// * real3
// ****************************************************************************
struct __attribute__ ((aligned(8))) real3 {
 public:
  // Les formules 4*(3*WARP_BASE(a)+0)+WARP_OFFSET(a) fonctionnent grâce au fait
  // que real3 est bien une struct avec les x,y et z en positions [0],[1] et [2]
  __attribute__ ((aligned(8))) real x;
  __attribute__ ((aligned(8))) real y;
  __attribute__ ((aligned(8))) real z;
 public:
  // Constructors
  inline real3(): x(0.0), y(0.0), z(0.0){}
  inline real3(double d) {x=d; y=d; z=d;}
  inline real3(double _x,double _y,double _z): x(_x), y(_y), z(_z){}
  inline real3(real f):x(f), y(f), z(f){}
  inline real3(real _x, real _y, real _z):x(_x), y(_y), z(_z){}
  inline real3(double *_x, double *_y, double *_z): x(*_x), y(*_y), z(*_z){}

  // Logicals
  //friend inline real3 operator&(const real3 &a, const real3 &b) { return real3((a.x&b.x), (a.y&b.y), (a.z&b.z)); }
  //friend inline real3 operator|(const real3 &a, const real3 &b) { return real3((a.x|b.x), (a.y|b.y), (a.z|b.z)); }
  //friend inline real3 operator^(const real3 &a, const real3 &b) { return real3((a.x^b.x), (a.y^b.y), (a.z^b.z)); }

  // Arithmetic operators
  friend inline real3 operator+(const real3 &a, const real3& b) { return real3((a.x+b.x), (a.y+b.y), (a.z+b.z));}
  friend inline real3 operator-(const real3 &a, const real3& b) { return real3((a.x-b.x), (a.y-b.y), (a.z-b.z));}
  friend inline real3 operator*(const real3 &a, const real3& b) { return real3((a.x*b.x), (a.y*b.y), (a.z*b.z));}
  friend inline real3 operator/(const real3 &a, const real3& b) { return real3((a.x/b.x), (a.y/b.y), (a.z/b.z));}
  friend inline real3 operator+(const volatile real3 &a, const volatile real3& b) { return real3((a.x+b.x), (a.y+b.y), (a.z+b.z));}

  // op= operators
  inline real3& operator+=(const real3& b) { return *this=real3((x+b.x),(y+b.y),(z+b.z));}
  inline real3& operator-=(const real3& b) { return *this=real3((x-b.x),(y-b.y),(z-b.z));}
  inline real3& operator*=(const real3& b) { return *this=real3((x*b.x),(y*b.y),(z*b.z));}
  inline real3& operator/=(const real3& b) { return *this=real3((x/b.x),(y/b.y),(z/b.z));}

  inline real3 operator-()const {return real3(0.0-x, 0.0-y, 0.0-z);}

  // op= operators with real
  inline real3& operator+=(real f){return *this=real3(x+f,y+f,z+f);}
  inline real3& operator-=(real f){return *this=real3(x-f,y-f,z-f);}
  inline real3& operator*=(real f){return *this=real3(x*f,y*f,z*f);}
  inline real3& operator/=(real f){return *this=real3(x/f,y/f,z/f);}

  inline real3& operator+=(double f){return *this=real3(x+f,y+f,z+f);}
  inline real3& operator-=(double f){return *this=real3(x-f,y-f,z-f);}
  inline real3& operator*=(double f){return *this=real3(x*f,y*f,z*f);}
  inline real3& operator/=(double f){return *this=real3(x/f,y/f,z/f);}
  friend inline real dot3(real3 u, real3 v){
    return real(u.x*v.x+u.y*v.y+u.z*v.z);
  }
  friend inline real norm(real3 u){ return real(square_root(dot3(u,u)));}

  friend inline real3 cross(real3 u, real3 v){
    return real3(((u.y*v.z)-(u.z*v.y)), ((u.z*v.x)-(u.x*v.z)), ((u.x*v.y)-(u.y*v.x)));
  }
};


#endif //  _KN_STD_REAL3_H_
