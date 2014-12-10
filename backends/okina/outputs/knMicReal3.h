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
#ifndef _KN_MIC_REAL3_H_
#define _KN_MIC_REAL3_H_


// ****************************************************************************
// * Real3
// ****************************************************************************
struct __attribute__ ((aligned(64))) real3 {
 public:
  // Les formules 4*(3*WARP_BASE(a)+0)+WARP_OFFSET(a) fonctionnent grâce au fait
  // que real3 est bien une struct avec les x,y et z en positions [0],[1] et [2]
  real x;
  real y;
  real z;
 public:
  // Constructors
  inline real3(): x(_mm512_setzero_pd()), y(_mm512_setzero_pd()), z(_mm512_setzero_pd()){}
  inline real3(double d): x(_mm512_set1_pd(d)), y(_mm512_set1_pd(d)), z(_mm512_set1_pd(d)){}
  inline real3(double _x,double _y,double _z): x(_mm512_set1_pd(_x)), y(_mm512_set1_pd(_y)), z(_mm512_set1_pd(_z)){}
  inline real3(real f):x(f), y(f), z(f){}
  inline real3(real _x, real _y, real _z):x(_x), y(_y), z(_z){}
  inline real3(double *_x, double *_y, double *_z): x(_mm512_load_pd(_x)), y(_mm512_load_pd(_y)), z(_mm512_load_pd(_z)){}

  // Logicals
  friend inline real3 operator&(const real3 &a, const real3 &b) { return real3(a.x&b.y, a.y&b.y, a.z&b.z); }
  friend inline real3 operator|(const real3 &a, const real3 &b) { return real3(a.x|b.x, a.y|b.y, a.z|b.z); }
  friend inline real3 operator^(const real3 &a, const real3 &b) { return real3(a.x^b.x, a.y^b.y, a.z^b.z); }

  // Arithmetic operators
  friend inline real3 operator+(const real3 &a, const real3& b) { return real3(_mm512_add_pd(a.x,b.x), _mm512_add_pd(a.y,b.y), _mm512_add_pd(a.z,b.z));}
  friend inline real3 operator-(const real3 &a, const real3& b) { return real3(_mm512_sub_pd(a.x,b.x), _mm512_sub_pd(a.y,b.y), _mm512_sub_pd(a.z,b.z));}
  friend inline real3 operator*(const real3 &a, const real3& b) { return real3(_mm512_mul_pd(a.x,b.x), _mm512_mul_pd(a.y,b.y), _mm512_mul_pd(a.z,b.z));}
  friend inline real3 operator/(const real3 &a, const real3& b) { return real3(_mm512_div_pd(a.x,b.x), _mm512_div_pd(a.y,b.y), _mm512_div_pd(a.z,b.z));}

  // op= operators
  inline real3& operator+=(const real3& b) { return *this=real3(_mm512_add_pd(x,b.x),_mm512_add_pd(y,b.y),_mm512_add_pd(z,b.z));}
  inline real3& operator-=(const real3& b) { return *this=real3(_mm512_sub_pd(x,b.x),_mm512_sub_pd(y,b.y),_mm512_sub_pd(z,b.z));}
  inline real3& operator*=(const real3& b) { return *this=real3(_mm512_mul_pd(x,b.x),_mm512_mul_pd(y,b.y),_mm512_mul_pd(z,b.z));}
  inline real3& operator/=(const real3& b) { return *this=real3(_mm512_div_pd(x,b.x),_mm512_div_pd(y,b.y),_mm512_div_pd(z,b.z));}

  //inline real3 operator-(){return real3(-x, -y, -z);}
  inline real3 operator-()const {return real3(-x, -y, -z);}

  // op= operators with real
  inline real3& operator+=(real f){return *this=real3(x+f,y+f,z+f);}
  inline real3& operator-=(real f){return *this=real3(x-f,y-f,z-f);}
  inline real3& operator*=(real f){return *this=real3(x*f,y*f,z*f);}
  inline real3& operator/=(real f){return *this=real3(x/f,y/f,z/f);}

  inline real3& operator+=(double f){return *this=real3(x+f,y+f,z+f);}
  inline real3& operator-=(double f){return *this=real3(x-f,y-f,z-f);}
  inline real3& operator*=(double f){return *this=real3(x*f,y*f,z*f);}
  inline real3& operator/=(double f){return *this=real3(x/f,y/f,z/f);}
  
  friend inline real dot3(real3 u, real3 v){ return real(u.x*v.x+u.y*v.y+u.z*v.z);}
  friend inline real norm(real3 u){ return real(_mm512_sqrt_pd(dot3(u,u)));}
  friend inline real3 cross(real3 u, real3 v){
    return
      real3(_mm512_sub_pd( _mm512_mul_pd(u.y,v.z) , _mm512_mul_pd(u.z,v.y) ),
            _mm512_sub_pd( _mm512_mul_pd(u.z,v.x) , _mm512_mul_pd(u.x,v.z) ),
            _mm512_sub_pd( _mm512_mul_pd(u.x,v.y) , _mm512_mul_pd(u.y,v.x) ));
  }
};

#endif // _KN_MIC_REAL3_H_
