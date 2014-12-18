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

#pragma pack(push,8)

// ****************************************************************************
// * REAL3 DEFINITION
// ****************************************************************************
class __attribute__ ((aligned(8))) Real3 {
public:
  __attribute__ ((aligned(8))) double x;
  __attribute__ ((aligned(8))) double y;
  __attribute__ ((aligned(8))) double z;

  // Constructors
  __device__ inline Real3(): x(0.0), y(0.0), z(0.0){}
  __device__ inline Real3(double f):x(f), y(f), z(f){}
  __device__ inline Real3(double _x, double _y, double _z):x(_x), y(_y), z(_z){}
  __device__ inline Real3(double *_x, double *_y, double *_z):x(*_x), y(*_y), z(*_z){}

  // Arithmetic operators
  friend __device__ inline real3 operator+(const real3 &a, const real3& b) { return real3((a.x+b.x), (a.y+b.y), (a.z+b.z));}
  friend __device__ inline real3 operator-(const real3 &a, const real3& b) { return real3((a.x-b.x), (a.y-b.y), (a.z-b.z));}
  friend __device__ inline real3 operator*(const real3 &a, const real3& b) { return real3((a.x*b.x), (a.y*b.y), (a.z*b.z));}
  friend __device__ inline real3 operator/(const real3 &a, const real3& b) { return real3((a.x/b.x), (a.y/b.y), (a.z/b.z));}

  // op= operators
  __device__ inline real3& operator+=(const real3& b) {return *this=real3((x+b.x),(y+b.y),(z+b.z));}
  __device__ inline real3& operator-=(const real3& b) {return *this=real3((x-b.x),(y-b.y),(z-b.z));}

  __device__ inline real3 operator-()const {return real3(-x, -y, -z);}
  __device__ friend inline real dot(real3 u, real3 v){return real(u.x*v.x+u.y*v.y+u.z*v.z);}
  
  __device__ friend inline real3 cross(real3 u, real3 v){
    return real3(((u.y*v.z)-(u.z*v.y)), ((u.z*v.x)-(u.x*v.z)), ((u.x*v.y)-(u.y*v.x)));
  }
};


/******************************************************************************
 * Gather avec des real3
 ******************************************************************************/
__device__ inline void gather3ki(const int a, real3 *data, real3 *gthr, int i){
  const double *p=(double *)data;
  const double value=p[3*a+i];
  if (i==0) gthr->x=value;
  if (i==1) gthr->y=value;
  if (i==2) gthr->z=value;
}

__device__ inline void gather3k(const int a, real3 *data, real3 *gthr){
  gather3ki(a, data, gthr, 0);
  gather3ki(a, data, gthr, 1);
  gather3ki(a, data, gthr, 2);
}

__device__ inline real gatherk_and_zero_neg_ones(const int a, real *data){
  if (a>=0) return *(data+a);
  return 0.0;
}

__device__ inline void gatherFromNode_k(const int a, real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,data);
}

__device__ inline void gatherFromNode_3kiArray8(const int a, const int corner,
                                                real3 *data, real3 *gthr, int i){
  const double *p=(double *)data;
  const double value=(a<0)?0.0:p[3*8*a+3*corner+i];
  if (i==0) gthr->x=value;
  if (i==1) gthr->y=value;
  if (i==2) gthr->z=value;
}

__device__ inline void gatherFromNode_3kArray8(const int a, const int corner,
                                               real3 *data, real3 *gthr){
  gatherFromNode_3kiArray8(a,corner, data, gthr, 0);
  gatherFromNode_3kiArray8(a,corner, data, gthr, 1);
  gatherFromNode_3kiArray8(a,corner, data, gthr, 2);
}
