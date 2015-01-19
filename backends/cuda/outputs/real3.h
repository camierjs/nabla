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

//#pragma pack(push,8)


// ****************************************************************************
// * REAL DEFINITION
// ****************************************************************************
class __attribute__ ((aligned(8))) real {
public:
  double d;
  // Constructors
  __device__ inline Real(): d(0.0){}
  __device__ inline Real(double f):d(f){}
  __device__ inline Real(double *_x):d(*_x){}
  
  // Convertors
  __device__ inline operator double() const { return d; }

  // Arithmetic operators
  friend __device__ inline real operator+(const real &a, const real& b) { return __dadd_rn(a,b); }
  friend __device__ inline real operator-(const real &a, const real& b) { return __dsub_rn(a,b); }
  friend __device__ inline real operator*(const real &a, const real& b) { return __dmul_rn(a,b); }
  friend __device__ inline real operator/(const real &a, const real& b) { return __ddiv_rn(a,b); }
  
  friend __device__ real operator +(const real &a, const double &f) { return __dadd_rn(a,f); }
  friend __device__ real operator -(const real &a, const double &f) { return __dsub_rn(a,f); } 
  friend __device__ real operator *(const real &a, const double &f) { return __dmul_rn(a,f); } 
  friend __device__ real operator /(const real &a, const double &f) { return __ddiv_rn(a,f); }

  friend __device__ real operator +(const double &f,const real &a) { return __dadd_rn(f,a); }
  friend __device__ real operator -(const double &f,const real &a) { return __dsub_rn(f,a); } 
  friend __device__ real operator *(const double &f,const real &a) { return __dmul_rn(f,a); } 
  friend __device__ real operator /(const double &f,const real &a) { return __ddiv_rn(f,a); }

  // op= operators
  __device__ inline real& operator+=(const real& b) {
    return *this=real(__dadd_rn(this->d,b));
  }
  __device__ inline real& operator-=(const real& b) {
    return *this=real(__dsub_rn(this->d,b));
  }
  __device__ inline real& operator*=(const real& b) {
    return *this=real(__dmul_rn(this->d,b));
  }
  __device__ inline real& operator/=(const real& b) {
    return *this=real(__ddiv_rn(this->d,b));
  }
};


// ****************************************************************************
// * REAL3 DEFINITION
// ****************************************************************************
class __attribute__ ((aligned(8))) real3 {
public:
  real x;
  real y;
  real z;

  // Constructors
  __device__ inline Real3(): x(0.0), y(0.0), z(0.0){}
  __device__ inline Real3(double f):x(f), y(f), z(f){}
  __device__ inline Real3(real f):x(f), y(f), z(f){}
  __device__ inline Real3(real _x, real _y, real _z):x(_x), y(_y), z(_z){}
  __device__ inline Real3(real *_x, real *_y, real *_z):x(*_x), y(*_y), z(*_z){}

  // Arithmetic operators
  friend __device__ inline real3 operator+(const real3 &a, const real3& b) {
    const double x=__dadd_rn(a.x,b.x);
    const double y=__dadd_rn(a.y,b.y);
    const double z=__dadd_rn(a.z,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator-(const real3 &a, const real3& b) {
    const double x=__dsub_rn(a.x,b.x);
    const double y=__dsub_rn(a.y,b.y);
    const double z=__dsub_rn(a.z,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator*(const real3 &a, const real3& b) {
    const double x=__dmul_rn(a.x,b.x);
    const double y=__dmul_rn(a.y,b.y);
    const double z=__dmul_rn(a.z,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator/(const real3 &a, const real3& b) {
    const double x=__ddiv_rn(a.x,b.x);
    const double y=__ddiv_rn(a.y,b.y);
    const double z=__ddiv_rn(a.z,b.z);
    return real3(x,y,z);
  }
  
  friend __device__ inline real3 operator+(const real a, const real3 b) {
    const double x=__dadd_rn(a,b.x);
    const double y=__dadd_rn(a,b.y);
    const double z=__dadd_rn(a,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator-(const real a, const real3 b) {
    const double x=__dsub_rn(a,b.x);
    const double y=__dsub_rn(a,b.y);
    const double z=__dsub_rn(a,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator*(const real a, const real3 b) {
    const double x=__dmul_rn(a,b.x);
    const double y=__dmul_rn(a,b.y);
    const double z=__dmul_rn(a,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator/(const real a, const real3 b) {
    const double x=__ddiv_rn(a,b.x);
    const double y=__ddiv_rn(a,b.y);
    const double z=__ddiv_rn(a,b.z);
    return real3(x,y,z);
  }
  
  friend __device__ inline real3 operator+(const double a, const real3 b) {
    const double x=__dadd_rn(a,b.x);
    const double y=__dadd_rn(a,b.y);
    const double z=__dadd_rn(a,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator-(const double a, const real3 b) {
    const double x=__dsub_rn(a,b.x);
    const double y=__dsub_rn(a,b.y);
    const double z=__dsub_rn(a,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator*(const double a, const real3 b) {
    const double x=__dmul_rn(a,b.x);
    const double y=__dmul_rn(a,b.y);
    const double z=__dmul_rn(a,b.z);
    return real3(x,y,z);
  }
  friend __device__ inline real3 operator/(const double a, const real3 b) {
    const double x=__ddiv_rn(a,b.x);
    const double y=__ddiv_rn(a,b.y);
    const double z=__ddiv_rn(a,b.z);
    return real3(x,y,z);
  }
  

  // op= operators
  __device__ inline real3& operator+=(const real3& b) {
    const double x=__dadd_rn(this->x,b.x);
    const double y=__dadd_rn(this->y,b.y);
    const double z=__dadd_rn(this->z,b.z);
    return *this=real3(x,y,z);
  }
  __device__ inline real3& operator+=(const real& b) {
    const double x=__dadd_rn(this->x,b);
    const double y=__dadd_rn(this->y,b);
    const double z=__dadd_rn(this->z,b);
    return *this=real3(x,y,z);
  }
  __device__ inline real3& operator-=(const real3& b) {
    const double x=__dsub_rn(this->x,b.x);
    const double y=__dsub_rn(this->y,b.y);
    const double z=__dsub_rn(this->z,b.z);
    return *this=real3(x,y,z);
  }
  __device__ inline real3& operator*=(const real3& b) {
    const double x=__dmul_rn(this->x,b.x);
    const double y=__dmul_rn(this->y,b.y);
    const double z=__dmul_rn(this->z,b.z);
    return *this=real3(x,y,z);
  }
  __device__ inline real3& operator*=(const real& b) {
    const double x=__dmul_rn(this->x,b);
    const double y=__dmul_rn(this->y,b);
    const double z=__dmul_rn(this->z,b);
    return *this=real3(x,y,z);
  }
  __device__ inline real3& operator*=(const double b) {
    const double x=__dmul_rn(this->x,b);
    const double y=__dmul_rn(this->y,b);
    const double z=__dmul_rn(this->z,b);
    return *this=real3(x,y,z);
  }
  __device__ inline real3& operator/=(const real3& b) {
    const double x=__ddiv_rn(this->x,b.x);
    const double y=__ddiv_rn(this->y,b.y);
    const double z=__ddiv_rn(this->z,b.z);
    return *this=real3(x,y,z);
  }

  __device__ inline real3 operator-()const {return real3(-x, -y, -z);}
  
  __device__ friend inline real dot3(const real3 u, const real3 v){
    const Real _x=__dmul_rn(u.x,v.x);
    const Real _y=__dmul_rn(u.y,v.y);
    const Real _z=__dmul_rn(u.z,v.z);
    const Real xy=__dadd_rn(_x,_y);
    const Real result=__dadd_rn(xy,_z);
    return result;
  }  
  __device__ friend inline real3 cross3(real3 u, real3 v){
    const Real x=__dsub_rn(__dmul_rn(u.y,v.z),__dmul_rn(u.z,v.y));
    const Real y=__dsub_rn(__dmul_rn(u.z,v.x),__dmul_rn(u.x,v.z));
    const Real z=__dsub_rn(__dmul_rn(u.x,v.y),__dmul_rn(u.y,v.x));
    return real3(x,y,z);
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


// ****************************************************************************
// * opTernary
// ****************************************************************************

__device__ inline integer opTernary(const bool cond,
                         const int ifStatement,
                         const int elseStatement){
  //debug()<<"opTernary bool int int";
  if (cond) return integer(ifStatement);
  return integer(elseStatement);
}

__device__ inline real opTernary(const bool cond,
                      const double ifStatement,
                      const double elseStatement){
  //debug()<<"opTernary bool double double";
  if (cond) return real(ifStatement);
  return real(elseStatement);
}

__device__ inline real opTernary(const bool cond,
                      const double ifStatement,
                      const real&  elseStatement){
  //debug()<<"opTernary bool double real";
  if (cond) return real(ifStatement);
  return elseStatement;
}

__device__ inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const double elseStatement){
  //debug()<<"opTernary bool real double";
  if (cond) return ifStatement;
  return real(elseStatement);
}
__device__ inline real3 opTernary(const bool cond,
                      const real3& ifStatement,
                      const double elseStatement){
  if (cond) return ifStatement;
  return real3(elseStatement);
}

__device__ inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const real& elseStatement){
  //debug()<<"opTernary bool real real";
  if (cond) return ifStatement;
  return elseStatement;
}

__device__ inline real3 opTernary(const bool cond,
                       const double ifStatement,
                       const real3&  elseStatement){
  //debug()<<"opTernary bool double real3";
  if (cond) return Real3(ifStatement);
  return elseStatement;
}

