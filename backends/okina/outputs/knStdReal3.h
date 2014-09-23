#ifndef _KN_STD_REAL3_H_
#define _KN_STD_REAL3_H_


// ****************************************************************************
// * real3
// ****************************************************************************
class real3 {
 public:
  // Les formules 4*(3*WARP_BASE(a)+0)+WARP_OFFSET(a) fonctionnent grâce au fait
  // que real3 est bien une struct avec les x,y et z en positions [0],[1] et [2]
  real x;
  real y;
  real z;
 public:
  // Constructors
  inline real3(): x(0.0), y(0.0), z(0.0){}
  inline real3(double d) {x=d; y=d; z=d;}
  inline real3(double _x,double _y,double _z): x(_x), y(_y), z(_z){}
  inline real3(real f):x(f), y(f), z(f){}
  inline real3(real _x, real _y, real _z):x(_x), y(_y), z(_z){}
  inline real3(double *_x, double *_y, double *_z): x(*_x), y(*_y), z(*_z){}

  // Logicals
  friend inline real3 operator&(const real3 &a, const real3 &b) { return real3((a.x&b.x), (a.y&b.y), (a.z&b.z)); }
  friend inline real3 operator|(const real3 &a, const real3 &b) { return real3( (a.x|b.x),  (a.y|b.y),  (a.z|b.z)); }
  friend inline real3 operator^(const real3 &a, const real3 &b) { return real3((a.x^b.x), (a.y^b.y), (a.z^b.z)); }

  // Arithmetic operators
  friend inline real3 operator+(const real3 &a, const real3& b) { return real3((a.x+b.x), (a.y+b.y), (a.z+b.z));}
  friend inline real3 operator-(const real3 &a, const real3& b) { return real3((a.x-b.x), (a.y-b.y), (a.z-b.z));}
  friend inline real3 operator*(const real3 &a, const real3& b) { return real3((a.x*b.x), (a.y*b.y), (a.z*b.z));}
  friend inline real3 operator/(const real3 &a, const real3& b) { return real3((a.x/b.x), (a.y/b.y), (a.z/b.z));}

  // op= operators
  inline real3& operator+=(const real3& b) { return *this=real3((x+b.x),(y+b.y),(z+b.z));}
  inline real3& operator-=(const real3& b) { return *this=real3((x-b.x),(y-b.y),(z-b.z));}
  inline real3& operator*=(const real3& b) { return *this=real3((x*b.x),(y*b.y),(z*b.z));}
  inline real3& operator/=(const real3& b) { return *this=real3((x/b.x),(y/b.y),(z/b.z));}

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
  friend inline real norm(real3 u){ return real(rsqrt(dot3(u,u)));}
  friend inline real3 cross(real3 u, real3 v){
    return real3(((u.y*v.z)-(u.z*v.y)), ((u.z*v.x)-(u.x*v.z)), ((u.x*v.y)-(u.y*v.x)));
  }
};


#endif //  _KN_STD_REAL3_H_
