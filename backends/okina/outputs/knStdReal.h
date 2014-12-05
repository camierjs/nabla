#ifndef _KN_STD_REAL_H_
#define _KN_STD_REAL_H_

#pragma pack(push,8)

// ****************************************************************************
// * real
// ****************************************************************************
class __attribute__ ((aligned(8))) real {
 protected:
  double vec;
 public:
  // Constructors
  inline real(): vec(0.0){}
  inline real(int i):vec((double)i){}
  inline real(long i):vec((double)i){}
  inline real(double d):vec(d){}
  inline real(double *x):vec(*x){}

  // Convertors
  inline operator double() const { return vec; }
  
  // Logicals
  friend inline real operator &(const real &a, const real &b) { return (a&b); }
  friend inline real operator |(const real &a, const real &b) { return (a|b); }
  friend inline real operator ^(const real &a, const real &b) { return (a^b); }

  // Arithmetics
  //friend inline real operator +(const real &a, const real &b) { return (a+b); }
  //friend inline real operator -(const real &a, const real &b) { return (a-b); }
  //friend inline real operator *(const real &a, const real &b) { return (a*b); }
  //friend inline real operator /(const real &a, const real &b) { return (a/b); }
  
  inline real& operator +=(const real &a) { return *this = (vec+a); }
  inline real& operator -=(const real &a) { return *this = (vec-a); }
  inline real& operator *=(const real &a) { return *this = (vec*a); }
  inline real& operator /=(const real &a) { return *this = (vec/a); }
  inline real& operator &=(const real &a) { return *this = (vec&a); }
  inline real& operator |=(const real &a) { return *this = (vec|a); }
  inline real& operator ^=(const real &a) { return *this = (vec^a); }
  
  //inline real operator -() const { return -(*this); }

  // Mixed vector-scalar operations
  inline real& operator *=(const double &f) { return *this = (vec*(f)); }
  inline real& operator /=(const double &f) { return *this = (vec/(f)); }
  inline real& operator +=(const double &f) { return *this = (vec+(f)); }
  inline real& operator +=(double &f) { return *this = (vec+(f)); }
  inline real& operator -=(const double &f) { return *this = (vec-(f)); }
  
  //friend inline real operator +(const real &a, const double &f) { return (a+f); }
  //friend inline real operator -(const real &a, const double &f) { return (a-f); } 
  //friend inline real operator *(const real &a, const double &f) { return (a*f); } 
  //friend inline real operator /(const real &a, const double &f) { return (a/f); }

  //friend inline real operator +(const double &f, const real &a) { return (f+a); }
  //friend inline real operator -(const double &f, const real &a) { return (f-a); } 
  //friend inline real operator *(const double &f, const real &a) { return (f*a); } 
  //friend inline real operator /(const double &f, const real &a) { return (f/a); }

  friend inline real sqrt(const real &a) { return ::sqrt(a); }
  friend inline real ceil(const real &a)   { return ::ceilf((a)); }
  friend inline real floor(const real &a)  { return ::floor((a)); }
  friend inline real trunc(const real &a)  { return ::trunc((a)); }
  
  friend inline real min(const real &r, const real &s){ return ::fmin(r,s);}
  friend inline real max(const real &r, const real &s){ return ::fmax(r,s);}
  
  friend inline real rcbrt(const real &a){
    return real(::cbrt(a));
  }
  friend inline real norm(real u){ return ::fabs(u);}

  // Element Access Only, no modifications to elements
  inline const double& operator[](int i) const  {
  // Assert enabled only during debug
    assert((0 <= i));
    double *dp = (double*)&vec;
    return *(dp+i);
  }
  // Element Access and Modification
  inline double& operator[](int i) {
  // Assert enabled only during debug
    assert((0 <= i));
    double *dp = (double*)&vec;
    return *(dp+i);
  }

};

inline double ReduceMinToDouble(Real r){ return r; }

inline double ReduceMaxToDouble(Real r){ return r; }

//inline real min(const double r, const double s){ return ::fmin(r,s);}
//inline real max(const double r, const double s){ return ::fmax(r,s);}

#endif //  _KN_STD_REAL_H_
