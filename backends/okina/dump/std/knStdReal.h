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
  inline operator double() const volatile { return vec; }
  inline operator double() const { return vec; }
  
  // Logicals
  //friend inline real operator &(const real &a, const real &b) { return (a&b); }
  //friend inline real operator |(const real &a, const real &b) { return (a|b); }
  //friend inline real operator ^(const real &a, const real &b) { return (a^b); }

  // Arithmetics
  //friend inline real operator +(const real &a, const real &b) { return (a+b); }
  //friend inline real operator -(const real &a, const real &b) { return (a-b); }
  //friend inline real operator *(const real &a, const real &b) { return (a*b); }
  //friend inline real operator /(const real &a, const real &b) { return (a/b); }
  
  inline real& operator +=(const real &a) { return *this = (vec+a); }
  inline real& operator -=(const real &a) { return *this = (vec-a); }
  inline real& operator *=(const real &a) { return *this = (vec*a); }
  inline real& operator /=(const real &a) { return *this = (vec/a); }
  //inline real& operator &=(const real &a) { return *this = (vec&a); }
  //inline real& operator |=(const real &a) { return *this = (vec|a); }
  //inline real& operator ^=(const real &a) { return *this = (vec^a); }
  
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
  
  friend inline real cube_root(const real &a){
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
