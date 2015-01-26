///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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
#ifndef _KN_BACKUP_H_
#define _KN_BACKUP_H_



// ****************************************************************************
// * bits operations
// ****************************************************************************
//inline __m256d operator&&(const __m256d m, const __m256d n) {return _mm256_and_pd(m,n);}
//inline __m256d operator&(const real& r, const int i) {return _mm256_and_pd(r,_mm256_set1_pd(i));}
//inline __m256d operator&(const real& r, const __m256d m) {return _mm256_and_pd(r,m);}
////inline real& operator|(const real& r) {return *this = _mm256_or_pd(m256,r);}
////inline real& operator^(const real& r) {return *this = _mm256_xor_pd(m256,r);}


/*inline real rabs(const real &r){
  const double *dp = (double*) &r;
  return _mm256_set_pd(::fabs(*dp+3),::fabs(*dp+2),::fabs(*dp+1),::fabs(*dp+0));
  }*/

//inline __m256d operator-(const real &a)const { return _mm256_xor_pd(_mm256_set1_pd(-0.0), a); }

/*inline __m256d rabs(const real &a){
  static const union{
    int i[8];
    __m256d m;
  } __f64vec4_abs_mask = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff,
                           0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};
  return _mm256_and_pd(a, __f64vec4_abs_mask.m);
  }*/

/*//inline double fabs(const double r){ return ::fabs(r);}
inline __m256d operator<(const __m256d v, const double d){
  return _mm256_cmp_pd(v, _mm256_set1_pd(d), _CMP_LT_OS);
}
inline __m256d operator<(const __m256d v, const __m256d w){
  return _mm256_cmp_pd(v, w, _CMP_LT_OS);
}
inline __m256d operator>(const __m256d v, const __m256d w){
  return _mm256_cmp_pd(v, w, _CMP_GT_OS);
}
inline __m256d operator>=(const __m256d v, const __m256d w){
  return _mm256_cmp_pd(v, w, _CMP_GE_OS);
  }*/





// ****************************************************************************
// * != operator
// ****************************************************************************
/*inline __m256d operator!=(const real& r, double d){
  return _mm256_cmp_pd(r, _mm256_set1_pd(d), _CMP_NEQ_OQ);
}
*/

//inline bool operator==(double d, real& v){  return v==d;}

/*inline bool operator==(const real& r, const double d){
  const __m256d a = _mm256_cmp_pd(r, _mm256_set1_pd(d), _CMP_EQ_OQ);
  const double *dp = (double*) &a;
  if ((*dp==0.0) && (*(dp+1)==0.0) && (*(dp+2)==0.0) && (*(dp+3)==0.0) )
    return false;
  return true; 
  }*/

/*inline bool operator==(const real& r, const real& s){
  const __m256d a = _mm256_cmp_pd(r, s, _CMP_EQ_OQ);
  const double *dp = (double*) &a;
  if ((*dp==0.0) && (*(dp+1)==0.0) && (*(dp+2)==0.0) && (*(dp+3)==0.0) )
    return false;
  return true; 
}*/



// ****************************************************************************
// * <= operator
// ****************************************************************************
/*inline __m256d operator<=(const real& r, const double d){
  return _mm256_cmp_pd(r, _mm256_set1_pd(d), _CMP_LE_OQ);
  }*/

/*inline __m256d operator<=(const real& r, const real& s){
  return _mm256_cmp_pd(r, s, _CMP_LE_OQ);
}

inline bool operator<=(const real& r, const double d){
  const __m256d a = _mm256_cmp_pd(r, _mm256_set1_pd(d), _CMP_LE_OQ);
  const double *dp = (double*) &a;
  if ((*dp==0.0) && (*(dp+1)==0.0) && (*(dp+2)==0.0) && (*(dp+3)==0.0) )
    return false;
  return true; 
}
inline bool operator>=(const real& r, const double d){
  const __m256d a = _mm256_cmp_pd(r, _mm256_set1_pd(d), _CMP_GE_OQ);
  const double *dp = (double*) &a;
  if ((*dp==0.0) && (*(dp+1)==0.0) && (*(dp+2)==0.0) && (*(dp+3)==0.0) )
    return false;
  return true; 
  }*/

// ****************************************************************************
// * +,-,*,/,neg operators
// ****************************************************************************
/*inline real operator+(double f, const real& v){return v+f;}
inline real operator-(double f, const real& v){return real(f)-v;}
inline real operator*(double f, const real& v){return v*f;}
inline real operator/(double f, const real& v){return real(f)/v;}
*/

// ****************************************************************************
// * double3
// ****************************************************************************
/*struct __attribute__ ((aligned(32))) double3 {
public:
  double x;
  double y;
  double z;

// Constructors
  inline double3(): x(0.0), y(0.0), z(0.0){}
  inline double3(const double f):x(f), y(f), z(f){}
  inline double3(const double _x,const double _y,const double _z):x(_x), y(_y), z(_z){}
  inline double3& operator=(const double3& r) {x=r.x;y=r.y;z=r.z; return *this;}
  // Arithmetic operators
  inline double3 operator+(const double3& v)const{ return double3(x+v.x, y+v.y, z+v.z);}
  inline double3 operator-(const double3& v)const{ return double3(x-v.x, y-v.y, z-v.z);}
  inline double3 operator*(const double3& v)const{ return double3(x*v.x, y*v.y, z*v.z);}
  inline double3 operator/(const double3& v)const{ return double3(x/v.x, y/v.y, z/v.z);}
  // Binary operators with double3
  inline bool operator<(double3 u)const{ return true;}
  // Arithmetic operators with doubles
  inline double3 operator+(double f)const{ return double3(x+f, y+f, z+f);}
  inline double3 operator-(double f)const{ return double3(x-f, y-f, z-f);}
  inline double3 operator*(double f)const{ return double3(x*f, y*f, z*f);}
  inline double3 operator/(double f)const{ return double3(x/f, y/f, z/f);}
 
  // op= operators
  inline double3& operator+=(const double3& v){
    x=x+v.x; y=y+v.y; z=z+v.z; return *this;}
  inline double3& operator-=(const double3& v){
    x=x-v.x; y=y-v.y; z=z-v.x; return *this;}
  inline double3& operator*=(const double3& v){
    x=x*v.x; y=y*v.y; z=z*v.z; return *this;}
  inline double3& operator/=(const double3& v){
    x=x/v.x; y=y/v.y; z=z/v.z; return *this;}

  // op= operators with doubles
  inline double3& operator=(double f){ x=f; y=f; z=f; return *this;}
  inline double3& operator+=(double f){ x=x+f; y=y+f; z=z+f; return *this;}
  inline double3& operator-=(double f){ x=x-f; y=y-f; z=z-f; return *this;}
  inline double3& operator*=(double f){ x=x*f; y=y*f; z=z*f; return *this;}
  inline double3& operator/=(double f){ x=x/f; y=y/f; z=z/f; return *this;}

  inline bool operator==(const real& r) const {return true;}
  inline bool operator==(const real3& r) const {return true;}
  inline bool operator==(const double d) const {return (x==d);}
};

inline double3 operator+(double f, const double3& v){return v+f;}
inline double3 operator-(double f, const double3& v){return double3(f)-v;}
inline double3 operator*(double f, const double3& v){return v*f;}
inline double3 operator/(double f, const double3& v){return double3(f)/v;}

inline real dot3(double3 u, double3 v){ return real(u.x*v.x+u.y*v.y+u.z*v.z);}
inline real norm(double3 u){ return real(sqrt(dot3(u,u)));}
inline double3 cross(double3 u, double3 v){
  return double3(u.y*v.z - u.z*v.y ,
                 u.z*v.x - u.x*v.z ,
                 u.x*v.y - u.y*v.x);
}
*/
/*inline bool operator==(double d, const real3& v){
//#warning double == real3 operation
  return real(d)==v;
  }*/



#endif //  _KN_BACKUP_H_
