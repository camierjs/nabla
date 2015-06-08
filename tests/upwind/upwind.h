#ifndef __OKINA_upwind_H__
#define __OKINA_upwind_H__


// *****************************************************************************
// * Okina includes
// *****************************************************************************
 // from nabla->simd->includes
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <assert.h>
#include <stdarg.h>
//#include <mathimf.h>
#include <iostream>
#include <sstream>
#include <fstream>
int omp_get_max_threads(void){return 1;}
int omp_get_thread_num(void){return 0;}
 // fromnabla->parallel->includes()


// *****************************************************************************
// * Defines
// *****************************************************************************
#define integer Integer
#define real Real
#define WARP_SIZE (1<<WARP_BIT)
#define WARP_ALIGN (8<<WARP_BIT)
#define NABLA_NB_GLOBAL_WARP WARP_SIZE
#define reducemin(a) 0.0
#define rabs(a) fabs(a)
#define set(a) a
#define set1(cst) cst
#define square_root(u) sqrt(u)
#define cube_root(u) cbrt(u)
#define store(u,_u) (*u=_u)
#define load(u) (*u)
#define zero() 0.0
#define DBG_MODE (false)
#define DBG_LVL (DBG_INI)
#define DBG_OFF 0x0000ul
#define DBG_CELL_VOLUME 0x0001ul
#define DBG_CELL_CQS 0x0002ul
#define DBG_GTH 0x0004ul
#define DBG_NODE_FORCE 0x0008ul
#define DBG_INI_EOS 0x0010ul
#define DBG_EOS 0x0020ul
#define DBG_DENSITY 0x0040ul
#define DBG_MOVE_NODE 0x0080ul
#define DBG_INI 0x0100ul
#define DBG_INI_CELL 0x0200ul
#define DBG_INI_NODE 0x0400ul
#define DBG_LOOP 0x0800ul
#define DBG_FUNC_IN 0x1000ul
#define DBG_FUNC_OUT 0x2000ul
#define DBG_VELOCITY 0x4000ul
#define DBG_BOUNDARIES 0x8000ul
#define DBG_ALL 0xFFFFul
#define opAdd(u,v) (u+v)
#define opSub(u,v) (u-v)
#define opDiv(u,v) (u/v)
#define opMul(u,v) (u*v)
#define opMod(u,v) (u%v)
#define opScaMul(u,v) dot3(u,v)
#define opVecMul(u,v) cross(u,v)
#define dot dot3
#define knAt(a) 
#define fatal(a,b) exit(-1)
#define synchronize(a) _Pragma("omp barrier")
#define mpi_reduce(how,what) how##ToDouble(what)
#define reduce(how,what) how##ToDouble(what)
#define xyz int
#define GlobalIteration global_iteration
#define MD_DirX 0
#define MD_DirY 1
#define MD_DirZ 2
#define File std::ofstream&
#define file(name,ext) std::ofstream name(#name "." #ext)


// *****************************************************************************
// * Typedefs
// *****************************************************************************
typedef struct real3 Real3;


// *****************************************************************************
// * Forwards
// *****************************************************************************
inline std::ostream& info(){std::cout.flush();std::cout<<"\n";return std::cout;}
inline std::ostream& debug(){std::cout.flush();std::cout<<"\n";return std::cout;}
inline int WARP_BASE(int a){ return (a>>WARP_BIT);}
inline int WARP_OFFSET(int a){ return (a&(WARP_SIZE-1));}
inline int WARP_NFFSET(int a){ return ((WARP_SIZE-1)-WARP_OFFSET(a));}
static void nabla_ini_node_coords(void);
static void verifCoords(void);
#ifndef _KN_STD_INTEGER_H_
#define _KN_STD_INTEGER_H_

// ****************************************************************************
// * Standard integer
// ****************************************************************************
class integer {
public:
  int vec;
public:
  // Constructors
  inline integer():vec(0){}
  inline integer(int i):vec(i){}
  //inline integer(int i3, int i2, int i1, int i0){vec=_mm_set_epi32(i3, i2, i1, i0);}
  
  // Convertors
  inline operator int() const { return vec; }

  //inline integer operator&(const integer &b) { return (vec&b); }
  //inline integer operator|(const integer &b) { return (vec|b); }
  //inline integer operator^(const integer &b) { return (vec^b); }
      
  inline integer& operator&=(const integer &a) { return *this = (integer) (vec&a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) (vec|a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) (vec^a); }

  inline integer& operator+=(const integer &a) { return *this = (integer)(vec+a); }
  inline integer& operator-=(const integer &a) { return *this = (integer)(vec-a); }   

  //friend inline bool operator==(const integer &a, const int i){ return a==i; }
};

//inline integer operator&(const integer &a, const integer &b) { return (a&b); }
//inline integer operator|(const integer &a, const integer &b) { return (a|b); }
//inline integer operator^(const integer &a, const integer &b) { return (a^b); }

//inline integer operator&(const integer &a, const int &b) { return integer(a&b); }
//inline integer operator|(const integer &a, const int &b) { return integer(a|b); }
//inline integer operator^(const integer &a, const int &b) { return integer(a^b); }

#endif //  _KN_STD_INTEGER_H_
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
#ifndef _KN_STD_REAL3_H_
#define _KN_STD_REAL3_H_

#pragma pack(push,8)

// ****************************************************************************
// * real3
// ****************************************************************************
class __attribute__ ((aligned(8))) real3 {
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
  friend inline real3 operator&(const real3 &a, const real3 &b) { return real3((a.x&b.x), (a.y&b.y), (a.z&b.z)); }
  friend inline real3 operator|(const real3 &a, const real3 &b) { return real3((a.x|b.x), (a.y|b.y), (a.z|b.z)); }
  friend inline real3 operator^(const real3 &a, const real3 &b) { return real3((a.x^b.x), (a.y^b.y), (a.z^b.z)); }

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
#ifndef _KN_STD_TERNARY_H_
#define _KN_STD_TERNARY_H_


// ****************************************************************************
// * opTernary
// ****************************************************************************

inline integer opTernary(const bool cond,
                         const int ifStatement,
                         const int elseStatement){
  //debug()<<"opTernary bool int int";
  if (cond) return integer(ifStatement);
  return integer(elseStatement);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const double elseStatement){
  //debug()<<"opTernary bool double double";
  if (cond) return real(ifStatement);
  return real(elseStatement);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const real&  elseStatement){
  //debug()<<"opTernary bool double real";
  if (cond) return real(ifStatement);
  return elseStatement;
}

inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const double elseStatement){
  //debug()<<"opTernary bool real double";
  if (cond) return ifStatement;
  return real(elseStatement);
}
inline real3 opTernary(const bool cond,
                      const real3& ifStatement,
                      const double elseStatement){
  if (cond) return ifStatement;
  return real3(elseStatement);
}

inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const real& elseStatement){
  //debug()<<"opTernary bool real real";
  if (cond) return ifStatement;
  return elseStatement;
}

inline real3 opTernary(const bool cond,
                       const double ifStatement,
                       const real3&  elseStatement){
  //debug()<<"opTernary bool double real3";
  if (cond) return Real3(ifStatement);
  return elseStatement;
}

#endif //  _KN_STD_TERNARY_H_
#ifndef _KN_STD_GATHER_H_
#define _KN_STD_GATHER_H_


/******************************************************************************
 * Gather: (X is the data @ offset x)       a            b       c   d
 * data:   |....|....|....|....|....|....|..A.|....|....|B...|...C|..D.|....|      
 * gather: |ABCD|
 ******************************************************************************/
inline void gatherk_load(const int a, real *data, real *gthr){
  *gthr=*(data+a);
}

inline void gatherk(const int a, real *data, real *gthr){
  gatherk_load(a,data,gthr);
}


inline real gatherk_and_zero_neg_ones(const int a, real *data){
  if (a>=0) return *(data+a);
  return 0.0;
}

inline void gatherFromNode_k(const int a, real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,data);
}


/******************************************************************************
 * Gather avec des real3
 ******************************************************************************/
inline void gather3ki(const int a, real3 *data, real3 *gthr, int i){
  //debug()<<"gather3ki, i="<<i;
  double *p=(double *)data;
  double value=p[3*a+i];
  if (i==0) (*gthr).x=value;
  if (i==1) (*gthr).y=value;
  if (i==2) (*gthr).z=value;
}

inline void gather3k(const int a, real3 *data, real3 *gthr){
  //debug()<<"gather3k";
  gather3ki(a, data, gthr, 0);
  gather3ki(a, data, gthr, 1);
  gather3ki(a, data, gthr, 2);
  //debug()<<"gather3k done";
}



/******************************************************************************
 * Gather avec des real3[nodes(#8)]
 ******************************************************************************/
inline void gatherFromNode_3kiArray8(const int a, const int corner,
                                     real3 *data, real3 *gthr, int i){
  //debug()<<"gather3ki, i="<<i;
  double *p=(double *)data;
  double value=(a<0)?0.0:p[3*8*a+3*corner+i];
  if (i==0) (*gthr).x=value;
  if (i==1) (*gthr).y=value;
  if (i==2) (*gthr).z=value;
}

inline void gatherFromNode_3kArray8(const int a, const int corner,
                                    real3 *data, real3 *gthr){
  //debug()<<"gather3k";
  gatherFromNode_3kiArray8(a,corner, data, gthr, 0);
  gatherFromNode_3kiArray8(a,corner, data, gthr, 1);
  gatherFromNode_3kiArray8(a,corner, data, gthr, 2);
  //debug()<<"gather3k done";
}


#endif //  _KN_STD_GATHER_H_
#ifndef _KN_STD_SCATTER_H_
#define _KN_STD_SCATTER_H_


// *****************************************************************************
// * Scatter: (X is the data @ offset x)
// * scatter: |ABCD| and offsets:    a                 b       c   d
// * data:    |....|....|....|....|..A.|....|....|....|B...|...C|..D.|....|....|
// * ! à la séquence car quand c et d sont sur le même warp, ça percute
// ******************************************************************************
inline void scatterk(const int a, real *scatter, real *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  p[a]=s[0];
}


// *****************************************************************************
// * Scatter for real3
// *****************************************************************************
inline void scatter3k(const int a, real3 *scatter, real3 *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  p[3*a+0]=s[0];
  p[3*a+1]=s[1];
  p[3*a+2]=s[2];
}

#endif //  _KN_STD_SCATTER_H_
#ifndef _KN_STD_OSTREAM_H_
#define _KN_STD_OSTREAM_H_


// ****************************************************************************
// * INTEGERS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const integer &a){
  int *ip = (int*)&(a);
  //return os << "["<<*ip<<"]";
  return os << *ip;
}


// ****************************************************************************
// * REALS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const Real &a){
  double *fp = (double*)&a;
  //return os << "["<<*fp<<"]";
  return os << *fp;
}


// ****************************************************************************
// * REALS_3
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const Real3 &a){
  double *x = (double*)&(a.x);
  double *y = (double*)&(a.y);
  double *z = (double*)&(a.z);
  //return os << "[("<<*x<<","<<*y<<","<<*z<< ")]";
  return os << "("<<*x<<","<<*y<<","<<*z<< ")";
}

#endif // _KN_STD_OSTREAM_H_
#ifndef _KN_DBG_HPP_
#define _KN_DBG_HPP_

#include <stdarg.h>

/******************************************************************************
 * Outils de traces
 *****************************************************************************/

void dbg(const unsigned int flag, const char *format, ...){
  if (!DBG_MODE) return;
  if ((flag&DBG_LVL)==0) return;
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  fflush(stdout);
  va_end(args);
}

#define dbgFuncIn()  do{dbg(DBG_FUNC_IN,"\n\t > %s",__FUNCTION__);}while(0)
#define dbgFuncOut() do{dbg(DBG_FUNC_OUT,"\n\t\t < %s",__FUNCTION__);}while(0)



inline void dbgReal3(const unsigned int flag, real3& v){
  if (!DBG_MODE) return;
  if ((flag&DBG_LVL)==0) return;
  double x[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  double y[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  double z[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  store(x, v.x);
  store(y, v.y);
  store(z, v.z);
//  for(int i=WARP_SIZE-1;i>=0;--i)
  for(int i=0;i<WARP_SIZE;i+=1)
    printf("\n\t\t\t[%.14f,%.14f,%.14f]", x[i], y[i], z[i]);
  fflush(stdout);
  }


inline void dbgReal(const unsigned int flag, real v){
  if (!DBG_MODE) return;
  if ((flag&DBG_LVL)==0) return;
  double x[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  store(x, v);
  printf("[");
//  for(int i=WARP_SIZE-1;i>=0;--i)
  for(int i=0;i<WARP_SIZE;i+=1)
    printf("%.14f ", x[i]);
  printf("]");
  fflush(stdout);
}


#endif // _KN_DBG_HPP_
#ifndef _KN_MATH_HPP_
#define _KN_MATH_HPP_

#endif // _KN_MATH_HPP_

static inline void iniGlobals();
static inline void testForQuit();
static inline Real u0_Test1_for_linear_advection_smooth_data(Real x );
static inline Real u0_Test2_for_linear_advection_discontinuous_data(Real x );

// ********************************************************
// * MESH GENERATION
// ********************************************************
const int NABLA_NB_NODES_X_AXIS = X_EDGE_ELEMS+1;
const int NABLA_NB_NODES_Y_AXIS = 0;
const int NABLA_NB_NODES_Z_AXIS = 0;

const int NABLA_NB_CELLS_X_AXIS = X_EDGE_ELEMS;
const int NABLA_NB_CELLS_Y_AXIS = 0;
const int NABLA_NB_CELLS_Z_AXIS = 0;

const double NABLA_NB_NODES_X_TICK = LENGTH/(NABLA_NB_CELLS_X_AXIS);
const double NABLA_NB_NODES_Y_TICK = 0.0;
const double NABLA_NB_NODES_Z_TICK = 0.0;

const int NABLA_NB_NODES        = (NABLA_NB_NODES_X_AXIS);
const int NABLA_NODES_PADDING   = (((NABLA_NB_NODES%WARP_SIZE)==0)?0:1);
const int NABLA_NB_NODES_WARP   = (NABLA_NODES_PADDING+NABLA_NB_NODES/WARP_SIZE);
const int NABLA_NB_CELLS        = (NABLA_NB_CELLS_X_AXIS);
 const int NABLA_NB_CELLS_WARP   = (NABLA_NB_CELLS/WARP_SIZE);


// ********************************************************
// * MESH CONNECTIVITY
// ********************************************************
int cell_node[2*NABLA_NB_CELLS]               __attribute__ ((aligned(WARP_ALIGN)));
int node_cell[2*NABLA_NB_NODES]               __attribute__ ((aligned(WARP_ALIGN)));
int node_cell_corner[2*NABLA_NB_NODES]        __attribute__ ((aligned(WARP_ALIGN)));
int cell_next[1*NABLA_NB_CELLS]               __attribute__ ((aligned(WARP_ALIGN)));
int cell_prev[1*NABLA_NB_CELLS]               __attribute__ ((aligned(WARP_ALIGN)));
int node_cell_and_corner[2*2*NABLA_NB_NODES]  __attribute__ ((aligned(WARP_ALIGN)));




/*********************************************************
 * Forward enumerates
 *********************************************************/
#define FOR_EACH_CELL(c) for(int c=0;c<NABLA_NB_CELLS;c+=1)
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<8;n+=1)

#define FOR_EACH_CELL_WARP(c) for(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)
#define FOR_EACH_CELL_WARP_SHARED(c,local) for(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)

#define FOR_EACH_CELL_WARP_NODE(n)\
  for(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\
    for(int n=8-1;n>=0;--n)

#define FOR_EACH_NODE(n) /**/for(int n=0;n<NABLA_NB_NODES;n+=1)
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=8*n;c<8;c+=1,nc+=1)

#define FOR_EACH_NODE_WARP(n) for(int n=0;n<NABLA_NB_NODES_WARP;n+=1)

#define FOR_EACH_NODE_WARP_CELL(c)\
    for(int c=0;c<8;c+=1)


// ********************************************************
// * Variables
// ********************************************************
real/*3*/ node_coord[NABLA_NB_NODES_WARP] __attribute__ ((aligned(WARP_ALIGN)));
real node_u[NABLA_NB_NODES_WARP] __attribute__ ((aligned(WARP_ALIGN)));
real node_unp1[NABLA_NB_NODES_WARP] __attribute__ ((aligned(WARP_ALIGN)));
real node_dtxp[NABLA_NB_NODES_WARP] __attribute__ ((aligned(WARP_ALIGN)));
real node_dtxm[NABLA_NB_NODES_WARP] __attribute__ ((aligned(WARP_ALIGN)));
real global_dtx[NABLA_NB_GLOBAL_WARP] __attribute__ ((aligned(WARP_ALIGN)));


// ********************************************************
// * Options
// ********************************************************
#define option_a 1.0
#define al 1.0
#define bt 8.0
#define xmin -1.0
#define xmax +1.0
#define CFL 0.8
#define test 1
#define time_steps 8
#define option_dtt_initial 0.0
#define option_stoptime 1.0


// ********************************************************
// * Temps de la simulation
// ********************************************************
Real global_deltat[1];
int global_iteration;
double global_time;


#endif // __OKINA_upwind_H__
