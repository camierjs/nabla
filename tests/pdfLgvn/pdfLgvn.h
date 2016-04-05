#ifndef __BACKEND_pdfLgvn_H__
#define __BACKEND_pdfLgvn_H__


// *****************************************************************************
// * Includes
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
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;
//int hlt_level;
//bool *hlt_exit;
int omp_get_max_threads(void){return 1;}
int omp_get_thread_num(void){return 0;}
 // from nabla->parallel->includes()


// *****************************************************************************
// * Defines
// *****************************************************************************
#define __host__ 
#define __global__ 
#define WARP_BIT 0
#define WARP_SIZE 1
#define WARP_ALIGN 8
#define NABLA_NB_GLOBAL 1
#define Bool bool
#define Integer int
#define real Real
#define Real2 real3
#define real2 real3
#define rabs(a) fabs(a)
#define set(a) a
#define set1(cst) cst
#define square_root(u) sqrt(u)
#define cube_root(u) cbrt(u)
#define store(u,_u) (*u=_u)
#define load(u) (*u)
#define zero() 0.0
#define DBG_MODE (false)
#define DBG_LVL (DBG_ALL)
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
#define ReduceMinToDouble(a) a
#define ReduceMaxToDouble(a) a
#define knAt(a) 
#define fatal(a,b) exit(-1)
#define mpi_reduce(how,what) how##ToDouble(what)
#define xyz int
#define GlobalIteration global_iteration
#define MD_DirX 0
#define MD_DirY 1
#define MD_DirZ 2
#define MD_Plus 0
#define MD_Negt 4
#define MD_Shift 3
#define MD_Mask 7
#define File std::ofstream&
#define file(name,ext) std::ofstream name(#name "." #ext)
#define xs_node_cell(c) xs_node_cell[n*NABLA_NODE_PER_CELL+c]
#define xs_face_cell(c) xs_face_cell[f+NABLA_NB_FACES*c]
#define xs_face_node(n) xs_face_node[f+NABLA_NB_FACES*n]
#define synchronize(v) 


// *****************************************************************************
// * Typedefs
// *****************************************************************************
typedef int integer;
typedef double real;
typedef struct real3 Real3;
typedef struct real3x3 Real3x3;


// *****************************************************************************
// * Forwards
// *****************************************************************************
// no Forwards here!

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
#ifndef _NABLA_LIB_TERNARY_H_
#define _NABLA_LIB_TERNARY_H_

inline int opTernary(const bool cond,
                     const int ifStatement,
                     const int elseStatement){
  if (cond) return ifStatement;
  return elseStatement;
}

inline real opTernary(const bool cond,
                      const real ifStatement,
                      const real elseStatement){
  if (cond) return ifStatement;
  return elseStatement;
}

inline real3 opTernary(const bool cond,
                       const real3& ifStatement,
                       const double elseStatement){
  if (cond) return ifStatement;
  return real3(elseStatement);
}


inline real3 opTernary(const bool cond,
                       const double ifStatement,
                       const real3&  elseStatement){
  if (cond) return Real3(ifStatement);
  return elseStatement;
}

#endif //  _NABLA_LIB_TERNARY_H_
#ifndef _NABLA_LIB_GATHER_H_
#define _NABLA_LIB_GATHER_H_

inline real rgatherk(const int a, const real *data){
  return data[a];
}
inline real3 rgather3k(const int a, const real3 *data){
  return data[a];
}
inline real3x3 rgather3x3k(const int a, const real3x3 *data){
  return data[a];
}

inline real rGatherAndZeroNegOnes(const int a, const real *data){
  if (a>=0) return data[a];
  return 0.0;
}
inline real3 rGatherAndZeroNegOnes(const int a, const real3 *data){
  if (a>=0) return data[a];
  return 0.0;
}
inline real3x3 rGatherAndZeroNegOnes(const int a,const  real3x3 *data){
  if (a>=0) return data[a];
  return 0.0;
}
inline real3 rGatherAndZeroNegOnes(const int a, const int corner, const real3 *data){
  const int i=3*8*a+3*corner;
  const double *p=(double*)data;
  if (a>=0) return real3(p[i+0],p[i+1],p[i+2]);
  return 0.0;
}

#endif //  _NABLA_LIB_GATHER_H_

#ifndef _NABLA_LIB_SCATTER_H_
#define _NABLA_LIB_SCATTER_H_

// *****************************************************************************
// * Scatter: (X is the data @ offset x)
// * scatter: |ABCD| and offsets:    a                 b       c   d
// * data:    |....|....|....|....|..A.|....|....|....|B...|...C|..D.|....|....|
// * ! à la séquence car quand c et d sont sur le même warp, ça percute
// ******************************************************************************
inline void scatterk(const int a, real *gathered, real *data){
  if (a<0) return; // Skipping to fake write
  data[a]=*gathered;
}

inline void scatter3k(const int a, real3 *gathered, real3 *data){
  if (a<0) return; // Skipping to fake write
  double *p=(double *)data;
  p[3*a+0]=gathered->x;
  p[3*a+1]=gathered->y;
  p[3*a+2]=gathered->z;
}

inline void scatter3x3k(const int a, real3x3 *gathered, real3x3 *data){
  if (a<0) return; // Skipping to fake write
  data[a]=*gathered;
}

#endif //  _NABLA_LIB_SCATTER_H_
#ifndef _NABLA_OSTREAM_H_
#define _NABLA_OSTREAM_H_

std::ostream& info(){
  std::cout.flush();
  std::cout<<"\n";
  return std::cout;
}

std::ofstream devNull("/dev/null");

std::ostream& debug(){
  if (getenv("NABLA_LAMBDA_ALEPH_DEBUG")==NULL) return devNull;
  std::cout.flush();
  std::cout<<"\n";
  return std::cout;
}

std::ostream& operator<<(std::ostream &os, const Real3 &a){
  double *x = (double*)&(a.x);
  double *y = (double*)&(a.y);
  double *z = (double*)&(a.z);
  return os << "("<<*x<<","<<*y<<","<<*z<< ")";
}

std::ostream& operator<<(std::ostream &os, const Real3x3 &a){
  return os << "(" << a.x <<","<< a.y <<","<< a.z <<")";
}

#endif // _NABLA_OSTREAM_H_
#ifndef _NABLA_LIB_DBG_HPP_
#define _NABLA_LIB_DBG_HPP_

#include <stdarg.h>

// ****************************************************************************
// * Outils de traces
// ****************************************************************************
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
  double x[1];
  double y[1];
  double z[1];
  store(x, v.x);
  store(y, v.y);
  store(z, v.z);
  printf("\n\t\t\t[%.14f,%.14f,%.14f]", x[0], y[0], z[0]);
  fflush(stdout);
}

inline void dbgReal(const unsigned int flag, real v){
  if (!DBG_MODE) return;
  if ((flag&DBG_LVL)==0) return;
  double x[1];
  store(x, v);
  printf("[");
  printf("%.14f ", x[0]);
  printf("]");
  fflush(stdout);
}

#endif // _NABLA_LIB_DBG_HPP_

static inline void ini(integer* global_piece_x,integer* global_ix_start,integer* global_ix_end,integer* global_nx_max,real* global_xb,real* global_dx,integer* global_nvtot,real* global_vb,real* global_dvx,real* global_dvy,real* global_dvz,real* global_cb,real* global_dc,real* global_vbcol,integer* global_bsize,integer* global_l1,integer* global_l2,integer* global_sizeAvx,integer* global_sizeAvy,integer* global_sizeAc);
static inline void initialise(real* global_Lx,real* global_kx);
#ifndef _LAMBDA_ITEMS_H_
#define _LAMBDA_ITEMS_H_

// aux cells
bool _isOwn_(int c){
  if (c>=0) return true;
  assert(c>=0);
  return false;
}

// aux faces
bool _isSubDomainBoundaryOutside_(const int NABLA_NB_FACES,
                                  const int *xs_face_cell,
                                  int f){
  assert(f>=0);
  assert(xs_face_cell[0*NABLA_NB_FACES+f]>=0);
  assert(xs_face_cell[1*NABLA_NB_FACES+f]<=0);
  // On ne retourne true que quand la direction est 'X+'
  if (xs_face_cell[1*NABLA_NB_FACES+f]==0) return true;
  return false;
}

#endif //_LAMBDA_ITEMS_H_


// ********************************************************
// * xHookVariablesPrefix
// ********************************************************
// Options
#define steps_p_period 200
#define nb_period 5.0
#define frequence 0.01
#define nx 128
#define xmin -60.0
#define xmax +60.0
#define nvx 128
#define vxmin -8.0
#define vxmax +8.0
#define nvy 1
#define vymin -0.5
#define vymax +0.5
#define nvz 1
#define vzmin -0.5
#define vzmax +0.5
#define nc 1
#define cmin -0.5
#define cmax +0.5
#define my_proc 0
#define nb_proc 1
#define C1 2.73
#define Cc 3.
#define C0 1.73
#define Cc0 0.
#define Cepsilon 1.
#define Comega 0.9
#define reynolds_xx 1.
#define reynolds_yy 1.
#define mean_vx 0.
#define mean_vy 0.
#define c_variance 0.5
#define mean_c 0.
#define zmt_lowbound -10.
#define zmt_highbound 10.
#define reynolds_xx_dirac 0.001
#define reynolds_yy_dirac 0.01
#define c_variance_dirac 0.01
#define pi 3.14159265359
#define Ck opSub(opAdd(opMul(3.,C1),opMul(2.,Comega)),4.)
#define beta opMul(3.,Comega)
#define dissipation_rate opMul(beta,Comega)
#define tau0 opMul(Comega,dissipation_rate)
#define omega0 opMul(tau0,Comega)

// ****************************************************************************
// * nablaMshStruct
// ****************************************************************************
typedef struct nablaMshStruct{
	int NABLA_NODE_PER_CELL;
	int NABLA_CELL_PER_NODE;
	int NABLA_CELL_PER_FACE;
	int NABLA_NODE_PER_FACE;
	int NABLA_FACE_PER_CELL;

	int NABLA_NB_NODES_X_AXIS;
	int NABLA_NB_NODES_Y_AXIS;
	int NABLA_NB_NODES_Z_AXIS;

	int NABLA_NB_CELLS_X_AXIS;
	int NABLA_NB_CELLS_Y_AXIS;
	int NABLA_NB_CELLS_Z_AXIS;

	int NABLA_NB_FACES_X_INNER;
	int NABLA_NB_FACES_Y_INNER;
	int NABLA_NB_FACES_Z_INNER;
	int NABLA_NB_FACES_X_OUTER;
	int NABLA_NB_FACES_Y_OUTER;
	int NABLA_NB_FACES_Z_OUTER;
	int NABLA_NB_FACES_INNER;
	int NABLA_NB_FACES_OUTER;
	int NABLA_NB_FACES;

	double NABLA_NB_NODES_X_TICK;
	double NABLA_NB_NODES_Y_TICK;
	double NABLA_NB_NODES_Z_TICK;

	int NABLA_NB_NODES;
	int NABLA_NODES_PADDING;
	int NABLA_NB_CELLS;
	int NABLA_NB_NODES_WARP;
	int NABLA_NB_CELLS_WARP;
}nablaMesh;


// *********************************************************
// * Forward enumerates
// *********************************************************
#define FOR_EACH_PARTICLE(p) for(int p=0;p<NABLA_NB_PARTICLES;p+=1)
#define FOR_EACH_PARTICLE_WARP(p) for(int p=0;p<NABLA_NB_PARTICLES;p+=1)

#define FOR_EACH_CELL(c) for(int c=0;c<NABLA_NB_CELLS;c+=1)
#define FOR_EACH_CELL_WARP(c) for(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)
#define FOR_EACH_OUTER_CELL_WARP(c) for(int c=0;c<NABLA_NB_CELLS_WARP;c+=nxtOuterCellOffset(c))
 #define FOR_EACH_CELL_WARP_SHARED(c,local) for(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)
#define FOR_EACH_CELL_WARP_NODE(n) for(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\
    for(int n=NABLA_NODE_PER_CELL-1;n>=0;--n)

#define FOR_EACH_CELL_SHARED(c,local) for(int c=0;c<NABLA_NB_CELLS;c+=1)

#define FOR_EACH_NODE_MSH(n) for(int n=0;n<msh.NABLA_NB_NODES;n+=1)
#define FOR_EACH_NODE(n) for(int n=0;n<NABLA_NB_NODES;n+=1)
#define FOR_EACH_NODE_WARP(n) for(int n=0;n<NABLA_NB_NODES_WARP;n+=1)
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=NABLA_NODE_PER_CELL*n;c<NABLA_NODE_PER_CELL;c+=1,nc+=1)

#define FOR_EACH_NODE_CELL_MSH(c) for(int c=0,nc=msh.NABLA_NODE_PER_CELL*n;c<msh.NABLA_NODE_PER_CELL;c+=1,nc+=1)

#define FOR_EACH_NODE_WARP_CELL(c)\
    for(int c=0;c<NABLA_NODE_PER_CELL;c+=1)

#define FOR_EACH_FACE(f) for(int f=0;f<NABLA_NB_FACES;f+=1)
#define FOR_EACH_FACE_WARP(f) for(int f=0;f<NABLA_NB_FACES;f+=1)
#define FOR_EACH_INNER_FACE(f) for(int f=0;f<NABLA_NB_FACES_INNER;f+=1)
#define FOR_EACH_INNER_FACE_WARP(f) for(int f=0;f<NABLA_NB_FACES_INNER;f+=1)
#define FOR_EACH_OUTER_FACE(f) for(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)
#define FOR_EACH_OUTER_FACE_WARP(f) for(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)
// Pour l'instant en étant que multi-threadé, les 'own' sont les 'all'
#define FOR_EACH_OWN_INNER_FACE(f) for(int f=0;f<NABLA_NB_FACES_INNER;f+=1)
#define FOR_EACH_OWN_INNER_FACE_WARP(f) for(int f=0;f<NABLA_NB_FACES_INNER;f+=1)
#define FOR_EACH_OWN_OUTER_FACE(f) for(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)
#define FOR_EACH_OWN_OUTER_FACE_WARP(f) for(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)

#define FOR_EACH_FACE_CELL(c)\
    for(int c=0;c<NABLA_NODE_PER_FACE;c+=1)


#endif // __BACKEND_pdfLgvn_H__
