#ifndef __CUDA_lulesh_H__
#define __CUDA_lulesh_H__


// *****************************************************************************
// * Includes
// * Tesla:  sm_10: ISA_1, Basic features
// *         sm_11: + atomic memory operations on global memory
// *         sm_12: + atomic memory operations on shared memory
// *                + vote instructions
// *         sm_13: + double precision floating point support
// * Fermi:  sm_20: + Fermi support
// *         sm_21
// * Kepler: sm_30: + Kepler support
// *         sm_35
// *****************************************************************************
#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <assert.h>
#include <stdarg.h>
#include <cuda.h>


// *****************************************************************************
// * Cartesian stuffs
// *****************************************************************************
#define MD_DirX 0
#define MD_DirY 1
#define MD_DirZ 2
//#warning empty libCartesianInitialize
//__device__ void libCartesianInitialize(void){}


// *****************************************************************************
// * Typedefs
// *****************************************************************************
typedef int integer;


// ****************************************************************************
// * ERROR HANDLING
// ****************************************************************************
static void HandleError( cudaError_t err,
                         const char *file,
                         int line){
  if (err != cudaSuccess) {
    printf("%s in %s at line %d",
           cudaGetErrorString( err ),
           file, line );
    exit( EXIT_FAILURE );
  }
}
#define CUDA_HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ )) 


// *****************************************************************************
// * Defines
// *****************************************************************************
#define real3 Real3
#define Real double
#define real double
#define ReduceMinToDouble(what) what
#define cuda_exit(what) *global_deltat=-1.0
#define norm fabs
#define rabs fabs
#define rsqrt sqrt
#define opAdd(u,v) (u+v)
#define opSub(u,v) (u-v)
#define opDiv(u,v) (u/v)
#define opMul(u,v) (u*v)
#define opScaMul(a,b) dot(a,b)
#define opVecMul(a,b) cross(a,b)
#define opTernary(cond,ifStatement,elseStatement) (cond)?ifStatement:elseStatement
#define knAt(a) 
#define fatal(a,b) return
#define synchronize(a) 
#define reducemin(a) a
#define mpi_reduce(how,what) what
#define xyz int
#define GlobalIteration *global_iteration
#define PAD_DIV(nbytes, align) (((nbytes)+(align)-1)/(align))


// *****************************************************************************
// * Forwards
// *****************************************************************************
inline std::ostream& info(){std::cout.flush();std::cout<<"\n";return std::cout;}
inline std::ostream& debug(){std::cout.flush();std::cout<<"\n";return std::cout;}
void gpuEnum(void);


// ********************************************************
// * MESH GENERATION
// ********************************************************
#define NABLA_NB_NODES_X_AXIS   (X_EDGE_ELEMS+1)
#define NABLA_NB_NODES_Y_AXIS   (Y_EDGE_ELEMS+1)
#define NABLA_NB_NODES_Z_AXIS   (Z_EDGE_ELEMS+1)

#define NABLA_NB_CELLS_X_AXIS    X_EDGE_ELEMS
#define NABLA_NB_CELLS_Y_AXIS    Y_EDGE_ELEMS
#define NABLA_NB_CELLS_Z_AXIS    Z_EDGE_ELEMS

#define BLOCKSIZE                  256
#define CUDA_NB_THREADS_PER_BLOCK  256

#define NABLA_NB_NODES_X_TICK (LENGTH/(NABLA_NB_NODES_X_AXIS-1))
#define NABLA_NB_NODES_Y_TICK (LENGTH/(NABLA_NB_NODES_Y_AXIS-1))
#define NABLA_NB_NODES_Z_TICK (LENGTH/(NABLA_NB_NODES_Z_AXIS-1))

#define NABLA_NB_NODES (NABLA_NB_NODES_X_AXIS*NABLA_NB_NODES_Y_AXIS*NABLA_NB_NODES_Z_AXIS)
#define NABLA_NB_CELLS (NABLA_NB_CELLS_X_AXIS*NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS)

#define NABLA_NB_GLOBAL 1


// ********************************************************
// * MESH CONNECTIVITY
// ********************************************************
__builtin_align__(8) int *node_cell;
__builtin_align__(8) int *cell_node;
// ****************************************************************************
// * OPERATIONS DEFINITIONS
// ****************************************************************************
//#define zero() (0.0)
//#define set1(a) (a)
//#define add(a,b) ((a)+(b))
//#define sub(a,b) ((a)-(b))
//#define mul(a,b) ((a)*(b))
//#define div(a,b) ((a)/(b))
//#define mod(a,b) ((a)%(b))


// ****************************************************************************
// * REAL3 DEFINITION
// ****************************************************************************
struct Real3 {
public:
  double x;
  double y;
  double z;

  // Constructors
  __device__ inline Real3(): x(0.0), y(0.0), z(0.0){}
  __device__ inline Real3(const double f):x(f), y(f), z(f){}
  __device__ inline Real3(const double _x, const double _y, const double _z):x(_x), y(_y), z(_z){}
  __device__ inline Real3(const double *_x, const double *_y, const double *_z):x(*_x), y(*_y), z(*_z){}

  // Arithmetic operators
  __device__ inline Real3 operator+(const Real3& v) const{ return Real3(x+v.x, y+v.y, z+v.z);}
  __device__ inline Real3 operator-(const Real3& v) const{ return Real3(x-v.x, y-v.y, z-v.z);}
  __device__ inline Real3 operator*(const Real3& v) const{ return Real3(x*v.x, y*v.y, z*v.z);}
  __device__ inline Real3 operator/(const Real3& v) const{ return Real3(x/v.x, y/v.y, z/v.z);}

  // Arithmetic operators with doubles
  __device__ inline Real3 operator+(double f)const{ return Real3(x+f, y+f, z+f);}
  __device__ inline Real3 operator-(double f)const{ return Real3(x-f, y-f, z-f);}
  __device__ inline Real3 operator-()const{ return Real3(-x, -y, -z);}
  __device__ inline Real3 operator*(double f)const{ return Real3(x*f, y*f, z*f);}
  __device__ inline Real3 operator/(double f)const{ return Real3(x/f, y/f, z/f);}

  // op= operators
  __device__ inline Real3& operator+=(const Real3& v){ x+=v.x; y+=v.y; z+=v.z; return *this;}
  __device__ inline Real3& operator-=(const Real3& v){ x-=v.x; y-=v.y; z-=v.x; return *this;}
  __device__ inline Real3& operator*=(const Real3& v){ x*=v.x; y*=v.y; z*=v.z; return *this;}
  __device__ inline Real3& operator/=(const Real3& v){ x/=v.x; y/=v.y; z/=v.z; return *this;}

  // op= operators with doubles
  __device__ inline Real3& operator+=(double f){ x+=f; y+=f; z+=f; return *this;}
  __device__ inline Real3& operator-=(double f){ x-=f; y-=f; z-=f; return *this;}
  __device__ inline Real3& operator*=(double f){ x*=f; y*=f; z*=f; return *this;}
  __device__ inline Real3& operator/=(double f){ x/=f; y/=f; z/=f; return *this;}

};

__device__ inline Real3 operator+(double f, const Real3& v){return v+f;}
__device__ inline Real3 operator-(double f, const Real3& v){return Real3(f)-v;}
__device__ inline Real3 operator*(double f, const Real3& v){return v*f;}
__device__ inline Real3 operator/(double f, const Real3& v){return Real3(f)/v;}

__device__ inline void doubletoReal3(double d, Real3 *r){*r=Real3(d);}
__device__ inline void cpy3(Real3 u, Real3 *r){ *r=u;}
__device__ inline double dot(Real3 u, Real3 v){ return u.x*v.x+u.y*v.y+u.z*v.z;}
__device__ inline Real3 sqrt(Real3 u){ return Real3(sqrt(u.x), sqrt(u.y), sqrt(u.z));}
__device__ inline double norm(Real3 u){ return sqrt(dot(u,u));}
__device__ inline Real3 cross(Real3 u, Real3 v){
  return Real3(((u.y*v.z)-(u.z*v.y)),
               ((u.z*v.x)-(u.x*v.z)),
               ((u.x*v.y)-(u.y*v.x)));
}
__device__ inline void cross3(Real3 u, Real3 v, Real3 *r){ *r=cross(u,v);}
//__device__ inline double ReduceMinToDouble(Real r){ return r; }


// ****************************************************************************
// * OPERATIONS TYPEDEFS
// ****************************************************************************
//typedef Real3 real3;


// ****************************************************************************
// * INTEGERS
// ****************************************************************************
/*std::ostream& operator<<(std::ostream &os, const integer &a){
  return os << "["<<a.vec<<"]";
  }*/


// ****************************************************************************
// * REALS
// ****************************************************************************
/*std::ostream& operator<<(std::ostream &os, const Real &a){
  return os << "["<<a.vec<<"]";
  }*/


// ****************************************************************************
// * REALS_3
// ****************************************************************************
__global__ void operator<<(std::ostream os, const Real3 r){
  //return os;// << "[("<<r.x<<","<<r.y<<","<<r.z<< ")]";
}

__device__ void operator<<(std::ostream &os, const Real3 &r){
  //return os;// << "[("<<r.x<<","<<r.y<<","<<r.z<< ")]";
}


__global__ void ini();
__device__ inline void calcElemShapeFunctionDerivatives(const Real * __restrict__ x , const Real * __restrict__ y , const Real * __restrict__ z , Real * __restrict__ _Bx , Real * __restrict__ _By , Real * __restrict__ _Bz , Real * rtn );
__device__ inline void CalcElemVelocityGradient(const Real * const xvel , const Real * const yvel , const Real * const zvel , const Real b [ ] [ 8 ] , const Real detJ , Real * const d );
__device__ inline void sumElemFaceNormal(Real * _B0x , Real * _B0y , Real * _B0z , Real * _B1x , Real * _B1y , Real * _B1z , Real * _B2x , Real * _B2y , Real * _B2z , Real * _B3x , Real * _B3y , Real * _B3z , const int ia , const int ib , const int ic , const int id , const Real * __restrict__ _Xx , const Real * __restrict__ _Xy , const Real * __restrict__ _Xz );
__device__ inline void calcElemFBHourglassForce(const Real * xd , const Real * yd , const Real * zd , const Real * hourgam0 , const Real * hourgam1 , const Real * hourgam2 , const Real * hourgam3 , const Real * hourgam4 , const Real * hourgam5 , const Real * hourgam6 , const Real * hourgam7 , const Real coefficient , Real * __restrict__ hgfx , Real * __restrict__ hgfy , Real * __restrict__ hgfz );
__device__ inline void _computeHourglassModes(const int i1 , const Real _determ , const Real * _dvdx , const Real * _dvdy , const Real * _dvdz , const Real gamma [ 4 ] [ 8 ] , const Real * x8n , const Real * y8n , const Real * z8n , Real * __restrict__ hourgam0 , Real * __restrict__ hourgam1 , Real * __restrict__ hourgam2 , Real * __restrict__ hourgam3 , Real * __restrict__ hourgam4 , Real * __restrict__ hourgam5 , Real * __restrict__ hourgam6 , Real * __restrict__ hourgam7 );
__device__ inline Real _calcElemVolume(const Real x0 , const Real x1 , const Real x2 , const Real x3 , const Real x4 , const Real x5 , const Real x6 , const Real x7 , const Real y0 , const Real y1 , const Real y2 , const Real y3 , const Real y4 , const Real y5 , const Real y6 , const Real y7 , const Real z0 , const Real z1 , const Real z2 , const Real z3 , const Real z4 , const Real z5 , const Real z6 , const Real z7 );
__device__ inline Real AreaFace(const Real x0 , const Real x1 , const Real x2 , const Real x3 , const Real y0 , const Real y1 , const Real y2 , const Real y3 , const Real z0 , const Real z1 , const Real z2 , const Real z3 );
__device__ inline Real calcElemCharacteristicLength(const Real x [ 8 ] , const Real y [ 8 ] , const Real z [ 8 ] , const Real _volume );
__global__ void calcMonotonicQForElems();