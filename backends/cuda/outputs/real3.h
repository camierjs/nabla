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

