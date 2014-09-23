#ifndef _KN_OSTREAM_H_
#define _KN_OSTREAM_H_


// ****************************************************************************
// * INTEGERS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const integer &a){
  int *ip = (int*)&(a);
  return os << "["<<*(ip+0)<<","<<*(ip+1)<<"]";
}


// ****************************************************************************
// * REALS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const __m128d v){
  double *fp = (double*)&v;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<"]";
}

std::ostream& operator<<(std::ostream &os, const Real &a){
  double *fp = (double*)&a;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<"]";
}


// ****************************************************************************
// * REALS_3
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const Real3 &a){
  double *x = (double*)&(a.x);
  double *y = (double*)&(a.y);
  double *z = (double*)&(a.z);
  return os << "[("<<*(x+0)<<","<<*(y+0)<<","<<*(z+0)<< "), "
            <<  "("<<*(x+1)<<","<<*(y+1)<<","<<*(z+1)<< ")]";
}


#endif // _KN_OSTREAM_H_
