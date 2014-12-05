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
