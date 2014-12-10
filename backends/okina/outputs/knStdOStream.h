// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
