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
#ifndef _KN_OSTREAM_H_
#define _KN_OSTREAM_H_


// ****************************************************************************
// * INTEGERS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const integer &a){
  int *ip = (int*)&(a);
  return os << "["<<*(ip+0)<<","<<*(ip+1)<<","<<*(ip+2)<<","<<*(ip+3)<<"]";
}


// ****************************************************************************
// * REALS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const __m256d v){
  double *fp = (double*)&v;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3)<<"]";
}

std::ostream& operator<<(std::ostream &os, const Real &a){
  double *fp = (double*)&a;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3)<<"]";
}


// ****************************************************************************
// * REALS_3
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const Real3 &a){
  double *x = (double*)&(a.x);
  double *y = (double*)&(a.y);
  double *z = (double*)&(a.z);
  return os << "[("<<*(x+0)<<","<<*(y+0)<<","<<*(z+0)<< "), "
            <<  "("<<*(x+1)<<","<<*(y+1)<<","<<*(z+1)<< "), "
            <<  "("<<*(x+2)<<","<<*(y+2)<<","<<*(z+2)<< "), "
            <<  "("<<*(x+3)<<","<<*(y+3)<<","<<*(z+3)<< ")]";
}


#endif // _KN_OSTREAM_H_
