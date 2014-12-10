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
#ifndef _KN_MIC_OSTREAM_H_
#define _KN_MIC_OSTREAM_H_


// ****************************************************************************
// * INTEGERS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const integer &a){
  int *ip = (int*)&(a);
  return os << "["<<*(ip+0)<<","<<*(ip+1)<<","<<*(ip+2)<<","<<*(ip+3)<<","<<*(ip+4)<<","<<*(ip+5)<<","<<*(ip+6)<<","<<*(ip+7)<<"]";
}


// ****************************************************************************
// * REALS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const __m512d v){
  const double *fp = (double*)&v;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3)
            <<","<<*(fp+4)<<","<<*(fp+5)<<","<<*(fp+6)<<","<<*(fp+7)<<"]";
}

std::ostream& operator<<(std::ostream &os, const real &a){
  const double *fp = (double*)&a;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3)
            <<","<<*(fp+4)<<","<<*(fp+5)<<","<<*(fp+6)<<","<<*(fp+7)<<"]";
}


// ****************************************************************************
// * REALS_3
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const real3 &a){
  double *x = (double*)&(a.x);
  double *y = (double*)&(a.y);
  double *z = (double*)&(a.z);
  return os << "[("<<*(x+0)<<","<<*(y+0)<<","<<*(z+0)<< "), "
            <<  "("<<*(x+1)<<","<<*(y+1)<<","<<*(z+1)<< "), "
            <<  "("<<*(x+2)<<","<<*(y+2)<<","<<*(z+2)<< "), "
            <<  "("<<*(x+3)<<","<<*(y+3)<<","<<*(z+3)<< "), "
            <<  "("<<*(x+4)<<","<<*(y+4)<<","<<*(z+4)<< "), "
            <<  "("<<*(x+5)<<","<<*(y+5)<<","<<*(z+5)<< "), "
            <<  "("<<*(x+6)<<","<<*(y+6)<<","<<*(z+6)<< "), "
            <<  "("<<*(x+7)<<","<<*(y+7)<<","<<*(z+7)<< ")]";
}


#endif // _KN_MIC_OSTREAM_H_
