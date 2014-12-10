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
