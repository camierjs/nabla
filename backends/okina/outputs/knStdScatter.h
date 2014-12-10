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
