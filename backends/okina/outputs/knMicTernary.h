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
#ifndef _KN_TERNARY_H_
#define _KN_TERNARY_H_


// ****************************************************************************
// * opTernary
// ****************************************************************************

inline integer opTernary(const __mmask8 cond,
                         const int ifStatement,
                         const int elseStatement){
  return _mm512_mask_blend_epi64(cond, integer(elseStatement), integer(ifStatement));
}

inline real opTernary(const __mmask8 cond,
                      const __m512d ifStatement,
                      const __m512d elseStatement){
  return _mm512_mask_blend_pd(cond, elseStatement, ifStatement);
}

inline real opTernary(const __mmask8 cond,
                      const double ifStatement,
                      const double elseStatement){
  return _mm512_mask_blend_pd(cond, _mm512_set1_pd(elseStatement), _mm512_set1_pd(ifStatement));
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const double elseStatement){
  if (cond) return _mm512_set1_pd(ifStatement);
  return _mm512_set1_pd(elseStatement);
}

inline real opTernary(const __mmask8 cond,
                      const double ifStatement,
                      const real&  elseStatement){
  return _mm512_mask_blend_pd(cond, elseStatement, _mm512_set1_pd(ifStatement));
}

inline real opTernary(const __mmask8 cond,
                      const real& ifStatement,
                      const double elseStatement){
  return _mm512_mask_blend_pd(cond, _mm512_set1_pd(elseStatement), ifStatement);
}

inline real opTernary(const __mmask8 cond,
                      const real& ifStatement,
                      const real& elseStatement){
  return _mm512_mask_blend_pd(cond, elseStatement, ifStatement);
}

inline real opTernary(const __mmask8 cond,
                      const __m512d ifStatement,
                      const real& elseStatement){
  return _mm512_mask_blend_pd(cond, elseStatement, ifStatement);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const real& elseStatement){
  if (cond) return _mm512_set1_pd(ifStatement);
  return elseStatement;
}

inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const real& elseStatement){
  if (cond) return ifStatement;
  return elseStatement;
}


inline real3 opTernary(const __mmask8 cond,
                       const double ifStatement,
                       const real3&  elseStatement){
  //debug()<<"opTernary4";
  return real3(_mm512_mask_blend_pd(cond, elseStatement.x, _mm512_set1_pd(ifStatement)),
               _mm512_mask_blend_pd(cond, elseStatement.y, _mm512_set1_pd(ifStatement)),
               _mm512_mask_blend_pd(cond, elseStatement.z, _mm512_set1_pd(ifStatement)));
}



#endif //  _KN_TERNARY_H_
