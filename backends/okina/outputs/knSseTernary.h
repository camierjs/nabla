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

inline integer opTernary(const __m128d cond,
                         const int ifStatement,
                         const int elseStatement){
  //double *fp = (double*)&cond;
  const int mask = _mm_movemask_pd(cond);
  //debug()<<"opTernary int, cond="<< "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3) <<"], if="<<ifStatement<<", else="<<elseStatement<<", mask="<<mask;
  //debug()<<((mask&1)==1)?ifStatement:elseStatement;
  //debug()<<((mask&2)==2)?ifStatement:elseStatement;
  //debug()<<((mask&4)==4)?ifStatement:elseStatement;
  //debug()<<((mask&8)==8)?ifStatement:elseStatement;
  return _mm_set_epi32(((mask&8)==8)?ifStatement:elseStatement,
                       ((mask&4)==4)?ifStatement:elseStatement,
                       ((mask&2)==2)?ifStatement:elseStatement,
                       ((mask&1)==1)?ifStatement:elseStatement);
}

inline integer opTernary(const __m128d cond,
                         const int ifStatement,
                         const Integer &elseStatement){
  const int mask = _mm_movemask_pd(cond);
  return _mm_set_epi32(((mask&8)==8)?ifStatement:elseStatement[3],
                       ((mask&4)==4)?ifStatement:elseStatement[2],
                       ((mask&2)==2)?ifStatement:elseStatement[1],
                       ((mask&1)==1)?ifStatement:elseStatement[0]);
}


inline integer opTernary(const Integer cond,
                         const Integer ifStatement,
                         const Integer &elseStatement){
  //int *ip = (int*)&cond;
  const int mask = _mm_movemask_epi8(cond);
  //debug()<<"opTernary int, cond="<< "["<<*(ip+0)<<","<<*(ip+1)<<","<<*(ip+2)<<","<<*(ip+3) <<"], if="<<ifStatement<<", else="<<elseStatement<<", mask="<<mask;
  return _mm_set_epi32(((mask&8)==8)?ifStatement[3]:elseStatement[3],
                       ((mask&4)==4)?ifStatement[2]:elseStatement[2],
                       ((mask&2)==2)?ifStatement[1]:elseStatement[1],
                       ((mask&1)==1)?ifStatement[0]:elseStatement[0]);
}

inline real opTernary(const __m128d cond,
                      const __m128d ifStatement,
                      const __m128d elseStatement){
  return _mm_blendv_pd(elseStatement, ifStatement, cond);
}

inline real opTernary(const __m128d cond,
                      const double ifStatement,
                      const double elseStatement){
  return _mm_blendv_pd(_mm_set1_pd(elseStatement), _mm_set1_pd(ifStatement), cond);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const double elseStatement){
  if (cond) return _mm_set1_pd(ifStatement);
  return _mm_set1_pd(elseStatement);
}

inline real opTernary(const __m128d cond,
                      const double ifStatement,
                      const real&  elseStatement){
  return _mm_blendv_pd(elseStatement, _mm_set1_pd(ifStatement), cond);
}

inline real opTernary(const __m128d cond,
                      const real& ifStatement,
                      const double elseStatement){
  return _mm_blendv_pd(_mm_set1_pd(elseStatement), ifStatement, cond);
}

inline real opTernary(const __m128d cond,
                      const real& ifStatement,
                      const real& elseStatement){
  return _mm_blendv_pd(elseStatement, ifStatement, cond);
}

inline real opTernary(const __m128d cond,
                      const __m128d ifStatement,
                      const real& elseStatement){
  return _mm_blendv_pd(elseStatement, ifStatement, cond);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const real& elseStatement){
  if (cond) return _mm_set1_pd(ifStatement);
  return elseStatement;
}

inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const real& elseStatement){
  if (cond) return ifStatement;
  return elseStatement;
}


inline real3 opTernary(const __m128d cond,
                       const double ifStatement,
                       const real3&  elseStatement){
  return real3(_mm_blendv_pd(elseStatement.x, _mm_set1_pd(ifStatement), cond),
               _mm_blendv_pd(elseStatement.y, _mm_set1_pd(ifStatement), cond),
               _mm_blendv_pd(elseStatement.z, _mm_set1_pd(ifStatement), cond));
}



#endif //  _KN_TERNARY_H_
