///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
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
