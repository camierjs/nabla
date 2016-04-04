///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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

inline integer opTernary(const __mmask8 cond,
                         const int ifStatement,
                         const integer elseStatement){
  return _mm512_mask_blend_epi64(cond, elseStatement, integer(ifStatement));
}
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
