#ifndef _KN_TERNARY_H_
#define _KN_TERNARY_H_


// ****************************************************************************
// * opTernary
// ****************************************************************************

inline integer opTernary(const __m256d cond,
                         const int ifStatement,
                         const int elseStatement){
  //double *fp = (double*)&cond;
  const int mask = _mm256_movemask_pd(cond);
  //debug()<<"opTernary int, cond="<< "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3) <<"], if="<<ifStatement<<", else="<<elseStatement<<", mask="<<mask;
  //debug()<<((mask&1)==1)?ifStatement:elseStatement;
  //debug()<<((mask&2)==2)?ifStatement:elseStatement;
  //debug()<<((mask&4)==4)?ifStatement:elseStatement;
  //debug()<<((mask&8)==8)?ifStatement:elseStatement;
  return _mm_set_epi32(((mask&8)==8)?ifStatement:elseStatement,
                       ((mask&4)==4)?ifStatement:elseStatement,
                       ((mask&2)==2)?ifStatement:elseStatement,
                       ((mask&1)==1)?ifStatement:elseStatement
                       );
}

inline real opTernary(const __m256d cond,
                      const __m256d ifStatement,
                      const __m256d elseStatement){
  return _mm256_blendv_pd(elseStatement, ifStatement, cond);
}

inline real opTernary(const __m256d cond,
                      const double ifStatement,
                      const double elseStatement){
  return _mm256_blendv_pd(_mm256_set1_pd(elseStatement), _mm256_set1_pd(ifStatement), cond);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const double elseStatement){
  if (cond) return _mm256_set1_pd(ifStatement);
  return _mm256_set1_pd(elseStatement);
}

inline real opTernary(const __m256d cond,
                      const double ifStatement,
                      const real&  elseStatement){
  return _mm256_blendv_pd(elseStatement, _mm256_set1_pd(ifStatement), cond);
}

inline real opTernary(const __m256d cond,
                      const real& ifStatement,
                      const double elseStatement){
  return _mm256_blendv_pd(_mm256_set1_pd(elseStatement), ifStatement, cond);
}

inline real opTernary(const __m256d cond,
                      const real& ifStatement,
                      const real& elseStatement){
  return _mm256_blendv_pd(elseStatement, ifStatement, cond);
}

inline real opTernary(const __m256d cond,
                      const __m256d ifStatement,
                      const real& elseStatement){
  return _mm256_blendv_pd(elseStatement, ifStatement, cond);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const real& elseStatement){
  if (cond) return _mm256_set1_pd(ifStatement);
  return elseStatement;
}

inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const real& elseStatement){
  if (cond) return ifStatement;
  return elseStatement;
}


inline real3 opTernary(const __m256d cond,
                       const double ifStatement,
                       const real3&  elseStatement){
  //debug()<<"opTernary4";
  return real3(_mm256_blendv_pd(elseStatement.x, _mm256_set1_pd(ifStatement), cond),
               _mm256_blendv_pd(elseStatement.y, _mm256_set1_pd(ifStatement), cond),
               _mm256_blendv_pd(elseStatement.z, _mm256_set1_pd(ifStatement), cond));
}



#endif //  _KN_TERNARY_H_
