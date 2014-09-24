#ifndef _KN_STD_TERNARY_H_
#define _KN_STD_TERNARY_H_


// ****************************************************************************
// * opTernary
// ****************************************************************************

inline integer opTernary(const bool cond,
                         const int ifStatement,
                         const int elseStatement){
  //debug()<<"opTernary bool int int";
  if (cond) return integer(ifStatement);
  return integer(elseStatement);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const double elseStatement){
  //debug()<<"opTernary bool double double";
  if (cond) return real(ifStatement);
  return real(elseStatement);
}

inline real opTernary(const bool cond,
                      const double ifStatement,
                      const real&  elseStatement){
  //debug()<<"opTernary bool double real";
  if (cond) return real(ifStatement);
  return elseStatement;
}

inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const double elseStatement){
  //debug()<<"opTernary bool real double";
  if (cond) return ifStatement;
  return real(elseStatement);
}

inline real opTernary(const bool cond,
                      const real& ifStatement,
                      const real& elseStatement){
  //debug()<<"opTernary bool real real";
  if (cond) return ifStatement;
  return elseStatement;
}

inline real3 opTernary(const bool cond,
                       const double ifStatement,
                       const real3&  elseStatement){
  //debug()<<"opTernary bool double real3";
  if (cond) return Real3(ifStatement);
  return elseStatement;
}

#endif //  _KN_STD_TERNARY_H_
