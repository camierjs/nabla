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
inline real3 opTernary(const bool cond,
                      const real3& ifStatement,
                      const double elseStatement){
  if (cond) return ifStatement;
  return real3(elseStatement);
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
