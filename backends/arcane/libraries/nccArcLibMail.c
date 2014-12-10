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
#include "nabla.h"


// ****************************************************************************
// * nccArcLibMailHeader
// * Let's be careful to quote subjects containing spaces.
// ****************************************************************************
char* nccArcLibMailHeader(void){
  return "\
\n#include <ostream>                                                    \
\n#include <arcane/utils/PlatformUtils.h>                               \
\nstruct mail{                                                          \
public:                                                                 \
  mail(){ oss<<\"echo -n \\\"\"; }                                      \
  ~mail(){                                                              \
    oss << \"\\\" | /bin/mail -s '[nahea@\";                            \
    oss << Arcane::platform::getHostName();                             \
    oss << \"] \";                                                      \
    oss << Arcane::platform::getCurrentDateTime();                      \
    oss << \"' \";                                                      \
    oss << Arcane::platform::getUserName();                             \
    const std::string& tmp = oss.str();                                 \
    const char *cstr=tmp.c_str();                                       \
    //std::cout << cstr;                                                \
\n  system(cstr);                                                       \
  }                                                                     \
 template <typename T>                                                  \
  mail& operator<<(T const& t){                                         \
    oss << t;                                                           \
    return *this;                                                       \
  }                                                                     \
private:                                                                \
  std::ostringstream oss;                                               \
};";
}


// ****************************************************************************
// * nccArcLibMailPrivates
// ****************************************************************************
char* nccArcLibMailPrivates(void){
  return "";
}


// ****************************************************************************
// * nccArcLibMailIni
// ****************************************************************************
void nccArcLibMailIni(nablaMain *arc){}


// ****************************************************************************
// * nccArcLibMailDelete
// ****************************************************************************
char *nccArcLibMailDelete(void){
  return "";
}

