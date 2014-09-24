/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcLibMail.c        													  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2013.04.11																	  *
 * Updated  : 2013.04.11																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2013.04.11	camierjs	Creation															  *
 *****************************************************************************/
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

