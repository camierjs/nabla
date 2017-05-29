///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
#include "nabla.h"


// ****************************************************************************
// * uCa
// ****************************************************************************
static const bool uCa(char **a,char **p,
                      const char *utf, const size_t w,
                      const char *ascii, const size_t len){
  assert(w>1);
  if (strncmp(*p,utf,w)!=0) return false;
  //printf("[36muCa='%s'[0m",*a);
  strcat(*a,ascii);
  //printf("[36muCa='%s'[0m",*a);
  *a+=len;
  //printf("[36muCa='%s'[0m",*a);
  *p+=w-1; // il y a le p++ qui suit
  return true;
}

// ****************************************************************************
// * u2a - αβγδεζηθικλμνξοπρςστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
// ****************************************************************************
static void u2a(char **a, char **p){
//  if (uCa(a,p,"","")) return;
  //printf("\n[36ma='%s'[0m",*a);
  //printf("\n[36mp='%s'[0m",*p);
 
  if (uCa(a,p,"²",2,"^^2",3)) return;
  if (uCa(a,p,"³",2,"^^3",3)) return;
  if (uCa(a,p,"∑",3,"N_ARY_SUM",9)) return;

  if (uCa(a,p,"∛",3,"cube_root",9)) return;
  if (uCa(a,p,"√",3,"square_root",11)) return;
  if (uCa(a,p,"⋅",3,"opScaMul",8)) return;
  if (uCa(a,p,"⨯",3,"opVecMul",8)) return;
  if (uCa(a,p,"⤫",3,"cross2D",7)) return;
  if (uCa(a,p,"⊗",3,"opProdTens",10)) return;
  if (uCa(a,p,"⨂",3,"opProdTensVec",13)) return;
  if (uCa(a,p,"⊛",3,"opMatrixProduct",15)) return;
  
  if (uCa(a,p,"à",2,"a",1)) return;
  if (uCa(a,p,"é",2,"e",1)) return;
  if (uCa(a,p,"è",2,"e",1)) return;
  if (uCa(a,p,"ï",2,"i",1)) return;
  
  //↑↗→↘↓↙←↖⊠⊡
  if (uCa(a,p,"↑",3,"_NP",3)) return; // North Point
  if (uCa(a,p,"↗",3,"_NE",3)) return; // North East
  if (uCa(a,p,"→",3,"_EP",3)) return; // East Point
  if (uCa(a,p,"↘",3,"_SE",3)) return; // South East
  if (uCa(a,p,"↓",3,"_SP",3)) return; // South Point
  if (uCa(a,p,"↙",3,"_SW",3)) return; // South West
  if (uCa(a,p,"←",3,"_WP",3)) return; // West Point
  if (uCa(a,p,"↖",3,"_NW",3)) return; // North West
  if (uCa(a,p,"⊠",3,"_BP",3)) return; // Back Point
  if (uCa(a,p,"⊡",3,"_FP",3)) return; // Front Point
  
  if (uCa(a,p,"∨",3,"or",2)) return;
  if (uCa(a,p,"∧",3,"and",3)) return;
  
  if (uCa(a,p,"ℵ",3,"Aleph",5)) return;
  if (uCa(a,p,"∀",3,"forall",6)) return;
  if (uCa(a,p,"𝜕",4,"partial",7)) return;
  if (uCa(a,p,"∞",3,"__builtin_inff()",16)) return;
  
  if (uCa(a,p,"ℝ³ˣ³",9,"Real3x3",7)) return;
  if (uCa(a,p,"ℝ³",5,"Real3",5)) return;
  if (uCa(a,p,"ℝ²",5,"Real2",5)) return;
  if (uCa(a,p,"ℝ",3,"Real",4)) return;
  if (uCa(a,p,"ℕ",3,"Integer",7)) return;
  if (uCa(a,p,"ℤ",3,"Integer",7)) return;
  if (uCa(a,p,"ℾ",3,"Bool",4)) return;
  
  if (uCa(a,p,"ⁿ⁺¹",8,"np1",3)) return;

  if (uCa(a,p,"½",2,"0.5",3)) return;
  if (uCa(a,p,"¼",2,"0.25",4)) return;
  if (uCa(a,p,"⅓",3,"(1./3.)",7)) return;
  if (uCa(a,p,"⅛",3,"0.125",5)) return;
  
  if (uCa(a,p,"α",2,"greek_alpha",11)) return;
  if (uCa(a,p,"β",2,"greek_beta",10)) return;
  if (uCa(a,p,"γ",2,"greek_gamma",11)) return;
  if (uCa(a,p,"δ",2,"greek_delta",11)) return;
  if (uCa(a,p,"ε",2,"greek_epsilon",13)) return;
  if (uCa(a,p,"ζ",2,"greek_zeta",10)) return;
  if (uCa(a,p,"η",2,"greek_eta",9)) return;
  if (uCa(a,p,"θ",2,"greek_theta",11)) return;
  if (uCa(a,p,"ι",2,"greek_iota",10)) return;
  if (uCa(a,p,"κ",2,"greek_kappa",11)) return;
  if (uCa(a,p,"λ",2,"greek_lambda",12)) return;
  if (uCa(a,p,"μ",2,"greek_mu",8)) return;
  if (uCa(a,p,"ν",2,"greek_nu",8)) return;
  if (uCa(a,p,"ξ",2,"greek_xi",8)) return;
  if (uCa(a,p,"ο",2,"greek_omicron",13)) return;
  if (uCa(a,p,"π",2,"greek_pi",8)) return;
  if (uCa(a,p,"ρ",2,"greek_rho",9)) return;
  //if (uCa(a,p,"ς","GREEK_SMALL_LETTER_FINAL_SIGMA",2)) return;
  if (uCa(a,p,"σ",2,"greek_sigma",11)) return;
  if (uCa(a,p,"τ",2,"greek_tau",9)) return;
  if (uCa(a,p,"υ",2,"greek_upsilon",13)) return;
  if (uCa(a,p,"φ",2,"greek_phi",9)) return;
  if (uCa(a,p,"χ",2,"greek_chi",9)) return;
  if (uCa(a,p,"ψ",2,"greek_psi",9)) return;
  if (uCa(a,p,"ω",2,"greek_omega",11)) return;
  
  if (uCa(a,p,"Α",2,"greek_capital_alpha",19)) return;
  if (uCa(a,p,"Β",2,"greek_capital_beta",18)) return;
  if (uCa(a,p,"Γ",2,"greek_capital_gamma",19)) return;
  if (uCa(a,p,"Δ",2,"greek_capital_delta",19)) return;
  if (uCa(a,p,"Ε",2,"greek_capital_epsilon",21)) return;
  if (uCa(a,p,"Ζ",2,"greek_capital_zeta",18)) return;
  if (uCa(a,p,"Η",2,"greek_capital_eta",17)) return;
  if (uCa(a,p,"Θ",2,"greek_capital_theta",19)) return;
  if (uCa(a,p,"Ι",2,"greek_capital_iota",18)) return;
  if (uCa(a,p,"Κ",2,"greek_capital_kappa",19)) return;
  if (uCa(a,p,"Λ",2,"greek_capital_lambda",20)) return;
  if (uCa(a,p,"Μ",2,"greek_capital_mu",16)) return;
  if (uCa(a,p,"Ν",2,"greek_capital_nu",16)) return;
  if (uCa(a,p,"Ξ",2,"greek_capital_xi",16)) return;
  if (uCa(a,p,"Ο",2,"greek_capital_omicron",21)) return;
  if (uCa(a,p,"Π",2,"greek_capital_pi",16)) return;
  if (uCa(a,p,"Ρ",2,"greek_capital_rho",17)) return;
  if (uCa(a,p,"Σ",2,"greek_capital_sigma",19)) return;
  if (uCa(a,p,"Τ",2,"greek_capital_tau",17)) return;
  if (uCa(a,p,"Υ",2,"greek_capital_upsilon",21)) return;
  if (uCa(a,p,"Φ",2,"greek_capital_phi",17)) return;
  if (uCa(a,p,"Χ",2,"greek_capital_chi",17)) return;
  if (uCa(a,p,"Ψ",2,"greek_capital_psi",17)) return;
  if (uCa(a,p,"Ω",2,"greek_capital_omega",19)) return;
  printf("\nerror: Can not recognize '%s'",*p);
  assert(NULL);
}

// ****************************************************************************
// * utf2ascii
// ****************************************************************************
char* utf2ascii(const char *utf){
  if (utf==NULL) return NULL;
  char *dup=sdup(utf);
  char *p=dup;
  int nb_utf=0;
  for(;*p!=0;p++) if (*p<0) nb_utf+=1;
  // On divise par deux le nombre de wide chars
  // (qui prenne 2 emplacements)
  nb_utf>>=1; 
  //if (nb_utf>0) printf("[1;35m%s[0mnb_utf=%d",utf,nb_utf);
  //printf("[%s] =#%d",utf,nb_utf);
  //const int utf_size_max = 21;
  const int ascii_size = NABLA_MAX_FILE_NAME;//1+strlen(utf)+utf_size_max*nb_utf;
  //if (nb_utf>0) printf(", ascii_size=%d+1",ascii_size-1);
  char *ascii=(char*)calloc(ascii_size,sizeof(char));
  char *bkp=ascii;
  for(p=dup;*p!=0;p++)
    if (*p<0) u2a(&ascii,&p);
    else *ascii++=*p;
  //*ascii=0;
  //if (nb_utf>0) printf(" ascii => [1;35m%s[0m\n",bkp);
  //printf("=> [%s]\n",sdup(bkp));
  char *rtn=sdup(bkp);
  free(bkp);
  return rtn;
}
