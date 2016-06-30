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
#include "nabla.h"


// ****************************************************************************
// * uCa
// ****************************************************************************
static const bool uCa(char **a,char **p,const char *utf,const char *ascii,const size_t w){
  assert(w>1);
  if (strncmp(*p,utf,w)!=0) return false;
  //printf("[36muCa='%s'[0m",*a);
  strcat(*a,ascii);
  //printf("[36muCa='%s'[0m",*a);
  *a+=strlen(ascii);
  //printf("[36muCa='%s'[0m",*a);
  *p+=w-1; // il y a le p++ qui suit
  return true;
}

// ****************************************************************************
// * u2a - αβγδεζηθικλμνξοπρςστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
// ****************************************************************************
static void u2a(char **a, char **p){
//  if (uCa(a,p,"","")) return;
 
  if (uCa(a,p,"²","^^2",2)) return;
  if (uCa(a,p,"³","^^3",2)) return;
  
  if (uCa(a,p,"∛","cube_root",3)) return;
  if (uCa(a,p,"√","square_root",3)) return;
  if (uCa(a,p,"⋅","opScaMul",3)) return;
  if (uCa(a,p,"⨯","opVecMul",3)) return;
  if (uCa(a,p,"⤫","cross2D",3)) return;
  if (uCa(a,p,"⊗","opProdTens",3)) return;
  if (uCa(a,p,"⨂","opProdTensVec",3)) return;
  if (uCa(a,p,"⊛","opMatrixProduct",3)) return;
  
  if (uCa(a,p,"à","a",2)) return;
  if (uCa(a,p,"é","e",2)) return;
  if (uCa(a,p,"è","e",2)) return;
  if (uCa(a,p,"ï","i",2)) return;
  
  
  if (uCa(a,p,"∨","or",3)) return;
  if (uCa(a,p,"∧","and",3)) return;
  
  if (uCa(a,p,"ℵ","Aleph",3)) return;
  if (uCa(a,p,"∀","forall",3)) return;
  if (uCa(a,p,"𝜕","partial",4)) return;
  if (uCa(a,p,"∞","__builtin_inff()",3)) return;
  
  if (uCa(a,p,"ℝ³ˣ³","Real3x3",9)) return;
  if (uCa(a,p,"ℝ³","Real3",5)) return;
  if (uCa(a,p,"ℝ²","Real2",5)) return;
  if (uCa(a,p,"ℝ","Real",3)) return;
  if (uCa(a,p,"ℕ","Integer",3)) return;
  if (uCa(a,p,"ℤ","Integer",3)) return;
  if (uCa(a,p,"ℾ","Bool",3)) return;
  
  if (uCa(a,p,"ⁿ⁺¹","np1",8)) return;

  if (uCa(a,p,"½","0.5",2)) return;
  if (uCa(a,p,"¼","0.25",2)) return;
  if (uCa(a,p,"⅓","(1./3.)",3)) return;
  if (uCa(a,p,"⅛","0.125",3)) return;
  
  if (uCa(a,p,"α","greek_alpha",2)) return;
  if (uCa(a,p,"β","greek_beta",2)) return;
  if (uCa(a,p,"γ","greek_gamma",2)) return;
  if (uCa(a,p,"δ","greek_delta",2)) return;
  if (uCa(a,p,"ε","greek_epsilon",2)) return;
  if (uCa(a,p,"ζ","greek_zeta",2)) return;
  if (uCa(a,p,"η","greek_eta",2)) return;
  if (uCa(a,p,"θ","greek_theta",2)) return;
  if (uCa(a,p,"ι","greek_iota",2)) return;
  if (uCa(a,p,"κ","greek_kappa",2)) return;
  if (uCa(a,p,"λ","greek_lambda",2)) return;
  if (uCa(a,p,"μ","greek_mu",2)) return;
  if (uCa(a,p,"ν","greek_nu",2)) return;
  if (uCa(a,p,"ξ","greek_xi",2)) return;
  if (uCa(a,p,"ο","greek_omicron",2)) return;
  if (uCa(a,p,"π","greek_pi",2)) return;
  if (uCa(a,p,"ρ","greek_rho",2)) return;
  //if (uCa(a,p,"ς","GREEK_SMALL_LETTER_FINAL_SIGMA",2)) return;
  if (uCa(a,p,"σ","greek_sigma",2)) return;
  if (uCa(a,p,"τ","greek_tau",2)) return;
  if (uCa(a,p,"υ","greek_upsilon",2)) return;
  if (uCa(a,p,"φ","greek_phi",2)) return;
  if (uCa(a,p,"χ","greek_chi",2)) return;
  if (uCa(a,p,"ψ","greek_psi",2)) return;
  if (uCa(a,p,"ω","greek_omega",2)) return;
  
  if (uCa(a,p,"Α","greek_capital_alpha",2)) return;
  if (uCa(a,p,"Β","greek_capital_beta",2)) return;
  if (uCa(a,p,"Γ","greek_capital_gamma",2)) return;
  if (uCa(a,p,"Δ","greek_capital_delta",2)) return;
  if (uCa(a,p,"Ε","greek_capital_epsilon",2)) return;
  if (uCa(a,p,"Ζ","greek_capital_zeta",2)) return;
  if (uCa(a,p,"Η","greek_capital_eta",2)) return;
  if (uCa(a,p,"Θ","greek_capital_theta",2)) return;
  if (uCa(a,p,"Ι","greek_capital_iota",2)) return;
  if (uCa(a,p,"Κ","greek_capital_kappa",2)) return;
  if (uCa(a,p,"Λ","greek_capital_lambda",2)) return;
  if (uCa(a,p,"Μ","greek_capital_mu",2)) return;
  if (uCa(a,p,"Ν","greek_capital_nu",2)) return;
  if (uCa(a,p,"Ξ","greek_capital_xi",2)) return;
  if (uCa(a,p,"Ο","greek_capital_omicron",2)) return;
  if (uCa(a,p,"Π","greek_capital_pi",2)) return;
  if (uCa(a,p,"Ρ","greek_capital_rho",2)) return;
  if (uCa(a,p,"Σ","greek_capital_sigma",2)) return;
  if (uCa(a,p,"Τ","greek_capital_tau",2)) return;
  if (uCa(a,p,"Υ","greek_capital_upsilon",2)) return;
  if (uCa(a,p,"Φ","greek_capital_phi",2)) return;
  if (uCa(a,p,"Χ","greek_capital_chi",2)) return;
  if (uCa(a,p,"Ψ","greek_capital_psi",2)) return;
  if (uCa(a,p,"Ω","greek_capital_omega",2)) return;
  printf("\nerror: Can not recognize '%s'",*p);
  assert(NULL);
}

// ****************************************************************************
// * utf2ascii
// ****************************************************************************
char* utf2ascii(const char *utf){
  if (utf==NULL) return NULL;
  char *dup=strdup(utf);
  char *p=dup;
  int nb_utf=0;
  for(;*p!=0;p++) if (*p<0) nb_utf+=1;
  // On divise par deux le nombre de wide chars
  // (qui prenne 2 emplacements)
  nb_utf>>=1; 
  //if (nb_utf>0) printf("[1;35m%s[0mnb_utf=%d",utf,nb_utf);
  //printf("[%s] =#%d",utf,nb_utf);
  const int utf_size_max = 21;
  const int ascii_size = 1+strlen(utf)+utf_size_max*nb_utf;
  //if (nb_utf>0) printf(", ascii_size=%d+1",ascii_size-1);
  char *ascii=(char*)calloc(ascii_size,sizeof(char));
  char *rtn=ascii;
  for(p=dup;*p!=0;p++)
    if (*p<0) u2a(&ascii,&p);
    else *ascii++=*p;
  //*ascii=0;
  //if (nb_utf>0) printf(" ascii => [1;35m%s[0m\n",rtn);
  //printf("=> [%s]\n",strdup(rtn));
  return rtn;
}
