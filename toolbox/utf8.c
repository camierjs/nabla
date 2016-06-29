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


// *****************************************************************************
// * p?[c|s]
// *****************************************************************************
static inline bool p2c(char *p,
                       unsigned short hex,
                       char *letter){
  if (*(unsigned short*)p!=hex) return false;
  *p=letter[0];
  *(p+1)=letter[1];
  return true;
}
static inline bool p2s(char *p,
                       unsigned short hex,
                       const char *str,
                       char **bkp){
  if (*(unsigned short*)p!=hex) return false;
  *bkp=strdup(str); 
  return true;
}
static inline bool p3c(char *p,
                       const unsigned int hex,
                       const char *letter){
  register unsigned int mask = (*(unsigned int*)p)&0x00FFFFFFl;
  //dbg("\n[p3c] p=%08p vs hex=0x%x", mask, hex);
  if (mask!=hex) return false;
  *p=letter[0];
  *(p+1)=letter[1];
  *(p+2)=letter[2];
  //dbg("\n[p3c] HIT!");
  return true;
}
static inline bool p4c(char *p,
                       const unsigned int hex,
                       const char *letter){
  register unsigned int mask = (*(unsigned int*)p)&0xFFFFFFFFl;
  //dbg("\n[p3c] p=%08p vs hex=0x%x", mask, hex);
  if (mask!=hex) return false;
  *p=letter[0];
  *(p+1)=letter[1];
  *(p+2)=letter[2];
  *(p+3)=letter[3];
  //dbg("\n[p3c] HIT!");
  return true;
}
static inline bool p4c3(char *w,
                        const char *p,
                        const unsigned int hex,
                        const char *letter){
  register unsigned int mask = (*(unsigned int*)p)&0xFFFFFFFFl;
  if (mask!=hex) return false;
  *w=letter[0];
  *(w+1)=letter[1];
  *(w+2)=letter[2];
  return true;
}
static inline bool p3s(char *p,
                       const unsigned int hex,
                       const char *str, char **bkp){
  register unsigned int mask = (*(unsigned int*)p)&0x00FFFFFFl;
  if (mask!=hex) return false;
  *bkp=strdup(str); 
  return true;
}
static inline bool p4s(char *p,
                       const unsigned int hex,
                       const char *str, char **bkp){
  register unsigned int mask = (*(unsigned int*)p)&0xFFFFFFFFl;
  if (mask!=hex) return false;
  *bkp=strdup(str); 
  return true;
}


// ****************************************************************************
// UTF8 codes > 3 bytes are not currently supported
// Pour ces caractères, on va les réduire si l'on peut
// ****************************************************************************
void toolUtf8SupThree(char **read){//read
  char *r=*read;
  char *w=r;//write
  //dbg("\n\t\t[nUtf8SupThree] in \"%s\"", r);
  for(w=r;*r!=0;r++){
    //dbg("\n\t\t\t[nUtf8SupThree] '%c'", *r);
    if (p4c3(w,r,0x959c9df0,"∂")){// 𝜕 -> ∂
      //dbg("\n\t\t\t\t[nUtf8SupThree] HIT 𝜕->∂!");
      r+=3;//+1 avec le r++
      w+=3;
      continue;
    }
    // Sinon on écrit normal, (copie)
    *w++=*r;
  }
  *w='\0';
  //dbg("\n\t\t[nUtf8SupThree] out \"%s\"", *read);
}


// ****************************************************************************
// * OK, that should be done more properly.
// * nUtf8
// * αβγδεζηθικλμνξοπρςστυφχψω
// * od -vt x2 /tmp/try
// * 0000000 b1ce b2ce b3ce b4ce b5ce b6ce b7ce b8ce
// * 0000020 b9ce bace bbce bcce bdce bece bfce 80cf
// * 0000040 81cf 3030 82cf 83cf 3030 84cf 85cf 86cf
// * 0000060 87cf 88cf 89cf 000a
// * ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
// * ½  ⅓  ¼  ⅛ 
// ****************************************************************************
void toolUtf8(char **bkp){
  char *p=*bkp;
  if (p==NULL) return;
  //dbg("\n[nUtf8] '%s'",p);
  
  if (strcmp(p,"δt")==0) { // "δt"
    //dbg("\n[nUtf8] hits deltat!");
    *bkp=strdup("deltat"); 
    return;
  }
  if (strcmp(p,"ℝ³⨯ℝ³")==0) { // ℝ³⨯ℝ³: 84e2 c29d e2b3 afa8 84e2 c29d 00b3
    //dbg("\n[nUtf8] hits (Real^3)x(Real^3)!");
    *bkp=strdup("Real3x3"); 
    return;
  }  
  if (strcmp(p,"ℝ³")==0) { // ℝ³: 84e2 c29d 00b3!
    //dbg("\n[nUtf8] hits Real^3!");
    *bkp=strdup("Real3"); 
    return;
  }  
  if (strcmp(p,"ℝ²")==0) {
    //dbg("\n[nUtf8] hits Real^2!");
    *bkp=strdup("Real2");
    return;
  }  
  if (strcmp(p,"ℾ")==0) { 
    //dbg("\n[nUtf8] hits Bool!");
    *bkp=strdup("Bool"); 
    return;
  }  
  if (strcmp(p,"ⁿ⁺¹")==0) {
    //dbg("\n[nUtf8] hits 'ⁿ⁺¹'!");
    *bkp=strdup("np1"); 
    return;
  }  
  // By skipping, we can: strncmp(n->children->token,"ℵ",3)
  /*if (*(unsigned int*)p==0x00b584e2) { // "ℵ"
    //dbg("\n[nUtf8] hits single ℵ ");
    *bkp=strdup("aleph"); 
    return;
    }*/
  //dbg("\n\t\t[nUtf8] '%s':", *bkp);
  for(;*p!=0;p++){
    //dbg("\n\t\t\t%c, 0x%x 0x%x",*p, *p,*(unsigned short*)p);//αβγδεζηθικλμνξοπρςστυφχψω
    // αβγδεζηθικλμνξοπρςστυφχψω
    if (p2c(p,0xb1ce,"al")) p+=1; // α = alpha    → 'al'
    if (p2c(p,0xb2ce,"bt")) p+=1; // β = beta     → 'bt'
    if (p2c(p,0xb3ce,"gm")) p+=1; // γ = gamma    → 'gm'
    if (p2c(p,0xb4ce,"dt")) p+=1; // δ = delta    → 'dt'
    if (p2c(p,0xb5ce,"ep")) p+=1; // ε = epsilon  → 'ep'
    if (p2c(p,0xb6ce,"zt")) p+=1; // ζ = zeta     → 'zt'
    if (p2c(p,0xb7ce,"et")) p+=1; // η = eta      → 'et'
    if (p2c(p,0xb8ce,"th")) p+=1; // θ = theta    → 'th'
    if (p2c(p,0xb9ce,"it")) p+=1; // ι = iota     → 'it'
    if (p2c(p,0xbace,"kp")) p+=1; // κ = kappa    → 'kp'
    //if (p2s(p,0xbace,"kappa",bkp)) p+=1;       // κ = kappa    → 'kappa'
    if (p2c(p,0xbbce,"lm")) p+=1; // λ = lambda   → 'lm'
    if (p2c(p,0xbcce,"mu")) p+=1; // μ = mu       → 'mu'
    if (p2c(p,0xbdce,"nu")) p+=1; // ν = nu       → 'nu'
    if (p2c(p,0xbece,"xi")) p+=1; // ξ = xi       → 'xi'
    if (p2c(p,0xbfce,"om")) p+=1; // ο = omicron  → 'om'
    if (p2c(p,0x80cf,"pi")) p+=1; // π = pi       → 'pi'
    if (p2c(p,0x81cf,"rh")) p+=1; // ρ = rho      → 'rh'
    if (p2c(p,0x82cf,"vg")) p+=1; // ς = varsigma → 'vg' (GREEK SMALL LETTER FINAL SIGMA)
    if (p2c(p,0x83cf,"sg")) p+=1; // σ = sigma    → 'sg'
    if (p2c(p,0x84cf,"tt")) p+=1; // τ = tau      → 'tt'
    if (p2c(p,0x85cf,"up")) p+=1; // υ = upsilon  → 'up'
    if (p2c(p,0x86cf,"p2")) p+=1; // φ = phi      → 'p2'
    if (p2c(p,0x87cf,"ci")) p+=1; // χ = chi      → 'ci'
    if (p2c(p,0x88cf,"p3")) p+=1; // ψ = psi      → 'p3'
    if (p2c(p,0x89cf,"mg")) p+=1; // ω = omega    → 'mg'
    // ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
    if (p2c(p,0x91ce,"Al")) p+=1; // α = Alpha    → 'Al'
    if (p2c(p,0x92ce,"Bt")) p+=1; // β = Beta     → 'Bt'
    if (p2c(p,0x93ce,"Gm")) p+=1; // γ = Gamma    → 'Gm'
    if (p2c(p,0x94ce,"Dt")) p+=1; // δ = Delta    → 'Dt'
    if (p2c(p,0x95ce,"Ep")) p+=1; // ε = Epsilon  → 'Ep'
    if (p2c(p,0x96ce,"Zt")) p+=1; // ζ = Zeta     → 'Zt'
    if (p2c(p,0x97ce,"Et")) p+=1; // η = Eta      → 'Et'
    if (p2c(p,0x98ce,"Th")) p+=1; // θ = Theta    → 'Th'
    if (p2c(p,0x99ce,"It")) p+=1; // ι = Iota     → 'It'
    if (p2c(p,0x9ace,"Kp")) p+=1; // κ = Kappa    → 'Kp'
    if (p2c(p,0x9bce,"Lm")) p+=1; // λ = Lambda   → 'Lm'
    if (p2c(p,0x9cce,"Mu")) p+=1; // μ = Mu       → 'Mu'
    if (p2c(p,0x9dce,"Nu")) p+=1; // ν = Nu       → 'Nu'
    if (p2c(p,0x9ece,"Xi")) p+=1; // ξ = Xi       → 'Xi'
    if (p2c(p,0x9fce,"Om")) p+=1; // ο = Omicron  → 'Om'
    if (p2c(p,0xa0ce,"Pi")) p+=1; // π = Pi       → 'Pi'
    if (p2c(p,0xa1ce,"Rh")) p+=1; // ρ = Rho      → 'Rh'
    if (p2c(p,0xa3ce,"Sg")) p+=1; // σ = Sigma    → 'Sg'
    if (p2c(p,0xa4ce,"Tt")) p+=1; // τ = Tau      → 'Tt'
    if (p2c(p,0xa5ce,"Up")) p+=1; // υ = Upsilon  → 'Up'
    if (p2c(p,0xa6ce,"P2")) p+=1; // φ = Phi      → 'P2'
    if (p2c(p,0xa7ce,"Ci")) p+=1; // χ = Chi      → 'Ci'
    if (p2c(p,0xa8ce,"P3")) p+=1; // ψ = Psi      → 'P3'
    if (p2c(p,0xa9ce,"Mg")) p+=1; // Ω = Omega    → 'Mg'
    // Partial 𝝏
    if (p4c(p,0x8f9d9df0,"Part")){
      //dbg("\n\t\t\t\t[nUtf8SupThree] HIT p4c ∂->Part!");
      p+=3; // ∂(!=𝝏) = Partial → 'Part'
    }
    if (p4c(p,0x959c9df0,"Part")){
      //dbg("\n\t\t\t\t[nUtf8SupThree] HIT p4c 𝜕->Part!");
      p+=3; // 𝝏 = Partial → 'Part'
    }
    // Aleph ℵ
    // By skipping, we can: strncmp(n->children->token,"ℵ",3)
    //if (p3c(p,0xb584e2,"Ale")) p+=2; // ℵ = Alef  → 'Ale'
    // Fractions
    if (p2s(p,0xbdc2,"0.5",bkp)) p+=1;       // ½
    if (p2s(p,0xbcc2,"0.25",bkp)) p+=1;      // ¼
    if (p3s(p,0x9385e2,"(1./3.)",bkp)) p+=2; // ⅓
    if (p3s(p,0x9b85e2,"0.125",bkp)) p+=2;   // ⅛
    // Infinity
    if (p3s(p,0x9e88e2,"/*wtf huge val*/__builtin_inff()",bkp)) p+=2;   // ∞
    // Sqrt
    if (p3s(p,0x9a88e2,"square_root",bkp)) p+=2;
    // Cbrt
    if (p3s(p,0x9b88e2,"cube_root",bkp)) p+=2;
    
    if (p3s(p,0xa788e2,"&&",bkp)) p+=2;
    if (p3s(p,0xa888e2,"||",bkp)) p+=2;

    // Double Struck Types
    if (p3s(p,0x9d84e2,"Real",bkp)) p+=2; // ℝ
    if (p3s(p,0x9584e2,"Integer",bkp)) p+=2; // ℕ: Should be Natural
    if (p3s(p,0xa484e2,"Integer",bkp)) p+=2; // ℤ
    
    // Les opérateurs suivant ont été transformés en opXYZ()
    // Pas besoin de les modifier dans ce qui sera généré
    // Operators 
    //p2c(p,0x221A,"sq"); // SQRT_OP → 'sq'
    //p2c(p,0xa8e2,"cr"); // CROSS_OP → 'cr'
    //p2c(p,0x22C5,"cd"); // CENTER_DOT_OP → 'cd'
    //p2c(p,0x2297,"ct"); // CIRCLE_TIMES_OP → 'ct'
 }
}


// ****************************************************************************
// * uCa
// ****************************************************************************
static bool uCa(char **a,char **p,const char *utf,const char *ascii){
  if (strncmp(*p,utf,2)) return false;
  strcat(*a,ascii);
  *a+=strlen(ascii);
  *p+=1;
  return true;
}

// ****************************************************************************
// * u2a - αβγδεζηθικλμνξοπρςστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
// ****************************************************************************
static void u2a(char **a, char **p){
  if (uCa(a,p,"α","greek_alpha")) return;
  if (uCa(a,p,"β","greek_beta")) return;
  if (uCa(a,p,"γ","greek_gamma")) return;
  if (uCa(a,p,"δ","greek_delta")) return;
  if (uCa(a,p,"ε","greek_epsilon")) return;
  if (uCa(a,p,"ζ","greek_zeta")) return;
  if (uCa(a,p,"η","greek_eta")) return;
  if (uCa(a,p,"θ","greek_theta")) return;
  if (uCa(a,p,"ι","greek_iota")) return;
  if (uCa(a,p,"κ","greek_kappa")) return;
  if (uCa(a,p,"λ","greek_lambda")) return;
  if (uCa(a,p,"μ","greek_mu")) return;
  if (uCa(a,p,"ν","greek_nu")) return;
  if (uCa(a,p,"ξ","greek_xi")) return;
  if (uCa(a,p,"ο","greek_omicron")) return;
  if (uCa(a,p,"π","greek_pi")) return;
  if (uCa(a,p,"ρ","greek_rho")) return;
  //if (uCa(a,p,"ς","GREEK_SMALL_LETTER_FINAL_SIGMA")) return;
  if (uCa(a,p,"σ","greek_sigma")) return;
  if (uCa(a,p,"τ","greek_tau")) return;
  if (uCa(a,p,"υ","greek_upsilon")) return;
  if (uCa(a,p,"φ","greek_phi")) return;
  if (uCa(a,p,"χ","greek_chi")) return;
  if (uCa(a,p,"ψ","greek_psi")) return;
  if (uCa(a,p,"ω","greek_omega")) return;
  
  if (uCa(a,p,"Α","greek_capital_alpha")) return;
  if (uCa(a,p,"Β","greek_capital_beta")) return;
  if (uCa(a,p,"Γ","greek_capital_gamma")) return;
  if (uCa(a,p,"Δ","greek_capital_delta")) return;
  if (uCa(a,p,"Ε","greek_capital_epsilon")) return;
  if (uCa(a,p,"Ζ","greek_capital_zeta")) return;
  if (uCa(a,p,"Η","greek_capital_eta")) return;
  if (uCa(a,p,"Θ","greek_capital_theta")) return;
  if (uCa(a,p,"Ι","greek_capital_iota")) return;
  if (uCa(a,p,"Κ","greek_capital_kappa")) return;
  if (uCa(a,p,"Λ","greek_capital_lambda")) return;
  if (uCa(a,p,"Μ","greek_capital_mu")) return;
  if (uCa(a,p,"Ν","greek_capital_nu")) return;
  if (uCa(a,p,"Ξ","greek_capital_xi")) return;
  if (uCa(a,p,"Ο","greek_capital_omicron")) return;
  if (uCa(a,p,"Π","greek_capital_pi")) return;
  if (uCa(a,p,"Ρ","greek_capital_rho")) return;
  if (uCa(a,p,"Σ","greek_capital_sigma")) return;
  if (uCa(a,p,"Τ","greek_capital_tau")) return;
  if (uCa(a,p,"Υ","greek_capital_upsilon")) return;
  if (uCa(a,p,"Φ","greek_capital_phi")) return;
  if (uCa(a,p,"Χ","greek_capital_chi")) return;
  if (uCa(a,p,"Ψ","greek_capital_psi")) return;
  if (uCa(a,p,"Ω","greek_capital_omega")) return;
  assert(NULL);
}

// ****************************************************************************
// * utf2ascii
// ****************************************************************************
char *utf2ascii(char *utf){
  assert(utf);
  int nb_utf=0;
  char *p=utf;
  //printf("\n[1;35m%s[0m:",utf);
  for(;*p!=0;p++)
    if (*p<0) nb_utf+=1;
  nb_utf>>=1;
  //printf("nb_utf=%d\n",nb_utf);
  const int utf_size_max = 16;
  char *ascii=(char*)calloc(strlen(utf)+utf_size_max*nb_utf,sizeof(char));
  char *rtn=ascii;
  for(p=utf;*p!=0;p++)
    if (*p<0) u2a(&ascii,&p);
    else *ascii++=*p;
  return rtn;
}
