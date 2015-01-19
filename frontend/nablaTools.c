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


/******************************************************************************
 * nprintf
 ******************************************************************************/
int nprintf(const nablaMain *nabla, const char *debug, const char *format, ...){
  int rtn;
  va_list args;
  if ((dbgGet()&DBG_CYC)!=0) if (debug!=NULL) fprintf(nabla->entity->src, debug);
  if (format==NULL) return 0;
  va_start(args, format);
  if ((rtn=vfprintf(nabla->entity->src, format, args))<0)
    exit(printf("[nprintf] error\n"));
  if (fflush(nabla->entity->src)!=0)
    exit(printf("[nprintf] Could not flush to file\n"));
  va_end(args);
  return rtn;
}


/******************************************************************************
 * hprintf
 ******************************************************************************/
int hprintf(const nablaMain *nabla, const char *debug, const char *format, ...){
  int rtn;
  va_list args;
  if ((dbgGet()&DBG_CYC)!=0) if (debug!=NULL) fprintf(nabla->entity->src, debug);
  if (format==NULL) return 0;
  va_start(args, format);
  if ((rtn=vfprintf(nabla->entity->hdr, format, args))<0)
    exit(printf("[nprintf] error\n"));
  if (fflush(nabla->entity->hdr)!=0)
    exit(printf("[nprintf] Could not flush to file\n"));
  va_end(args);
  return rtn;
}


/******************************************************************************
 * strDownCase
 ******************************************************************************/
char *toolStrDownCase(const char * str){
  char *p=strdup(str);
  char *bkp=p;
  for(;*p!=0;p++){
    if (*p>64 && *p<91) *p+=32;
  }
  return bkp;
}


/******************************************************************************
 * strUpCase
 ******************************************************************************/
char *toolStrUpCase(const char * str){
  char *p=strdup(str);
  char *bkp=p;
  for(;*p!=0;p++)
    if ((*p>=97)&&(*p<=122)) *p-=32;
  return bkp;
}

/******************************************************************************
 * ''' to ' '
 ******************************************************************************/
char *trQuote(const char * str){
  char *p=strdup(str);
  char *bkp=p;
  for(;*p!=0;p++)
    if (*p==0x27) *p=0x20;
  return bkp;
}


// *****************************************************************************
// * op2name
// *****************************************************************************
char *op2name(char *op){
  //printf("[%s] ",op);
  if (strncmp(op,"√",3)==0) return "opSqrt";
  if (strncmp(op,"∛",3)==0) return "opCbrt";
  
  if (strncmp(op,"⋅",3)==0) return "opScaMul";
  if (strncmp(op,"⨯",3)==0) return "opVecMul";
  if (strncmp(op,"⤫",3)==0) return "cross2D";
  
  if (strncmp(op,"⊗",3)==0) return "opProdTens";
  if (strncmp(op,"⨂",3)==0) return "opProdTensVec";
  
  if (strncmp(op,"⊛",3)==0) return "opMatrixProduct";
  
  switch (op[0]){
  case ('*') : return "opMul";
  case ('/') : return "opDiv";
  case ('%') : return "opMod";
  case ('+') : return "opAdd";
  case ('-') : return "opSub";
  case ('?') : return "opTernary";
  default: return "opUnknown"; 
  }
  return "opUnknown";
}



// *****************************************************************************
// * p?[c|s]
// *****************************************************************************
static inline bool p2c(char *p, unsigned short hex, char *letter){
  if (*(unsigned short*)p!=hex) return false;
  *p=letter[0];
  *(p+1)=letter[1];
  return true;
}
static inline bool p2s(char *p, unsigned short hex, const char *str, char **bkp){
  if (*(unsigned short*)p!=hex) return false;
  *bkp=strdup(str); 
  return true;
}
static inline bool p3c(char *p, const unsigned int hex, const char *letter){
  register unsigned int mask = (*(unsigned int*)p)&0x00FFFFFFl;
  //dbg("\n[p3c] p=%08p vs hex=0x%x", mask, hex);
  if (mask!=hex) return false;
  *p=letter[0];
  *(p+1)=letter[1];
  *(p+2)=letter[2];
  //dbg("\n[p3c] HIT!");
  return true;
}
static inline bool p4c(char *p, const unsigned int hex, const char *letter){
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
static inline bool p4c3(char *w, const char *p, const unsigned int hex, const char *letter){
  register unsigned int mask = (*(unsigned int*)p)&0xFFFFFFFFl;
  if (mask!=hex) return false;
  *w=letter[0];
  *(w+1)=letter[1];
  *(w+2)=letter[2];
  return true;
}
static inline bool p3s(char *p, const unsigned int hex, const char *str, char **bkp){
  register unsigned int mask = (*(unsigned int*)p)&0x00FFFFFFl;
  if (mask!=hex) return false;
  *bkp=strdup(str); 
  return true;
}
static inline bool p4s(char *p, const unsigned int hex, const char *str, char **bkp){
  register unsigned int mask = (*(unsigned int*)p)&0xFFFFFFFFl;
  if (mask!=hex) return false;
  *bkp=strdup(str); 
  return true;
}

// ****************************************************************************
// UTF8 codes > 3 bytes are not currently supported
// Pour ces caractères, on va les réduire si l'on peut
// ****************************************************************************
void nUtf8SupThree(char **read){//read
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

/******************************************************************************
 * nUtf8
 * αβγδεζηθικλμνξοπρςστυφχψω
 * od -vt x2 /tmp/try
 * 0000000 b1ce b2ce b3ce b4ce b5ce b6ce b7ce b8ce
 * 0000020 b9ce bace bbce bcce bdce bece bfce 80cf
 * 0000040 81cf 3030 82cf 83cf 3030 84cf 85cf 86cf
 * 0000060 87cf 88cf 89cf 000a
 * ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
 * ½  ⅓  ¼  ⅛ 
 ******************************************************************************/
void nUtf8(char **bkp){
  char *p=*bkp;
  if (p==NULL) return;
  //dbg("\n[nUtf8] '%s'",p);
  if (*(unsigned int*)p==0x0074b4ce) { // "δt"
    //dbg("\n[nUtf8] hits deltat!");
    *bkp=strdup("deltat"); 
    return;
  }
  /*if (*(unsigned int*)p==0x00b584e2) { // "ℵ"
    dbg("\n[nUtf8] hits single ℵ ");
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
    if (p3c(p,0xb584e2,"Ale")) p+=2; // ℵ = Alef  → 'Ale'
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
    
    // Les opérateurs suivant ont été transformés en opXYZ()
    // Pas besoin de les modifier dans ce qui sera généré
    // Operators 
    //p2c(p,0x221A,"sq"); // SQRT_OP → 'sq'
    //p2c(p,0xa8e2,"cr"); // CROSS_OP → 'cr'
    //p2c(p,0x22C5,"cd"); // CENTER_DOT_OP → 'cd'
    //p2c(p,0x2297,"ct"); // CIRCLE_TIMES_OP → 'ct'
 }
}


// *****************************************************************************
// * nablaMakeTempFile
// *****************************************************************************
int nablaMakeTempFile(const char *entity_name, char **unique_temporary_file_name){
  int n,size = NABLA_MAX_FILE_NAME;
  if ((*unique_temporary_file_name=malloc(size))==NULL)
    error(!0,0,"[nablaMakeTempFile] Could not malloc our unique_temporary_file_name!");
  n=snprintf(*unique_temporary_file_name, size, "/tmp/nabla_%s_XXXXXX", entity_name);
  if (n > -1 && n < size)
    return mkstemp(*unique_temporary_file_name);
  error(!0,0,"[nablaMakeTempFile] Error in snprintf into unique_temporary_file_name!");
  return -1;
}
