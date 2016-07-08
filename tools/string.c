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
// * strDownCase
// ****************************************************************************
char *toolStrDownCase(const char * str){
  char *p=sdup(str);
  char *bkp=p;
  for(;*p!=0;p++){
    if (*p>64 && *p<91) *p+=32;
  }
  return bkp;
}


// ****************************************************************************
// * path2namespace
// ****************************************************************************
char *pth2nmspc(const char *str){
  char *p=sdup(str);
  char *f=p;//forecast
  char *bkp=p;
  for(int i=0;*f!=0;p++,f++,i++){
    if (*f=='/'){
      char nxt_char=*++f;
      if (nxt_char>=97 && nxt_char<=122) nxt_char-=32; // to upper
      *p=nxt_char;
      continue;
    }
    if (i==0 && *f>=97 && *f<=122) *p-=32; // to upper
    if ( i>0 && *f>64  && *f<91) *p+=32; // to lower
    else *p=*f;
  }
  *p=0;
  return bkp;
}


/******************************************************************************
 * strUpCase
 ******************************************************************************/
char *toolStrUpCase(const char * str){
  char *p=sdup(str);
  char *bkp=p;
  for(;*p!=0;p++)
    if ((*p>=97)&&(*p<=122)) *p-=32;
  return bkp;
}


// ****************************************************************************
// * toolStrUpCaseAndSwap
// ****************************************************************************
char *toolStrUpCaseAndSwap(const char * str, const char cHit, const char cNew){
  char *p=sdup(str);
  char *bkp=p;
  for(;*p!=0;p++){
    if (*p==cHit) *p=cNew;
    if ((*p>=97)&&(*p<=122)) *p-=32;
  }
  return bkp;
}


/******************************************************************************
 * ''' to ' '
 ******************************************************************************/
char *toolStrQuote(const char * str){
  char *p=sdup(str);
  char *bkp=p;
  for(;*p!=0;p++)
    if (*p==0x27) *p=0x20;
  return bkp;
}


// *****************************************************************************
// * 
// *****************************************************************************
const char* mkktemp(const char *prefix){
  char *rtn,*unique_temporary_kernel_name=NULL;
  const int size = NABLA_MAX_FILE_NAME;

  dbg("\n\t\t[mkktemp] with prefix: '%s'", prefix);

  dbg("\n\t\t[mkktemp] calloc(%d,%d)",size,sizeof(char));
  if ((unique_temporary_kernel_name=calloc(size,sizeof(char)))==NULL)
    nablaError("[mkktemp] Could not calloc our unique_temporary_kernel_name!");
  
  dbg("\n\t\t[mkktemp] snprintf");
  const int n=snprintf(unique_temporary_kernel_name, size, "/tmp/nabla_%sXXXXXX", prefix);
  
  if (n > -1 && n < size)
    if (mkstemp(unique_temporary_kernel_name)==-1)
      nablaError("[mkktemp] Could not mkstemp our unique_temporary_kernel_name!");
  
  assert(strrchr(prefix,'_')==NULL);
  rtn=sdup(strrchr(unique_temporary_kernel_name,'_')+1);
  free(unique_temporary_kernel_name);
  return rtn;
}


// *****************************************************************************
// * 
// *****************************************************************************
void toolUnlinkKtemp(nablaJob *job){
  char *kernel_name=NULL;
  const int size = NABLA_MAX_FILE_NAME;
  
  if ((kernel_name=calloc(size,sizeof(char)))==NULL)
    nablaError("[mkktemp] Could not calloc our unique_temporary_kernel_name!");

  for(;job!=NULL;job=job->next){
    if (!job->has_to_be_unlinked) continue;
    snprintf(kernel_name, size, "/tmp/nabla_%s", job->name);
    dbg("\n\t\t[toolUnlinkKtemp] kernel_name to unlink: '%s'", kernel_name);
    if (unlink(kernel_name)!=0)
      nablaError("Error while removing '%s' file", kernel_name);
  }
  free(kernel_name);
}
