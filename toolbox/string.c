///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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
char *toolStrQuote(const char * str){
  char *p=strdup(str);
  char *bkp=p;
  for(;*p!=0;p++)
    if (*p==0x27) *p=0x20;
  return bkp;
}


// *****************************************************************************
// * 
// *****************************************************************************
const char* mkktemp(const char *prefix){
  char *unique_temporary_kernel_name=NULL;
  int n,size = NABLA_MAX_FILE_NAME;
  
  if ((unique_temporary_kernel_name=malloc(size))==NULL)
    nablaError("[mkktemp] Could not malloc our unique_temporary_kernel_name!");
  n=snprintf(unique_temporary_kernel_name, size, "/tmp/nabla_%sXXXXXX", prefix);
  
  if (n > -1 && n < size)
    if (mkstemp(unique_temporary_kernel_name)==-1)
      nablaError("[mkktemp] Could not mkstemp our unique_temporary_kernel_name!");
  assert(strrchr(prefix,'_')==NULL);
  return strdup(strrchr(unique_temporary_kernel_name,'_')+1);
}


// *****************************************************************************
// * 
// *****************************************************************************
void toolUnlinkKtemp(nablaJob *job){
  char *kernel_name=NULL;
  int size = NABLA_MAX_FILE_NAME;
  
  if ((kernel_name=malloc(size))==NULL)
    nablaError("[mkktemp] Could not malloc our unique_temporary_kernel_name!");

  for(;job!=NULL;job=job->next){
    if (!job->has_to_be_unlinked) continue;
    snprintf(kernel_name, size, "/tmp/nabla_%s", job->name);
    dbg("\n\t\t[toolUnlinkKtemp] kernel_name to unlink: '%s'", kernel_name);
    if (unlink(kernel_name)!=0)
      nablaError("Error while removing '%s' file", kernel_name);
  }
}