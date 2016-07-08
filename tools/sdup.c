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
// * sdup
// ****************************************************************************
char *strdup(const char *s);

typedef struct nStrDup{
  char *str;
  struct nStrDup *next; 
} nStrDup;

static nStrDup *duped=NULL;
static nStrDup *dplst=NULL;

static nStrDup* snew(const char* str){
  nStrDup *new=(nStrDup *)calloc(1,sizeof(nStrDup));
  assert(new);
  new->str=strdup(str);
  return new;
}

static nStrDup *sadd(const char* str){
  nStrDup *dup=duped;
  if (duped == NULL) return dplst=duped=snew(str);
  //for(;dup->next!=NULL;dup=dup->next);
  dup=dplst;
  return dplst=dup->next=snew(str);
}

char *sdup(const char *s){
  //printf("[1;33m[sdup] '%s'[0m\n",s);
  return sadd(s)->str;
}

void sfree(void){
  //printf("[1;31m[sfree][0m\n");    
  for(nStrDup *this,*dup=this=duped;dup!=NULL;){
    dup=(this=dup)->next;
    //printf("\t[1;33m[sfree] '%s'[0m\n",this->str);    
    free(this->str);//if (this->str!=NULL){ free(this->str); this->str=NULL;}
    free(this);//if (this!=NULL){ free(this);this=NULL;}
  }
}
