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
#include "nabla.tab.h"


void nMiddlePowerFree(nablaPowerType *ptype){
  for(nablaPowerType *this,*ptp=this=ptype;ptp!=NULL;free(this))
    ptp=(this=ptp)->next;
}
nablaPowerType *nMiddlePowerTypeNew(void){
	nablaPowerType *ptype = (nablaPowerType*) calloc(1,sizeof(nablaPowerType));
 	assert(ptype);
  	return ptype; 
}
nablaPowerType *nMiddlePowerTypeLast(nablaPowerType *ptype) {
   while(ptype->next != NULL) ptype = ptype->next;
   return ptype;
}
nablaPowerType *nMiddlePowerTypeAdd(nablaVariable *var, nablaPowerType *ptype) {
  assert(ptype);
  if (var->power_type==NULL) var->power_type=ptype;
  else nMiddlePowerTypeLast(var->power_type)->next=ptype;
  return NABLA_OK;
}

// ****************************************************************************
// * dfsPowerArgs
// ****************************************************************************
static void dfsPowerArgs(astNode *n, nablaPowerType *ptype){
  if (n->tokenid==IDENTIFIER){
    dbg("\n\t\t\t\t[1;33m[dfsPower] arg '%s'[0m",n->token);
    ptype->args[ptype->nargs++]=sdup(n->token);
  }
  if (n->children != NULL) dfsPowerArgs(n->children,ptype);
  if (n->next != NULL) dfsPowerArgs(n->next,ptype);
}


// ****************************************************************************
// * dfsPower
// ****************************************************************************
static void dfsPower(astNode *n, nablaPowerType *ptype){
  if (n->ruleid==ruleToId(rule_power_dimension)){
    if (n->children->ruleid==ruleToId(rule_power_function)){
      dbg("\n\t\t\t[1;33m[dfsPower] power '%s' function[0m",n->children->children->token);
      ptype->id=sdup(n->children->children->token);
      dfsPowerArgs(n->children->children->next,ptype);
    }
    if (n->children->tokenid==IDENTIFIER){
      ptype->id=sdup(n->children->token);
      dbg("\n\t\t\t[1;33m[dfsPower] power '%s' ident[0m",n->children->token);
    }
  }
  if (n->children != NULL) dfsPower(n->children,ptype);
  if (n->next != NULL) dfsPower(n->next,ptype);
}


// ****************************************************************************
// * nMiddlePower
// ****************************************************************************
void nMiddlePower(astNode *n,nablaMain *nabla,nablaVariable *var){
  nablaPowerType *ptype=nMiddlePowerTypeNew();
  assert(ptype);
  dfsPower(n,ptype);
  nMiddlePowerTypeAdd(var,ptype);
}
