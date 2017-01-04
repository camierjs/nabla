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


// *************************************************************
// * nHLTNew
// *************************************************************
nablaHLT *nHLTNew(nablaMain *nabla){
  nablaHLT *HLT = (nablaHLT *)calloc(1,sizeof(nablaHLT));
  assert(HLT != NULL);
  return HLT; 
}

// *************************************************************
// * nHLTCompare
// *************************************************************
int nHLTCompare(const nablaHLT *one,const nablaHLT *two){
  const int d1=one->depth;
  const int d2=two->depth;
  assert(d1>0 && d2>0);

  // On est sur la 'root' ligne en temps
  if (d1==1&&d2==1){
    if (one->at[0]==two->at[0]) return 0;
    if (one->at[0]<two->at[0]) return -1;
    if (one->at[0]>two->at[0]) return +1;
  }

  // Sinon, c'est qu'on teste les niveaux HLT
  // On va tester niveau par niveau jusqu'à trouver le résultat
  /*for(k=1;;k+=1){
    if ()
    }*/

  assert(false);
  return 0;
}


// *************************************************************
// * nSwirlNew
// *************************************************************
nablaSwirl *nSwirlNew(nablaMain *nabla){
  nablaSwirl *swirl = (nablaSwirl *)calloc(1,sizeof(nablaSwirl));
  assert(swirl != NULL);
  return swirl; 
}


// *************************************************************
// * nSwirlLast
// *************************************************************
nablaSwirl *nSwirlLast(nablaSwirl *swirls) {
  while(swirls->next != NULL)
    swirls = swirls->next;
  return swirls;
}


// *************************************************************
// * nSwirlAdd
// *************************************************************
nablaSwirl *nSwirlAdd(nablaMain *nabla, nablaSwirl *swirl) {
  assert(swirl != NULL);
  if (nabla->swirls == NULL)
    nabla->swirls=swirl;
  else
    nSwirlLast(nabla->swirls)->next=swirl;
  return NABLA_OK;
}



// *************************************************************
// * nSwirlFind
// *************************************************************
nablaSwirl *nSwirlFind(nablaSwirl *swirl, nablaHLT *at) {
  assert(at!=NULL);
  //dbg("\n\t[nablaSwirlFind] looking for '%s'", name);
  // Some backends use the fact it will return NULL
  //assert(swirl != NULL);  assert(name != NULL);
  while(swirl != NULL) {
    //dbg(" ?%s", swirl->name);
    if(nHLTCompare(swirl->at, at) == 0){
      //dbg(" Yes!");
      return swirl;
    }
    swirl = swirl->next;
  }
  //dbg(" Nope!");
  return NULL;
}
