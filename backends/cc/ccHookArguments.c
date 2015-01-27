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
#include "ccHook.h"
#include "nabla.tab.h"


/*****************************************************************************
  * Dump d'extra arguments
 *****************************************************************************/
void ccAddExtraArguments(nablaMain *nabla, nablaJob *job, int *numParams){
  nprintf(nabla,"\n\t\t/*ccAddExtraArguments*/",NULL);
}



/*****************************************************************************
  * Dump dans le src des arguments nabla en in comme en out
 *****************************************************************************/
void ccDumpNablaArgumentList(nablaMain *nabla, astNode *n, int *numParams){
  nprintf(nabla,"\n\t\t/*ccDumpNablaArgumentList*/",NULL);
}


/*****************************************************************************
  * Dump dans le src l'appel des fonction de debug des arguments nabla  en out
 *****************************************************************************/
void ccDumpNablaDebugFunctionFromOutArguments(nablaMain *nabla, astNode *n, bool in_or_out){
  nprintf(nabla,"\n\t\t/*ccDumpNablaDebugFunctionFromOutArguments*/",NULL);
}


// *****************************************************************************
// * Ajout des variables d'un job trouvé depuis une fonction @ée
// *****************************************************************************
void ccAddNablaVariableList(nablaMain *nabla, astNode *n, nablaVariable **variables){
  nprintf(nabla,"\n/*ccAddNablaVariableList*/",NULL);
}


// ****************************************************************************
// * Dump d'extra connectivity
// ****************************************************************************
void ccAddExtraConnectivitiesArguments(nablaMain *nabla, int *numParams){
  return;
}


// ****************************************************************************
// * ccHookReturnFromArgument
// ****************************************************************************
void ccHookReturnFromArgument(nablaMain *nabla, nablaJob *job){
  const char *rtnVariable=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nprintf(nabla, NULL, "\
\n\tint threads = omp_get_max_threads();\
\n\tReal %s_per_thread[threads];", rtnVariable);
}

void ccHookAddArguments(struct nablaMainStruct *nabla,nablaJob *fct){
  if (fct->parse.function_call_name!=NULL){
    //nprintf(nabla, "/*ShouldDumpParamsInCc*/", "/*Arg*/");
    int numParams=1;
    nablaJob *called=nablaJobFind(fct->entity->jobs,fct->parse.function_call_name);
    ccAddExtraArguments(nabla, called, &numParams);
    if (called->nblParamsNode != NULL)
      ccDumpNablaArgumentList(nabla,called->nblParamsNode,&numParams);
  }
}
