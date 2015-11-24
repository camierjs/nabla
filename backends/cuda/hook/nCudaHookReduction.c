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
#include "nabla.tab.h"


void nCudaHookReduction(struct nablaMainStruct *nabla, astNode *n){
  int fakeNumParams=0;
  const astNode *item_node = n->children->next->children;
  const astNode *global_var_node = n->children->next->next;
  const astNode *reduction_operation_node = global_var_node->next;
  astNode *item_var_node = reduction_operation_node->next;
  astNode *at_single_cst_node = dfsFetch(n, rulenameToId("at_constant"));
  assert(at_single_cst_node!=NULL);
  char *global_var_name = global_var_node->token;
  char *item_var_name = item_var_node->token;
  // Préparation du nom du job
  char job_name[NABLA_MAX_FILE_NAME];
  job_name[0]=0;
  strcat(job_name,"cudaReduction_");
  strcat(job_name,global_var_name);
  // Rajout du job de reduction
  nablaJob *redjob = nMiddleJobNew(nabla->entity);
  redjob->is_an_entry_point=true;
  redjob->is_a_function=false;
  redjob->scope  = strdup("NoGroup");
  redjob->region = strdup("NoRegion");
  redjob->item   = strdup(item_node->token);
  redjob->return_type  = strdup("void");
  redjob->name   = strdup(job_name);
  redjob->name_utf8 = strdup(job_name);
  redjob->xyz    = strdup("NoXYZ");
  redjob->direction  = strdup("NoDirection");
  redjob->called_variables=nMiddleVariableNew(nabla);
  redjob->called_variables->item=strdup("cell");
  redjob->called_variables->name=item_var_name;
  // On annonce que c'est un job de reduction pour lancer le deuxieme etage de reduction dans la boucle
  redjob->reduction = true;
  redjob->reduction_name = strdup(global_var_name);

    // Init flush
  redjob->when_index = 0;
  redjob->whens[0] = 0.0;
  // On parse le at_single_cst_node pour le metre dans le redjob->whens[redjob->when_index-1]
  nMiddleAtConstantParse(redjob,at_single_cst_node,nabla,redjob->at);
  nMiddleStoreWhen(redjob,nabla,NULL);
  assert(redjob->when_index>0);
  dbg("\n\t[cudaHookReduction] @ %f",redjob->whens[redjob->when_index-1]);

  nMiddleJobAdd(nabla->entity, redjob);
  const bool min_reduction = reduction_operation_node->tokenid==MIN_ASSIGN;
  const double reduction_init = min_reduction?1.0e20:-1.0e20;
  // Génération de code associé à ce job de réduction
  nprintf(nabla, NULL, "\n\
// ******************************************************************************\n\
// * Kernel de reduction de la variable '%s' vers la globale '%s'\n\
// ******************************************************************************\n\
__global__ void %s(", item_var_name, global_var_name, job_name);
  nCudaHookAddExtraParameters(nabla,redjob,&fakeNumParams);
  nprintf(nabla, NULL,",Real *cell_%s){ // @ %s\n\
\t//const double reduction_init=%e;\n\
\tCUDA_INI_CELL_THREAD(tcid);\n\
\t/**global_%s=*/Reduce%sToDouble((double)(cell_%s[tcid]));\n\
}\n\n", item_var_name,
          at_single_cst_node->token,
          reduction_init,
          global_var_name,
          min_reduction?"Min":"Max",
          item_var_name);
}
