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
#include <strings.h>
#include "nabla.tab.h"
#include "backends/arcane/arcane.h"


// ****************************************************************************
// *arcaneHookReduction
// ****************************************************************************
void arcaneHookReduction(nablaMain *nabla, astNode *n){
  dbg("\n\t\t[arcaneHookReduction]");
  astNode *nabla_items = dfsFetch(n->children,rulenameToId("nabla_items"));
  assert(nabla_items);
  astNode *global_var_node = n->children->next;
  assert(global_var_node);
  astNode *reduction_operation_node = global_var_node->next;
  assert(reduction_operation_node);
  astNode *item_var_node = reduction_operation_node->next;
  assert(item_var_node);
  astNode *at_single_cst_node = dfsFetch(n->children->next, rulenameToId("at_constant"));
  assert(at_single_cst_node);
  const char *global_var_name = global_var_node->token;
  assert(global_var_name);
  const char *item_var_name = item_var_node->token;
  assert(item_var_name);
  dbg("\n\t\t[arcaneHookReduction] global_var_name=%s",global_var_name);
  dbg("\n\t\t[arcaneHookReduction] item_var_name=%s",item_var_name);
  dbg("\n\t\t[arcaneHookReduction] item=%s",nabla_items->token);
  // Préparation du nom du job
  char job_name[NABLA_MAX_FILE_NAME];
  job_name[0]=0;
  strcat(job_name,"arcaneReduction_");
  strcat(job_name,global_var_name);
  strcat(job_name,"_");
  dbg("\n\t\t[arcaneHookReduction] job_name=%s",job_name);

  // Rajout du job de reduction
  nablaJob *redjob = nMiddleJobNew(nabla->entity);
  redjob->when_index = 0;
  redjob->whens[0] = 0.0;
  // On parse le at_single_cst_node pour le metre dans le redjob->whens[redjob->when_index-1]
  nMiddleAtConstantParse(redjob,at_single_cst_node,nabla);
  nMiddleStoreWhen(redjob,nabla);
  assert(redjob->when_index>0);
  dbg("\n\t[arcaneHookReduction] @ %f",redjob->whens[redjob->when_index-1]);
  // On construit mtn la string representant le 'double'
  char *ftoa=calloc(1024,1);
  sprintf(ftoa,"%a",redjob->whens[redjob->when_index-1]);
  // S'il est negatif, on l'évite
  if (ftoa[0]=='-') ftoa+=1;
  // On évite aussi le '.'
  char *locate_dot_sign_in_ftoa=index(ftoa,'.');
  assert(locate_dot_sign_in_ftoa);
  *locate_dot_sign_in_ftoa='_';
  // Ainsi que le '+'
  char *locate_exp_sign_in_ftoa=rindex(ftoa,'+');
  assert(locate_exp_sign_in_ftoa);
  *locate_exp_sign_in_ftoa='_';
  dbg("\n\t\t[arcaneHookReduction] ftoa=%s",ftoa);
  strcat(job_name,ftoa);
  dbg("\n\t\t[arcaneHookReduction] job_name=%s",job_name);

  redjob->is_an_entry_point=true;
  redjob->is_a_function=false;
  redjob->scope  = strdup("NoGroup");
  redjob->region = strdup("NoRegion");
  redjob->item   = strdup(nabla_items->token);
  redjob->return_type  = strdup("void");
  redjob->name   = strdup(job_name);
  redjob->name_utf8 = strdup(job_name);
  redjob->xyz    = strdup("NoXYZ");
  redjob->direction  = strdup("NoDirection");
  

  nMiddleJobAdd(nabla->entity, redjob);
  const bool min_reduction = reduction_operation_node->tokenid==MIN_ASSIGN;
  const double reduction_init = min_reduction?1.0e20:-1.0e20;
  const char* mix = min_reduction?"in":"ax";
  // Génération de code associé à ce job de réduction
  nablaVariable *var=nMiddleVariableFind(nabla->variables, item_var_name);
  const char cnf=var->item[0];
  assert(var!=NULL);
  nprintf(nabla, NULL, "\n\n\
// ******************************************************************************\n\
// * Kernel de reduction de la variable '%s' vers la globale '%s'\n\
// ******************************************************************************\n\
void %s%s::%s(){\n",
          item_var_name,
          global_var_name,
          nabla->name,
          isAnArcaneModule(nabla)?"Module":"Service",
          job_name);
  
  nprintf(nabla, NULL, "\n\tm_global_%s=%f;\n",global_var_name,reduction_init);
  if (cnf=='c') nprintf(nabla, NULL, "\tENUMERATE_CELL(cell,ownCells()){");
  if (cnf=='n') nprintf(nabla, NULL, "\tENUMERATE_NODE(node,ownNodes()){");
  nprintf(nabla, NULL, "\n\t\tm_global_%s=::fm%s(m_global_%s(),m_%s_%s[%s]);",
          global_var_name,mix,global_var_name,
          var->item,item_var_name,var->item);
  nprintf(nabla, NULL, "\n\t}");
  
  nprintf(nabla, NULL, "\n\
   m_global_%s=mpi_reduce(Parallel::ReduceM%s,m_global_%s());\n}\n\n",
          global_var_name,mix,global_var_name);
}
