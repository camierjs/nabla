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
#include "nabla.tab.h"

// ****************************************************************************
// * hookReduction
// ****************************************************************************
void kHookReduction(struct nablaMainStruct *nabla, astNode *n){
  const astNode *item_node = dfsFetch(n->children,rulenameToId("nabla_items"));
  assert(item_node);
  const astNode *global_var_node = n->children->next;
  const astNode *reduction_operation_node = global_var_node->next;
  const astNode *item_var_node = reduction_operation_node->next;
  astNode *at_single_cst_node = dfsFetch(n->children->next, rulenameToId("at_constant"));
  assert(at_single_cst_node!=NULL);
  char *global_var_name = global_var_node->token;
  char *item_var_name = item_var_node->token;


  // Rajout du job de reduction
  nablaJob *redjob = nMiddleJobNew(nabla->entity);
  redjob->is_an_entry_point=true;
  redjob->is_a_function=false;
  redjob->scope  = strdup("NoGroup");
  redjob->region = strdup("NoRegion");
  redjob->item   = strdup(item_node->token);
  redjob->return_type  = strdup("void");
  redjob->xyz    = strdup("NoXYZ");
  redjob->direction  = strdup("NoDirection");
  // Init flush
  redjob->when_index = 0;
  redjob->whens[0] = 0.0;
  // On parse le at_single_cst_node pour le metre dans le redjob->whens[redjob->when_index-1]
  nMiddleAtConstantParse(redjob,at_single_cst_node,nabla);
  nMiddleStoreWhen(redjob,nabla);
  assert(redjob->when_index>0);
  dbg("\n\t[hookReduction] @ %f",redjob->whens[redjob->when_index-1]);
  
    // Préparation du nom du job
  char job_name[NABLA_MAX_FILE_NAME];
  const unsigned long *adrs = (unsigned long*)&redjob->whens[redjob->when_index-1];
  snprintf(job_name,NABLA_MAX_FILE_NAME,"reduction_%s_at_0x%lx",global_var_name,*adrs);
  redjob->name   = strdup(job_name);
  redjob->name_utf8 = strdup(job_name);

  nMiddleJobAdd(nabla->entity, redjob);
  const bool min_reduction = reduction_operation_node->tokenid==MIN_ASSIGN;
  const double reduction_init = min_reduction?1.0e20:-1.0e20;
  //const char* mix = min_reduction?"in":"ax";
 
  // Génération de code associé à ce job de réduction
  nprintf(nabla, NULL, "\n\
// ******************************************************************************\n\
// * Kernel de reduction de la variable '%s' vers la globale '%s'\n\
// ******************************************************************************\n\
#ifndef STRUCT_MIN_FUNCTOR\n\
#define STRUCT_MIN_FUNCTOR\n\
struct minFunctor {\n\
\tdouble value;\n\
\tKOKKOS_INLINE_FUNCTION minFunctor():value(1.0e20){}\n\
\t//KOKKOS_INLINE_FUNCTION minFunctor(const minFunctor& f):value(f.value){}\n\
\tKOKKOS_INLINE_FUNCTION minFunctor(double v):value(v){}\n\
\tKOKKOS_INLINE_FUNCTION void operator+=(const volatile minFunctor& f) volatile {\n\
\t\tvalue = fmin(value,f.value);\n\
\t}\n};\n\
#endif // STRUCT_MIN_FUNCTOR\n\
void %s(const int NABLA_NB_CELLS_WARP,\n\t\t\t\t\t\tconst int NABLA_NB_CELLS,",
          item_var_name,
          global_var_name,job_name);
  
  nablaVariable *global_var=nMiddleVariableFind(nabla->variables, global_var_name);
  assert(global_var);

  nprintf(nabla, NULL, "\n\t\t\t\t\t\tKokkos::View<%s*>& %s_%s",
          global_var->type,
          global_var->item,
          global_var->name);
  
  nablaVariable *local_var=nMiddleVariableFind(nabla->variables, item_var_name);
  assert(local_var);
  nprintf(nabla, NULL, ",\n\t\t\t\t\t\tconst Kokkos::View<%s*>& %s_%s",
          local_var->type,
          local_var->item,
          local_var->name);

  // Et on les rajoute pour faire croire qu'on a fait le DFS
  nablaVariable* new_global_var=nMiddleVariableNew(nabla);
  new_global_var->name=strdup(global_var->name);
  new_global_var->type=strdup(global_var->type);
  new_global_var->out=true;
  new_global_var->item=strdup("global");
  redjob->used_variables=new_global_var;

  nablaVariable* new_local_var=nMiddleVariableNew(nabla);
  new_local_var->name=strdup(local_var->name);
  new_local_var->type=strdup(local_var->type);
  new_local_var->in=true;
  new_local_var->item=strdup(local_var->item);
  redjob->used_variables->next=new_local_var;

  nprintf(nabla, NULL,"){\n\
\tminFunctor mix=%e;\n\
\tKokkos::parallel_reduce(NABLA_NB_%sS, KOKKOS_LAMBDA (const int i, minFunctor& lmix) {\n\
\t//const double val=%s_%s[i];\n\
\t//printf(\"\\n\\tbefore lmix=%%f, val=%%f\",lmix.value,val);\n\
\tminFunctor tmp(%s_%s[i]);\n\
\tlmix+=tmp;\n\
\t//printf(\"\\n\\tafter lmix=%%f\",lmix.value);\n\
\t}, mix);\n\
\t%s_%s[0]=mix.value;\n}\n",
          reduction_init,
          (item_node->token[0]=='c')?"CELL":(item_node->token[0]=='n')?"NODE":"NULL",
          local_var->item, local_var->name,
          //mix,
          local_var->item, local_var->name,
          global_var->item, global_var->name
          );
}
