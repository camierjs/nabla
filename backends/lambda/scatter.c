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
// * Flush de la 'vraie' variable depuis celle déclarée en in/out
// ****************************************************************************
/*static void lambdaHookFlushRealVariable(nablaJob *job, nablaVariable *var){
  // On informe la suite que cette variable est en train d'être scatterée
  nablaVariable *real_variable=nMiddleVariableFind(job->entity->main->variables, var->name);
  if (real_variable==NULL)
    nablaError("Could not find real variable from scattered variables!");
  real_variable->is_gathered=false;
  }*/


// ****************************************************************************
// * Scatter
// ****************************************************************************
char* lambdaHookScatter(nablaVariable* var){
  char scatter[1024];
  snprintf(scatter, 1024,
           "\n\tscatter%sk(cell_node[n*NABLA_NB_CELLS+c], &gathered_%s_%s, %s_%s);",
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           var->item, var->name);
  return strdup(scatter);
}


// ****************************************************************************
// * Filtrage du SCATTER
// ****************************************************************************
char* lambdaHookFilterScatter(astNode *n,nablaJob *job){
  char *scatters=NULL;
  int nbToScatter=0;
  
  if (job->parse.selection_statement_in_compound_statement){
    nprintf(job->entity->main,
            "/*selection_statement_in_compound_statement, nothing to do*/",
            "/*if=>!lambdaScatter*/");
    return "";
  }
  
  if ((scatters=calloc(NABLA_MAX_FILE_NAME,sizeof(char)))==NULL)
    nablaError("[lambdaHookFilterScatter] Could not malloc our scatters!");

  // On récupère le nombre de variables potentielles à scatterer
//#warning variables_to_gather_scatter seem NULL!
  //for(var=job->variables_to_gather_scatter;var!=NULL;var=var->next){
  for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
    dbg("\n\t\t\t\t[lambdaHookFilterScatter] var '%s'", var->name);
    nprintf(job->entity->main, NULL, "\n\t/*?var %s*/",var->name);
    if (!var->is_gathered) continue;
    nprintf(job->entity->main, NULL, "/*gathered!*/");
    if (!var->out) continue;
    nprintf(job->entity->main, NULL, "/*out!*/");    

    nprintf(job->entity->main, NULL, "/*%s*/",var->name);
    nbToScatter+=1;
    strcat(scatters,lambdaHookScatter(var));
  }
  return strdup(scatters);
}

