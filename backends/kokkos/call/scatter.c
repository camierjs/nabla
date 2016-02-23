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
static void callFlushRealVariable(nablaJob *job, nablaVariable *var){
  // On informe la suite que cette variable est en train d'être scatterée
  nablaVariable *real_variable=nMiddleVariableFind(job->entity->main->variables, var->name);
  if (real_variable==NULL)
    nablaError("Could not find real variable from scattered variables!");
  real_variable->is_gathered=false;
}


// ****************************************************************************
// * Scatter
// ****************************************************************************
static char* callScatter(nablaVariable* var){
  char scatter[1024];
  snprintf(scatter, 1024, "\tscatter%sk(ia, &gathered_%s_%s, %s_%s);",
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           var->item, var->name);
  return strdup(scatter);
}



// ****************************************************************************
// * Filtrage du SCATTER
// ****************************************************************************
char* callFilterScatter(nablaJob *job){
  int i;
  char scatters[1024];
  nablaVariable *var;
  scatters[0]='\0';
  int nbToScatter=0;
  int filteredNbToScatter=0;
  
  if (job->parse.selection_statement_in_compound_statement){
    nprintf(job->entity->main, "/*selection_statement_in_compound_statement, nothing to do*/",
            "/*if=>!scatter*/");
    return "";
  }
  
  // On récupère le nombre de variables potentielles à scatterer
  for(var=job->variables_to_gather_scatter;var!=NULL;var=var->next)
    nbToScatter+=1;

  // S'il y en a pas, on a rien d'autre à faire
  if (nbToScatter==0) return "";
  
  for(var=job->variables_to_gather_scatter;var!=NULL;var=var->next){
    //nprintf(job->entity->main, NULL, "\n\t\t// scatter on %s for variable %s_%s", job->item, var->item, var->name);
    //nprintf(job->entity->main, NULL, "\n\t\t// scatter enum_enum=%c", job->parse.enum_enum);
    if (job->parse.enum_enum=='\0') continue;
    filteredNbToScatter+=1;
  }
  //nprintf(job->entity->main, NULL, "/*filteredNbToScatter=%d*/", filteredNbToScatter);

  // S'il reste rien après le filtre, on a rien d'autre à faire
  if (filteredNbToScatter==0) return "";
  
  for(i=0,var=job->variables_to_gather_scatter;var!=NULL;var=var->next,i+=1){
    // Si c'est pas le scatter de l'ordre de la déclaration, on continue
    if (i!=job->parse.iScatter) continue;
    callFlushRealVariable(job,var);
    // Pour l'instant, on ne scatter pas les node_coord
    if (strcmp(var->name,"coord")==0) continue;
    // Si c'est le cas d'une variable en 'in', pas besoin de la scaterer
    if (var->inout==enum_in_variable) continue;
    strcat(scatters,callScatter(var));
  }
  job->parse.iScatter+=1;
  return strdup(scatters);
}


