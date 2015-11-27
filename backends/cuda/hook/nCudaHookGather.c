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

// ****************************************************************************
// * Gather for Cells
// ****************************************************************************
static char* cuGatherCells(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration) return "";
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
gather%sk(cell_node[n*NABLA_NB_CELLS+tcid],\n\t\t\t\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":"real3",
           strcmp(var->type,"real")==0?"":"3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item,
           var->name);
  return strdup(gather);
}

// ****************************************************************************
// * Gather for Nodes
// ****************************************************************************
static char* cuGatherNodes(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration) return "";
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
//#warning continue node_cell_corner\n\
//if (node_cell_corner[8*tnid+i]==-1) continue;\n\
gatherFromNode_%sk%s(node_cell[8*tnid+i],\n\
%s\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":"real3",
           strcmp(var->type,"real")==0?"":"3",
           var->dim==0?"":"Array8",
           var->dim==0?"":"\t\t\t\t\t\tnode_cell_corner[8*tnid+i],\n\t\t\t",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item,
           var->name);
  return strdup(gather);
}

// ****************************************************************************
// * Gather for Faces
// ****************************************************************************
static char* cuGatherFaces(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration) return "";
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\
\n\t\t\t%s gathered_%s_%s=%s(0.0);\
\n\t\t\tgatherFromFace_%sk%s(face_node[NABLA_NB_FACES*n+tfid],\
\n\t\t\t\t\t%s\
\n\t\t\t\t\t%s_%s,\
\n\t\t\t\t\t&gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3", var->item, var->name, // ligne #1
           strcmp(var->type,"real")==0?"real":"real3", strcmp(var->type,"real")==0?"":"3", // ligne #3
           var->dim==0?"":"Array8", // fin ligne #3
           var->dim==0?"":"\t\t\t\t\t\tface_node[4*n+tfid],\n\t\t\t", // ligne #4
           var->item, var->name, // ligne #5
           var->item, var->name  // ligne #6
           );
  return strdup(gather);
}

// ****************************************************************************
// * Gather switch
// ****************************************************************************
char* cuHookGather(nablaJob *job,nablaVariable* var, enum_phase phase){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return cuGatherCells(job,var,phase);
  if (itm=='n') return cuGatherNodes(job,var,phase);
  if (itm=='f') return cuGatherFaces(job,var,phase);
  nablaError("Could not distinguish job item inGather!");
  return NULL;
}




// ****************************************************************************
// * Filtrage du GATHER
// * Une passe devrait être faite à priori afin de déterminer les contextes
// * d'utilisation: au sein d'un forall, postfixed ou pas, etc.
// * Et non pas que sur leurs déclarations en in et out
// ****************************************************************************
char* cuHookFilterGather(nablaJob *job){
  int i;
  char gathers[1024];
  nablaVariable *var;
  gathers[0]='\0';
  int nbToGather=0;
  int filteredNbToGather=0;

  // Si l'on a trouvé un 'selection_statement_in_compound_statement'
  // dans le corps du kernel, on débraye les gathers
  // *ou pas*
  if (job->parse.selection_statement_in_compound_statement){
    //nprintf(job->entity->main,
    //"/*selection_statement_in_compound_statement, nothing to do*/",
    //"/*if=>!cuGather*/");
    //return "";
  }
  
  // On récupère le nombre de variables potentielles à gatherer
  for(var=job->variables_to_gather_scatter;var!=NULL;var=var->next)
    nbToGather+=1;
  //nprintf(job->entity->main, NULL, "/* nbToGather=%d*/", nbToGather);
  
  // S'il y en a pas, on a rien d'autre à faire
  if (nbToGather==0) return "";

  // On filtre suivant s'il y a des forall
  for(var=job->variables_to_gather_scatter;var!=NULL;var=var->next){
    //nprintf(job->entity->main, NULL, "\n\t\t// cuGather on %s for variable %s_%s", job->item, var->item, var->name);
    //nprintf(job->entity->main, NULL, "\n\t\t// cuGather enum_enum=%c", job->parse.enum_enum);
    if (job->parse.enum_enum=='\0') continue;
    filteredNbToGather+=1;
  }
  //nprintf(job->entity->main, NULL, "/*filteredNbToGather=%d*/", filteredNbToGather);

  // S'il reste rien après le filtre, on a rien d'autre à faire
  if (filteredNbToGather==0) return "";
  
  strcat(gathers,job->entity->main->call->simd->gather(job,var,enum_phase_declaration));
  
  for(i=0,var=job->variables_to_gather_scatter;var!=NULL;var=var->next,i+=1){
    // Si c'est pas le gather de l'ordre de la déclaration, on continue
    if (i!=job->parse.iGather) continue;
    strcat(gathers,job->entity->main->call->simd->gather(job,var,enum_phase_function_call));
    // On informe la suite que cette variable est en train d'être gatherée
    nablaVariable *real_variable=nMiddleVariableFind(job->entity->main->variables,
                                                     var->name);
    if (real_variable==NULL)
      nablaError("Could not find real variable from gathered variables!");
    real_variable->is_gathered=true;
  }
  job->parse.iGather+=1;
  return strdup(gathers);
}




