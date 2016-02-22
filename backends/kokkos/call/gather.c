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
// * Gather for Cells
// ****************************************************************************
static char* callGatherCells(nablaJob *job, nablaVariable* var, GATHER_SCATTER_PHASE phase){
  bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  
  // Phase de déclaration
  if (phase==GATHER_SCATTER_DECL)
    return strdup("//int cw,ia;\n\t\t"); //__attribute__((unused))

  char gather[1024];

  if (var->item[0]=='n'){
    // Phase function call
    snprintf(gather, 1024, "\n\t\t\t\
%s gathered_%s_%s=%s(0.0);\n\t\t\t\
gather%sk(cell_node[n*NABLA_NB_CELLS+c],%s_%s%s,&gathered_%s_%s);\n\t\t\t",
             strcmp(var->type,"real")==0?"real":dim1D?"real":"real3",
             var->item, var->name,
             strcmp(var->type,"real")==0?"real":dim1D?"real":"real3",
             strcmp(var->type,"real")==0?"":dim1D?"":"3",
             var->item, var->name,
             strcmp(var->type,"real")==0?"":"",
             var->item, var->name);
  }

  if (var->item[0]=='f'){
    // Phase function call
     snprintf(gather, 1024, "\n\t\t\t\
%s gathered_%s_%s=%s(0.0);\n\t\t\t\
gather%sk(cell_face[f*NABLA_NB_CELLS+c],%s_%s%s,&gathered_%s_%s);\n\t\t\t",
             strcmp(var->type,"real")==0?"real":"real3",
             var->item, var->name,
             strcmp(var->type,"real")==0?"real":"real3",
             strcmp(var->type,"real")==0?"":"3",
             var->item, var->name,
             strcmp(var->type,"real")==0?"":"",
             var->item, var->name);
  }

  
  return strdup(gather);
}


// ****************************************************************************
// * Gather for Nodes
// * En STD, le gather aux nodes est le même qu'aux cells
// ****************************************************************************
static char* callGatherNodes(nablaJob *job,
                                   nablaVariable* var,
                                   GATHER_SCATTER_PHASE phase){
  bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  
  // Phase de déclaration
  if (phase==GATHER_SCATTER_DECL){
    return strdup("");
  }
  
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t\
%s gathered_%s_%s=%s(0.0);\n\t\t\t\
gatherFromNode_%sk%s(node_cell[NABLA_NODE_PER_CELL*n+c],%s %s_%s, &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":dim1D?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":dim1D?"real":"real3",
           strcmp(var->type,"real")==0?"":dim1D?"":"3",
           var->dim==0?"":"Array8",
           var->dim==0?"":"node_cell_corner[NABLA_NODE_PER_CELL*n+c],",
           var->item, var->name,
           var->item, var->name);
  return strdup(gather);
}


// ****************************************************************************
// * Gather for Faces
// ****************************************************************************
static char* callGatherFaces(nablaJob *job,
                                   nablaVariable* var,
                                   GATHER_SCATTER_PHASE phase){
  // Phase de déclaration
  if (phase==GATHER_SCATTER_DECL){
    return strdup("int nw;\n\t\t");
  }
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\
\n\t\t\t%s gathered_%s_%s=%s(0.0);\
\n\t\t\tgatherFromFaces_%sk%s(face_node[NABLA_NB_FACES*n+f],%s\
\n\t\t\t\t\t%s_%s, &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3", var->item, var->name, // ligne #1
           strcmp(var->type,"real")==0?"real":"real3", strcmp(var->type,"real")==0?"":"3", // ligne #3
           var->dim==0?"":"Array8", // fin ligne #3
           var->dim==0?"":"\t\t\t\t\t\tnode_cell_corner[8*nw+f],\n\t\t\t", // ligne #4
           var->item, var->name, // ligne #5
           var->item, var->name  // ligne #6
           );
           return strdup(gather);
}


// ****************************************************************************
// * Gather switch
// ****************************************************************************
char* callGather(nablaJob *job,nablaVariable* var,
                       GATHER_SCATTER_PHASE phase){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return callGatherCells(job,var,phase);
  if (itm=='n') return callGatherNodes(job,var,phase);
  if (itm=='f') return callGatherFaces(job,var,phase);
  nablaError("Could not distinguish job item in callGather for job '%s'!", job->name);
  return NULL;
}


// ****************************************************************************
// * Devrait être un Call!
// * Filtrage du GATHER
// * Une passe devrait être faite à priori afin de déterminer les contextes
// * d'utilisation: au sein d'un forall, postfixed ou pas, etc.
// * Et non pas que sur leurs déclarations en in et out
// ****************************************************************************
char* callFilterGather(astNode *n,nablaJob *job,GATHER_SCATTER_PHASE phase){
  char *gather_src_buffer=NULL;
  
  if ((gather_src_buffer=calloc(NABLA_MAX_FILE_NAME,sizeof(char)))==NULL)
    nablaError("[callFilterGather] Could not malloc our gather_src_buffer!");

  for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
    dbg("\n\t\t\t\t[callFilterGather] var '%s'", var->name);
    if (!var->is_gathered) continue;
    if (!dfsUsedInThisForall(job->entity->main,job,n,var->name)) continue;
    nprintf(job->entity->main, NULL,
            "\n\t\t// gather %s for variable '%s'",
            (phase==GATHER_SCATTER_DECL)?"DECL":"CALL",
            var->name);
    dbg("\n\t\t\t\t[callFilterGather] strcat");
    strcat(gather_src_buffer,
           job->entity->main->call->simd->gather(job,var,phase));
  }
  dbg("\n\t\t\t\t[callFilterGather] gather_src_buffer='%s'",
      gather_src_buffer?gather_src_buffer:"NULL");
  return gather_src_buffer;
}

