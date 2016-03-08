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


// ****************************************************************************
// * Gather for Cells
// ****************************************************************************
static char* xGatherCells(nablaJob *job,
                          nablaVariable* var){
  const bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  char gather[1024];

  if (var->item[0]=='n')
    snprintf(gather, 1024, "\n\t\t\t\
%s gathered_%s_%s=rgather%sk(cell_node[n*NABLA_NB_CELLS+c],%s_%s%s);\n\t\t\t",
             strcmp(var->type,"real")==0?"real":dim1D?"real":strcmp(var->type,"real3x3")==0?"real3x3":"real3",
             var->item,
             var->name,
             strcmp(var->type,"real")==0?"":dim1D?"":strcmp(var->type,"real3x3")==0?"3x3":"3",
             var->item,
             var->name,
             strcmp(var->type,"real")==0?"":"");

  if (var->item[0]=='f')
     snprintf(gather, 1024, "\n\t\t\t\
%s gathered_%s_%s=rgather%sk(cell_face[f*NABLA_NB_CELLS+c],%s_%s%s);\n\t\t\t",
              strcmp(var->type,"real")==0?"real":strcmp(var->type,"real3x3")==0?"real3x3":"real3",
              var->item,
              var->name,
              strcmp(var->type,"real")==0?"":strcmp(var->type,"real3x3")==0?"3x3":"3",
              var->item,
              var->name,
              strcmp(var->type,"real")==0?"":"");
  
  return strdup(gather);
}


// ****************************************************************************
// * Gather for Nodes
// * En STD, le gather aux nodes est le même qu'aux cells
// ****************************************************************************
static char* xGatherNodes(nablaJob *job,
                          nablaVariable* var){
  bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t\
%s gathered_%s_%s=rGatherAndZeroNegOnes(node_cell[NABLA_NODE_PER_CELL*n+c],%s %s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":dim1D?"real":"real3",
           var->item,
           var->name,
           var->dim==0?"":"node_cell_corner[NABLA_NODE_PER_CELL*n+c],",
           var->item,
           var->name);
  return strdup(gather);
}


// ****************************************************************************
// * Gather for Faces
// ****************************************************************************
static char* xGatherFaces(nablaJob *job,
                          nablaVariable* var){
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t\
%s gathered_%s_%s=rGatherAndZeroNegOnes(face_node[NABLA_NB_FACES*n+f],%s %s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":strcmp(var->type,"real3x3")==0?"real3x3":"real3",
           var->item,
           var->name,
           var->dim==0?"":"node_cell_corner[NABLA_NODE_PER_CELL*n+f],",
           var->item, var->name);
  return strdup(gather);
}


// ****************************************************************************
// * Gather switch
// ****************************************************************************
static char* xGather(nablaJob *job,nablaVariable* var){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return xGatherCells(job,var);
  if (itm=='n') return xGatherNodes(job,var);
  if (itm=='f') return xGatherFaces(job,var);
  nablaError("Could not distinguish job item in xGather for job '%s'!", job->name);
  return NULL;
}


// ****************************************************************************
// * Filtrage du GATHER
// * Une passe devrait être faite à priori afin de déterminer les contextes
// * d'utilisation: au sein d'un forall, postfixed ou pas, etc.
// * Et non pas que sur leurs déclarations en in et out
// ****************************************************************************
char* xFilterGather(astNode *n,nablaJob *job){
  char *gather_src_buffer=NULL;  
  if ((gather_src_buffer=calloc(NABLA_MAX_FILE_NAME,sizeof(char)))==NULL)
    nablaError("[xFilterGather] Could not malloc our gather_src_buffer!");
  
  nprintf(job->entity->main, NULL,"/*filterGather*/");

  for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
    if (!var->is_gathered) continue;
    dbg("\n\t\t\t\t[xFilterGather] var '%s'", var->name);
    nprintf(job->entity->main, NULL, "/* '%s' is gathered",var->name);
    if (!dfsUsedInThisForall(job->entity->main,job,n,var->name)){
      nprintf(job->entity->main, NULL, " but NOT used InThisForall! */");
      continue;
    }
    nprintf(job->entity->main, NULL, " and IS used InThisForall! */");
    dbg("\n\t\t\t\t[xFilterGather] strcat");
    strcat(gather_src_buffer,xGather(job,var));
  }
  dbg("\n\t\t\t\t[xFilterGather] gather_src_buffer='%s'",
      gather_src_buffer?gather_src_buffer:"NULL");
  return gather_src_buffer;
}

