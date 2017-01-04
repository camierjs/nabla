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

//#warning no WARP_BIT here

// ****************************************************************************
// * Scatter
// ****************************************************************************
char* xCallScatterCells(nablaJob *job,
                        nablaVariable* var){
  const bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  char scatter[1024];
  snprintf(scatter, 1024,
           "\n\t\tscatter%sk(xs_cell_node[NABLA_NB_CELLS*n+c], &gathered_%s_%s, %s_%s);",
           strcmp(var->type,"real")==0?"":dim1D?"":strcmp(var->type,"real3x3")==0?"3x3":"3",
           var->item, var->name,
           var->item, var->name);
  return sdup(scatter);
}


// ****************************************************************************
// * Scatter for Faces
// ****************************************************************************
static char* xCallScatterFaces(nablaJob *job,
                              nablaVariable* var){
  const bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  char scatter[1024];
  if (var->item[0]=='n')
    snprintf(scatter, 1024,
             "\n\t\tscatter%sk(xs_face_node[NABLA_NB_FACES*(n<<WARP_BIT)+f], &gathered_%s_%s, %s %s_%s);\n\t\t\t",
             strcmp(var->type,"real")==0?"":dim1D?"":strcmp(var->type,"real3x3")==0?"3x3":"3",
             var->item, var->name,
             var->dim==0?"":"node_cell_corner[NABLA_NODE_PER_CELL*n+f],",
             var->item, var->name);
  if (var->item[0]=='c')
    snprintf(scatter, 1024,
             "\n\t\tscatter%sk(xs_face_cell[NABLA_NB_FACES*(c<<WARP_BIT)+f], &gathered_%s_%s, %s %s_%s);\n\t\t\t",
             strcmp(var->type,"real")==0?"":dim1D?"":strcmp(var->type,"real3x3")==0?"3x3":"3",
             var->item, var->name,
             var->dim==0?"":"/*xCallScatterFaces var->dim==1*/,",
             var->item, var->name);
  
  return sdup(scatter);
}


// ****************************************************************************
// * Gather switch
// ****************************************************************************
char* xCallScatter(nablaJob *job,
                   nablaVariable* var){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return xCallScatterCells(job,var);
  //if (itm=='n') return xCallScatterNodes(job,var);
  if (itm=='f') return xCallScatterFaces(job,var);
  nablaError("Could not distinguish job item in xCallScatter for job '%s'!", job->name);
  return NULL;
}



// ****************************************************************************
// * Filtrage du SCATTER
// ****************************************************************************
char* xCallFilterScatter(astNode *n,nablaJob *job){
  char *scatters=NULL;
    
  if ((scatters=calloc(NABLA_MAX_FILE_NAME,sizeof(char)))==NULL)
    nablaError("[xCallFilterScatter] Could not calloc our scatters!");

  // On récupère le nombre de variables potentielles à scatterer
  for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
    dbg("\n\t\t\t\t[xCallFilterScatter] var '%s'", var->name);
    if (!var->is_gathered) continue;
    //nprintf(job->entity->main, NULL, "\n\t/*gathered %s?*/",var->name);
    if (!var->out) continue;
    //nprintf(job->entity->main, NULL, "/*out: %s*/",var->name);
    strcat(scatters,xCallScatter(job,var));
  }
  char *rtn=sdup(scatters);
  free(scatters);
  return rtn;
}
