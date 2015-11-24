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


/***************************************************************************** 
 * Upcase de la cha�ne donn�e en argument
 *****************************************************************************/
static inline char *itemUPCASE(const char *itm){
  if (itm[0]=='c') return "CELLS";
  if (itm[0]=='n') return "NODES";
  if (itm[0]=='f') return "FACES";
  if (itm[0]=='g') return "GLOBAL";
  if (itm[0]=='p') return "PARTICLES";
  dbg("\n\t[itemUPCASE] itm=%s", itm);
  exit(NABLA_ERROR|fprintf(stderr, "\n[itemUPCASE] Error with given item\n"));
  return NULL;
}


/***************************************************************************** 
 * enums pour les diff�rents dumps � faire: d�claration, malloc et free
 *****************************************************************************/
typedef enum {
  CUDA_VARIABLES_DECLARATION=0,
  CUDA_VARIABLES_MALLOC,
  CUDA_VARIABLES_FREE
} CUDA_VARIABLES_SWITCH;


// Pointeur de fonction vers une qui dump ce que l'on souhaite
typedef NABLA_STATUS (*pFunDump)(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth);


/***************************************************************************** 
 * Dump d'un MALLOC d'une variables dans le fichier source
 *****************************************************************************/
static NABLA_STATUS cudaGenerateSingleVariableMalloc(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth){
  if (var->dim==0){
    fprintf(nabla->entity->src,
            "\n\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&%s_%s%s%s, NABLA_NB_%s*sizeof(%s)));",
            var->item,
            var->name,
            postfix?postfix:"",
            depth?depth:"",
            itemUPCASE(var->item),
            postfix?"real":var->type);
  }else{
    fprintf(nabla->entity->src,
            "\n\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&%s_%s, NABLA_NB_%s*8*sizeof(%s)));",
            var->item,
            var->name, 
            itemUPCASE(var->item),
            postfix?"real":var->type);
  }
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'un FREE d'une variables dans le fichier source
 *****************************************************************************/
static NABLA_STATUS cudaGenerateSingleVariableFree(nablaMain *nabla,
                                                   nablaVariable *var,
                                                   char *postfix,
                                                   char *depth){  
  if (var->dim==0)
    fprintf(nabla->entity->src,
            "\n\tCUDA_HANDLE_ERROR(cudaFree(%s_%s%s%s));",
            var->item,
            var->name,
            postfix?postfix:"",
            depth?depth:"");
  else
    fprintf(nabla->entity->src,
            "\n\tCUDA_HANDLE_ERROR(cudaFree(%s_%s));",
            var->item,
            var->name);
  
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'une variables dans le fichier
 *****************************************************************************/
static NABLA_STATUS cudaGenerateSingleVariable(nablaMain *nabla,
                                               nablaVariable *var,
                                               char *postfix,
                                               char *depth){  
  if (var->dim==0)
    fprintf(nabla->entity->hdr,"\n__builtin_align__(8) %s *%s_%s%s%s; %s host_%s_%s%s%s[NABLA_NB_%s];",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"", depth?depth:"",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"", depth?depth:"",
            itemUPCASE(var->item));
  if (var->dim==1)
    fprintf(nabla->entity->hdr,"\n__builtin_align__(8) %s *%s_%s%s; %s host_%s_%s%s[NABLA_NB_%s][%ld];",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"",
            itemUPCASE(var->item),
            var->size);
  return NABLA_OK;
}


/***************************************************************************** 
 * Retourne quelle fonction selon l'enum donn�
 *****************************************************************************/
pFunDump witch2func(CUDA_VARIABLES_SWITCH witch){
  switch (witch){
  case (CUDA_VARIABLES_DECLARATION): return cudaGenerateSingleVariable;
  case (CUDA_VARIABLES_MALLOC): return cudaGenerateSingleVariableMalloc;
  case (CUDA_VARIABLES_FREE): return cudaGenerateSingleVariableFree;
  default: exit(NABLA_ERROR|fprintf(stderr, "\n[witch2switch] Error with witch\n"));
  }
}



/***************************************************************************** 
 * Dump d'une variables de dimension 1
 *****************************************************************************/
static NABLA_STATUS cudaGenericVariableDim1(nablaMain *nabla, nablaVariable *var, pFunDump fDump){
  //int i;
  //char depth[]="[0]";
  dbg("\n[cudaGenerateVariableDim1] variable %s", var->name);
  //for(i=0;i<NABLA_HARDCODED_VARIABLE_DIM_1_DEPTH;++i,depth[1]+=1) fDump(nabla, var, NULL, depth);
  fDump(nabla, var, NULL, "/*8*/");
  return NABLA_OK;
}

/***************************************************************************** 
 * Dump d'une variables de dimension 0
 *****************************************************************************/
static NABLA_STATUS cudaGenericVariableDim0(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  dbg("\n[cudaGenerateVariableDim0] variable %s", var->name);
  if (strcmp(var->type,"real3")!=0)
    return fDump(nabla, var, NULL, NULL);
  else
    return fDump(nabla, var, NULL, NULL);
  /*return fDump(nabla, var, "_x", NULL)|
           fDump(nabla, var, "_y", NULL)|
           fDump(nabla, var, "_z", NULL);*/
  return NABLA_ERROR;
}

/***************************************************************************** 
 * Dump d'une variables
 *****************************************************************************/
static NABLA_STATUS cudaGenericVariable(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  if (!var->axl_it) return NABLA_OK;
  if (var->item==NULL) return NABLA_ERROR;
  if (var->name==NULL) return NABLA_ERROR;
  if (var->type==NULL) return NABLA_ERROR;
  if (var->dim==0) return cudaGenericVariableDim0(nabla,var,fDump);
  if (var->dim==1) return cudaGenericVariableDim1(nabla,var,fDump);
  dbg("\n[cudaGenericVariable] variable dim error: %d", var->dim);
  exit(NABLA_ERROR|fprintf(stderr, "\n[cudaGenericVariable] Error with given variable\n"));
}


/***************************************************************************** 
 * Dump des options
 *****************************************************************************/
static void cudaOptions(nablaMain *nabla){
  nablaOption *opt;

  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Options\n\
// ********************************************************");
  for(opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->hdr,
            "\n#define %s %s",
            opt->name,
            opt->dflt);
}


/***************************************************************************** 
 * Dump des globals
 *****************************************************************************/
static void cudaGlobals(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Globals, cot� DEVICE\n\
// ********************************************************\n\
__builtin_align__(8) Real *global_time;\n\
__builtin_align__(8) Real *global_deltat;\n\
__builtin_align__(8) int *global_iteration;\n\
__builtin_align__(8) Real *global_device_shared_reduce_results;\n\
\n\
\n\
// ********************************************************\n\
// * Globals, cot� HOST\n\
// ********************************************************\n\
double host_time;\n");
}


/***************************************************************************** 
 * Dump des variables
 *****************************************************************************/
void cudaVariablesPrefix(nablaMain *nabla){
  nablaVariable *var;

  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Variables\n\
// ********************************************************");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (cudaGenericVariable(nabla, var, witch2func(CUDA_VARIABLES_DECLARATION))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[cudaVariables] Error with variable %s\n", var->name));
    if (cudaGenericVariable(nabla, var, witch2func(CUDA_VARIABLES_MALLOC))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[cudaVariables] Error with variable %s\n", var->name));
  }
  cudaOptions(nabla);
  cudaGlobals(nabla);
}


void cudaVariablesPostfix(nablaMain *nabla){
  nablaVariable *var;
  for(var=nabla->variables;var!=NULL;var=var->next)
    if (cudaGenericVariable(nabla, var, witch2func(CUDA_VARIABLES_FREE))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[cudaVariables] Error with variable %s\n", var->name));
}




// *****************************************************************************
// * Ajout des variables d'un job trouv� depuis une fonction @�e
// *****************************************************************************
void cudaAddNablaVariableList(nablaMain *nabla, astNode *n, nablaVariable **variables){
  if (n==NULL) return;
  if (n->tokenid!=0) dbg("\n\t\t\t[cudaAddNablaVariableList] token is '%s'",n->token);

  // Si on tombe sur la '{', on arr�te; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) {
    dbg("\n\t\t\t[cudaAddNablaVariableList] '{', returning");
    return;
  }
  
  if (n->tokenid=='@'){
    return;
    dbg("\n\t\t\t[cudaAddNablaVariableList] '@', returning");
  }
    
  if (n->ruleid==rulenameToId("direct_declarator")){
    dbg("\n\t\t\t[cudaAddNablaVariableList] Found a direct_declarator!");
    dbg("\n\t\t\t[cudaAddNablaVariableList] Now looking for: '%s'",n->children->token);
    nablaVariable *hit=nMiddleVariableFind(nabla->variables, n->children->token);
    dbg("\n\t\t\t[cudaAddNablaVariableList] Got the direct_declarator '%s' on %ss", hit->name, hit->item);
    // Si on ne trouve pas de variable, c'est pas normal
    if (hit == NULL)
      return exit(NABLA_ERROR|fprintf(stderr, "\n\t\t[cudaAddNablaVariableList] Variable error\n"));
    dbg("\n\t\t\t[cudaAddNablaVariableList] Now testing if its allready in our growing variables list");
    nablaVariable *allready_here=nMiddleVariableFind(*variables, hit->name);
    if (allready_here!=NULL){
      dbg("\n\t\t\t[cudaAddNablaVariableList] allready_here!");
    }else{
      // Cr�ation d'une nouvelle called_variable
      nablaVariable *new = nMiddleVariableNew(NULL);
      new->name=strdup(hit->name);
      new->item=strdup(hit->item);
      new->type=strdup(hit->type);
      new->dim=hit->dim;
      new->size=hit->size;
      // Rajout � notre liste
      if (*variables==NULL){
        dbg("\n\t\t\t[cudaAddNablaVariableList] first hit");
        *variables=new;
      }else{
        dbg("\n\t\t\t[cudaAddNablaVariableList] last hit");
        nMiddleVariableLast(*variables)->next=new;
      }
    }
  }
  if (n->children != NULL) cudaAddNablaVariableList(nabla, n->children, variables);
  if (n->next != NULL) cudaAddNablaVariableList(nabla, n->next, variables);
}

