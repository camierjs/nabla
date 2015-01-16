// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
#include "nabla.h"
#include "nabla.tab.h"

/*****************************************************************************
 * Cuda libraries
 *****************************************************************************/
void cudaHookLibraries(astNode * n, nablaEntity *entity){
  fprintf(entity->src, "\n/*lib %s*/",n->children->token);
}


// ****************************************************************************
// * cudaInlines
// ****************************************************************************
void cudaInlines(nablaMain *nabla){
  fprintf(nabla->entity->src,"#include \"%sEntity.h\"\n", nabla->entity->name);
}

// ****************************************************************************
// * cudaPragmas
// ****************************************************************************
char *nCudaPragmaGccIvdep(void){ return ""; }
char *nCudaPragmaGccAlign(void){ return "__align__(8)"; }


/***************************************************************************** 
 * 
 *****************************************************************************/
extern char real3_h[];
static void cudaHeaderReal3(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,real3_h+NABLA_GPL_HEADER);
}

extern char extra_h[];
static void cudaHeaderExtra(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,extra_h+NABLA_GPL_HEADER);
}

extern char cuMsh_h[];
static void cudaHeaderMesh(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,cuMsh_h+NABLA_GPL_HEADER);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void cudaHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __CUDA_%s_H__\n#define __CUDA_%s_H__",
          nabla->entity->name,nabla->entity->name);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void cudaHeaderIncludes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Includes\n\
// *****************************************************************************\n\
#include <iostream>\n\
#include <cstdio>\n\
#include <cstdlib>\n\
#include <sys/time.h>\n\
#include <stdlib.h>\n\
#include <stdio.h>\n\
#include <assert.h>\n\
#include <string.h>\n\
#include <vector>\n\
#include <math.h>\n\
#include <assert.h>\n\
#include <stdarg.h>\n\
#include <cuda_runtime.h>\n\
cudaError_t cudaCalloc(void **devPtr, size_t size){\n\
   if (cudaSuccess==cudaMalloc(devPtr,size))\n\
      return cudaMemset(*devPtr,0,size);\n\
   return cudaErrorMemoryAllocation;\n\
}\n");
//  #warning CUDA Cartesian here
  fprintf(nabla->entity->hdr,"\n\n\
// *****************************************************************************\n\
// * Cartesian stuffs\n\
// *****************************************************************************\n\
#define MD_DirX 0\n#define MD_DirY 1\n#define MD_DirZ 2\n\
//#warning empty libCartesianInitialize\n\
//__device__ void libCartesianInitialize(void){}\n");
}


/***************************************************************************** 
 * 
 *****************************************************************************/
extern char debug_h[];
static __attribute__((unused)) void cudaHeaderDebug(nablaMain *nabla){
  nablaVariable *var;
  fprintf(nabla->entity->hdr,debug_h+NABLA_GPL_HEADER);
  hprintf(nabla,NULL,"\n\n\
// *****************************************************************************\n \
// * Debug macro functions\n\
// *****************************************************************************");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (strcmp(var->item, "global")==0) continue;
    if (strcmp(var->name, "deltat")==0) continue;
    if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->name, "coord")==0) continue;
    //continue;
    hprintf(nabla,NULL,"\ndbg%sVariable%sDim%s(%s);",
            (var->item[0]=='n')?"Node":"Cell",
            (strcmp(var->type,"real3")==0)?"XYZ":"",
            (var->dim==0)?"0":"1",
            var->name);
    continue;
    hprintf(nabla,NULL,"// dbg%sVariable%sDim%s_%s();",
            (var->item[0]=='n')?"Node":"Cell",
            (strcmp(var->type,"real3")==0)?"XYZ":"",
            (var->dim==0)?"0":"1",
            var->name);
  }
}


/***************************************************************************** 
 * 
 *****************************************************************************/
extern char error_h[];
static void cudaHeaderHandleErrors(nablaMain *nabla){
  fprintf(nabla->entity->hdr,error_h);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void cudaDfsForCalls(struct nablaMainStruct *nabla,
                            nablaJob *fct, astNode *n,
                            const char *namespace,
                            astNode *nParams){
  int nb_called;
  nablaVariable *var;
  // On scan en dfs pour chercher ce que cette fonction va appeler
  dbg("\n\t[cudaDfsForCalls] On scan en DFS pour chercher ce que cette fonction va appeler");
  nb_called=dfsScanJobsCalls(&fct->called_variables,nabla,n);
  dbg("\n\t[cudaDfsForCalls] nb_called = %d", nb_called);
  if (nb_called!=0){
    int numParams=1;
    cudaAddExtraConnectivitiesParameters(nabla,&numParams);
    dbg("\n\t[cudaDfsForCalls] dumping variables found:");
    for(var=fct->called_variables;var!=NULL;var=var->next){
      dbg("\n\t\t[cudaDfsForCalls] variable %s %s %s", var->type, var->item, var->name);
      nprintf(nabla, NULL, ",\n\t\t/*used_called_variable*/%s *%s_%s",var->type, var->item, var->name);
    }
  }
  // Maintenant qu'on a tous les called_variables potentielles, on remplit aussi le hdr
  // On remplit la ligne du hdr
  hprintf(nabla, NULL, "\n%s %s %s%s(",
          nabla->hook->entryPointPrefix(nabla,fct),
          fct->rtntp,
          namespace?"Entity::":"",
          fct->name);
  // On va chercher les paramètres standards pour le hdr
  dumpParameterTypeList(nabla->entity->hdr, nParams);
  hprintf(nabla, NULL, ");");
}


static void cudaHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __CUDA_%s_H__\n",nabla->entity->name);
}

static char* cudaEntryPointPrefix(struct nablaMainStruct *nabla, nablaJob *entry_point){
  if (entry_point->is_an_entry_point) return "__global__";
  return "__device__ inline";
}

static void cudaIteration(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*ITERATION*/", "cuda_iteration()");
}
static void cudaExit(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*EXIT*/", "cudaExit(global_deltat)");
}
static void cudaTime(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*TIME*/", "*global_time");
}
static void cudaFatal(struct nablaMainStruct *nabla){
  nprintf(nabla, NULL, "fatal");
}
static void cudaAddCallNames(struct nablaMainStruct *nabla,nablaJob *fct,astNode *n){
  nablaJob *foundJob;
  char *callName=n->next->children->children->token;
  nprintf(nabla, "/*function_got_call*/", "/*%s*/",callName);
  fct->parse.function_call_name=NULL;
  if ((foundJob=nablaJobFind(fct->entity->jobs,callName))!=NULL){
    if (foundJob->is_a_function!=true){
      nprintf(nabla, "/*isNablaJob*/", NULL);
      fct->parse.function_call_name=strdup(callName);
    }else{
      nprintf(nabla, "/*isNablaFunction*/", NULL);
    }
  }else{
    nprintf(nabla, "/*has not been found*/", NULL);
  }
}
static void cudaAddArguments(struct nablaMainStruct *nabla,nablaJob *fct){
  // En Cuda, par contre il faut les y mettre
  if (fct->parse.function_call_name!=NULL){
    nprintf(nabla, "/*ShouldDumpParamsInCuda*/", "/*cudaAddArguments*/");
    int numParams=1;
    nablaJob *called=nablaJobFind(fct->entity->jobs,fct->parse.function_call_name);
    cudaAddExtraArguments(nabla, called, &numParams);
    nprintf(nabla, "/*ShouldDumpParamsInCuda*/", "/*cudaAddArguments done*/");
    if (called->nblParamsNode != NULL)
      cudaDumpNablaArgumentList(nabla,called->nblParamsNode,&numParams);
  }
}
static void cudaTurnTokenToOption(struct nablaMainStruct *nabla,nablaOption *opt){
  nprintf(nabla, "/*tt2o cuda*/", "%s", opt->name);
}

static  void cudaHookReduction(struct nablaMainStruct *nabla, astNode *n){
  int fakeNumParams=0;
  const astNode *item_node = n->children->next->children;
  const astNode *global_var_node = n->children->next->next;
  const astNode *reduction_operation_node = global_var_node->next;
  astNode *item_var_node = reduction_operation_node->next;
  const astNode *at_single_cst_node = item_var_node->next->next->children->next->children;
  char *global_var_name = global_var_node->token;
  char *item_var_name = item_var_node->token;
  // Préparation du nom du job
  char job_name[NABLA_MAX_FILE_NAME];
  job_name[0]=0;
  strcat(job_name,"cudaReduction_");
  strcat(job_name,global_var_name);
  // Rajout du job de reduction
  nablaJob *redjob = nablaJobNew(nabla->entity);
  redjob->is_an_entry_point=true;
  redjob->is_a_function=false;
  redjob->scope  = strdup("NoGroup");
  redjob->region = strdup("NoRegion");
  redjob->item   = strdup(item_node->token);
  redjob->rtntp  = strdup("void");
  redjob->name   = strdup(job_name);
  redjob->name_utf8 = strdup(job_name);
  redjob->xyz    = strdup("NoXYZ");
  redjob->drctn  = strdup("NoDirection");
  //redjob->nblParamsNode=item_var_node;
  //nablaVariable *item_var=nablaVariableFind(nabla,item_var_name);
  //assert(item_var!=NULL);
  redjob->called_variables=nablaVariableNew(nabla);
  redjob->called_variables->item=strdup("cell");
  redjob->called_variables->name=item_var_name;
  // On annonce que c'est un job de reduction pour lancer le deuxieme etage de reduction dans la boucle
  redjob->reduction = true;
  assert(at_single_cst_node->parent->ruleid==rulenameToId("at_single_constant"));
  dbg("\n\t[cudaHookReduction] @ %s",at_single_cst_node->token);
  sprintf(&redjob->at[0],at_single_cst_node->token);
  redjob->whenx  = 1;
  redjob->whens[0] = atof(at_single_cst_node->token);
  nablaJobAdd(nabla->entity, redjob);
  const double reduction_init = (reduction_operation_node->tokenid==MIN_ASSIGN)?1.0e20:0.0;
  // Génération de code associé à ce job de réduction
  nprintf(nabla, NULL, "\n\
// ******************************************************************************\n\
// * Kernel de reduction de la variable '%s' vers la globale '%s'\n\
// ******************************************************************************\n\
__global__ void %s(", item_var_name, global_var_name, job_name);
  cudaHookAddExtraParameters(nabla,redjob,&fakeNumParams);
nprintf(nabla, NULL,",Real *cell_%s){ // @ %s\n\
\t//const double reduction_init=%e;\n\
\tCUDA_INI_CELL_THREAD(tcid);\n\
\t*global_%s=ReduceMinToDouble(cell_%s[tcid]);\n\
}\n\n", item_var_name,at_single_cst_node->token, reduction_init,global_var_name,item_var_name);
}


/*****************************************************************************
 * nccCuda
 *****************************************************************************/
NABLA_STATUS nccCuda(nablaMain *nabla,
                   astNode *root,
                   const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  nablaBackendSimdHooks nablaCudaSimdHooks={
    nccCudaBits,
    nccCudaGather,
    nccCudaScatter,
    cudaTypedef,
    cudaDefines,
    cudaForwards,
    nccCudaPrevCell,
    nccCudaNextCell,
    nccCudaIncludes
  };
  nablaBackendHooks cudaBackendHooks={
    // Jobs stuff
    cudaHookPrefixEnumerate,
    cudaHookDumpEnumerateXYZ,
    cudaHookDumpEnumerate,
    cudaHookPostfixEnumerate,
    cudaHookItem,
    cudaHookSwitchToken,
    cudaHookTurnTokenToVariable,
    cudaHookSystem,
    cudaHookAddExtraParameters,
    cudaHookDumpNablaParameterList,
    cudaHookTurnBracketsToParentheses,
    cudaHookJobDiffractStatement,
    cudaHookFunctionName,
    cudaHookFunction,
    cudaHookJob,
    cudaHookReduction,
    cudaIteration,
    cudaExit,
    cudaTime,
    cudaFatal,
    cudaAddCallNames,
    cudaAddArguments,
    cudaTurnTokenToOption,
    cudaEntryPointPrefix,
    cudaDfsForCalls,
    NULL, // primary_expression_to_return
    NULL // returnFromArgument
  };
  nabla->simd=&nablaCudaSimdHooks;
  nabla->hook=&cudaBackendHooks;
  
  nablaBackendPragmaHooks cudaPragmaGCCHooks={
    nCudaPragmaGccIvdep,
    nCudaPragmaGccAlign
  };
  nabla->pragma=&cudaPragmaGCCHooks;

  // Rajout de la variable globale 'iteration'
  nablaVariable *iteration = nablaVariableNew(nabla);
  nablaVariableAdd(nabla, iteration);
  iteration->axl_it=false;
  iteration->item=strdup("global");
  iteration->type=strdup("integer");
  iteration->name=strdup("iteration");

  // Rajout de la variable globale 'time'
  /*nablaVariable *time = nablaVariableNew(nabla);
  nablaVariableAdd(nabla, time);
  time->axl_it=false;
  time->item=strdup("global");
  time->type=strdup("real");
  time->name=strdup("time");*/
  
  // Rajout de la variable globale 'min_array'
  // Pour la réduction aux blocs
  nablaVariable *device_shared_reduce_results = nablaVariableNew(nabla);
  nablaVariableAdd(nabla, device_shared_reduce_results);
  device_shared_reduce_results->axl_it=false;
  device_shared_reduce_results->item=strdup("global");
  device_shared_reduce_results->type=strdup("real");
  device_shared_reduce_results->name=strdup("device_shared_reduce_results");

  // Ouverture du fichier source du entity
  sprintf(srcFileName, "%sEntity.cu", nabla->name);
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  // Ouverture du fichier header du entity
  sprintf(hdrFileName, "%sEntity.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  
  // Dump des includes dans le header file, puis des typedefs, defines, debug & errors stuff
  cudaHeaderPrefix(nabla);
  cudaHeaderIncludes(nabla);
  nablaTypedefs(nabla,cudaTypedef);
  cudaHeaderHandleErrors(nabla);
  nablaDefines(nabla,cudaDefines);
  nablaForwards(nabla,cudaForwards);
  cudaDefineEnumerates(nabla);
   
  // Génération du maillage
  cudaMesh(nabla);
  cudaMeshConnectivity(nabla);

  // Rajout de la classe Real3 et des extras
  cudaHeaderDebug(nabla);
  cudaHeaderReal3(nabla);
  cudaHeaderExtra(nabla);
  cudaHeaderMesh(nabla);

  // Dump dans le fichier source
  cudaInlines(nabla);
  nccCudaMainMeshConnectivity(nabla);
  
  // Parse du code préprocessé et lance les hooks associés
  nablaMiddlendParseAndHook(root,nabla);
  nccCudaMainVarInitKernel(nabla);

  // Partie PREFIX
  nccCudaMainPrefix(nabla);
  cudaVariablesPrefix(nabla);
  nccCudaMainMeshPrefix(nabla);
  
  // Partie Pré Init
  nccCudaMainPreInit(nabla);
  nccCudaMainVarInitCall(nabla);
      
  // Dump des entry points dans le main
  nccCudaMain(nabla);

  // Partie Post Init
  nccCudaMainPostInit(nabla);
  
  // Partie POSTFIX
  cudaHeaderPostfix(nabla); 
  nccCudaMainMeshPostfix(nabla);
  cudaVariablesPostfix(nabla);
  nccCudaMainPostfix(nabla);
  return NABLA_OK;
}
