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


/*****************************************************************************
 * Backend CUDA PREFIX - Génération du 'main'
 *****************************************************************************/
#define CUDA_MAIN_PREFIX "\n\n\n\n\
/*****************************************************************************\n\
 * Backend CUDA - 'main'\n\
 *****************************************************************************/\n\
int main(void){\n\
\t__align__(8)int iteration=1;\n\
\tfloat gputime=0.0;\n\
\tstruct timeval st, et;\n\
\n\
\tconst dim3 dimJobBlock=dim3(BLOCKSIZE,1,1);\n\
\tconst dim3 dimNodeGrid=dim3(PAD_DIV(NABLA_NB_NODES,dimJobBlock.x),1,1);\n\
\tconst dim3 dimCellGrid=dim3(PAD_DIV(NABLA_NB_CELLS,dimJobBlock.x),1,1);\n\
\n\
//#warning dimFuncBlock set for dimJobBlock\n\
\tconst dim3 dimFuncBlock=dim3(BLOCKSIZE,1,1);\n\
\tconst dim3 dimFuncGrid=dim3(PAD_DIV(NABLA_NB_CELLS,dimJobBlock.x),1,1);\n\
\tgpuEnum();\n\
\tassert((PAD_DIV(NABLA_NB_CELLS,dimJobBlock.x)<65535));  // Max grid dimensions:  (65535, 65535, 65535) \n\
\tassert((PAD_DIV(NABLA_NB_NODES,dimJobBlock.x)<65535));  // Max grid dimensions:  (65535, 65535, 65535) \n\
\tassert((dimJobBlock.x<=1024)); // Max threads per block:  1024 \n\
\tprintf(\"NABLA_NB_NODES=%%d,NABLA_NB_CELLS=%%d\",\n\
\t\tNABLA_NB_NODES,NABLA_NB_CELLS);\n\n\
\t// Allocation coté CUDA des variables globales\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&global_deltat, sizeof(double)));\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&global_iteration, sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&global_time, sizeof(double)));\n\
\tCUDA_HANDLE_ERROR(cudaMalloc((void**)&global_min_array, sizeof(double)*CUDA_NB_THREADS_PER_BLOCK));\n"


/*****************************************************************************
 * Backend CUDA INIT - Génération du 'main'
 *****************************************************************************/
#define CUDA_MAIN_PREINIT "\n\
\tnabla_ini_node_coords<<<dimNodeGrid,dimJobBlock>>>(node_cell,node_cell_corner,%s);\n\
\tnabla_ini_cell_connectivity<<<dimCellGrid,dimJobBlock>>>(cell_node);\n\
\tnabla_rescan_connectivity<<<dimCellGrid,dimJobBlock>>>(cell_node,cell_prev,cell_next,node_cell,node_cell_corner);\n\
\t//dbgCoords();\n\
\t//Initialisation du temps et du deltaT\n\
\tprintf(\"\\nInitialisation du temps et du deltaT\");\n\
\thost_time=0.0;\n\n\
\tCUDA_HANDLE_ERROR(cudaMemcpy(global_time, &host_time, sizeof(double), cudaMemcpyHostToDevice));\n\
\t{// Initialisation et boucle de calcul"


/*****************************************************************************
 * Backend CUDA POSTFIX - Génération du 'main'
 *****************************************************************************/
#define CUDA_MAIN_POSTINIT "\n\t}\n\n\
\tgettimeofday(&et, NULL);\n\
\tgputime = ((et.tv_sec-st.tv_sec)*1000.+ (et.tv_usec - st.tv_usec)/1000.0);\n\
\tprintf(\"\\ngpuTime=%%.2fs\\n\", gputime/1000.0);\n\
"

/*****************************************************************************
 * Backend CUDA POSTFIX - Génération du 'main'
 *****************************************************************************/
#define CUDA_MAIN_POSTFIX "\n\
\tCUDA_HANDLE_ERROR(cudaFree(global_deltat));\n\
\tCUDA_HANDLE_ERROR(cudaFree(global_time));\n\
\tCUDA_HANDLE_ERROR(cudaFree(global_iteration));\n\
\treturn 0;\n\
}\n"


/*****************************************************************************
 * Backend CUDA GPU ENUM - Génération de 'gpuEnum'
 *****************************************************************************/
#define CUDA_MAIN_GPUENUM "\n\n\n\n/*****************************************************************************\n\
 * Backend CUDA - 'gpuEnum'\n\
 *****************************************************************************/\n\
void gpuEnum(void){\n\
  int count=0;\n\
  cudaDeviceProp  prop;\n\
  cudaGetDeviceCount(&count);\n\
  CUDA_HANDLE_ERROR(cudaGetDeviceCount(&count));\n\
  //CUDA_HANDLE_ERROR_WITH_SUCCESS(cudaGetDeviceCount(&count));\n\
  for (int i=0; i<count; i++) {\n\
    CUDA_HANDLE_ERROR( cudaGetDeviceProperties(&prop,i));\n\
    printf( \"--- General Information for device #%%d ---\\n\", i );\n\
    printf( \"\tName:  %%s\\n\", prop.name );\n\
    printf( \"\tCompute capability:  %%d.%%d\\n\", prop.major, prop.minor );\n\
    printf( \"\tClock rate:  %%d\\n\", prop.clockRate );\n\
    printf( \"\tDevice copy overlap:  \" );\n\
    if (prop.deviceOverlap) printf( \"Enabled\\n\" );\n\
    else printf( \"Disabled\\n\");\n\
    printf( \"\tKernel execution timeout :  \" );\n\
    if (prop.kernelExecTimeoutEnabled) printf( \"Enabled\\n\" );\n\
    else printf( \"Disabled\\n\" );\n\
    printf(\"--- Memory Information for device #%%d ---\\n\", i );\n\
    printf(\"\tTotal global mem:  %%ld\\n\", prop.totalGlobalMem );\n\
    printf(\"\tTotal constant Mem:  %%ld\\n\", prop.totalConstMem );\n\
    printf(\"\tMax mem pitch:  %%ld\\n\", prop.memPitch );\n\
    printf(\"\tTexture Alignment:  %%ld\\n\", prop.textureAlignment );\n\
    printf(\"--- Multi Processing Information for device #%%d ---\\n\", i );\n\
    printf(\"\tMultiprocessor count:  %%d\\n\", prop.multiProcessorCount );\n\
    printf(\"\tShared mem per mp:  %%ld\\n\", prop.sharedMemPerBlock );\n\
    printf(\"\tRegisters per mp:  %%d\\n\", prop.regsPerBlock );\n\
    printf(\"\tThreads in warp:  %%d\\n\", prop.warpSize );\n\
    printf(\"\tMax threads per block:  %%d\\n\",prop.maxThreadsPerBlock );\n\
    printf(\"\tMax thread dimensions:  (%%d, %%d, %%d)\\n\",\n\
            prop.maxThreadsDim[0], prop.maxThreadsDim[1],\n\
            prop.maxThreadsDim[2] );\n\
    printf( \"\tMax grid dimensions:  (%%d, %%d, %%d)\\n\",\n\
            prop.maxGridSize[0], prop.maxGridSize[1],\n\
            prop.maxGridSize[2] );\n\
    printf(\"\\n\");\n\
  }\n\
  CUDA_HANDLE_ERROR(cudaDeviceReset());\
  CUDA_HANDLE_ERROR(cudaSetDevice(0));\n\
}\n\
"


/*****************************************************************************
 * nccCudaMainPrefix
 *****************************************************************************/
NABLA_STATUS nccCudaMainPrefix(nablaMain *nabla){
  dbg("\n[nccCudaMainPrefix]");
  fprintf(nabla->entity->src, CUDA_MAIN_PREFIX);
  return NABLA_OK;
}


/*****************************************************************************
 * nccCudaMainPreInit
 *****************************************************************************/
NABLA_STATUS nccCudaMainPreInit(nablaMain *nabla){
  dbg("\n[nccCudaMainPreInit]");
  if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA)
    fprintf(nabla->entity->src, CUDA_MAIN_PREINIT, "node_coord");
  else
    fprintf(nabla->entity->src, CUDA_MAIN_PREINIT, "node_coordx,node_coordy,node_coordz");
  return NABLA_OK;
}


// *****************************************************************************
// * nccCudaMainVarInitKernel
// * We now calloc things, so this init is now anymore usefull
// * #warning Formal parameter space overflowed (256 bytes max)
// *****************************************************************************/
NABLA_STATUS nccCudaMainVarInitKernel(nablaMain *nabla){ return NABLA_OK; }
/*
NABLA_STATUS nccCudaMainVarInitKernel(nablaMain *nabla){
  int i,iVar;
  nablaVariable *var;
  dbg("\n[nccCudaMainVarInit]");
  nprintf(nabla,NULL,"\n\n\
// ******************************************************************************\n\
// * Kernel d'initialisation des variables aux NOEUDS\n\
// ******************************************************************************\n\
__global__ void nabla_ini_node_variables(");
  // Variables aux noeuds
  for(iVar=0,var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    if (strcmp(var->name, "coord")==0) continue;
    if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"%sReal *",(iVar!=0)?", ":"");
    if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"%sReal3 *",(iVar!=0)?", ":"");
    if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"%sinteger *",(iVar!=0)?", ":"");
    nprintf(nabla,NULL,"%s_%s",var->item,var->name);
    iVar+=1;
  }
  nprintf(nabla,NULL,"){\n\tCUDA_INI_NODE_THREAD(tnid);");
  // Variables aux noeuds
  nprintf(nabla,NULL,"\n\t{");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    if (strcmp(var->name, "coord")==0) continue;
    nprintf(nabla,NULL,"\n\t\t%s_%s[tnid]=",var->item,var->name);
    if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
    if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"Real3(0.0);");
    if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0.0;");
  }
  nprintf(nabla,NULL,"\n\t}");  
  nprintf(nabla,NULL,"\n}");

 
  nprintf(nabla,NULL,"\n\n\
// ******************************************************************************\n\
// * Kernel d'initialisation des variables aux MAILLES\n\
// ******************************************************************************\n\
  __global__ void nabla_ini_cell_variables(");
  // Variables aux mailles
//  #warning Formal parameter space overflowed (256 bytes max)
  for(iVar=0,var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    if (var->dim==0){
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"%sReal *",(iVar!=0)?", ":"");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"%sReal3 *",(iVar!=0)?", ":"");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"%sinteger *",(iVar!=0)?", ":"");
      nprintf(nabla,NULL,"%s_%s",var->item,var->name);
    }else{
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"%sReal *",(iVar!=0)?", ":"");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"%sReal3 *",(iVar!=0)?", ":"");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"%sinteger *",(iVar!=0)?", ":"");
      nprintf(nabla,NULL,"%s_%s",var->item,var->name);
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,NULL);
    }
    iVar+=1;
  }
  nprintf(nabla,NULL,"){\n\tCUDA_INI_CELL_THREAD(tcid);");
  // Variables aux mailles real
  nprintf(nabla,NULL,"\n{");
  for(iVar=0,var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    if (var->dim==0){
      nprintf(nabla,NULL,"\n\t\t%s_%s[tcid]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"Real3(0.0);");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0.0;");
    }else{
      nprintf(nabla,NULL,"\n\t\t");
      for(i=0;i<8;i++)
        nprintf(nabla,NULL,"%s_%s[tcid+%d]=",var->item,var->name,i);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"Real3(0.0);");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0.0;");
    }
    iVar+=1;
    if (iVar==16)break;
  }
  nprintf(nabla,NULL,"\n\t}");
  nprintf(nabla,NULL,"\n}");
  return NABLA_OK;
}*/


/*****************************************************************************
 * nccCudaMainVarInitCall
 *****************************************************************************/
NABLA_STATUS nccCudaMainVarInitCall(nablaMain *nabla){
  nablaVariable *var;
  dbg("\n[nccCudaMainVarInitCall]");

  // We now use memset 0 to calloc things
  /*{
    int iVar;
    // Variables aux noeuds
    nprintf(nabla,NULL,"\n\t\tnabla_ini_node_variables<<<dimNodeGrid,dimJobBlock>>>(");
    for(iVar=0,var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='n') continue;
      if (strcmp(var->name, "coord")==0) continue;
      nprintf(nabla,NULL,"%s%s_%s",(iVar!=0)?", ":"",var->item,var->name);
      iVar+=1;
    }
    nprintf(nabla,NULL,");");
  
    // Variables aux mailles
    nprintf(nabla,NULL,"\n\t\tnabla_ini_cell_variables<<<dimCellGrid,dimJobBlock>>>(");
    for(iVar=0,var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='c') continue;
      nprintf(nabla,NULL,"%s%s_%s",(iVar!=0)?", ":"",var->item,var->name);
      iVar+=1;
      //if (iVar==16) break;
    }
    nprintf(nabla,NULL,");");
  }*/
  
  //nprintf(nabla,NULL,"\n#warning HWed nccCudaMainVarInitCall\n\t\t//nccCudaMainVarInitCall:");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (strcmp(var->name, "deltat")==0) continue;
    if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->name, "coord")==0) continue;
    // Si les variables sont globales, on ne les debug pas
    if (var->item[0]=='g') continue;
    //if (strcmp(var->name, "min_array")==0) continue;
    //if (strcmp(var->name, "iteration")==0) continue;
    //if (strcmp(var->name, "dtt_courant")==0) continue;
    //if (strcmp(var->name, "dtt_hydro")==0) continue;
    //if (strcmp(var->name, "elemBC")==0) continue;
    //#warning continue dbgsVariable
    //continue;
    nprintf(nabla,NULL,"\n\t\t//printf(\"\\ndbgsVariable %s\"); dbg%sVariable%sDim%s_%s();",
            var->name,
            (var->item[0]=='n')?"Node":"Cell",
            (strcmp(var->type,"real3")==0)?"XYZ":"",
            (var->dim==0)?"0":"1",
            var->name);
  }
  
  return NABLA_OK;
}


/*****************************************************************************
 * nccCudaMainPostInit
 *****************************************************************************/
NABLA_STATUS nccCudaMainPostInit(nablaMain *nabla){
  dbg("\n[nccCudaMainPostInit]");
  fprintf(nabla->entity->src, CUDA_MAIN_POSTINIT);
  return NABLA_OK;
}


/*****************************************************************************
 * nccCudaMain
 *****************************************************************************/
NABLA_STATUS nccCudaMain(nablaMain *n){
  nablaVariable *var;
  nablaJob *entry_points;
  int i,numParams,number_of_entry_points;
  bool is_into_compute_loop=false;

  dbg("\n[nccCudaMain]");
  number_of_entry_points=nablaNumberOfEntryPoints(n);
  entry_points=nablaEntryPointsSort(n);
  
  // Et on rescan afin de dumper
  for(i=0;i<number_of_entry_points;++i){
    dbg("%s\n\t[nccCudaMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    // Si l'on passe pour la première fois la frontière du zéro, on écrit le code pour boucler
    if (entry_points[i].whens[0]>=0 && is_into_compute_loop==false){
      is_into_compute_loop=true;
      nprintf(n, NULL,"\
\n\t\t__align__(8) double new_delta_t=0.0;\
\n\t\t//cudaFuncSetCacheConfig(integrateStressForElems,cudaFuncCachePreferL1);\
\n\t\tgettimeofday(&st, NULL);\
\n\t\tCUDA_HANDLE_ERROR(cudaDeviceSynchronize());\
\n\t\twhile (new_delta_t>=0. && iteration<option_max_iterations){//host_time<=OPTION_TIME_END){\
\n\t\t\t//printf(\"\\nITERATION %%d\", iteration);\
\n\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(global_iteration, &iteration, sizeof(int), cudaMemcpyHostToDevice));");
    }
    nprintf(n, NULL, "\n%s%s<<<dim%sGrid,dim%sBlock>>>( // @ %f",
            is_into_compute_loop?"\t\t\t":"\t\t",
            entry_points[i].name,
            entry_points[i].item[0]=='c'?"Cell":entry_points[i].item[0]=='n'?"Node":"Func",
            entry_points[i].item[0]=='c'?"Job":entry_points[i].item[0]=='n'?"Job":"Func",
            entry_points[i].whens[0]);
    
    if (entry_points[i].stdParamsNode != NULL)
      numParams=dumpParameterTypeList(n->entity->src, entry_points[i].stdParamsNode);
    else nprintf(n,NULL,"/*NULL_stdParamsNode*/");
    
    nprintf(n,NULL,"/*numParams=%d*/",numParams);
    
    // On s'autorise un endroit pour insérer des arguments
    cudaAddExtraArguments(n, &entry_points[i], &numParams);
    
    // Et on dump les in et les out
    if (entry_points[i].nblParamsNode != NULL){
      cudaDumpNablaArgumentList(n,entry_points[i].nblParamsNode,&numParams);
    }else nprintf(n,NULL,"/*NULL_nblParamsNode*/");

    // Si on doit appeler des jobs depuis cette fonction @ée
    if (entry_points[i].called_variables != NULL){
      cudaAddExtraConnectivitiesArguments(n,&numParams);
      // Et on rajoute les called_variables en paramètre d'appel
      dbg("\n\t[nccCudaMain] Et on rajoute les called_variables en paramètre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }else nprintf(n,NULL,"/*NULL_called_variables*/");

    
    nprintf(n, NULL, ");");
    cudaDumpNablaDebugFunctionFromOutArguments(n,entry_points[i].nblParamsNode,true);
    
    nprintf(n, NULL, "\n\t\t\t\tCUDA_CHECK_LAST_KERNEL(\"cudaCheck_%s\");\
\n\t\t\t\tCUDA_HANDLE_ERROR(cudaDeviceSynchronize());\n",entry_points[i].name);
 }
  nprintf(n, NULL,"\
\n\t\t\tCUDA_CHECK_LAST_KERNEL(\"cudaDeviceSynchronize\");\
\n\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(&new_delta_t, global_deltat, sizeof(double), cudaMemcpyDeviceToHost));\
\n\t\t\t//new_delta_t=option_dtt_initial;\
\n\t\t\tprintf(\"\\n\\t[#%%d] got new_delta_t=%%.21e\", iteration, new_delta_t); \
\n\t\t\t//CUDA_HANDLE_ERROR(cudaMemcpy(&host_min_array, global_min_array, CUDA_NB_THREADS_PER_BLOCK*sizeof(double), cudaMemcpyDeviceToHost));\
\n\t\t\t//for(int i=0;i<CUDA_NB_THREADS_PER_BLOCK;i+=1){printf(\" host_min_array[%%d]=%%f\",i,host_min_array[i]);}\
\n\t\t\thost_time+=new_delta_t;\
\n\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(global_time, &host_time, sizeof(double), cudaMemcpyHostToDevice));\
\n\t\t\tif (new_delta_t>=0.) printf(\"\\n\\t[#%%d] time=%%.21e, delta_t=%%.21e\\r\", iteration, host_time, new_delta_t);\
\n\t\t\titeration+=1;\
\n\t\t}");
  return NABLA_OK;
}


/*****************************************************************************
 * nccCudaMainPostfix
 *****************************************************************************/
NABLA_STATUS nccCudaMainPostfix(nablaMain *nabla){
  dbg("\n[nccCudaMainPostfix]");
  fprintf(nabla->entity->src, CUDA_MAIN_POSTFIX);
  fprintf(nabla->entity->src, CUDA_MAIN_GPUENUM);
  return NABLA_OK;
}


