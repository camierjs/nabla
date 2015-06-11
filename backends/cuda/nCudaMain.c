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
//#warning Reduction for cells\n\
\tconst int reduced_size=(NABLA_NB_CELLS%%CUDA_NB_THREADS_PER_BLOCK)==0?\
(NABLA_NB_CELLS/CUDA_NB_THREADS_PER_BLOCK):\
(1+NABLA_NB_CELLS/CUDA_NB_THREADS_PER_BLOCK);\n\
\tdouble *host_reduce_results=(double*)malloc(reduced_size*sizeof(double));\n\
\n\
\tconst dim3 dimJobBlock=dim3(BLOCKSIZE,1,1);\n\
\tconst dim3 dimNodeGrid=dim3(PAD_DIV(NABLA_NB_NODES,dimJobBlock.x),1,1);\n\
\tconst dim3 dimCellGrid=dim3(PAD_DIV(NABLA_NB_CELLS,dimJobBlock.x),1,1);\n\
\n\
//#warning dimFuncBlock set for dimJobBlock\n\
\tconst dim3 dimFuncBlock=dim3(BLOCKSIZE,1,1);\n\
\tconst dim3 dimFuncGrid=dim3(PAD_DIV(NABLA_NB_CELLS,dimJobBlock.x),1,1);\n\
\tgpuEnum();\n                                                          \
\tassert((PAD_DIV(NABLA_NB_CELLS,dimJobBlock.x)<65535));  // Max grid dimensions:  (65535, 65535, 65535) \n\
\tassert((PAD_DIV(NABLA_NB_NODES,dimJobBlock.x)<65535));  // Max grid dimensions:  (65535, 65535, 65535) \n\
\tassert((dimJobBlock.x<=1024)); // Max threads per block:  1024 \n\
\tprintf(\"NABLA_NB_NODES=%%d,NABLA_NB_CELLS=%%d\",\n\
\t\tNABLA_NB_NODES,NABLA_NB_CELLS);\n\n\
\t// Allocation coté CUDA des variables globales\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&global_deltat, sizeof(double)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&global_iteration, sizeof(int)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&global_time, sizeof(double)));\n\
\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&global_device_shared_reduce_results, sizeof(double)*reduced_size));\n"


/*****************************************************************************
 * Backend CUDA INIT - Génération du 'main'
 *****************************************************************************/
#define CUDA_MAIN_PREINIT "\n\
\tnabla_ini_node_coords<<<dimNodeGrid,dimJobBlock>>>(node_cell,node_cell_corner,node_cell_corner_idx,%s);\n\
\tnabla_ini_cell_connectivity<<<dimCellGrid,dimJobBlock>>>(cell_node);\n\
\tnabla_set_next_prev<<<dimCellGrid,dimJobBlock>>>(cell_node,cell_prev,cell_next,node_cell,node_cell_corner,node_cell_corner_idx);\n\
\thost_set_corners();\n\
\tCUDA_HANDLE_ERROR(cudaMemcpy(node_cell, &host_node_cell, 8*NABLA_NB_NODES*sizeof(int), cudaMemcpyHostToDevice));\n\
\tCUDA_HANDLE_ERROR(cudaMemcpy(node_cell_corner, &host_node_cell_corner, 8*NABLA_NB_NODES*sizeof(int), cudaMemcpyHostToDevice));\n\
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
#define CUDA_MAIN_GPUENUM "\n\n\n\n\
// *****************************************************************************\n\
// * Backend CUDA - 'gpuEnum'\n\
// *****************************************************************************\n\
void gpuEnum(void){\n\
  int count=0;\n\
  cudaDeviceProp  prop;\n\
  cudaGetDeviceCount(&count);\n\
  CUDA_HANDLE_ERROR(cudaGetDeviceCount(&count));\n\
  //CUDA_HANDLE_ERROR_WITH_SUCCESS(cudaGetDeviceCount(&count));\n\
  for (int i=0; i<count; i++) {\n\
    CUDA_HANDLE_ERROR( cudaGetDeviceProperties(&prop,i));\n\
    printf(\"\\33[7m%%s, sm_%%d%%d, %%d thr/Warp, %%d thr/Block\\33[m\\n\",\
 prop.name, prop.major, prop.minor, prop.warpSize,prop.maxThreadsPerBlock);\n}\n\
  CUDA_HANDLE_ERROR(cudaDeviceReset());\
  CUDA_HANDLE_ERROR(cudaSetDevice(0));\n\
}\n"


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
  //if ((nabla->colors&BACKEND_COLOR_SOA)!=BACKEND_COLOR_SOA)
    fprintf(nabla->entity->src, CUDA_MAIN_PREINIT, "node_coord");
  //else
    //fprintf(nabla->entity->src, CUDA_MAIN_PREINIT, "node_coordx,node_coordy,node_coordz");
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
    //if (strcmp(var->name, "iteration")==0) continue;
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
  number_of_entry_points=nMiddleNumberOfEntryPoints(n);
  entry_points=nMiddleEntryPointsSort(n,number_of_entry_points);
  
  // Et on rescan afin de dumper
  for(i=0;i<number_of_entry_points;++i){
    if (strcmp(entry_points[i].name,"ComputeLoopEnd")==0)continue;
    if (strcmp(entry_points[i].name,"ComputeLoopBegin")==0)continue;
    dbg("%s\n\t[nccCudaMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    // Si l'on passe pour la première fois la frontière du zéro, on écrit le code pour boucler
    if (entry_points[i].whens[0]>=0 && is_into_compute_loop==false){
      is_into_compute_loop=true;
      nprintf(n, NULL,"\
\n\t\t__align__(8) double new_delta_t=0.0;\
\n\t\t__align__(8) double reduced;\
\n//#warning Should get rid of courant_or_hydro\
\n//\t\tint courant_or_hydro=0;        \
\n\t\t//cudaFuncSetCacheConfig(...); \
\n\t\tCUDA_HANDLE_ERROR(cudaDeviceSynchronize());\
\n\t\tgettimeofday(&st, NULL);\
\n\t\t//while (new_delta_t>=0. && iteration<option_max_iterations){//host_time<=OPTION_TIME_END){\
\n\t\twhile(host_time<option_stoptime && iteration<option_max_iterations){\
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
      numParams=nMiddleDumpParameterTypeList(n->entity->src, entry_points[i].stdParamsNode);
    else nprintf(n,NULL,"/*NULL_stdParamsNode*/");
    
    //nprintf(n,NULL,"/*numParams=%d*/",numParams);
    
    // On s'autorise un endroit pour insérer des arguments
    cudaAddExtraArguments(n, &entry_points[i], &numParams);
    
    // Et on dump les in et les out
    if (entry_points[i].nblParamsNode != NULL){
      cudaDumpNablaArgumentList(n,entry_points[i].nblParamsNode,&numParams);
    }else nprintf(n,NULL,"/*NULL_nblParamsNode*/");

    // Si on doit appeler des jobs depuis cette fonction @ée
    if (entry_points[i].called_variables != NULL){
      if (!entry_points[i].reduction)
        cudaAddExtraConnectivitiesArguments(n,&numParams);
      // Et on rajoute les called_variables en paramètre d'appel
      dbg("\n\t[nccCudaMain] Et on rajoute les called_variables en paramètre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }else nprintf(n,NULL,"/*NULL_called_variables*/");

    
    nprintf(n, NULL, ");");
    cudaDumpNablaDebugFunctionFromOutArguments(n,entry_points[i].nblParamsNode,true);

    if (entry_points[i].reduction==true){
      nprintf(n, NULL,"\
\n\t\t\t//CUDA_CHECK_LAST_KERNEL(\"cudaDeviceSynchronize\");\
\n\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(host_reduce_results, global_device_shared_reduce_results,reduced_size*sizeof(double), cudaMemcpyDeviceToHost)); \
\n\t\t\t//CUDA_HANDLE_ERROR(cudaDeviceSynchronize());\
\n\t\t\treduced=host_reduce_results[0];\
\n\t\t\tfor(int i=0;i<reduced_size;i+=1){\
\n\t\t\t\t//printf(\"\\n\\treduced=%%.21e, host_reduce_results[%%d/%%d]=%%.21e\",reduced,i,host_reduce_results[i],reduced_size);\
\n\t\t\t\treduced=min(reduced,host_reduce_results[i]);\
\n\t\t\t}\n\
//\t\t\tif ((courant_or_hydro%%2)==0)\n\
//\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(global_dtt_courant, &reduced, sizeof(double), cudaMemcpyHostToDevice));\n\
//\t\t\telse\n\
//\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(global_dtt_hydro, &reduced, sizeof(double), cudaMemcpyHostToDevice));\n\
\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(global_%s, &reduced, sizeof(double), cudaMemcpyHostToDevice));\n\
//\t\t\tcourant_or_hydro+=1;\n",entry_points[i].reduction_name);
    }
    nprintf(n, NULL, "\n\t\t\t\t//CUDA_CHECK_LAST_KERNEL(\"cudaCheck_%s\");\
\n\t\t\t\t//CUDA_HANDLE_ERROR(cudaDeviceSynchronize());\n",entry_points[i].name);
 }
  nprintf(n, NULL,"\
\n\t\t\t//CUDA_CHECK_LAST_KERNEL(\"cudaDeviceSynchronize\");\
\n\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(&new_delta_t, global_deltat, sizeof(double), cudaMemcpyDeviceToHost));\
\n\t\t\t//printf(\"\\n\\t[#%%d] got new_delta_t=%%.21e, reduced blocs=%%d\", iteration, new_delta_t, reduced_size);\
\n\t\t\thost_time+=new_delta_t;\
\n\t\t\tCUDA_HANDLE_ERROR(cudaMemcpy(global_time, &host_time, sizeof(double), cudaMemcpyHostToDevice));\
\n\t\t\t//if (new_delta_t>=0.) printf(\"\\n\\t[#%%d] \\r\",iteration);\
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


