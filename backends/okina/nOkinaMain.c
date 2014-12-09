/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccOkinaMain.c																  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2013.01.03      															  *
 * Updated  : 2013.01.03																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2013.01.03	camierjs	Creation															  *
 *****************************************************************************/
#include "nabla.h"


// ****************************************************************************
// * Backend OKINA PREFIX - G�n�ration du 'main'
// * look at c++/4.7/bits/ios_base.h for cout options
// ****************************************************************************
#define OKINA_MAIN_PREFIX "\n\n\n\
// ******************************************************************************\n\
// * Main d'Okina\n\
// ******************************************************************************\n\
int main(int argc, char *argv[]){\n\
\tfloat cputime=0.0;\n\
\tstruct timeval st, et;\n\
\t//int iteration=1;\n\
#ifdef __AVX__\n\
\t//avxTest();\n\
#endif\n\
#if defined(__MIC__)||defined(__AVX512F__)\n\
\t//micTestReal();\n\
\t//micTestReal3();\n\
#endif\n\
\tprintf(\"%%d noeuds, %%d mailles\",NABLA_NB_NODES,NABLA_NB_CELLS);\n\
\tnabla_ini_variables();\n\
\tnabla_ini_node_coords();\n\
\t// Initialisation de la pr�cision du cout\n\
\tstd::cout.precision(21);\n\
\t//std::cout.setf(std::ios::floatfield);\n\
\tstd::cout.setf(std::ios::scientific, std::ios::floatfield);\n\
\t// Initialisation du temps et du deltaT\n\
\tglobal_time=0.0;\n\
\tglobal_iteration=1;\n\
\tglobal_deltat[0] = set1(option_dtt_initial);// @ 0;\n\
\t//printf(\"\\n\\33[7;32m[main] time=%%e, Global Iteration is #%%d\\33[m\",global_time,global_iteration);"



/*****************************************************************************
 * Backend OKINA INIT - G�n�ration du 'main'
 *****************************************************************************/
#define OKINA_MAIN_PREINIT "\n\t//OKINA_MAIN_PREINIT"


/*****************************************************************************
 * Backend OKINA POSTFIX - G�n�ration du 'main'
 *****************************************************************************/
#define OKINA_MAIN_POSTINIT "\n\t//OKINA_MAIN_POSTINIT"


/*****************************************************************************
 * Backend OKINA POSTFIX - G�n�ration du 'main'
\n\tprintf(\"\\n\\t\\33[7m[#%%04d]\\33[m time=%%e, delta_t=%%e\", iteration+=1, global_time, *(double*)&global_del *****************************************************************************/
#define OKINA_MAIN_POSTFIX "\n//OKINA_MAIN_POSTFIX\
\n\tglobal_time+=*(double*)&global_deltat[0];\
\n\tglobal_iteration+=1;\
\n\t//printf(\"\\ntime=%%e, dt=%%e\\n\", global_time, *(double*)&global_deltat[0]);\
\n\t}\
\tgettimeofday(&et, NULL);\n\
\tcputime = ((et.tv_sec-st.tv_sec)*1000.+ (et.tv_usec - st.tv_usec)/1000.0);\n\
\tprintf(\"\\n\\t\\33[7m[#%%04d] Elapsed time = %%12.6e(s)\\33[m\\n\", global_iteration-1, cputime/1000.0);\n\
\n}\n"


/*****************************************************************************
 * nccOkinaMainPrefix
 *****************************************************************************/
NABLA_STATUS nccOkinaMainPrefix(nablaMain *nabla){
  dbg("\n[nccOkinaMainPrefix]");
  fprintf(nabla->entity->src, OKINA_MAIN_PREFIX);
  return NABLA_OK;
}


/*****************************************************************************
 * nccOkinaMainPreInit
 *****************************************************************************/
NABLA_STATUS nccOkinaMainPreInit(nablaMain *nabla){
  dbg("\n[nccOkinaMainPreInit]");
  fprintf(nabla->entity->src, OKINA_MAIN_PREINIT);
  return NABLA_OK;
}


/*****************************************************************************
 * nccOkinaMainVarInitKernel
 *****************************************************************************/
NABLA_STATUS nccOkinaMainVarInitKernel(nablaMain *nabla){
  //int i,iVar;
  nablaVariable *var;
  dbg("\n[nccOkinaMainVarInit]");
  nprintf(nabla,NULL,"\n\
// ******************************************************************************\n\
// * Kernel d'initialisation des variables\n\
// ******************************************************************************\n\
void nabla_ini_variables(void){");
  // Variables aux noeuds
  nprintf(nabla,NULL,"\n\tFOR_EACH_NODE_WARP(n){");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    if (strcmp(var->name, "coord")==0) continue;
    nprintf(nabla,NULL,"\n\t\t%s_%s[n]=",var->item,var->name);
    if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
    if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
    if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
  }
  nprintf(nabla,NULL,"\n\t}");  
  // Variables aux mailles real
  nprintf(nabla,NULL,"\n\tFOR_EACH_CELL_WARP(c){");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    if (var->dim==0){
      nprintf(nabla,NULL,"\n\t\t%s_%s[c]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
    }else{
      nprintf(nabla,NULL,"\n\t\tFOR_EACH_CELL_WARP_NODE(n)");
      nprintf(nabla,NULL," %s_%s[n+8*c]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
    }
  }
  nprintf(nabla,NULL,"\n\t}");
  nprintf(nabla,NULL,"\n}");
  return NABLA_OK;
}


/*****************************************************************************
 * nccOkinaMainVarInitKernel
 *****************************************************************************/
NABLA_STATUS nccOkinaMainVarInitCall(nablaMain *nabla){
  nablaVariable *var;
  dbg("\n[nccOkinaMainVarInitCall]");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (strcmp(var->name, "deltat")==0) continue;
    if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->name, "coord")==0) continue;
    nprintf(nabla,NULL,"\n\t//printf(\"\\ndbgsVariable %s\"); dbg%sVariable%sDim%s_%s();",
            var->name,
            (var->item[0]=='n')?"Node":"Cell",
            (strcmp(var->type,"real3")==0)?"XYZ":"",
            (var->dim==0)?"0":"1",
            var->name);
  }
  return NABLA_OK;
}


/*****************************************************************************
 * nccOkinaMainPostInit
 *****************************************************************************/
NABLA_STATUS nccOkinaMainPostInit(nablaMain *nabla){
  dbg("\n[nccOkinaMainPostInit]");
  fprintf(nabla->entity->src, OKINA_MAIN_POSTINIT);
  return NABLA_OK;
}


/*****************************************************************************
 * nccOkinaMain
 *****************************************************************************/
NABLA_STATUS nccOkinaMain(nablaMain *n){
  nablaVariable *var;
  nablaJob *entry_points;
  int i,numParams,number_of_entry_points;
  bool is_into_compute_loop=false;
  double last_when;
  
  dbg("\n[nccOkinaMain]");
  number_of_entry_points=nablaNumberOfEntryPoints(n);
  entry_points=nablaEntryPointsSort(n);
  
  // Et on rescan afin de dumper
  for(i=0,last_when=entry_points[i].whens[0];i<number_of_entry_points;++i){
      dbg("%s\n\t[nccOkinaMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    // Si l'on passe pour la premi�re fois la fronti�re du z�ro, on �crit le code pour boucler
    if (entry_points[i].whens[0]>=0 && is_into_compute_loop==false){
      is_into_compute_loop=true;
      nprintf(n, NULL,"\
\n\tgettimeofday(&st, NULL);\n\
\twhile (global_time<option_stoptime){// && global_iteration!=option_max_iterations){");
    }
    
    // On sync si l'on d�couvre un temps logique diff�rent
    if (last_when!=entry_points[i].whens[0])
      nprintf(n, NULL, "\n%s%s",
              is_into_compute_loop?"\t\t":"\t",
              n->parallel->sync());
    last_when=entry_points[i].whens[0];
    
    // Dump de la tabulation et du nom du point d'entr�e
    nprintf(n, NULL, "\n%s/*@%f*/%s%s(",
            is_into_compute_loop?"\t\t":"\t",
            n->parallel->spawn(), 
            entry_points[i].name,
            entry_points[i].whens[0]);
    // Dump des arguments *ou pas*
    if (entry_points[i].stdParamsNode != NULL){
      //nprintf(n, "/*entry_points[i].stdParamsNode != NULL*/",NULL);
      //numParams=dumpParameterTypeList(n->entity->src, entry_points[i].stdParamsNode);
    }//else nprintf(n,NULL,"/*NULL_stdParamsNode*/");
    
    // On s'autorise un endroit pour ins�rer des arguments
    okinaAddExtraArguments(n, &entry_points[i], &numParams);
    
    // Et on dump les in et les out
    if (entry_points[i].nblParamsNode != NULL){
      okinaDumpNablaArgumentList(n,entry_points[i].nblParamsNode,&numParams);
    }else nprintf(n,NULL,"/*NULL_nblParamsNode*/");

    // Si on doit appeler des jobs depuis cette fonction @�e
    if (entry_points[i].called_variables != NULL){
      okinaAddExtraConnectivitiesArguments(n,&numParams);
      // Et on rajoute les called_variables en param�tre d'appel
      dbg("\n\t[nccOkinaMain] Et on rajoute les called_variables en param�tre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }else nprintf(n,NULL,"/*NULL_called_variables*/");
    nprintf(n, NULL, ");");
    okinaDumpNablaDebugFunctionFromOutArguments(n,entry_points[i].nblParamsNode,true);
    //nprintf(n, NULL, "\n");
  }
  return NABLA_OK;
}


static char *okinaSourceMeshAoS_vs_SoA(nablaMain *nabla){
  if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)==BACKEND_COLOR_OKINA_SOA)
    return "\
node_coordx[iNode]=x;\n\
node_coordy[iNode]=y;\n\
node_coordz[iNode]=z;\n";
  return "node_coord[iNode]=Real3(x,y,z);"; 
}

extern char knMsh_c[];
static void okinaSourceMesh(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->src,knMsh_c,okinaSourceMeshAoS_vs_SoA(nabla));
  //fprintf(nabla->entity->src,knMsh_c);
}

/*****************************************************************************
 * nccOkinaMainPostfix
 *****************************************************************************/
NABLA_STATUS nccOkinaMainPostfix(nablaMain *nabla){
  dbg("\n[nccOkinaMainPostfix] OKINA_MAIN_POSTFIX");
  fprintf(nabla->entity->src, OKINA_MAIN_POSTFIX);
  dbg("\n[nccOkinaMainPostfix] okinaSourceMesh");
  okinaSourceMesh(nabla);
  dbg("\n[nccOkinaMainPostfix] NABLA_OK");
  return NABLA_OK;
}


