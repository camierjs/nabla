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
// * Backend PREFIX - Génération du 'main'
// * look at c++/4.7/bits/ios_base.h for cout options
// ****************************************************************************
#define BACKEND_MAIN_PREFIX "\n\n\n\
static void nabla_ini_connectivity(void);\n\
static void nabla_free_variables(void);\n\
\n\
// ******************************************************************************\n \
// * Main\n\
// ******************************************************************************\n\
int main(int argc, char *argv[]){\n\
\tfloat cputime=0.0;\n\
\tstruct timeval st, et;\n\
\t//int iteration=1;\n\
\tprintf(\"%%d noeuds, %%d mailles & %%d faces\",NABLA_NB_NODES,NABLA_NB_CELLS,NABLA_NB_FACES);\n\
\t//nabla_malloc_variables();\n\
\t//nabla_ini_variables();\n\
\t//assert(argc>=2);\n\
\tif (argc==1)\n\
\t\tNABLA_NB_PARTICLES=1000;\n\
\telse\n\
\t\tNABLA_NB_PARTICLES=atoi(argv[1]);\n\
\tnabla_ini_connectivity();\n\
\t// Initialisation des swirls\n\
\thlt_level=0;\n\
\thlt_exit=(bool*)calloc(64,sizeof(bool));\n\
\t// Initialisation de la précision du cout\n\
\tstd::cout.precision(14);//21, 14 pour Arcane\n\
\t//std::cout.setf(std::ios::floatfield);\n\
\tstd::cout.setf(std::ios::scientific, std::ios::floatfield);\n\
\t// Initialisation du temps et du deltaT\n\
\tglobal_time[0]=set1(option_dtt_initial);// Arcane fait comme cela! 0.0;\n\
\tglobal_iteration[0]=1;\n\
\tglobal_deltat[0] = set1(option_dtt_initial);// @ 0;\n\
\t//printf(\"\\n\\33[7;32m[main] time=%%e, Global Iteration is #%%d\\33[m\",global_time[0],global_iteration[0]);\n"
NABLA_STATUS hookMainPrefix(nablaMain *nabla){
  dbg("\n[hookMainPrefix]");
  fprintf(nabla->entity->src, BACKEND_MAIN_PREFIX);
  return NABLA_OK;
}


// ****************************************************************************
// * Backend PREINIT - Génération du 'main'
// ****************************************************************************
#define BACKEND_MAIN_PREINIT "\n\n\t//BACKEND_MAIN_PREINIT\n\
\tnabla_ini_variables();\n"
NABLA_STATUS hookMainPreInit(nablaMain *nabla){
  dbg("\n[hookMainPreInit]");
  fprintf(nabla->entity->src, BACKEND_MAIN_PREINIT);
  return NABLA_OK;
}


// ****************************************************************************
// * hookMainVarInitKernel
// ****************************************************************************
NABLA_STATUS hookMainVarInitKernel(nablaMain *nabla){
  //int i,iVar;
  nablaVariable *var;
  dbg("\n[hookMainVarInit]");
  nprintf(nabla,NULL,"\n\
// ******************************************************************************\n\
// * Kernel d'initialisation des variables\n\
// ******************************************************************************\n\
void nabla_ini_variables(void){");
  // Variables Particulaires
  nprintf(nabla,NULL,"\n\tFOR_EACH_PARTICLE(p){");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='p') continue;
    nprintf(nabla,NULL,"\n\t\t%s_%s[p]=",var->item,var->name);
    if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
    if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
    if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
    if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
  }
  nprintf(nabla,NULL,"\n\t}");  
  // Variables aux noeuds
  nprintf(nabla,NULL,"\n\tFOR_EACH_NODE(n){");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    if (strcmp(var->name, "coord")==0) continue;
    nprintf(nabla,NULL,"\n\t\t%s_%s[n]=",var->item,var->name);
    if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
    if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
    if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
    if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
  }
  nprintf(nabla,NULL,"\n\t}");  
  // Variables aux mailles real
  nprintf(nabla,NULL,"\n\tFOR_EACH_CELL(c){");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    if (var->dim==0){
      nprintf(nabla,NULL,"\n\t\t%s_%s[c]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
      if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
      if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
      if (strcmp(var->type, "real3x3")==0) nprintf(nabla,NULL,"real3x3();");
    }else{
      nprintf(nabla,NULL,"\n\t\tFOR_EACH_CELL_NODE(n)");
      nprintf(nabla,NULL," %s_%s[n+NABLA_NODE_PER_CELL*c]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
      if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
    }
  }
  nprintf(nabla,NULL,"\n\t}");
  nprintf(nabla,NULL,"\n}");
  return NABLA_OK;
}


// ****************************************************************************
// * hookMainVarInitKernel
// ****************************************************************************
NABLA_STATUS hookMainVarInitCall(nablaMain *nabla){
  nablaVariable *var;
  dbg("\n[hookMainVarInitCall]");
  nprintf(nabla,NULL,"\n\
\t// ***************************************************************************\n\
\t// * hookMainVarInitCall\n\
\t// ***************************************************************************\n");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (strcmp(var->name, "deltat")==0) continue;
    if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->name, "coord")==0) continue;
    continue;
    nprintf(nabla,NULL,"\n\t//printf(\"\\ndbgsVariable %s\"); dbg%sVariable%sDim%s_%s();",
            var->name,
            (var->item[0]=='n')?"Node":"Cell",
            (strcmp(var->type,"real3")==0)?"XYZ":"",
            (var->dim==0)?"0":"1",
            var->name);
  }
  return NABLA_OK;
}


// ****************************************************************************
// * hookMain
// ****************************************************************************
NABLA_STATUS hookMainHLT(nablaMain *n){
  nablaVariable *var;
  nablaJob *entry_points;
  int i,numParams,number_of_entry_points;
  bool is_into_compute_loop=false;
  double last_when;
  n->HLT_depth=0;
  
  dbg("\n[hookMain]");
  number_of_entry_points=nMiddleNumberOfEntryPoints(n);
  entry_points=nMiddleEntryPointsSort(n,number_of_entry_points);
  
  // Et on rescan afin de dumper, on rajoute les +2 ComputeLoop[End|Begin]
  for(i=0,last_when=entry_points[i].whens[0];i<number_of_entry_points+2;++i){
    if (strncmp(entry_points[i].name,"hltDive",7)==0) continue;
    if (strcmp(entry_points[i].name,"ComputeLoopEnd")==0) continue;
    if (strcmp(entry_points[i].name,"ComputeLoopBegin")==0) continue;
    
    dbg("%s\n\t[hookMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    
    // Si l'on passe pour la première fois la frontière du zéro, on écrit le code pour boucler
    if (entry_points[i].whens[0]>=0 && is_into_compute_loop==false){
      is_into_compute_loop=true;
      nprintf(n, NULL,"\n\tgettimeofday(&st, NULL);\n\
\twhile (global_time[0]<option_stoptime){\
\t// && global_iteration!=option_max_iterations){");
    }
    
    if (entry_points[i].when_depth==(n->HLT_depth+1)){
      nprintf(n, NULL, "\n\n\t// DIVING in HLT! (depth=%d)",n->HLT_depth);
      nprintf(n, NULL, "\n\thlt_exit[hlt_level=%d]=true;\n\tdo{",n->HLT_depth);
      n->HLT_depth=entry_points[i].when_depth;
    }
    if (entry_points[i].when_depth==(n->HLT_depth-1)){
      nprintf(n, NULL, "\n\t// Poping from HLT!");
//#warning HWed 'redo_with_a_smaller_time_step'
      nprintf(n, NULL, "\n\t}while(hlt_exit[%d]);hlt_level-=1;\n",entry_points[i].when_depth);
      n->HLT_depth=entry_points[i].when_depth;
    }
    
    // \n ou if d'un IF after '@'
    if (entry_points[i].ifAfterAt!=NULL){
      dbg("\n\t[hookMain] dumpIfAfterAt!");
      nprintf(n, NULL, "\n\t\tif (");
      nMiddleDumpIfAfterAt(entry_points[i].ifAfterAt, n,false);
      nprintf(n, NULL, ") ");
    }else nprintf(n, NULL, "\n");
    
    // On provoque un parallel->sync
    // si l'on découvre un temps logique différent
    if (last_when!=entry_points[i].whens[0])
      nprintf(n, NULL, "%s%s",
              is_into_compute_loop?"\t\t":"\t",
              n->call->parallel->sync());
    last_when=entry_points[i].whens[0];
    
    // Dump de la tabulation et du nom du point d'entrée
    //#warning f or s?
    nprintf(n, NULL, "\n%s/*@%f*/%s%s(",
            is_into_compute_loop?"\t\t":"\t",
            entry_points[i].whens[0],
            n->call->parallel->spawn(),
            entry_points[i].name);
    
    // Dump des arguments *ou pas*
//    if (entry_points[i].stdParamsNode != NULL){
//      //nprintf(n, NULL,"/*entry_points[i].stdParamsNode != NULL*/");
//      numParams=nMiddleDumpParameterTypeList(n,n->entity->src,
//                                             entry_points[i].stdParamsNode);
//      //nprintf(n, NULL,"/*done*/");
//    }//else nprintf(n,NULL,"/*NULL_stdParamsNode*/");
    
    // On s'autorise un endroit pour insérer des arguments
    nMiddleArgsDumpFromDFS(n,&entry_points[i]);
    
    /*if (entry_points[i].used_options==NULL &&
        entry_points[i].used_variables==NULL)
        nMiddleArgsAddGlobal(n, &entry_points[i], &numParams);*/
    
    // Et on dump les in et les out
//   if (entry_points[i].nblParamsNode != NULL){
//      nMiddleArgsDump(n,entry_points[i].nblParamsNode,&numParams);
//    } //else nprintf(n,NULL,"/*NULL_nblParamsNode*/");

    // Si on doit appeler des jobs depuis cette fonction @ée
    if (entry_points[i].called_variables != NULL){
      if (!entry_points[i].reduction)
         nMiddleArgsAddExtra(n,&numParams);
      // Et on rajoute les called_variables en paramètre d'appel
      dbg("\n\t[hookMain] Et on rajoute les called_variables en paramètre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }//else nprintf(n,NULL,"/*NULL_called_variables*/");
    nprintf(n, NULL, ");");
  }
  return NABLA_OK;
}


// ****************************************************************************
// * Backend POSTINIT - Génération du 'main'
// ****************************************************************************
#define BACKEND_MAIN_POSTINIT "\n\t//BACKEND_MAIN_POSTINIT"
NABLA_STATUS hookMainPostInit(nablaMain *nabla){
  dbg("\n[hookMainPostInit]");
  fprintf(nabla->entity->src, BACKEND_MAIN_POSTINIT);
  return NABLA_OK;
}


// ****************************************************************************
// * Backend POSTFIX - Génération du 'main'
// ****************************************************************************
#define BACKEND_MAIN_POSTFIX "\n//BACKEND_MAIN_POSTFIX\
\n\tglobal_time[0]+=*(double*)&global_deltat[0];\
\n\tglobal_iteration[0]+=1;\
\n\t//printf(\"\\ntime=%%e, dt=%%e\\n\", global_time[0], *(double*)&global_deltat[0]);\
\n\t}\
\tgettimeofday(&et, NULL);\n\
\tcputime = ((et.tv_sec-st.tv_sec)*1000.+ (et.tv_usec - st.tv_usec)/1000.0);\n\
\tprintf(\"\\n\\t\\33[7m[#%%04d] Elapsed time = %%12.6e(s)\\33[m\\n\", global_iteration[0]-1, cputime/1000.0);\n\
\tnabla_free_variables();\n\
\treturn 0;\n\
\n}\n"
// ****************************************************************************
// * hookMainPostfix
// ****************************************************************************
NABLA_STATUS hookMainPostfix(nablaMain *nabla){
  fprintf(nabla->entity->src, BACKEND_MAIN_POSTFIX);
  return NABLA_OK;
}


