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
// * Backend OKINA PREFIX - Génération du 'main'
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
\t//printf(\"%%d noeuds, %%d mailles\",NABLA_NB_NODES,NABLA_NB_CELLS);\n\
\tnabla_ini_variables();\n\
\tnabla_ini_node_coords();\n\
\t// Initialisation de la précision du cout\n\
\tstd::cout.precision(21);\n\
\t//std::cout.setf(std::ios::floatfield);\n\
\tstd::cout.setf(std::ios::scientific, std::ios::floatfield);\n\
\t// Initialisation du temps et du deltaT\n\
\tglobal_time=0.0;\n\
\tglobal_iteration=1;\n\
\tglobal_deltat[0] = set1(option_dtt_initial);// @ 0;\n\
\t//printf(\"\\n\\33[7;32m[main] time=%%e,\
 Global Iteration is #%%d\\33[m\",global_time,global_iteration);"


// ****************************************************************************
// * Backend OKINA INIT - Génération du 'main'
// ****************************************************************************
#define OKINA_MAIN_PREINIT "\n\t//OKINA_MAIN_PREINIT"


// ****************************************************************************
// * Backend OKINA POSTFIX - Génération du 'main'
// ****************************************************************************
#define OKINA_MAIN_POSTINIT "\n\t//OKINA_MAIN_POSTINIT"


// ****************************************************************************
// * Backend OKINA POSTFIX - Génération du 'main'
// * \n\tprintf(\"\\n\\t\\33[7m[#%%04d]\\33[m time=%%e, delta_t=%%e\", iteration+=1, global_time, *(double*)&global_del
// ****************************************************************************
#define OKINA_MAIN_POSTFIX "\n//OKINA_MAIN_POSTFIX\
\n\tglobal_time+=*(double*)&global_deltat[0];\
\n\tglobal_iteration+=1;\
\n\t//printf(\"\\ntime=%%e, dt=%%e\\n\", global_time, *(double*)&global_deltat[0]);\
\n\t}\
\tgettimeofday(&et, NULL);\n\
\tcputime = ((et.tv_sec-st.tv_sec)*1000.+ (et.tv_usec - st.tv_usec)/1000.0);\n\
\tprintf(\"\\n\\t\\33[7m[#%%04d] Elapsed time = %%12.6e(s)\\33[m\\n\",\
 global_iteration-1, cputime/1000.0);\n\n}\n"


// ****************************************************************************
// * nOkinaMainPrefix
// ****************************************************************************
NABLA_STATUS nOkinaMainPrefix(nablaMain *nabla){
  dbg("\n[nOkinaMainPrefix]");
  fprintf(nabla->entity->src, OKINA_MAIN_PREFIX);
  return NABLA_OK;
}


// ****************************************************************************
// * nOkinaMainPreInit
// ****************************************************************************
NABLA_STATUS nOkinaMainPreInit(nablaMain *nabla){
  dbg("\n[nOkinaMainPreInit]");
  fprintf(nabla->entity->src, OKINA_MAIN_PREINIT);
  return NABLA_OK;
}


// ****************************************************************************
// * nOkinaMainPostInit
// ****************************************************************************
NABLA_STATUS nOkinaMainPostInit(nablaMain *nabla){
  dbg("\n[nOkinaMainPostInit]");
  fprintf(nabla->entity->src, OKINA_MAIN_POSTINIT);
  return NABLA_OK;
}


// ****************************************************************************
// * nOkinaMain
// ****************************************************************************
NABLA_STATUS nOkinaMain(nablaMain *n){
  nablaVariable *var;
  nablaJob *entry_points;
  int i,numParams,number_of_entry_points;
  bool is_into_compute_loop=false;
  double last_when;
  
  dbg("\n[nOkinaMain]");
  number_of_entry_points=nMiddleNumberOfEntryPoints(n);
  entry_points=nMiddleEntryPointsSort(n,number_of_entry_points);
  // Et on rescan afin de dumper, on rajoute les +2 ComputeLoopEnd|Begin
   for(i=0,last_when=entry_points[i].whens[0];i<number_of_entry_points+2;++i){
     if (strcmp(entry_points[i].name,"ComputeLoopEnd")==0)continue;
     if (strcmp(entry_points[i].name,"ComputeLoopBegin")==0)continue;
     dbg("%s\n\t[nOkinaMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    // Si l'on passe pour la première fois la frontière du zéro, on écrit le code pour boucler
    if (entry_points[i].whens[0]>=0 && is_into_compute_loop==false){
      is_into_compute_loop=true;
      nprintf(n, NULL,"\
\n\tgettimeofday(&st, NULL);\n\
\twhile (global_time<option_stoptime){// && global_iteration!=option_max_iterations){");
    }
    // On sync si l'on découvre un temps logique différent
    if (last_when!=entry_points[i].whens[0])
      nprintf(n, NULL, "\n%s%s",
              is_into_compute_loop?"\t\t":"\t",
              n->call->parallel->sync());
    last_when=entry_points[i].whens[0];
    // Dump de la tabulation et du nom du point d'entrée
//#warning ici float, mais ordre!! 
    nprintf(n, NULL, "\n%s/*@%f*/%s%s(",
            is_into_compute_loop?"\t\t":"\t",
            entry_points[i].whens[0],
            n->call->parallel->spawn(), 
            entry_points[i].name);
    // Dump des arguments *ou pas*
    if (entry_points[i].stdParamsNode != NULL){
      //nprintf(n, "/*entry_points[i].stdParamsNode != NULL*/",NULL);
      //numParams=dumpParameterTypeList(n->entity->src, entry_points[i].stdParamsNode);
    }//else nprintf(n,NULL,"/*NULL_stdParamsNode*/");
    
    // On s'autorise un endroit pour insérer des arguments
    nOkinaArgsExtra(n, &entry_points[i], &numParams);
    
    // Et on dump les in et les out
    if (entry_points[i].nblParamsNode != NULL){
      nOkinaArgsList(n,entry_points[i].nblParamsNode,&numParams);
    }else nprintf(n,NULL,"/*NULL_nblParamsNode*/");

    // Si on doit appeler des jobs depuis cette fonction @ée
    if (entry_points[i].called_variables != NULL){
      nOkinaArgsAddExtraConnectivities(n,&numParams);
      // Et on rajoute les called_variables en paramètre d'appel
      dbg("\n\t[nOkinaMain] Et on rajoute les called_variables en paramètre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }else nprintf(n,NULL,"/*NULL_called_variables*/");
    nprintf(n, NULL, ");");
    nOkinaArgsDumpNablaDebugFunctionFromOut(n,entry_points[i].nblParamsNode,true);
    //nprintf(n, NULL, "\n");
  }
  return NABLA_OK;
}


// ****************************************************************************
// * okinaSourceMesh
// ****************************************************************************
extern char knMsh1D_c[];
extern char knMsh3D_c[];
static char *nOkinaMainSourceMeshAoS_vs_SoA(nablaMain *nabla){
  return "node_coord[iNode]=real3(x,y,z);"; 
}
static void nOkinaMainSourceMesh(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  if ((nabla->entity->libraries&(1<<with_real))!=0)
    fprintf(nabla->entity->src,knMsh1D_c);
  else
    fprintf(nabla->entity->src,knMsh3D_c,nOkinaMainSourceMeshAoS_vs_SoA(nabla));
  //fprintf(nabla->entity->src,knMsh_c);
}


// ****************************************************************************
// * nOkinaMainPostfix
// ****************************************************************************
NABLA_STATUS nOkinaMainPostfix(nablaMain *nabla){
  dbg("\n[nOkinaMainPostfix] OKINA_MAIN_POSTFIX");
  fprintf(nabla->entity->src, OKINA_MAIN_POSTFIX);
  dbg("\n[nOkinaMainPostfix] okinaSourceMesh");
  nOkinaMainSourceMesh(nabla);
  dbg("\n[nOkinaMainPostfix] NABLA_OK");
  return NABLA_OK;
}


