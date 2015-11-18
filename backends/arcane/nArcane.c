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
#include "frontend/nablaAst.h"


static char* arcaneEntryPointPrefix(nablaMain *nabla,
                                    nablaJob *entry_point){
  return "";
}
static void arcaneIteration(nablaMain *nabla){
  nprintf(nabla, "/*ITERATION*/", "subDomain()->commonVariables().globalIteration()");
}
static void arcaneExit(nablaMain *nabla){
  nprintf(nabla, "/*EXIT*/","{\n\
if (m_hlt_dive){\n\
   m_hlt_exit=true;\n\
}else{\n\
   subDomain()->timeLoopMng()->stopComputeLoop(true);\n\
}}");
}
static void arcaneTime(nablaMain *nabla){
  nprintf(nabla, "/*TIME*/", "subDomain()->commonVariables().globalTime()");
}
static void arcaneFatal(struct nablaMainStruct *nabla){
  dbg("\n[arcaneFatal]");
  nprintf(nabla, NULL, "throw FatalErrorException");
} 
static void arcaneAddCallNames(nablaMain *nabla,
                               nablaJob *job,
                               astNode *n){
  dbg("\n[arcaneAddCallNames]");
  /*nothing to do*/
}
static void arcaneAddArguments(nablaMain *nabla,nablaJob *job){
  dbg("\n[arcaneAddArguments]");
  /*nothing to do*/
}
static void arcaneTurnTokenToOption(nablaMain *nabla,nablaOption *opt){
  nprintf(nabla, "/*tt2o arc*/", "options()->%s()", opt->name);
}


/*
 * nablaArcaneColor
 */
char *nablaArcaneColor(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_ALONE)==BACKEND_COLOR_ARCANE_ALONE) return "Module";
  if ((middlend->colors&BACKEND_COLOR_ARCANE_MODULE)==BACKEND_COLOR_ARCANE_MODULE) return "Module";
  if ((middlend->colors&BACKEND_COLOR_ARCANE_SERVICE)==BACKEND_COLOR_ARCANE_SERVICE) return "Service";
  exit(NABLA_ERROR|fprintf(stderr,"[nablaArcaneColor] Unable to switch!"));
  return NULL;
}
bool isAnArcaneAlone(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_ALONE)==BACKEND_COLOR_ARCANE_ALONE) return true;
  return false;
}
bool isAnArcaneModule(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_ALONE)==BACKEND_COLOR_ARCANE_ALONE) return true;
  if ((middlend->colors&BACKEND_COLOR_ARCANE_MODULE)==BACKEND_COLOR_ARCANE_MODULE) return true;
  return false;
}
bool isAnArcaneService(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_SERVICE)==BACKEND_COLOR_ARCANE_SERVICE) return true;
  return false;
}
char *nArcanePragmaGccIvdep(void){ return ""; }
char *nArcanePragmaGccAlign(void){ return ""; }


// ****************************************************************************
// *arcaneHookReduction
// ****************************************************************************
static void arcaneHookReduction(nablaMain *nabla, astNode *n){
  //int fakeNumParams=0;
  const astNode *item_node = n->children->next->children;
  const astNode *global_var_node = n->children->next->next;
  const astNode *reduction_operation_node = global_var_node->next;
  const astNode *item_var_node = reduction_operation_node->next;
  astNode *at_single_cst_node = dfsFetch(n, rulenameToId("at_constant"));
  assert(at_single_cst_node!=NULL);
  char *global_var_name = global_var_node->token;
  char *item_var_name = item_var_node->token;
  // Pr�paration du nom du job
  char job_name[NABLA_MAX_FILE_NAME];
  job_name[0]=0;
  strcat(job_name,"arcaneReduction_");
  strcat(job_name,global_var_name);
  // Rajout du job de reduction
  nablaJob *redjob = nMiddleJobNew(nabla->entity);
  redjob->is_an_entry_point=true;
  redjob->is_a_function=false;
  redjob->scope  = strdup("NoGroup");
  redjob->region = strdup("NoRegion");
  redjob->item   = strdup(item_node->token);
  redjob->return_type  = strdup("void");
  redjob->name   = strdup(job_name);
  redjob->name_utf8 = strdup(job_name);
  redjob->xyz    = strdup("NoXYZ");
  redjob->direction  = strdup("NoDirection");
  // Init flush
  redjob->when_index = 0;
  redjob->whens[0] = 0.0;
  // On parse le at_single_cst_node pour le metre dans le redjob->whens[redjob->when_index-1]
  nMiddleAtConstantParse(redjob,at_single_cst_node,nabla,redjob->at);
  nMiddleStoreWhen(redjob,nabla,NULL);
  assert(redjob->when_index>0);
  dbg("\n\t[arcaneHookReduction] @ %f",redjob->whens[redjob->when_index-1]);
  
  nMiddleJobAdd(nabla->entity, redjob);
  const bool min_reduction = reduction_operation_node->tokenid==MIN_ASSIGN;
  const double reduction_init = min_reduction?1.0e20:-1.0e20;
  const char* mix = min_reduction?"in":"ax";
  // G�n�ration de code associ� � ce job de r�duction
  nablaVariable *var=nMiddleVariableFind(nabla->variables, item_var_name);
  const char cnf=var->item[0];
  assert(var!=NULL);
  nprintf(nabla, NULL, "\n\n\
// ******************************************************************************\n\
// * Kernel de reduction de la variable '%s' vers la globale '%s'\n\
// ******************************************************************************\n\
void %s%s::%s(){\n",
          item_var_name,
          global_var_name,
          nabla->name,
          isAnArcaneModule(nabla)?"Module":"Service",
          job_name);
  
  nprintf(nabla, NULL, "\n\tm_global_%s=%f;\n",global_var_name,reduction_init);
  if (cnf=='c') nprintf(nabla, NULL, "\tENUMERATE_CELL(cell,ownCells()){");
  if (cnf=='n') nprintf(nabla, NULL, "\tENUMERATE_NODE(node,ownNodes()){");
  nprintf(nabla, NULL, "\n\t\tm_global_%s=::fm%s(m_global_%s(),m_%s_%s[%s]);",
          global_var_name,mix,global_var_name,
          var->item,item_var_name,var->item);
  nprintf(nabla, NULL, "\n\t}");
  
  nprintf(nabla, NULL, "\n\
   m_global_%s=mpi_reduce(Parallel::ReduceM%s,m_global_%s());\n}\n\n",
          global_var_name,mix,global_var_name);
}

// Typedefs, Defines & Forwards
const nHookHeader nablaArcaneHeaderHooks={
  NULL, // dump
  NULL, // open
  NULL, // enums
  NULL, // prefix
  NULL, // include
  NULL  // postfix
};

/*****************************************************************************
 * ncc
 *****************************************************************************/
NABLA_STATUS nccArcane(nablaMain *middlend,
                       astNode *root,
                       const char *nabla_entity_name){
  char cfgFileName[NABLA_MAX_FILE_NAME];
  char axlFileName[NABLA_MAX_FILE_NAME];
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  nablaEntity *entity=middlend->entity;  // On fait l'hypoth�se qu'il n'y a qu'un entity pour l'instant

  nCallSimd nablaArcaneSimdCalls={
    nccArcBits,
    nccArcGather,
    nccArcScatter,
    nccArcIncludes
  };
  nHookXyz nablaArcaneXyzHooks={
    nccArcSystemPrefix,
    nccArcPrevCell,
    nccArcNextCell,
    nccArcSystemPostfix
  };
  const nHookForAll nArcaneHookForAll={
    arcaneHookPrefixEnumerate,
    arcaneHookDumpEnumerate,
    arcaneHookItem,
    arcaneHookPostfixEnumerate
  };
  const nHookToken nArcaneHookToken={
    arcaneHookTokenPrefix,
    arcaneHookSwitchToken,
    arcaneHookTurnTokenToVariable,
    arcaneTurnTokenToOption,
    arcaneHookSystem,
    arcaneIteration,
    arcaneExit,
    arcaneTime,
    arcaneFatal,
    arcaneHookTurnBracketsToParentheses,
    arcaneHookIsTest,
    arcaneHookTokenPostfix
  };

  const nHookGrammar hookGrammar={
    arcaneHookFunction,
    arcaneJob,
    arcaneHookReduction,
    NULL, // primary_expression_to_return
    NULL // returnFromArgument
  };
  
  const nHookCall nArcaneHookCall={
    arcaneAddCallNames,
    arcaneAddArguments,
    arcaneEntryPointPrefix,
    NULL, // DFS for calls
    NULL, // addExtraParameters
    NULL  // dumpNablaParameterList
  };
  
  nHooks arcaneBackendHooks={
    &nArcaneHookForAll,
    &nArcaneHookToken,
    &hookGrammar,
    &nArcaneHookCall,
    &nablaArcaneXyzHooks, // Xyz
    NULL, // pragma
    &nablaArcaneHeaderHooks, // header
    NULL, // source
    NULL, // mesh
    NULL, // vars
    NULL // main
  };
  nCalls arcaneBackendCalls={
    NULL, // header
    &nablaArcaneSimdCalls, // simd
    NULL // parallel
  };

  middlend->call=&arcaneBackendCalls;
  middlend->hook=&arcaneBackendHooks;
  //middlend->hook->simd=&nablaArcaneSimdHooks;
  
  nHookPragma arcanePragmaGCCHooks={
    nArcanePragmaGccAlign
  };
  middlend->hook->pragma=&arcanePragmaGCCHooks;
    
  dbg("\n[nccArcane] Cr�ation du fichier ARCANE main.c dans le cas d'un module");
  if (isAnArcaneModule(middlend)==true)
    nccArcMain(middlend);

  dbg("\n[nccArcane] Ouverture du fichier SOURCE du middlend");
  sprintf(srcFileName, "%s%s.cc", entity->name, nablaArcaneColor(middlend));
  if ((middlend->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  dbg("\n[nccArcane] Ouverture du fichier HEADER du middlend dans le cas d'un module");
  if (isAnArcaneModule(middlend)==true){
    sprintf(hdrFileName, "%s%s.h", middlend->name, nablaArcaneColor(middlend));
    if ((middlend->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  }

  dbg("\n[nccArcane] Ouverture du fichier CONFIG pour ARCANE dans le cas d'un module");
  if (isAnArcaneModule(middlend)==true){
    sprintf(cfgFileName, "%s.config", middlend->name);
    if ((middlend->cfg=fopen(cfgFileName, "w")) == NULL) exit(NABLA_ERROR);
  }

  dbg("\n[nccArcane] Et du fichier AXL pour ARCANE");
  if (isAnArcaneModule(middlend))
    sprintf(axlFileName, "%s.axl", middlend->name);
  else
    sprintf(axlFileName, "%sService.axl", middlend->name);
  if ((middlend->axl=fopen(axlFileName, "w")) == NULL) exit(NABLA_ERROR);

  dbg("\n[nccArcane] Pour l'instant, nous n'avons pas d'options");
  middlend->options=NULL;
  
  dbg("\n[nccArcane] nccAxlGenerateHeader");
  nccAxlGenerateHeader(middlend);
  nccArcaneEntityHeader(middlend);
 
  // Dans le cas d'un service, on le fait maintenant
  if (isAnArcaneService(middlend)){
    if (nccArcaneEntityIncludes(entity)!=NABLA_OK) printf("error: in service HDR generation!\n");
    nccArcaneBeginNamespace(middlend);
  }
  
  // Premi�re passe pour les VARIABLES du fichier AXL
  if (isAnArcaneModule(middlend))
    nccArcConfigHeader(middlend);
  
  nMiddleGrammar(root,middlend);

  // Dans le cas d'un module, on le fait maintenant
  if (isAnArcaneModule(middlend)){
    dbg("\n[nccArcane] nccArcaneEntityIncludes initialization");
    if (nccArcaneEntityIncludes(entity)!=NABLA_OK) printf("error: in Includes generation!\n");
  }
   
  dbg("\n[nccArcane] nccArcaneEntityConstructor initialization");
  if (nccArcaneEntityConstructor(entity)!=NABLA_OK) printf("error: in EntityConstructor generation!\n");

  dbg("\n[nccArcane] nccArcaneEntityVirtuals initialization");
  if (nccArcaneEntityVirtuals(entity)!=NABLA_OK) printf("error: in EntityVirtuals generation!\n");

  { // Should be done by a hook
    // MATHEMATICA fonctions d'initialisation
    if ((entity->libraries&(1<<with_mathematica))!=0){
      dbg("\n[nccArcane] MATHEMATICA initialization");
      nccArcLibMathematicaIni(middlend);
    }
    // MAIL fonctions d'initialisation
    if ((entity->libraries&(1<<with_mail))!=0){
      dbg("\n[nccArcane] MAIL initialization");
      nccArcLibMailIni(middlend);
    }
    // PARTICLES fonctions d'initialisation
    if ((entity->libraries&(1<<with_particles))!=0){
      dbg("\n[nccArcane] PARTICLES initialization");
      nccArcLibParticlesIni(middlend);
    }
    // ALEPH fonctions d'initialisation
    if ((entity->libraries&(1<<with_aleph))!=0){
      dbg("\n[nccArcane] ALEPH initialization");
      if (isAnArcaneModule(middlend)) nccArcLibAlephIni(middlend);
      if (isAnArcaneService(middlend)) nccArcLibSchemeIni(middlend);
    }
    // CARTESIAN fonctions d'initialisation
    if ((entity->libraries&(1<<with_cartesian))!=0){
      dbg("\n[nccArcane] CARTESIAN initialization");
      nccArcLibCartesianIni(middlend);
    }
    // MATERIALS fonctions d'initialisation
    if ((entity->libraries&(1<<with_materials))!=0){
      dbg("\n[nccArcane] MATERIALS initialization");
      nccArcLibMaterialsIni(middlend);
    }
    // GMP fonctions d'initialisation
    if ((entity->libraries&(1<<with_gmp))!=0){
      dbg("\n[nccArcane] GMP initialization");
      nccArcLibGmpIni(middlend);
    }
    // DFT fonctions d'initialisation
    if ((entity->libraries&(1<<with_dft))!=0){
      dbg("\n[nccArcane] DFT initialization");
      nccArcLibDftIni(middlend);
    }
    // SLURM fonctions d'initialisation
    if ((entity->libraries&(1<<with_slurm))!=0){
      dbg("\n[nccArcane] SLURM initialization");
      nccArcLibSlurmIni(middlend);
    }
  }
  nArcaneHLTInit(middlend);
  
  dbg("\n[nccArcane] AXL generation");
  if (nccAxlGenerator(middlend)!=NABLA_OK) printf("error: in AXL generation!\n");

  dbg("\n[nccArcane] HDR generation");
  if (nccArcaneEntityGeneratorPrivates(entity)!=NABLA_OK) printf("error: in HDR generation!\n");

  if (isAnArcaneModule(middlend)==true){ // Fermeture du CONFIG dans le cas d'un module
    nccArcConfigFooter(middlend); 
    fclose(middlend->cfg);
  }

  // Fermeture de l'AXL
  fprintf(middlend->axl,"\n</%s>\n", (isAnArcaneModule(middlend)==true)?"module":"service");
  fclose(middlend->axl);
  
  // Fermeture du header du entity
  if (isAnArcaneModule(middlend)==true){ // Fermeture du HEADER dans le cas d'un module
    fclose(entity->hdr);
  }
  
  // Fermeture du source du entity
  fprintf(entity->src,"\n\
\n/*---------------------------------------------------------------------------*/\
\n/*---------------------------------------------------------------------------*/\
\nARCANE_REGISTER_%s_%s%s(%s%s%s%s%s);\n%s\n",
          isAnArcaneModule(middlend)?"MODULE":"SERVICE",
          toolStrUpCase(middlend->name),
          isAnArcaneService(middlend)?"SERVICE":"",
          isAnArcaneService(middlend)?middlend->service_name:"",
          isAnArcaneService(middlend)?",":"",
          middlend->name,
          isAnArcaneModule(middlend)?"Module":"Service",
          isAnArcaneService(middlend)?",SFP_None":"",
          isAnArcaneService(middlend)?"ARCANE_END_NAMESPACE\n":"");
  fclose(entity->src);
  dbgCloseTraceFile();
  return NABLA_OK;
}
