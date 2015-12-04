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
#include "frontend/ast.h"


static char* arcaneEntryPointPrefix(nablaMain *nabla,
                                    nablaJob *entry_point){
  return "";
}
static void arcaneIteration(nablaMain *nabla){
  nprintf(nabla, "/*ITERATION*/", "subDomain()->commonVariables().globalIteration()");
}
static void arcaneExit(nablaMain *nabla,nablaJob *job){
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
  dbg("\n\t\t[arcaneHookReduction]");
  const astNode *nabla_items = dfsFetch(n->children,rulenameToId("nabla_items"));
  assert(nabla_items);
  const astNode *global_var_node = n->children->next;
  assert(global_var_node);
  const astNode *reduction_operation_node = global_var_node->next;
  assert(reduction_operation_node);
  const astNode *item_var_node = reduction_operation_node->next;
  assert(item_var_node);
  astNode *at_single_cst_node = dfsFetch(n->children->next, rulenameToId("at_constant"));
  assert(at_single_cst_node);
  char *global_var_name = global_var_node->token;
  assert(global_var_name);
  char *item_var_name = item_var_node->token;
  assert(item_var_name);
  dbg("\n\t\t[arcaneHookReduction] global_var_name=%s",global_var_name);
  dbg("\n\t\t[arcaneHookReduction] item_var_name=%s",item_var_name);
  dbg("\n\t\t[arcaneHookReduction] item=%s",nabla_items->token);
  // Préparation du nom du job
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
  redjob->item   = strdup(nabla_items->token);
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
  // Génération de code associé à ce job de réduction
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
const hookHeader nablaArcaneHeaderHooks={
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
NABLA_STATUS nccArcane(nablaMain *nabla,
                       astNode *root,
                       const char *nabla_entity_name){
  char cfgFileName[NABLA_MAX_FILE_NAME];
  char axlFileName[NABLA_MAX_FILE_NAME];
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  nablaEntity *entity=nabla->entity;  // On fait l'hypothèse qu'il n'y a qu'un entity pour l'instant

  callSimd nablaArcaneSimdCalls={
    nccArcBits,
    nccArcGather,
    nccArcScatter,
    nccArcIncludes
  };
  hookXyz nablaArcaneXyzHooks={
    nccArcSystemPrefix,
    nccArcPrevCell,
    nccArcNextCell,
    nccArcSystemPostfix
  };
  const hookForAll nArcaneHookForAll={
    arcaneHookPrefixEnumerate,
    arcaneHookDumpEnumerate,
    arcaneHookItem,
    arcaneHookPostfixEnumerate
  };
  const hookToken nArcaneHookToken={
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

  const hookGrammar hookGrammar={
    arcaneHookFunction,
    arcaneJob,
    arcaneHookReduction,
    NULL, // primary_expression_to_return
    NULL, // returnFromArgument
    arcaneHookDfsVariable
  };
  
  const hookCall nArcaneHookCall={
    arcaneAddCallNames,
    arcaneAddArguments,
    arcaneEntryPointPrefix,
    NULL, // DFS for calls
    NULL, // addExtraParameters
    NULL  // dumpNablaParameterList
  };
  
  hooks arcaneBackendHooks={
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
  calls arcaneBackendCalls={
    NULL, // header
    &nablaArcaneSimdCalls, // simd
    NULL // parallel
  };

  nabla->call=&arcaneBackendCalls;
  nabla->hook=&arcaneBackendHooks;
  //nabla->hook->simd=&nablaArcaneSimdHooks;
  
  hookPragma arcanePragmaGCCHooks={
    nArcanePragmaGccAlign
  };
  nabla->hook->pragma=&arcanePragmaGCCHooks;
    
  dbg("\n[nccArcane] Création du fichier ARCANE main.c dans le cas d'un module");
  if (isAnArcaneModule(nabla)==true)
    nccArcMain(nabla);

  dbg("\n[nccArcane] Ouverture du fichier SOURCE du nabla");
  sprintf(srcFileName, "%s%s.cc", entity->name, nablaArcaneColor(nabla));
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  dbg("\n[nccArcane] Ouverture du fichier HEADER du nabla dans le cas d'un module");
  if (isAnArcaneModule(nabla)==true){
    sprintf(hdrFileName, "%s%s.h", nabla->name, nablaArcaneColor(nabla));
    if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  }

  dbg("\n[nccArcane] Ouverture du fichier CONFIG pour ARCANE dans le cas d'un module");
  if (isAnArcaneModule(nabla)==true){
    sprintf(cfgFileName, "%s.config", nabla->name);
    if ((nabla->cfg=fopen(cfgFileName, "w")) == NULL) exit(NABLA_ERROR);
  }

  dbg("\n[nccArcane] Et du fichier AXL pour ARCANE");
  if (isAnArcaneModule(nabla))
    sprintf(axlFileName, "%s.axl", nabla->name);
  else
    sprintf(axlFileName, "%sService.axl", nabla->name);
  if ((nabla->axl=fopen(axlFileName, "w")) == NULL) exit(NABLA_ERROR);

  dbg("\n[nccArcane] Pour l'instant, nous n'avons pas d'options");
  nabla->options=NULL;
  
  dbg("\n[nccArcane] nccAxlGenerateHeader");
  nccAxlGenerateHeader(nabla);
  nccArcaneEntityHeader(nabla);
 
  // Dans le cas d'un service, on le fait maintenant
  if (isAnArcaneService(nabla)){
    if (nccArcaneEntityIncludes(entity)!=NABLA_OK) printf("error: in service HDR generation!\n");
    nccArcaneBeginNamespace(nabla);
  }
  
  // Première passe pour les VARIABLES du fichier AXL
  if (isAnArcaneModule(nabla))
    nccArcConfigHeader(nabla);
  
  nMiddleGrammar(root,nabla);

  // Dans le cas d'un module, on le fait maintenant
  if (isAnArcaneModule(nabla)){
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
      nccArcLibMathematicaIni(nabla);
    }
    // MAIL fonctions d'initialisation
    if ((entity->libraries&(1<<with_mail))!=0){
      dbg("\n[nccArcane] MAIL initialization");
      nccArcLibMailIni(nabla);
    }
    // PARTICLES fonctions d'initialisation
    if ((entity->libraries&(1<<with_particles))!=0){
      dbg("\n[nccArcane] PARTICLES initialization");
      nccArcLibParticlesIni(nabla);
    }
    // ALEPH fonctions d'initialisation
    if ((entity->libraries&(1<<with_aleph))!=0){
      dbg("\n[nccArcane] ALEPH initialization");
      if (isAnArcaneModule(nabla)) nccArcLibAlephIni(nabla);
      if (isAnArcaneService(nabla)) nccArcLibSchemeIni(nabla);
    }
    // CARTESIAN fonctions d'initialisation
    if ((entity->libraries&(1<<with_cartesian))!=0){
      dbg("\n[nccArcane] CARTESIAN initialization");
      nccArcLibCartesianIni(nabla);
    }
    // MATERIALS fonctions d'initialisation
    if ((entity->libraries&(1<<with_materials))!=0){
      dbg("\n[nccArcane] MATERIALS initialization");
      nccArcLibMaterialsIni(nabla);
    }
    // GMP fonctions d'initialisation
    if ((entity->libraries&(1<<with_gmp))!=0){
      dbg("\n[nccArcane] GMP initialization");
      nccArcLibGmpIni(nabla);
    }
    // DFT fonctions d'initialisation
    if ((entity->libraries&(1<<with_dft))!=0){
      dbg("\n[nccArcane] DFT initialization");
      nccArcLibDftIni(nabla);
    }
    // SLURM fonctions d'initialisation
    if ((entity->libraries&(1<<with_slurm))!=0){
      dbg("\n[nccArcane] SLURM initialization");
      nccArcLibSlurmIni(nabla);
    }
  }
  nArcaneHLTInit(nabla);
  
  dbg("\n[nccArcane] AXL generation");
  if (nccAxlGenerator(nabla)!=NABLA_OK) printf("error: in AXL generation!\n");

  dbg("\n[nccArcane] HDR generation");
  if (nccArcaneEntityGeneratorPrivates(entity)!=NABLA_OK) printf("error: in HDR generation!\n");

  if (isAnArcaneModule(nabla)==true){ // Fermeture du CONFIG dans le cas d'un module
    nccArcConfigFooter(nabla); 
    fclose(nabla->cfg);
  }

  // Fermeture de l'AXL
  fprintf(nabla->axl,"\n</%s>\n", (isAnArcaneModule(nabla)==true)?"module":"service");
  fclose(nabla->axl);
  
  // Fermeture du header du entity
  if (isAnArcaneModule(nabla)==true){ // Fermeture du HEADER dans le cas d'un module
    fclose(entity->hdr);
  }
  
  // Fermeture du source du entity
  fprintf(entity->src,"\n\
\n/*---------------------------------------------------------------------------*/\
\n/*---------------------------------------------------------------------------*/\
\nARCANE_REGISTER_%s_%s%s(%s%s%s%s%s);\n%s\n",
          isAnArcaneModule(nabla)?"MODULE":"SERVICE",
          toolStrUpCase(nabla->name),
          isAnArcaneService(nabla)?"SERVICE":"",
          isAnArcaneService(nabla)?nabla->service_name:"",
          isAnArcaneService(nabla)?",":"",
          nabla->name,
          isAnArcaneModule(nabla)?"Module":"Service",
          isAnArcaneService(nabla)?",SFP_None":"",
          isAnArcaneService(nabla)?"ARCANE_END_NAMESPACE\n":"");
  fclose(entity->src);

  dbg("\n\t[nccArcane]  Deleting kernel names");
  toolUnlinkKtemp(nabla->entity->jobs);

  dbgCloseTraceFile();
  return NABLA_OK;
}
