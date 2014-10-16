/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nccArcane.c
 * Author   : Camier Jean-Sylvain
 * Created  : 18.01.2010
 * Updated  : 2012.08.24
 *****************************************************************************
 * Description:
 *****************************************************************************
 * Date			Author	Description
 * 18.01.2010	jscamier	Creation
 *****************************************************************************/
#include "nabla.h"

static void arcaneDfsForCalls(struct nablaMainStruct *nabla,nablaJob *job, astNode *n,const char *namespace,
                            astNode *nParams){}

static char* arcaneEntryPointPrefix(struct nablaMainStruct *nabla, nablaJob *entry_point){
  return "";
}
static void arcaneIteration(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*ITERATION*/", "subDomain()->commonVariables().globalIteration()");
}
static void arcaneExit(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*EXIT*/", "subDomain()->timeLoopMng()->stopComputeLoop(true)");
}
static void arcaneTime(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*TIME*/", "subDomain()->commonVariables().globalTime()");
}
static void arcaneFatal(struct nablaMainStruct *nabla){
  dbg("\n[arcaneFatal]");
  nprintf(nabla, NULL, "throw FatalErrorException");
} 
static void arcaneAddCallNames(struct nablaMainStruct *nabla,nablaJob *job,astNode *n){
  dbg("\n[arcaneAddCallNames]");
  /*nothing to do*/
}
static void arcaneAddArguments(struct nablaMainStruct *nabla,nablaJob *job){
  dbg("\n[arcaneAddArguments]");
  /*nothing to do*/
}
static void arcaneTurnTokenToOption(struct nablaMainStruct *nabla,nablaOption *opt){
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
  nablaEntity *entity=middlend->entity;  // On fait l'hypothèse qu'il n'y a qu'un entity pour l'instant
  nablaBackendSimdHooks nablaArcaneSimdHooks={
    nccArcBits,
    nccArcGather,
    nccArcScatter,
    NULL,//Typedefs
    NULL,//Defines
    NULL,//Forwards
    nccArcPrevCell,
    nccArcNextCell,
    nccArcIncludes
  };
  static nablaBackendHooks arcaneBackendHooks={
    // Jobs stuff
    arcaneHookPrefixEnumerate,
    arcaneHookDumpEnumerateXYZ,
    arcaneHookDumpEnumerate,
    arcaneHookPostfixEnumerate,
    arcaneHookItem,
    arcaneHookSwitchToken,
    arcaneHookTurnTokenToVariable,
    arcaneHookSystem,
    NULL,//addExtraParameters
    NULL,//dumpNablaParameterList
    arcaneHookTurnBracketsToParentheses,
    NULL,//diffractStatement
// Is this true ? #warning Arcane backend does not support Real3 globals
    // Other hooks
    arcaneHookFunctionName,
    arcaneHookFunction,
    arcaneJob,
    arcaneIteration,
    arcaneExit,
    arcaneTime,
    arcaneFatal,
    arcaneAddCallNames,
    arcaneAddArguments,
    arcaneTurnTokenToOption,
    arcaneEntryPointPrefix,
    arcaneDfsForCalls,
    NULL, // primary_expression_to_return
    NULL // returnFromArgument
  };
  middlend->simd=&nablaArcaneSimdHooks;
  middlend->hook=&arcaneBackendHooks;
  
  dbg("\n[nccArcane] Création du fichier ARCANE main.c dans le cas d'un module");
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
  
  // Première passe pour les VARIABLES du fichier AXL
  if (isAnArcaneModule(middlend))
    nccArcConfigHeader(middlend);
  
  nablaMiddlendParseAndHook(root,middlend);

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
    if ((entity->libraries&(1<<mathematica))!=0){
      dbg("\n[nccArcane] MATHEMATICA initialization");
      nccArcLibMathematicaIni(middlend);
    }
    // MAIL fonctions d'initialisation
    if ((entity->libraries&(1<<mail))!=0){
      dbg("\n[nccArcane] MAIL initialization");
      nccArcLibMailIni(middlend);
    }
    // PARTICLES fonctions d'initialisation
    if ((entity->libraries&(1<<particles))!=0){
      dbg("\n[nccArcane] PARTICLES initialization");
      nccArcLibParticlesIni(middlend);
    }
    // ALEPH fonctions d'initialisation
    if ((entity->libraries&(1<<aleph))!=0){
      dbg("\n[nccArcane] ALEPH initialization");
      if (isAnArcaneModule(middlend)) nccArcLibAlephIni(middlend);
      if (isAnArcaneService(middlend)) nccArcLibSchemeIni(middlend);
    }
    // CARTESIAN fonctions d'initialisation
    if ((entity->libraries&(1<<cartesian))!=0){
      dbg("\n[nccArcane] CARTESIAN initialization");
      nccArcLibCartesianIni(middlend);
    }
    // MATERIALS fonctions d'initialisation
    if ((entity->libraries&(1<<materials))!=0){
      dbg("\n[nccArcane] MATERIALS initialization");
      nccArcLibMaterialsIni(middlend);
    }
    // GMP fonctions d'initialisation
    if ((entity->libraries&(1<<gmp))!=0){
      dbg("\n[nccArcane] GMP initialization");
      nccArcLibGmpIni(middlend);
    }
    // DFT fonctions d'initialisation
    if ((entity->libraries&(1<<dft))!=0){
      dbg("\n[nccArcane] DFT initialization");
      nccArcLibDftIni(middlend);
    }
    // SLURM fonctions d'initialisation
    if ((entity->libraries&(1<<slurm))!=0){
      dbg("\n[nccArcane] SLURM initialization");
      nccArcLibSlurmIni(middlend);
    }
  }
  
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
