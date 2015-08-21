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
  * Dump d'extra arguments
 *****************************************************************************/
void lambdaAddExtraArguments(nablaMain *nabla,
                             nablaJob *job,
                             int *numParams){
  nprintf(nabla,"\n\t\t/*lambdaAddExtraArguments*/",NULL);
}


/*****************************************************************************
  * Dump dans le src des arguments nabla en in comme en out
 *****************************************************************************/
void lambdaDumpNablaArgumentList(nablaMain *nabla, astNode *n,
                                 int *numParams){
  nprintf(nabla,"\n\t\t/*lambdaDumpNablaArgumentList*/",NULL);
}


// ****************************************************************************
// * Forward Declarations
// ****************************************************************************
static char* lambdaHookForwards[]={
  "inline std::ostream& info(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline std::ostream& debug(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "static void nabla_ini_node_coords(void);",
  "static void verifCoords(void);",
  NULL
};


// ****************************************************************************
// * Defines
// ****************************************************************************
static nablaDefine lambdaHookDefines[]={
  {"real", "Real"},
  {"WARP_ALIGN", "8"},    
  {"NABLA_NB_GLOBAL_WARP","1"},
  {"rabs(a)","fabs(a)"},
  {"set(a)", "a"},
  {"set1(cst)", "cst"},
  {"square_root(u)", "sqrt(u)"},
  {"cube_root(u)", "cbrt(u)"},
  {"store(u,_u)", "(*u=_u)"},
  {"load(u)", "(*u)"},
  {"zero()", "0.0"},
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_INI)"},
  {"DBG_OFF", "0x0000ul"},
  {"DBG_CELL_VOLUME", "0x0001ul"},
  {"DBG_CELL_CQS", "0x0002ul"},
  {"DBG_GTH", "0x0004ul"},
  {"DBG_NODE_FORCE", "0x0008ul"},
  {"DBG_INI_EOS", "0x0010ul"},
  {"DBG_EOS", "0x0020ul"},
  {"DBG_DENSITY", "0x0040ul"},
  {"DBG_MOVE_NODE", "0x0080ul"},
  {"DBG_INI", "0x0100ul"},
  {"DBG_INI_CELL", "0x0200ul"},
  {"DBG_INI_NODE", "0x0400ul"},
  {"DBG_LOOP", "0x0800ul"},
  {"DBG_FUNC_IN", "0x1000ul"},
  {"DBG_FUNC_OUT", "0x2000ul"},
  {"DBG_VELOCITY", "0x4000ul"},
  {"DBG_BOUNDARIES", "0x8000ul"},
  {"DBG_ALL", "0xFFFFul"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opMod(u,v)", "(u%v)"},
  {"opScaMul(u,v)","dot3(u,v)"},
  {"opVecMul(u,v)","cross(u,v)"},    
  {"dot", "dot3"},
  {"knAt(a)",""},
  {"fatal(a,b)","exit(-1)"},
  {"synchronize(a)","_Pragma(\"omp barrier\")"},
  {"mpi_reduce(how,what)","how##ToDouble(what)"},
  {"reduce(how,what)","how##ToDouble(what)"},
  {"xyz","int"},
  {"GlobalIteration", "global_iteration"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {"File", "std::ofstream&"},
  {"file(name,ext)", "std::ofstream name(#name \".\" #ext)"},
  {NULL,NULL}
};


// ****************************************************************************
// * Typedefs
// ****************************************************************************
static nablaTypedef lambdaHookTypedef[]={
  {"struct real3","Real3"},
  {NULL,NULL}
};





// ****************************************************************************
// * nLambda
// ****************************************************************************
NABLA_STATUS nLambda(nablaMain *nabla,
                     astNode *root,
                     const char *nabla_entity_name){
  // Définition des hooks pour l'AVX ou le MIC
  nablaBackendSimdHooks nablaLambdaSimdHooks={
    lambdaHookBits,
    lambdaHookGather,
    lambdaHookScatter,
    lambdaHookTypedef,
    lambdaHookDefines,
    lambdaHookForwards,
    lambdaHookPrevCell,
    lambdaHookNextCell,
    lambdaHookIncludes
  };
  nabla->simd=&nablaLambdaSimdHooks;
    
  // Définition des hooks pour Cilk+ *ou pas*
  nablaBackendParallelHooks lambdaCilkHooks={
    nLambdaParallelCilkSync,
    nLambdaParallelCilkSpawn,
    nLambdaParallelCilkLoop,
    nLambdaParallelCilkIncludes
  };
  nablaBackendParallelHooks lambdaOpenMPHooks={
    nLambdaParallelOpenMPSync,
    nLambdaParallelOpenMPSpawn,
    nLambdaParallelOpenMPLoop,
    nLambdaParallelOpenMPIncludes
  };
  nablaBackendParallelHooks lambdaVoidHooks={
    nLambdaParallelVoidSync,
    nLambdaParallelVoidSpawn,
    nLambdaParallelVoidLoop,
    nLambdaParallelVoidIncludes
  };
  nabla->parallel=&lambdaVoidHooks;
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->parallel=&lambdaCilkHooks;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->parallel=&lambdaOpenMPHooks;

  
  nablaBackendPragmaHooks lambdaPragmaICCHooks ={
    lambdaHookPragmaIccIvdep,
    lambdaHookPragmaIccAlign
  };
  nablaBackendPragmaHooks lambdaPragmaGCCHooks={
    lambdaHookPragmaGccIvdep,
    lambdaHookPragmaGccAlign
  };
  // Par defaut, on met GCC
  nabla->pragma=&lambdaPragmaGCCHooks;
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->pragma=&lambdaPragmaICCHooks;

  // Hooks pour le source
  nHookSource nLHookSource={
    nLambdaHookSourceOpen,
    nLambdaHookSourceInclude
  };

// Hooks pour le header
  nHookHeader nLHookHeader={
    nLambdaHookHeaderOpen,
    nLambdaHookHeaderPrefix,
    nLambdaHookHeaderIncludes,
    nLambdaHookHeaderDump,
    nLambdaHookHeaderDefineEnumerates,
    nLambdaHookHeaderPostfix
  };
  
  // Hooks pour le maillage
  nHookMesh nLHookMesh={
    nLambdaHookMeshPrefix,
    nLambdaHookMeshCore,
    nLambdaHookMeshPostfix
  };
  
  // Hooks pour les variables
  nHookVars nLHookVars={
    nLambdaHookVarsInit,
    nLambdaHookVariablesPrefix,
    nLambdaHookVariablesPostfix
  };  

  // Hooks pour le main
  nHookMain nLHookMain={
    nLambdaHookMainPrefix,
    nLambdaHookMainPreInit,
    nLambdaHookMainVarInitKernel,
    nLambdaHookMainVarInitCall,
    nLambdaHookMain,
    nLambdaHookMainPostInit,
    nLambdaHookMainPostfix
  };  
   
  nablaBackendHooks lambdaBackendHooks={
    // Jobs stuff
    lambdaHookPrefixEnumerate,
    lambdaHookDumpEnumerateXYZ,
    lambdaHookDumpEnumerate,
    lambdaHookPostfixEnumerate,
    lambdaHookItem,
    lambdaHookSwitchToken,
    lambdaHookTurnTokenToVariable,
    lambdaHookSystem,
    lambdaHookAddExtraParameters,
    lambdaHookDumpNablaParameterList,
    lambdaHookTurnBracketsToParentheses,
    lambdaHookJobDiffractStatement,
    // Other hooks
    lambdaHookFunctionName,
    lambdaHookFunction,
    lambdaHookJob,
    lambdaHookReduction,
    lambdaHookIteration,
    lambdaHookExit,
    lambdaHookTime,
    lambdaHookFatal,
    lambdaHookAddCallNames,
    lambdaHookAddArguments,
    lambdaHookTurnTokenToOption,
    lambdaHookEntryPointPrefix,
    lambdaHookDfsForCalls,
    lambdaHookPrimaryExpressionToReturn,
    lambdaHookReturnFromArgument,
    &nLHookHeader,
    &nLHookSource,
    &nLHookMesh,
    &nLHookVars,
    &nLHookMain
  };
  nabla->hook=&lambdaBackendHooks;


  ///////////////////////////////////////////////////////////
  // Partie des hooks à remonter à termes dans le middlend //
  ///////////////////////////////////////////////////////////
  nabla->hook->vars->init(nabla);

  nabla->hook->source->open(nabla);
  nabla->hook->source->include(nabla);

  nabla->hook->header->open(nabla);
  nabla->hook->header->prefix(nabla);
  nabla->hook->header->includes(nabla);
  nabla->hook->header->dump(nabla);
  nabla->hook->header->enumerates(nabla);
 
  nabla->hook->mesh->core(nabla);
  
  
  // Parse du code préprocessé et lance les hooks associés
  nMiddleParseAndHook(root,nabla);
  
  nabla->hook->main->varInitKernel(nabla);
  nabla->hook->main->prefix(nabla);
  
  nabla->hook->vars->prefix(nabla);
  
  nabla->hook->mesh->prefix(nabla);
  nabla->hook->main->preInit(nabla);
  nabla->hook->main->varInitCall(nabla);
  nabla->hook->main->main(nabla);
  nabla->hook->main->postInit(nabla);
  
  // Partie POSTFIX
  nabla->hook->header->postfix(nabla); 
  nabla->hook->mesh->postfix(nabla);
  nabla->hook->vars->postfix(nabla);
  nabla->hook->main->postfix(nabla);
  return NABLA_OK;
}

