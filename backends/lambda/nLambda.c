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


// ****************************************************************************
// * ENUMERATES Hooks
// ****************************************************************************
static void lambdaDefineEnumerates(nablaMain *nabla){
  const char *parallel_prefix_for_loop=nabla->parallel->loop(nabla);
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define FOR_EACH_CELL(c) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<8;n+=1)\n\
\n\
#define FOR_EACH_CELL_WARP(c) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
#define FOR_EACH_CELL_WARP_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
\n\
#define FOR_EACH_CELL_WARP_NODE(n)\\\n\
  %sfor(int cn=c;cn>=c;--cn)\\\n\
    for(int n=8-1;n>=0;--n)\n\
\n\
#define FOR_EACH_NODE(n) /*%s*/for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=8*n;c<8;c+=1,nc+=1)\n\
\n\
#define FOR_EACH_NODE_WARP(n) %sfor(int n=0;n<NABLA_NB_NODES_WARP;n+=1)\n\
\n\
#define FOR_EACH_NODE_WARP_CELL(c)\\\n\
    for(int c=0;c<8;c+=1)\n",
          parallel_prefix_for_loop, // FOR_EACH_CELL
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_SHARED
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_NODE
          parallel_prefix_for_loop, // FOR_EACH_NODE
          parallel_prefix_for_loop  // FOR_EACH_NODE_WARP
          );
}


// ****************************************************************************
// * Forward Declarations
// ****************************************************************************
static char* lambdaForwards[]={
  "inline std::ostream& info(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline std::ostream& debug(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "static void nabla_ini_node_coords(void);",
  "static void verifCoords(void);",
  NULL
};


// ****************************************************************************
// * Defines
// ****************************************************************************
static nablaDefine lambdaDefines[]={
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
static nablaTypedef lambdaTypedef[]={
  {"struct real3","Real3"},
  {NULL,NULL}
};


// ****************************************************************************
// * lambdaInclude
// ****************************************************************************
static void lambdaInclude(nablaMain *nabla){
  fprintf(nabla->entity->src,"#include \"%s.h\"\n", nabla->entity->name);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void lambdaHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __LAMBDA_%s_H__\n#define __LAMBDA_%s_H__",
          nabla->entity->name,
          nabla->entity->name);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void lambdaHeaderIncludes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Lambda includes\n\
// *****************************************************************************\n\
%s // from nabla->simd->includes\n\
#include <sys/time.h>\n\
#include <stdlib.h>\n\
#include <stdio.h>\n\
#include <string.h>\n\
#include <vector>\n\
#include <math.h>\n\
#include <assert.h>\n\
#include <stdarg.h>\n\
#include <iostream>\n\
#include <sstream>\n\
#include <fstream>\n\
using namespace std;\n\
%s // from nabla->parallel->includes()",
          nabla->simd->includes(),
          nabla->parallel->includes());
}


// ****************************************************************************
// * lambdaHeader for Std, Avx or Mic
// ****************************************************************************
extern char lambdaStdReal_h[];
extern char lambdaStdReal3_h[];
extern char lambdaStdInteger_h[];
extern char lambdaStdGather_h[];
extern char lambdaStdScatter_h[];
extern char lambdaStdOStream_h[];
extern char lambdaStdTernary_h[];

static char *dumpExternalFile(char *file){
  return file+NABLA_LICENSE_HEADER;
}
static void lambdaHeaderTypes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaStdInteger_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaStdReal_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaStdReal3_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaStdTernary_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaStdGather_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaStdScatter_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaStdOStream_h));
}


// ****************************************************************************
// * lambdaHeader for Dbg
// ****************************************************************************
extern char lambdaDbg_h[];
static void lambdaHeaderDbg(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaDbg_h));
}


// ****************************************************************************
// * lambdaHeader for Maths
// ****************************************************************************
extern char lambdaMth_h[];
static void lambdaHeaderMth(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(lambdaMth_h));
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void lambdaHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __LAMBDA_%s_H__\n",nabla->entity->name);
}


// ****************************************************************************
// * nLambda
// ****************************************************************************
NABLA_STATUS nLambda(nablaMain *nabla,
                     astNode *root,
                     const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Définition des hooks pour l'AVX ou le MIC
  nablaBackendSimdHooks nablaLambdaSimdStdHooks={
    lambdaHookBits,
    lambdaHookGather,
    lambdaHookScatter,
    lambdaTypedef,
    lambdaDefines,
    lambdaForwards,
    lambdaHookPrevCell,
    lambdaHookNextCell,
    lambdaHookIncludes
  };
  nabla->simd=&nablaLambdaSimdStdHooks;
    
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
  
  static nablaBackendHooks lambdaBackendHooks={
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
    lambdaHookReturnFromArgument
  };
  nabla->hook=&lambdaBackendHooks;

  // Rajout de la variable globale 'iteration'
  nablaVariable *iteration = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, iteration);
  iteration->axl_it=false;
  iteration->item=strdup("global");
  iteration->type=strdup("integer");
  iteration->name=strdup("iteration");
 
  // Ouverture du fichier source du entity
  sprintf(srcFileName, "%s.cc", nabla->name);
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  // Ouverture du fichier header du entity
  sprintf(hdrFileName, "%s.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  
  // Dump dans le HEADER des includes, typedefs, defines, debug, maths & errors stuff
  lambdaHeaderPrefix(nabla);
  lambdaHeaderIncludes(nabla);
  nMiddleDefines(nabla,nabla->simd->defines);
  nMiddleTypedefs(nabla,nabla->simd->typedefs);
  nMiddleForwards(nabla,nabla->simd->forwards);

  // On inclue les fichiers lambda'SIMD'
  lambdaHeaderTypes(nabla);
  lambdaHeaderDbg(nabla);
  lambdaHeaderMth(nabla);

  //lambdaMesh(nabla);
  // Mesh structures and functions depends on the ℝ library that can be used
  if (isWithLibrary(nabla,with_real)){
    lambdaMesh1D(nabla);
  }else{
    lambdaMesh3D(nabla);
  }
  lambdaDefineEnumerates(nabla);

  // Dump dans le fichier SOURCE
  lambdaInclude(nabla);
  
  // Parse du code préprocessé et lance les hooks associés
  nMiddleParseAndHook(root,nabla);
  lambdaMainVarInitKernel(nabla);

  // Partie PREFIX
  lambdaMainPrefix(nabla);
  lambdaVariablesPrefix(nabla);
  lambdaMainMeshPrefix(nabla);
  
  // Partie Pré Init
  lambdaMainPreInit(nabla);
  lambdaMainVarInitCall(nabla);
      
  // Dump des entry points dans le main
  lambdaMain(nabla);

  // Partie Post Init
  lambdaMainPostInit(nabla);
  
  // Partie POSTFIX
  lambdaHeaderPostfix(nabla); 
  lambdaMainMeshPostfix(nabla);
  lambdaVariablesPostfix(nabla);
  lambdaMainPostfix(nabla);
  return NABLA_OK;
}

