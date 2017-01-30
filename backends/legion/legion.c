///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
#include "legion.h"

const nWhatWith legionCallHeaderDefines[]={{NULL,NULL}};

// ****************************************************************************
// * CALLS
// ****************************************************************************
const static callHeader xHeader={
  NULL,
  NULL,
  NULL
};
const static callSimd simd={
  NULL,
  xCallGather,
  xCallScatter,
  NULL,
  xCallUid
};
const static callParallel parallel={
  NULL,
  NULL,
  xParallelLoop,
  legionCallParallelIncludes
};
static backendCalls calls={
  &xHeader,
  &simd,
  &parallel
};


// ****************************************************************************
// * HOOKS
// ****************************************************************************
const static hookForAll forall={
  NULL,
  legionHookForAllDump,
  NULL,//xHookForAllItem, // traite les items
  legionHookForAllPostfix
};

  const static hookToken token={
  NULL,
  legionHookSwitchToken,
  legionHookTurnTokenToVariable,
  legionHookTurnTokenToOption,
  NULL,//xHookSystem,
  NULL,//xHookIteration,
  legionHookExit,
  NULL,//xHookError,
  legionHookTime,
  NULL,//xHookFatal,
  NULL,//xHookTurnBracketsToParentheses,
  NULL,//xHookIsTest,
  NULL
};

bool legionHookGramSkip(nablaMain *nabla){return false;}

const static hookGrammar gram={
  NULL,
  NULL,
  NULL,//legionHookReduction,
  NULL,
  NULL,
  NULL,//xHookDfsVariable,   // return true pour dire qu'on supporte le scan des in&out
  legionHookDfsExtra, // return false
  NULL,
  NULL,//legionHookEoe,
  NULL,
  legionHookGramSkip
};

const static hookCall call={
  xHookAddCallNames,        // utilisé pour dumper le nom des fonctions
  NULL,//xHookAddArguments, // rajoute les arguments des fonctions appelées
  NULL,//xHookEntryPointPrefix,    // retourne static inline
  NULL,//xHookDfsForCalls,  // Dump des variables appelées
  NULL,
  NULL
};

const static hookXyz xyz={
  NULL,
  NULL,//xHookPrevCell,
  NULL,//xHookNextCell,
  NULL,//xHookSysPostfix
};

const static hookHeader header={
  legionHookHeaderDump,
  NULL,//xHookHeaderDumpWithLibs,
  xHookHeaderOpen,
  NULL,//xHookHeaderDefineEnumerates,
  xHookHeaderPrefix, // #ifndef __BACKEND_pennant_H__
  legionHookHeaderIncludes,
  NULL,//xHookHeaderAlloc,
  legionHookHeaderPostfix // Avant le #endif // __BACKEND_pennantH__
};

const static hookSource source={
  xHookSourceOpen,    // Ouvre le fichier source
  xHookSourceInclude, // Rajoute l'include .h
  NULL
};

const static hookMesh mesh={
  NULL,//xHookMeshPrefix,
  NULL,//xHookMeshCore,
  NULL,//xHookMeshPostfix
};

const static hookVars vars={
  NULL,//xHookVariablesInit, // rajoute global int iteration
  legionHookVariablesPrefix, // Au dessus du main
  legionHookVariablesMalloc, // Juste apres la déclaration du main
  legionHookVariablesFree,   // A la fin du main
  NULL,
  NULL
};  

const static hookMain mains={
  legionHookMainPrefix,
  legionHookMainPreInit,
  NULL,//xHookMainVarInitKernel, // ne fait rien
  NULL,//xHookMainVarInitCall,
  NULL,//xHookMainHLT,
  NULL,//xHookMainPostInit, // ne fait rien
  legionHookMainPostfix
};  

static hooks legionHooks={
  &forall,
  &token,
  &gram,
  &call,
  &xyz,
  NULL,
  &header,
  &source,
  &mesh,
  &vars,
  &mains
};



// ****************************************************************************
// * nLegionDump
// ****************************************************************************
static void nLegionDumpGeneric(nablaMain *nabla, const char *filename, char *external){
  FILE *file;
  const int external_strlen = strlen(external);
  int fprintf_strlen=0;
  //printf("[33m[nLegionDumpGeneric] dumping (len=%d): %s",external_strlen,filename);
  //fflush(stdout);
  if ((file=fopen(filename, "w")) == NULL) exit(NABLA_ERROR);
  // On ne peut pas comparer les strlen, car les '%%' sont transformés en '%'
  if (external_strlen!=(fprintf_strlen=fprintf(file,external))){
    //printf(", got %d![m\n", fprintf_strlen);fflush(stdout);
    //exit(NABLA_ERROR);
  }
  if (fclose(file)!=0) exit(NABLA_ERROR);
  //printf(", done![m\n");fflush(stdout);
}
extern char legion_args_rg[];
extern char legion_common_rg[];
extern char legion_compile_rg[];
extern char legion_config_rg[];
extern char legion_distri_rg[];
extern char legion_geom_rg[];
extern char legion_hacks_rg[];
extern char legion_init_rg[];
extern char legion_input_rg[];
extern char legion_math_rg[];
extern char legion_mesh_rg[];
extern char legion_rg[];
extern char legion_tools_rg[];

extern char legion_cc[];
extern char legion_h[];

extern char makefile[];

extern char pennant_data_rg[];
extern char pennant_kernels_init_rg[];
extern char pennant_kernels_loop_rg[];
extern char pennant_main_init_rg[];
extern char pennant_main_loop_rg[];
extern char pennant_valid_rg[];
   
extern char sedovsmall_pnt[];
extern char sedovsmall_xy_std[];

// ****************************************************************************
// * Dump de tous les fichiers legion/pennant/sedov utiles pour la compilation
// ****************************************************************************
static void nLegionDump(nablaMain *nabla){
  nLegionDumpGeneric(nabla,"legion_args.rg",legion_args_rg);
  nLegionDumpGeneric(nabla,"legion_common.rg",legion_common_rg);
  nLegionDumpGeneric(nabla,"legion_compile.rg",legion_compile_rg);
  nLegionDumpGeneric(nabla,"legion_config.rg",legion_config_rg);
  nLegionDumpGeneric(nabla,"legion_distri.rg",legion_distri_rg);
  nLegionDumpGeneric(nabla,"legion_geom.rg",legion_geom_rg);
  nLegionDumpGeneric(nabla,"legion_hacks.rg",legion_hacks_rg);
  nLegionDumpGeneric(nabla,"legion_init.rg",legion_init_rg);
  nLegionDumpGeneric(nabla,"legion_input.rg",legion_input_rg);
  nLegionDumpGeneric(nabla,"legion_math.rg",legion_math_rg);
  nLegionDumpGeneric(nabla,"legion_mesh.rg",legion_mesh_rg);
  nLegionDumpGeneric(nabla,"legion.rg",legion_rg);
  nLegionDumpGeneric(nabla,"legion_tools.rg",legion_tools_rg);
  nLegionDumpGeneric(nabla,"legion.cc",legion_cc);
  nLegionDumpGeneric(nabla,"legion.h",legion_h);
  nLegionDumpGeneric(nabla,"makefile",makefile);
  
  nLegionDumpGeneric(nabla,"pennant_data.rg",pennant_data_rg);
  nLegionDumpGeneric(nabla,"pennant_kernels_init.rg",pennant_kernels_init_rg);
  nLegionDumpGeneric(nabla,"pennant_kernels_loop.rg",pennant_kernels_loop_rg);
  nLegionDumpGeneric(nabla,"pennant_main_init.rg",pennant_main_init_rg);
  nLegionDumpGeneric(nabla,"pennant_main_loop.rg",pennant_main_loop_rg);
  nLegionDumpGeneric(nabla,"pennant_valid.rg",pennant_valid_rg);
  
  nLegionDumpGeneric(nabla,"sedovsmall.pnt",sedovsmall_pnt);
  nLegionDumpGeneric(nabla,"sedovsmall.xy.std",sedovsmall_xy_std);
}

// ****************************************************************************
// * Legion
// ****************************************************************************
hooks* legion(nablaMain *nabla){
  nabla->call=&calls;
  nLegionDump(nabla);
  return &legionHooks;
}