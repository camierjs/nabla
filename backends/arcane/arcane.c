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
#include "backends/arcane/arcane.h"



static const hookHeader header={
  aHookHeaderDump,
  aHookHeaderOpen,
  aHookHeaderEnums,
  aHookHeaderPrefix,
  aHookHeaderIncludes,
  aHookHeaderPostfix
};

static const hookXyz xyz={
  nccArcSystemPrefix,
  NULL, NULL, NULL
};

static const hookForAll forall={
  arcaneHookPrefixEnumerate,
  arcaneHookDumpEnumerate,
  arcaneHookItem,
  arcaneHookPostfixEnumerate
};

static const hookToken token={
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

static const hookCall call={
  arcaneAddCallNames,
  arcaneAddArguments,
  arcaneEntryPointPrefix,
  NULL,//aHookDfsForCalls,
  NULL,//aHookAddExtraParametersDFS,
  NULL//aHookDumpNablaParameterListDFS
};

static const hookGrammar grammar={
  arcaneHookFunction,
  arcaneJob,
  arcaneHookReduction,
  NULL,//aHookPrimaryExpressionToReturn,
  NULL, // returnFromArgument
  arcaneHookDfsVariable
};

const static hookSource source={
  aHookSourceOpen,
  aHookSourceInclude
};

const static hookMesh mesh={
  aHookMeshPrefix,
  aHookMeshCore,
  aHookMeshPostfix
};

const static hookVars vars={
  aHookVariablesInit,
  aHookVariablesPrefix,
  aHookVariablesMalloc,
  aHookVariablesFree
};

const static hookMain mains={
  aHookMainPrefix,
  aHookMainPreInit,
  aHookMainVarInitKernel,
  aHookMainVarInitCall,
  aHookMainHLT,
  aHookMainPostInit,
  aHookMainPostfix
};

static hooks arcaneBackendHooks={
  &forall,
  &token,
  &grammar,
  &call,
  &xyz,
  NULL, // pragma
  &header,
  &source,
  &mesh,
  &vars,
  &mains
};

// ****************************************************************************
// * arcane
// ****************************************************************************
NABLA_STATUS arcaneOld(nablaMain *nabla,
                       astNode *root,
                       const char *nabla_entity_name){
  nabla->hook=&arcaneBackendHooks;
  
  dbg("\n* Backend ARCANE"); // org mode item
  
  aHookSourceOpen(nabla);
  aHookHeaderOpen(nabla);
  aHookHeaderDump(nabla);
    
  nMiddleGrammar(root,nabla);

  aHookVariablesPrefix(nabla);


  aHookMainPrefix(nabla);
  aHookMainPreInit(nabla);
  
  aHookHeaderPostfix(nabla);
  aHookMainPostfix(nabla);
  
  dbg("\n\t[nccArcane]  Deleting kernel names");
  toolUnlinkKtemp(nabla->entity->jobs);

  dbgCloseTraceFile();
  return NABLA_OK;
}


// ****************************************************************************
// * arcane with middlend/animate
// ****************************************************************************
hooks* arcane(nablaMain *nabla){
  dbg("\n* Backend ARCANE"); // org mode item
  return &arcaneBackendHooks;
}