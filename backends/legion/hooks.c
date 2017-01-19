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


// ****************************************************************************
// * Fonction produisant l'ENUMERATE_*
// ****************************************************************************
char* legionHookForAllDump(nablaJob *job){
  return "// legionHookForAllDump";
}

// **************************************************************************** 
// * legionHookForAllPostfix
// **************************************************************************** 
char* legionHookForAllPostfix(nablaJob *job){
  return "// legionHookForAllPostfix";
}


void legionHookSwitchToken(astNode *n, nablaJob *job){}

nablaVariable *legionHookTurnTokenToVariable(astNode * n,
                                             nablaMain *nabla,
                                             nablaJob *job){return NULL;}
void legionHookTurnTokenToOption(struct nablaMainStruct *nabla,nablaOption *opt){}
void legionHookExit(nablaMain *nabla, nablaJob *job){}
void legionHookTime(nablaMain *nabla){}


// ****************************************************************************
// * legionHookHeaderDump
// ****************************************************************************
void legionHookHeaderDump(nablaMain *nabla){
}

// ****************************************************************************
// * legionHookHeaderIncludes
// ****************************************************************************
void legionHookHeaderIncludes(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
#include <cstdio>\n\
#include <cassert>\n\
#include <cstdlib>\n\
//#include \"legion.h\"\n\
//using namespace Legion;\n\
\n\
typedef double Real;\n\
typedef double Real2;\n\
");
}


// ****************************************************************************
// * legionHookHeaderPostfix
// ****************************************************************************
void legionHookHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n//legionHookHeaderPostfix\n");  
  fprintf(nabla->entity->hdr,"\n#endif // __BACKEND_%sH__\n",
          nabla->entity->name);
}


// ****************************************************************************
// * legionsHookParallelIncludes
// ****************************************************************************
char *legionCallParallelIncludes(void){
  return "\n//legionCallParallelIncludes\n";
}


// ****************************************************************************
// * legionHookEoe - End Of Enumerate
// ****************************************************************************
char* legionHookEoe(nablaMain* nabla){
  return "//legionHookEoe";
}


// ****************************************************************************
// * legionHookDfsExtra
// ****************************************************************************
bool legionHookDfsExtra(nablaMain* nabla,nablaJob* job, bool arg_test){
  return false;
}

// ****************************************************************************
// * Dump des options dans le header
// ****************************************************************************
void legionHookVariablesPrefix(nablaMain *nabla){
  fprintf(nabla->entity->src, "\n// legionHookVariablesPrefix");
}


// ****************************************************************************
// * Malloc des variables
// ****************************************************************************
void legionHookVariablesMalloc(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n// legionHookVariablesMalloc");
}


// ****************************************************************************
// * Variables Postfix
// ****************************************************************************
void legionHookVariablesFree(nablaMain *nabla){
  fprintf(nabla->entity->src, "\n// legionHookVariablesFree");
}


// ****************************************************************************
// * legionHookMainPrefix
// ****************************************************************************
NABLA_STATUS legionHookMainPrefix(nablaMain *nabla){
  fprintf(nabla->entity->src, "\n\
// ******************************************************************************\n\
// * Main\n\
// ******************************************************************************\n\
int main(int argc, char *argv[]){");
  return NABLA_OK;
}


// ****************************************************************************
// * legionHookMainPreInit
// ****************************************************************************
NABLA_STATUS legionHookMainPreInit(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n// legionHookMainPreInit");
  return NABLA_OK;
}


// ****************************************************************************
// * legionHookMainPostfix
// ****************************************************************************
NABLA_STATUS legionHookMainPostfix(nablaMain *nabla){
  fprintf(nabla->entity->src, "\n// legionHookMainPostfix");
  fprintf(nabla->entity->src, "\n}");
  return NABLA_OK;
}
