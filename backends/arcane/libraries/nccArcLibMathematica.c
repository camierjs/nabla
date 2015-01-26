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
 * Préparation du header du module
 *****************************************************************************/
char* nccArcLibMathematicaHeader(void){
  return "\n\
#include \"mathlink.h\"\n\
#include <arcane/Timer.h>\n\
#include <arcane/IApplication.h>\n\
#include <arcane/IParallelMng.h>\n\
#include <arcane/AbstractService.h>\n\
#include <arcane/ServiceBuilder.h>\n\
//#include <arcane/FactoryService.h>\n\
#include <arcane/mathlink/mathlink.h>\n";
}


// ****************************************************************************
// * nccArcLibMathematicaPrivates
// ****************************************************************************
char* nccArcLibMathematicaPrivates(void){
  return   "\nprivate:\t//Mathematica stuffs\n\
\tvoid mathematicaIni(void);\n\
\tmathlink *m_mathlink;";
}


// ****************************************************************************
// * nccArcLibMathematicaIni
// ****************************************************************************
void nccArcLibMathematicaIni(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid %s%s::mathematicaIni(void){\
\n\tServiceBuilder<mathlink> serviceBuilder(subDomain());\
\n\tm_mathlink = serviceBuilder.createInstance(\"mathlink\");\
\n\tm_mathlink->link();\
\n}",arc->name,nablaArcaneColor(arc));
  nablaJob *mathlinkIniFunction=nablaJobNew(arc->entity);
  mathlinkIniFunction->is_an_entry_point=true;
  mathlinkIniFunction->is_a_function=true;
  mathlinkIniFunction->scope  = strdup("NoGroup");
  mathlinkIniFunction->region = strdup("NoRegion");
  mathlinkIniFunction->item   = strdup("\0");
  mathlinkIniFunction->rtntp  = strdup("void");
  mathlinkIniFunction->name   = strdup("mathematicaIni");
  mathlinkIniFunction->name_utf8   = strdup("mathematicaIni");
  mathlinkIniFunction->xyz    = strdup("NoXYZ");
  mathlinkIniFunction->drctn  = strdup("NoDirection");
  sprintf(&mathlinkIniFunction->at[0],"-1024.0");
  mathlinkIniFunction->whenx  = 1;
  mathlinkIniFunction->whens[0] = -1024.0;
  nablaJobAdd(arc->entity, mathlinkIniFunction);
}


// ****************************************************************************
// * nccArcLibMathematicaDelete
// ****************************************************************************
char *nccArcLibMathematicaDelete(void){
  return "\n\t\tif (m_mathlink) m_mathlink->unlink();";
}
