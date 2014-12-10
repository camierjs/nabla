// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
