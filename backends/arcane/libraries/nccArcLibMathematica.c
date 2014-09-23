/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcLibMathematica.c  													  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2013.04.03																	  *
 * Updated  : 2013.04.03																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2013.04.03	camierjs	Creation															  *
 *****************************************************************************/
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
  mathlinkIniFunction->group  = strdup("NoGroup");
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
