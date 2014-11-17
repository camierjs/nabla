/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcaneMain.c																  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2012.11.13																	  *
 * Updated  : 2012.11.13																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2012.11.13	camierjs	Creation															  *
 *****************************************************************************/
#include "nabla.h"


/*****************************************************************************
 * Backend ARCANE - Génération du fichier 'main.cc'
 *****************************************************************************/
#define ARC_MAIN "#include <iostream>\n//#include <mpi.h>\n#include <arcane/impl/ArcaneMain.h>\n\
using namespace Arcane;\n\
int main(int argc,char* argv[]){\n\
  int r = 0;\n\
  ArcaneMain::arcaneInitialize();\n\
  {\n\
    ApplicationInfo app_info(&argc,&argv,\"%s\",VersionInfo(1,0,0));\n\
    r = ArcaneMain::arcaneMain(app_info);\n\
  }\n\
  ArcaneMain::arcaneFinalize();\n\
  return r;\n\
}\n"


NABLA_STATUS nccArcMain(nablaMain *arc){
  if ((arc->main=fopen("main.cc", "w")) == NULL) exit(NABLA_ERROR); 
  fprintf(arc->main, ARC_MAIN, arc->name);
  fclose(arc->main);
  return NABLA_OK;
}
