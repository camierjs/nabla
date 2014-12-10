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
