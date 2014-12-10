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
 * HEADER
 *****************************************************************************/
char* nccArcLibCartesianHeader(void){
  // On prépare le header de l'entity
  return "\n#include \"arcane/MeshArea.h\"\n\
#include \"arcane/ISubDomain.h\"\n\
#include \"arcane/MeshAreaAccessor.h\"\n\
#include \"arcane/cea/ICartesianMesh.h\"\n\
#include \"arcane/cea/CellDirectionMng.h\"\n\
#include \"arcane/cea/FaceDirectionMng.h\"\n\
#include \"arcane/cea/NodeDirectionMng.h\"\n\n";
}


/***************************************************************************** 
 * PRIVATES
 *****************************************************************************/
char* nccArcLibCartesianPrivates(void){
  return "\nprivate:\n\
\tvoid libCartesianInitialize(void);\n\
\tICartesianMesh* m_cartesian_mesh;";
}


/******************************************************************************
 * Initialisation
 ******************************************************************************/
void nccArcLibCartesianIni(nablaMain *arc){
  fprintf(arc->entity->src, "\
\n\nvoid %s%s::libCartesianInitialize(void){\n\
\tIMesh* mesh = defaultMesh();\n\
\tm_cartesian_mesh = arcaneCreateCartesianMesh(mesh);\n\
\tm_cartesian_mesh->computeDirections();\n}",arc->name,nablaArcaneColor(arc));
  nablaJob *libCartesianInitialize=nablaJobNew(arc->entity);
  libCartesianInitialize->is_an_entry_point=true;
  libCartesianInitialize->is_a_function=true;
  libCartesianInitialize->scope  = strdup("NoGroup");
  libCartesianInitialize->region = strdup("NoRegion");
  libCartesianInitialize->item   = strdup("\0");
  libCartesianInitialize->rtntp  = strdup("void");
  libCartesianInitialize->name   = strdup("libCartesianInitialize");
  libCartesianInitialize->name_utf8 = strdup("libCartesianInitialize");
  libCartesianInitialize->xyz    = strdup("NoXYZ");
  libCartesianInitialize->drctn  = strdup("NoDirection");
  sprintf(&libCartesianInitialize->at[0],"-huge_valf");
  libCartesianInitialize->whenx  = 1;
  libCartesianInitialize->whens[0] = ENTRY_POINT_init;
  nablaJobAdd(arc->entity, libCartesianInitialize);
}
