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
char* nccArcLibMaterialsHeader(void){
  return "\n#include <arcane/materials/IMeshMaterial.h>\n\
using namespace Arcane::Materials;";
}


/***************************************************************************** 
 * PRIVATES
 *****************************************************************************/
char* nccArcLibMaterialsPrivates(void){
  return "\nprivate:\n\
\tvoid libMaterialsInitialize(void);\n\
\tIMeshMaterialMng* m_material_mng;";
}


/******************************************************************************
 * Initialisation
 ******************************************************************************/
void nccArcLibMaterialsIni(nablaMain *arc){
  fprintf(arc->entity->src, "\n\nvoid %s%s::libMaterialsInitialize(void){\n\
\t//m_material_mng = IMeshMaterialMng::getReference(defaultMesh());\n\
\n}",arc->name,nablaArcaneColor(arc));
//  #warning Création des matériaux
  //m_material_mng->registerMaterialInfo(mat_name);
//  #warning Création des milieux
}


/******************************************************************************
 * Création des matériaux et des milieux
 * Enregistrement des matériaux et des milieux.
 * La première chose à faire est d'enregistrer les caractéristiques des matériaux.
 * Pour l'instant, il s'agit uniquement du nom du matériau.
 * On enregistre uniquement les caractéristiques des matériaux et pas les
 * matériaux eux-même car ces derniers sont créés lors de la création des milieux.
 *
 * material_mng->registerMaterialInfo("MAT1");
 ******************************************************************************/



/******************************************************************************
 * Itération sur les mailles matériaux
 * Il existe trois classes pour faire référence aux notions de maille matériaux:
 *   - AllEnvCell est une classe permettant d'accéder à l'ensemble des milieux d'une maille.
 *   - EnvCell correspond à un milieu d'une maille et permet d'accéder aux valeurs
 *     de ce milieu pour cette maille et à l'ensemble des valeurs de cette maille pour les matériaux de ce milieu.
 *   - MatCell correspodant à une valeur d'un matériau d'un milieu d'une maille.
 * Il existe deux manières d'itérer sur les mailles matériau.
******************************************************************************/
