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
//  #warning Cr�ation des mat�riaux
  //m_material_mng->registerMaterialInfo(mat_name);
//  #warning Cr�ation des milieux
}


/******************************************************************************
 * Cr�ation des mat�riaux et des milieux
 * Enregistrement des mat�riaux et des milieux.
 * La premi�re chose � faire est d'enregistrer les caract�ristiques des mat�riaux.
 * Pour l'instant, il s'agit uniquement du nom du mat�riau.
 * On enregistre uniquement les caract�ristiques des mat�riaux et pas les
 * mat�riaux eux-m�me car ces derniers sont cr��s lors de la cr�ation des milieux.
 *
 * material_mng->registerMaterialInfo("MAT1");
 ******************************************************************************/



/******************************************************************************
 * It�ration sur les mailles mat�riaux
 * Il existe trois classes pour faire r�f�rence aux notions de maille mat�riaux:
 *   - AllEnvCell est une classe permettant d'acc�der � l'ensemble des milieux d'une maille.
 *   - EnvCell correspond � un milieu d'une maille et permet d'acc�der aux valeurs
 *     de ce milieu pour cette maille et � l'ensemble des valeurs de cette maille pour les mat�riaux de ce milieu.
 *   - MatCell correspodant � une valeur d'un mat�riau d'un milieu d'une maille.
 * Il existe deux mani�res d'it�rer sur les mailles mat�riau.
******************************************************************************/
