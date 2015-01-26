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
