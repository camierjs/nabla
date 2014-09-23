/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcLibMaterials.c    													  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2012.12.06																	  *
 * Updated  : 2012.12.06																	  *
 *****************************************************************************
 * Description:
 * file:///usr/local/arcane/share/doc/1.18.4/devdoc/html/d2/dff/arcane_materials.html
 *****************************************************************************
 * Date			Author	Description														  *
 * 2012.12.06	camierjs	Creation															  *
 *****************************************************************************/
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
