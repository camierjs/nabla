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
