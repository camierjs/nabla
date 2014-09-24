/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nccCudaFunction.c
 * Author   : Camier Jean-Sylvain
 * Created  : 2012.12.14
 * Updated  : 2012.12.14
 *****************************************************************************
 * Description:
 *****************************************************************************
 * Date			Author	Description
 * 2012.12.14	camierjs	Creation
 *****************************************************************************/
#include "nabla.h"


/*****************************************************************************
 * Hook pour dumper le nom de la fonction
 *****************************************************************************/
void cudaHookFunctionName(nablaMain *arc){
  //nprintf(arc, NULL, "%sEntity::", arc->name);
}


/*****************************************************************************
 * Génération d'un kernel associé à une fonction
 *****************************************************************************/
void cudaHookFunction(nablaMain *nabla, astNode *n){
  nablaJob *fct=nablaJobNew(nabla->entity);
  nablaJobAdd(nabla->entity, fct);
  nablaFctFill(nabla,fct,n,NULL);
}
