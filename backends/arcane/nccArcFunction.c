/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcaneFunction.c									       			  *
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
//#include "ncc.tab.h"

void arcaneHookFunctionName(nablaMain *arc){
  nprintf(arc, NULL, "%s%s::", arc->name, nablaArcaneColor(arc));
}

/*****************************************************************************
 * Prise en charge d'une fonction
 *****************************************************************************/
void arcaneHookFunction(nablaMain *arc, astNode *n){
  dbg("\n\t\t[arcaneHookFunction]");
  nablaJob *fct=nablaJobNew(arc->entity);
  nablaJobAdd(arc->entity, fct);
  nablaFctFill(arc,fct,n,arc->name);
}
