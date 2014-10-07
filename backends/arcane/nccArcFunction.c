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


// ****************************************************************************
// * functionGlobalVar
// ****************************************************************************
char *functionGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != '\0') return NULL; // On est bien une fonction
  if (var->item[0] != 'g') return NULL;  // On a bien affaire à une variable globale
  const bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  
  dbg("\n\t\t[functionGlobalVar] name=%s, scalar=%d, resolve=%d",var->name, scalar,resolve);

  //nprintf(arc, "/*0*/", "%s",(left_of_assignment_operator)?"":"()"); // "()" permet de récupérer les m_global_...()
  
  
  if (left_of_assignment_operator || !scalar) return "";
  return "()";
}


// ****************************************************************************
// * arcaneHookFunctionName
// ****************************************************************************
void arcaneHookFunctionName(nablaMain *arc){
  nprintf(arc, NULL, "%s%s::", arc->name, nablaArcaneColor(arc));
}


// *****************************************************************************
// * Prise en charge d'une fonction
// *****************************************************************************
void arcaneHookFunction(nablaMain *arc, astNode *n){
  dbg("\n\t\t[arcaneHookFunction]");
  nablaJob *fct=nablaJobNew(arc->entity);
  nablaJobAdd(arc->entity, fct);
  nablaFctFill(arc,fct,n,arc->name);
}
