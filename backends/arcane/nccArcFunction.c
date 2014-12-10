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
