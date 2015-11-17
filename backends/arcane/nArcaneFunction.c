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
  if (left_of_assignment_operator || !scalar) return "/*global_*/";
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
  nablaJob *fct=nMiddleJobNew(arc->entity);
  nMiddleJobAdd(arc->entity, fct);
  nMiddleFunctionFill(arc,fct,n,arc->name);
}
