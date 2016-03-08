///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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
#include "nabla.tab.h"



// ****************************************************************************
// * Prépare le nom de la variable
// ****************************************************************************
static void nvar(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  nprintf(nabla, "/*tt2a*/", "%s_%s", var->item, var->name);
}


// ****************************************************************************
// * Tokens to gathered  variables
// ****************************************************************************
static bool nOkinaHookTurnTokenToGatheredVariable(nablaMain *arc,
                                                 nablaVariable *var,
                                                 nablaJob *job){
  if (!var->is_gathered) return false;
  if (job->parse.enum_enum=='\0') return false;
  nprintf(arc, "/*gathered variable!*/", "gathered_%s_%s",var->item,var->name);
  return true;
}


// ****************************************************************************
// * Tokens to variables 'CELL Job' switch
// ****************************************************************************
static void nOkinaHookTurnTokenToVariableForCellJob(nablaMain *arc,
                                                    nablaVariable *var,
                                                    nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'c') return;
  
  //nprintf(arc, "/*CellJob*/","/*CellJob*/");
  
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if ((job->parse.function_call_arguments==true)&&(var->dim==1)){
    //nprintf(arc, "/*function_call_arguments,*/","&");
  }
  switch (var->item[0]){
  case ('c'):{
    nvar(arc,var,job);
    nprintf(arc, "/*CellVar*/",
            "%s",
            ((var->dim==0)? (isPostfixed==2)?"":"[c":
             (enum_enum!='\0')?"[n+8*c":
             (var->dim==1)?"[8*c":"[c"));
    job->parse.variableIsArray=(var->dim==1)?true:false;
    if (job->parse.postfix_constant==true
        && job->parse.variableIsArray==true)
      nprintf(arc, NULL,"+");
    else
      nprintf(arc, NULL,"]");
    break;
  }
  case ('n'):{
    nvar(arc,var,job);
    if (enum_enum=='f') nprintf(arc, "/*f*/", "[");
    if (enum_enum=='n') nprintf(arc, "/*n*/", "[cell_node[n*NABLA_NB_CELLS+c]]");
    if (isPostfixed!=2 && enum_enum=='\0'){
      if (job->parse.postfix_constant==true)
        nprintf(arc, "/*NodeVar + postfix_constant*/", "[");
      else
        nprintf(arc, "/*NodeVar 0*/", "[cell_node_");
    }
    if (isPostfixed==2 && enum_enum=='\0') nprintf(arc, "/*NodeVar 2&0*/", "[cell_node_");
    break;
  }
  case ('f'):{
    nvar(arc,var,job);
    if (enum_enum=='f') nprintf(arc, "/*FaceVar*/", "[f]");
    if (enum_enum=='\0') nprintf(arc, "/*FaceVar*/", "[cell->face");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;      // GLOBAL variable
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job nOkinaHookTurnTokenToVariableForCellJob\n"));
  }
}


// ****************************************************************************
// * Tokens to variables 'NODE Job' switch
// ****************************************************************************
static void nOkinaHookTurnTokenToVariableForNodeJob(nablaMain *arc,
                                                    nablaVariable *var,
                                                    nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'n') return;
  nprintf(arc, "/*NodeJob*/",NULL);

  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);

  switch (var->item[0]){
  case ('c'):{
    if (var->dim!=0)     nprintf(arc, "/*CellVar dim!0*/", "[c][c");
    if (enum_enum=='f')  nprintf(arc, "/*CellVar f*/", "[");
    if (enum_enum=='n')  nprintf(arc, "/*CellVar n*/", "[n]");
    if (enum_enum=='c' && (!isWithLibrary(arc,with_real)))  nprintf(arc, "/*CellVar c*/", "[c]");
    if (enum_enum=='c' && isWithLibrary(arc,with_real))  nprintf(arc, "/*CellVar c*/", "[node_cell[2*n+c]]");
    if (enum_enum=='\0') nprintf(arc, "/*CellVar 0*/", "[cell->node");
    break;
  }
  case ('n'):{
    if ((isPostfixed!=2) && enum_enum=='f')  nprintf(arc, "/*NodeVar !2f*/", "[n]");
    if ((isPostfixed==2) && enum_enum=='f')  ;//nprintf(arc, NULL);
    if ((isPostfixed==2) && enum_enum=='\0') nprintf(arc, "/*NodeVar 20*/", NULL);
    if ((isPostfixed!=2) && enum_enum=='n')  nprintf(arc, "/*NodeVar !2n*/", "[n]");
    if ((isPostfixed!=2) && enum_enum=='c')  nprintf(arc, "/*NodeVar !2c*/", "[n]");
    if ((isPostfixed!=2) && enum_enum=='\0') nprintf(arc, "/*NodeVar !20*/", "[n]");
    break;
  }
  case ('f'):{
    if (enum_enum=='f')  nprintf(arc, "/*FaceVar f*/", "[f]");
    if (enum_enum=='\0') nprintf(arc, "/*FaceVar 0*/", "[face]");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] NODES job nOkinaHookTurnTokenToVariableForNodeJob\n"));
  }
}


// ****************************************************************************
// * Tokens to variables 'FACE Job' switch
// ****************************************************************************
static void nOkinaHookTurnTokenToVariableForFaceJob(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'f') return;
  nprintf(arc, "/*FaceJob*/", NULL);
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);
  switch (var->item[0]){
  case ('c'):{
    nprintf(arc, "/*CellVar*/",
            "%s",
            ((var->dim==0)?
             ((enum_enum=='\0')?
              (isPostfixed==2)?"[":"[face->cell"
              :"[c")
             :"[cell][node->cell")); 
    break;
  }
  case ('n'):{
    nprintf(arc, "/*NodeVar*/", "[face->node");
    break;
  }
  case ('f'):{
    nprintf(arc, "/*FaceVar*/", "[face]");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job nOkinaHookTurnTokenToVariableForFaceJob\n"));
  }
}


// ****************************************************************************
// * Tokens to variables 'Std Function' switch
// ****************************************************************************
static void nOkinaHookTurnTokenToVariableForStdFunction(nablaMain *arc,
                                                       nablaVariable *var,
                                                       nablaJob *job){
  const char cnfg=job->item[0];
 
  // Preliminary pertinence test
  if (cnfg != '\0') return;
  nprintf(arc, "/*StdJob*/", NULL);// Fonction standard
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);
  switch (var->item[0]){
  case ('c'):{
    nprintf(arc, "/*CellVar*/", NULL);// CELL variable
    break;
  }
  case ('n'):{
    nprintf(arc, "/*NodeVar*/", NULL); // NODE variable
    break;
  }
  case ('f'):{
    nprintf(arc, "/*FaceVar*/", NULL);// FACE variable
    break;
  }
  case('g'):{
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] StdJob nOkinaHookTurnTokenToVariableForStdFunction\n"));
  }
}


// ****************************************************************************
// * Transformation de tokens en variables selon les contextes,
// * dans le cas d'un '[Cell|node]Enumerator'
// ****************************************************************************
nablaVariable *nOkinaHookVariablesTurnTokenToVariable(astNode * n,
                                                      nablaMain *arc,
                                                      nablaJob *job){
  nablaVariable *var=nMiddleVariableFind(arc->variables, n->token);
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL) return NULL;
  dbg("\n\t[nOkinaHookTurnTokenToVariable] %s_%s token=%s", var->item, var->name, n->token);

  // Set good isDotXYZ
  if (job->parse.isDotXYZ==0 && strcmp(var->type,"real3")==0 && job->parse.left_of_assignment_operator==true){
//    #warning Diffracting OFF
    //nprintf(arc, NULL, "/* DiffractingNOW */");
    //job->parse.diffracting=true;
    //job->parse.isDotXYZ=job->parse.diffractingXYZ=1;
  }
  //nprintf(arc, NULL, "\n\t/*nOkinaHookTurnTokenToVariable::isDotXYZ=%d, job->parse.diffractingXYZ=%d*/", job->parse.isDotXYZ, job->parse.diffractingXYZ);

  // Check whether this variable is being gathered
  if (nOkinaHookTurnTokenToGatheredVariable(arc,var,job)){
    return var;
  }
  
  // Check whether there's job for a cell job
  nOkinaHookTurnTokenToVariableForCellJob(arc,var,job);
  
  // Check whether there's job for a node job
  nOkinaHookTurnTokenToVariableForNodeJob(arc,var,job);
  
  // Check whether there's job for a face job
  nOkinaHookTurnTokenToVariableForFaceJob(arc,var,job);
  
  // Check whether there's job for a face job
  nOkinaHookTurnTokenToVariableForStdFunction(arc,var,job);
  return var;
}

