///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
// * Traitement des transformations '[', '(' & ''
// * This function should be killed
// ****************************************************************************
void xHookTurnBracketsToParentheses(nablaMain* nabla,
                                    nablaJob *job,
                                    nablaVariable *var,
                                    char cnfg){
  dbg("\n\t[hookTurnBracketsToParentheses] primaryExpression hits variable");
  if (  (cnfg=='c' && var->item[0]=='n')
        ||(cnfg=='c' && var->item[0]=='f')
        ||(cnfg=='n' && var->item[0]!='n')            
        ||(cnfg=='f' && var->item[0]!='f')
        ||(cnfg=='e' && var->item[0]!='e')
        ||(cnfg=='m' && var->item[0]!='m')
        ){
    job->parse.turnBracketsToParentheses=true;
  }else{
    if (job->parse.postfix_constant==true
        && job->parse.variableIsArray==true) return;
    if (job->parse.isDotXYZ==1)
      nprintf(nabla, "/*hookTurnBracketsToParentheses_X*/", NULL);
    if (job->parse.isDotXYZ==2)
      nprintf(nabla, "/*hookTurnBracketsToParentheses_Y*/", NULL);
    if (job->parse.isDotXYZ==3)
      nprintf(nabla, "/*hookTurnBracketsToParentheses_Z*/", NULL);
    job->parse.isDotXYZ=0;
    job->parse.turnBracketsToParentheses=false;
  }
}


/***************************************************************************** 
 * Traitement des tokens SYSTEM
 *****************************************************************************/
void xHookSystem(astNode *n,
                 nablaMain *nabla,
                 const char cnf,
                 char enum_enum){
  char *itm=(cnf=='c')?"cell":(cnf=='n')?"node":"face";
  //char *etm=(enum_enum=='c')?"c":(enum_enum=='n')?"n":"f";
  if (n->tokenid == LID)           nprintf(nabla, "/*chs*/", "[%s->localId()]",itm);//asInteger
  if (n->tokenid == SID)           nprintf(nabla, "/*chs*/", "[subDomain()->subDomainId()]");
  if (n->tokenid == THIS)          nprintf(nabla, "/*chs THIS*/", NULL);
  if (n->tokenid == NBNODE)        nprintf(nabla, "/*chs NBNODE*/", NULL);
  if (n->tokenid == NBCELL)        nprintf(nabla, "/*chs NBCELL*/", NULL);
  //if (n->tokenid == INODE)         nprintf(nabla, "/*chs INODE*/", NULL);
  if (n->tokenid == BOUNDARY_CELL) nprintf(nabla, "/*chs BOUNDARY_CELL*/", NULL);
  if (n->tokenid == FATAL)         nprintf(nabla, "/*chs*/", "throw FatalErrorException");

  if (n->tokenid == BACKCELL)      nprintf(nabla, "/*chs*/", "[xs_face_cell[f+NABLA_NB_FACES*0]]");
  if (n->tokenid == BACKCELLUID)   nprintf(nabla, "/*chs*/", "[xs_face_cell[f+NABLA_NB_FACES*0]]");
  if (n->tokenid == FRONTCELL)     nprintf(nabla, "/*chs*/", "[xs_face_cell[f+NABLA_NB_FACES*1]]");
  if (n->tokenid == FRONTCELLUID)  nprintf(nabla, "/*chs*/", "[xs_face_cell[f+NABLA_NB_FACES*1]]");
    
  if (n->tokenid == NEXTNODE)      nprintf(nabla, NULL, "[n])+nextNode))");
  if (n->tokenid == PREVNODE)      nprintf(nabla, NULL, "[n])-prevNode))");
  if (n->tokenid == PREVLEFT)      nprintf(nabla, "/*chs PREVLEFT*/", "[cn.previousLeft()]");
  if (n->tokenid == PREVRIGHT)     nprintf(nabla, "/*chs PREVRIGHT*/", "[cn.previousRight()]");
  if (n->tokenid == NEXTLEFT)      nprintf(nabla, "/*chs NEXTLEFT*/", "[cn.nextLeft()]");
  if (n->tokenid == NEXTRIGHT)     nprintf(nabla, "/*chs NEXTRIGHT*/", "[cn.nextRight()]");
  //error(!0,0,"Could not switch Hook System!");
}

// ****************************************************************************
// * Prépare le nom de la variable
// ****************************************************************************
static void nvar(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  nprintf(nabla, NULL, "%s_%s",var->item,var->name);
}


// ****************************************************************************
// * xHookPowerTypeDump
// ****************************************************************************
//static char* xHookPowerTypeDump(nablaVariable *var){  return sdup(var->power_type->id);}

// ****************************************************************************
// * Tokens to variables 'CELL Job' switch
// ****************************************************************************
static void xHookTurnTokenToVariableForCellJob(nablaMain *nabla,
                                               nablaVariable *var,
                                               nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'c') return;
  
  //nprintf(nabla, "/*CellJob*/","/*CellJob*/");
  
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if ((job->parse.function_call_arguments==true)&&(var->dim==1)){
    //nprintf(nabla, "/*function_call_arguments,*/","&");
  }
  switch (var->item[0]){
  case ('c'):{
    nvar(nabla,var,job);
    nprintf(nabla, "/*CellVar*/",
            "%s",((var->power_type)?"[c][s"://xHookPowerTypeDump(var): // PowerType
                  (var->dim==0)? (isPostfixed==2)?"":"[c":
                  (enum_enum!='\0')?"[n+NABLA_NODE_PER_CELL*c":
                  (var->dim==1)?"[NABLA_NODE_PER_CELL*c":
                  "[c"));
    job->parse.variableIsArray=(var->dim==1)?true:false;
    if (job->parse.postfix_constant && job->parse.variableIsArray)
      nprintf(nabla, NULL,"+");
    else
      nprintf(nabla, NULL,"]");
    break;
  }
  case ('n'):{
    nvar(nabla,var,job);
    if (enum_enum=='f') nprintf(nabla, "/*f*/", "[");
    if (enum_enum=='n') nprintf(nabla, "/*n*/", "[xs_cell_node[n*NABLA_NB_CELLS+c]]");
    if (isPostfixed!=2 && enum_enum=='\0'){
      if (job->parse.postfix_constant==true)
        nprintf(nabla, NULL, "/*NodeVar + postfix_constant*/[");
      else nprintf(nabla, "/*NodeVar 0*/", "[xs_cell_node_");
    }
    if (isPostfixed==2 && enum_enum=='\0') {
      if (job->parse.postfix_constant==true){
        nprintf(nabla, NULL, "/*NodeVar + postfix_constant*/[");
      }else
        nprintf(nabla, "/*NodeVar 2&0*/", "/*p2&0*/[xs_cell_node[c+NABLA_NB_CELLS*");
    }
    break;
  }
  case ('f'):{
    nvar(nabla,var,job);
    if (enum_enum=='f') nprintf(nabla, "/*FaceVar*/", "[f]");
    if (enum_enum=='\0') nprintf(nabla, "/*FaceVar*/", "[cell->face");
    break;
  }
  case ('g'):{
    nprintf(nabla, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;      // GLOBAL variable
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job xHookTurnTokenToVariableForCellJob\n"));
  }
}


// ****************************************************************************
// * Tokens to variables 'NODE Job' switch
// ****************************************************************************
static void xHookTurnTokenToVariableForNodeJob(nablaMain *nabla,
                                               nablaVariable *var,
                                               nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'n') return;
  //nprintf(nabla, NULL, "/*NodeJob*/");

  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(nabla,var,job);

  switch (var->item[0]){
  case ('c'):{
    if (var->dim!=0)     nprintf(nabla, "/*CellVar dim!0*/", "/**/");
    if (enum_enum=='f')  nprintf(nabla, "/*CellVar f*/", "[");
    if (enum_enum=='n')  nprintf(nabla, "/*CellVar n*/", "[n]");
    if (enum_enum=='c' && (!isWithLibrary(nabla,with_real)&&
                           !isWithLibrary(nabla,with_real2)))
      nprintf(nabla, "/*CellVar c*/", "/*3D*/[c]");
    if (enum_enum=='c' && isWithLibrary(nabla,with_real))
      nprintf(nabla, "/*CellVar c*/", "/*1D*/[xs_node_cell[2*n+c]]");
    if (enum_enum=='c' && isWithLibrary(nabla,with_real2))
      nprintf(nabla, "/*CellVar c*/", "/*2D*/[xs_node_cell[2*n+c]]");
    //if (enum_enum=='c')  nprintf(nabla, "/*CellVar c*/", "[c]");
    if (enum_enum=='\0') nprintf(nabla, "/*CellVar 0*/", "[cell->node");
    break;
  }
  case ('n'):{
    if ((isPostfixed!=2) && enum_enum=='f')  nprintf(nabla, "/*NodeVar !2f*/", "[n]");
    if ((isPostfixed==2) && enum_enum=='f')  ;//nprintf(nabla, NULL);
    if ((isPostfixed==2) && enum_enum=='\0') nprintf(nabla, "/*NodeVar 20*/", NULL);
    if ((isPostfixed!=2) && enum_enum=='n')  nprintf(nabla, "/*NodeVar !2n*/", "[n]");
    if ((isPostfixed!=2) && enum_enum=='c')  nprintf(nabla, "/*NodeVar !2c*/", "[n]");
    if ((isPostfixed!=2) && enum_enum=='\0') nprintf(nabla, "/*NodeVar !20*/", "[n]");
    break;
  }
  case ('f'):{
    if (enum_enum=='f')  nprintf(nabla, "/*FaceVar f*/", "[f]");
    if (enum_enum=='\0') nprintf(nabla, "/*FaceVar 0*/", "[face]");
    break;
  }
  case ('g'):{
    nprintf(nabla, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n[nlambda] NODES job xHookTurnTokenToVariableForNodeJob\n"));
  }
}


// ****************************************************************************
// * Tokens to variables 'FACE Job' switch
// ****************************************************************************
static void xHookTurnTokenToVariableForFaceJob(nablaMain *nabla,
                                               nablaVariable *var,
                                               nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'f') return;
  //nprintf(nabla, "/*FaceJob*/", NULL);
  
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(nabla,var,job);
  switch (var->item[0]){
  case ('c'):{
    nprintf(nabla, "/*CellVar*/",
            "%s",
            ((var->dim==0)?
             ((enum_enum=='\0')?
              (isPostfixed==2)?"[xs_face_cell[f+NABLA_NB_FACES*":"[face->cell"
              :"[c")
             :"[cell][node->cell")); 
    break;
  }
  case ('n'):{
    if (isPostfixed!=2) nprintf(nabla, "/*NodeVar*/", "[xs_face_node(n)]");
    else nprintf(nabla, "/*NodeVar*/", "[xs_face_node[f+NABLA_NB_FACES*");
    break;
  }
  case ('f'):{
    nprintf(nabla, "/*FaceVar*/", "[f]");
    break;
  }
  case ('g'):{
    nprintf(nabla, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n[ncc] FACES job xHookTurnTokenToVariableForFaceJob\n"));
  }
}


// ***************************************************************************
// * Tokens to variables 'PARTICLE Job' switch
// ***************************************************************************
static void xHookTurnTokenToVariableForParticleJob(nablaMain *nabla,
                                                   nablaVariable *var,
                                                   nablaJob *job){
  const char cnfg=job->item[0];

  // Preliminary pertinence test
  if (cnfg != 'p') return;
  nprintf(nabla, "/*ParticleJob*/", NULL);
  
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(nabla,var,job);
  
  switch (var->item[0]){
  case ('c'):{
    exit(NABLA_ERROR|fprintf(stderr,"xHookTurnTokenToVariableForParticleJob for cell"));
    break;
  }
  case ('n'):{
    exit(NABLA_ERROR|fprintf(stderr,"xHookTurnTokenToVariableForParticleJob for cell"));
    break;
  }
  case ('f'):{
    exit(NABLA_ERROR|fprintf(stderr,"xHookTurnTokenToVariableForParticleJob for cell"));
    break;
  }
  case ('p'):{
    nprintf(nabla, NULL, "[p]");
    break;
  }
  case ('g'):{
    nprintf(nabla, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\nPARTICLE job xHookTurnTokenToVariableForParticleJob\n"));
  }
}


// ****************************************************************************
// * Tokens to variables 'Std Function' switch
// ****************************************************************************
static void xHookTurnTokenToVariableForStdFunction(nablaMain *nabla,
                                                   nablaVariable *var,
                                                   nablaJob *job){
  const char cnfg=job->item[0];
  // Preliminary pertinence test
  if (cnfg != '\0') return;
  nprintf(nabla, "/*StdJob*/", NULL);// Fonction standard
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(nabla,var,job);
  switch (var->item[0]){
  case ('c'):{
    nprintf(nabla, "/*CellVar*/", NULL);// CELL variable
    break;
  }
  case ('n'):{
    nprintf(nabla, "/*NodeVar*/", NULL); // NODE variable
    break;
  }
  case ('f'):{
    nprintf(nabla, "/*FaceVar*/", NULL);// FACE variable
    break;
  }
  case('g'):{
    nprintf(nabla, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  case ('p'):{
    nprintf(nabla, "/*ParticleVar*/", NULL);
    break;
  }
  default:
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n[ncc] StdJob xHookTurnTokenToVariableForStdFunction\n"));
  }
}


// ****************************************************************************
// * Tokens to gathered  variables
// ****************************************************************************
static bool xHookTurnTokenToGatheredVariable(nablaMain *nabla,
                                             nablaVariable *var,
                                             nablaJob *job){
  if (!var->is_gathered) return false;
  if (job->parse.enum_enum=='\0') return false;
  return true;
}

// ****************************************************************************
// * Transformation de tokens en variables selon les contextes
// ****************************************************************************
nablaVariable *xHookTurnTokenToVariable(astNode * n,
                                        nablaMain *nabla,
                                        nablaJob *job){
  nablaVariable *var=nMiddleVariableFind(nabla->variables, n->token);
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL) return NULL;
  // On récupère la variable de ce job pour ces propriétés
  nablaVariable *used=nMiddleVariableFindWithSameJobItem(nabla,job,job->used_variables, n->token);
  assert(used);
  dbg("\n\t[xHookTurnTokenToVariable] %s_%s token=%s", var->item, var->name, n->token);

  // Si on est dans une expression d'Aleph, on garde la référence à la variable  telle-quelle
  if (job->parse.alephKeepExpression){
    nprintf(nabla, "/*xHookTurnTokenToVariable*/", "/*aleph*/%s_%s", var->item, var->name);
    return var;
  }

  // Check whether this variable is being gathered
  if (xHookTurnTokenToGatheredVariable(nabla,used,job)){
    nprintf(nabla, "/*gathered variable!*/", "gathered_%s_%s",var->item,var->name);
    return var;
  }
  
  // Check whether there's job for a cell job
  xHookTurnTokenToVariableForCellJob(nabla,var,job);
  
  // Check whether there's job for a node job
  xHookTurnTokenToVariableForNodeJob(nabla,var,job);
  
  // Check whether there's job for a face job
  xHookTurnTokenToVariableForFaceJob(nabla,var,job);
  
  // Check whether there's job for a standard function
  xHookTurnTokenToVariableForStdFunction(nabla,var,job);

  // Check whether there's job for a standard function
  xHookTurnTokenToVariableForParticleJob(nabla,var,job);
  
  return var;
}


