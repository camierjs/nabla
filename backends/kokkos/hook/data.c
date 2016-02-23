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
// * Traitement des transformations '[', '(' & ''
// * This function should be killed
// ****************************************************************************
void hookTurnBracketsToParentheses(nablaMain* nabla,
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
    if (!job->parse.selection_statement_in_compound_statement){
      //nprintf(nabla, "/*turnBracketsToParentheses@true*/", "/*b2p !if: %c%c*/", cnfg, var->item[0]);
      //nprintf(nabla, "/*turnBracketsToParentheses@true*/", NULL);
    }else{
      //nprintf(nabla, "/*turnBracketsToParentheses+if@true*/", "cell_node[", cnfg, var->item[0]);
    }
    //nprintf(nabla, NULL,"/*b2p@true*/");
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
void hookSystem(astNode * n,nablaMain *arc, const char cnf, char enum_enum){
  char *itm=(cnf=='c')?"cell":(cnf=='n')?"node":"face";
  //char *etm=(enum_enum=='c')?"c":(enum_enum=='n')?"n":"f";
  if (n->tokenid == LID)           nprintf(arc, "/*chs*/", "[%s->localId()]",itm);//asInteger
  if (n->tokenid == SID)           nprintf(arc, "/*chs*/", "[subDomain()->subDomainId()]");
  if (n->tokenid == THIS)          nprintf(arc, "/*chs THIS*/", NULL);
  if (n->tokenid == NBNODE)        nprintf(arc, "/*chs NBNODE*/", NULL);
  if (n->tokenid == NBCELL)        nprintf(arc, "/*chs NBCELL*/", NULL);
  //if (n->tokenid == INODE)         nprintf(arc, "/*chs INODE*/", NULL);
  if (n->tokenid == BOUNDARY_CELL) nprintf(arc, "/*chs BOUNDARY_CELL*/", NULL);
  if (n->tokenid == FATAL)         nprintf(arc, "/*chs*/", "throw FatalErrorException");

  if (n->tokenid == BACKCELL)      nprintf(arc, "/*chs*/", "[face_cell[f+NABLA_NB_FACES*0]]");
  if (n->tokenid == BACKCELLUID)   nprintf(arc, "/*chs*/", "[face_cell[f+NABLA_NB_FACES*0]]");
  if (n->tokenid == FRONTCELL)     nprintf(arc, "/*chs*/", "[face_cell[f+NABLA_NB_FACES*1]]");
  if (n->tokenid == FRONTCELLUID)  nprintf(arc, "/*chs*/", "[face_cell[f+NABLA_NB_FACES*1]]");
    
  if (n->tokenid == NEXTNODE)      nprintf(arc, NULL, "[n])+nextNode))");
  if (n->tokenid == PREVNODE)      nprintf(arc, NULL, "[n])-prevNode))");
  if (n->tokenid == PREVLEFT)      nprintf(arc, "/*chs PREVLEFT*/", "[cn.previousLeft()]");
  if (n->tokenid == PREVRIGHT)     nprintf(arc, "/*chs PREVRIGHT*/", "[cn.previousRight()]");
  if (n->tokenid == NEXTLEFT)      nprintf(arc, "/*chs NEXTLEFT*/", "[cn.nextLeft()]");
  if (n->tokenid == NEXTRIGHT)     nprintf(arc, "/*chs NEXTRIGHT*/", "[cn.nextRight()]");
  //error(!0,0,"Could not switch Hook System!");
}


/*****************************************************************************
 * Prépare le nom de la variable
 *****************************************************************************/
static void nvar(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  nprintf(nabla, NULL, "%s_%s",
          //var->is_gathered?"gathered_":"",
          var->item,
          var->name);
}


/*****************************************************************************
 * Postfix d'un .x|y|z slon le isDotXYZ
 *****************************************************************************/
static void setDotXYZ(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  switch (job->parse.isDotXYZ){
  case(0): break;
  case(1): {nprintf(nabla, "/*setDotX+flush*/", ""); break;}
  case(2): {nprintf(nabla, "/*setDotY+flush*/", ""); break;}
  case(3): {nprintf(nabla, "/*setDotZ+flush*/", ""); break;}
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[setDotXYZ] Switch isDotXYZ error\n"));
  }
  // Flush isDotXYZ
  job->parse.isDotXYZ=0;
  job->parse.turnBracketsToParentheses=false;
}


/*****************************************************************************
 * Tokens to gathered  variables
 *****************************************************************************/
static bool hookTurnTokenToGatheredVariable(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  //nprintf(arc, NULL, "/*gathered variable '%s' ?*/",var->name);
  if (!var->is_gathered) return false;
  if (job->parse.enum_enum=='\0') return false;
  nprintf(arc, "/*gathered variable!*/", "gathered_%s_%s",var->item,var->name);
  return true;
}


/*****************************************************************************
 * Tokens to variables 'CELL Job' switch
 *****************************************************************************/
static void hookTurnTokenToVariableForCellJob(nablaMain *arc,
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
        nprintf(arc, NULL, "/*NodeVar + postfix_constant*/[");
      else nprintf(arc, "/*NodeVar 0*/", "[cell_node_");
    }
    //if (isPostfixed==2 && enum_enum=='\0') nprintf(arc, "/*NodeVar 2&0*/", "[cell_node_");
    if (job->parse.postfix_constant!=true) setDotXYZ(arc,var,job);
    if (isPostfixed==2 && enum_enum=='\0') {
      if (job->parse.postfix_constant==true){
        nprintf(arc, NULL, "/*NodeVar + postfix_constant*/[");
      }else
        nprintf(arc, "/*NodeVar 2&0*/", "/*p2&0*/[cell_node[c+NABLA_NB_CELLS*");
    }
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
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job hookTurnTokenToVariableForCellJob\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'NODE Job' switch
 *****************************************************************************/
static void hookTurnTokenToVariableForNodeJob(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'n') return;
  nprintf(arc, NULL, "/*NodeJob*/");

  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);

  switch (var->item[0]){
  case ('c'):{
    if (var->dim!=0)     nprintf(arc, "/*CellVar dim!0*/", "/**/");
    if (enum_enum=='f')  nprintf(arc, "/*CellVar f*/", "[");
    if (enum_enum=='n')  nprintf(arc, "/*CellVar n*/", "[n]");
    if (enum_enum=='c' && (!isWithLibrary(arc,with_real)&&
                           !isWithLibrary(arc,with_real2)))
      nprintf(arc, "/*CellVar c*/", "/*3D*/[c]");
    if (enum_enum=='c' && isWithLibrary(arc,with_real))
      nprintf(arc, "/*CellVar c*/", "/*1D*/[node_cell[2*n+c]]");
    if (enum_enum=='c' && isWithLibrary(arc,with_real2))
      nprintf(arc, "/*CellVar c*/", "/*2D*/[node_cell[2*n+c]]");
    //if (enum_enum=='c')  nprintf(arc, "/*CellVar c*/", "[c]");
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
  default:
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\nNODE job hookTurnTokenToVariableForNodeJob\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'FACE Job' switch
 *****************************************************************************/
static void hookTurnTokenToVariableForFaceJob(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'f') return;
  //nprintf(arc, "/*FaceJob*/", NULL);
  
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);
  switch (var->item[0]){
  case ('c'):{
    nprintf(arc, "/*CellVar*/",
            "%s",
            ((var->dim==0)?
             ((enum_enum=='\0')?
              (isPostfixed==2)?"[face_cell[f+NABLA_NB_FACES*":"[face->cell"
              :"[c")
             :"[cell][node->cell")); 
    break;
  }
  case ('n'):{
    if (isPostfixed!=2) nprintf(arc, "/*NodeVar*/", "[face_node(n)]");
    else nprintf(arc, "/*NodeVar*/", "[face_node[f+NABLA_NB_FACES*");
    break;
  }
  case ('f'):{
    nprintf(arc, "/*FaceVar*/", "[f]");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] FACES job hookTurnTokenToVariableForFaceJob\n"));
  }
}


// ***************************************************************************
// * Tokens to variables 'PARTICLE Job' switch
// ***************************************************************************
static void hookTurnTokenToVariableForParticleJob(nablaMain *nabla,
                                                        nablaVariable *var,
                                                        nablaJob *job){
  const char cnfg=job->item[0];
  //char enum_enum=job->parse.enum_enum;
  //int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'p') return;
  nprintf(nabla, "/*ParticleJob*/", NULL);
  
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(nabla,var,job);
  
  switch (var->item[0]){
  case ('c'):{
    exit(NABLA_ERROR|fprintf(stderr,"hookTurnTokenToVariableForParticleJob for cell"));
    break;
  }
  case ('n'):{
    exit(NABLA_ERROR|fprintf(stderr,"hookTurnTokenToVariableForParticleJob for cell"));
    break;
  }
  case ('f'):{
    exit(NABLA_ERROR|fprintf(stderr,"hookTurnTokenToVariableForParticleJob for cell"));
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
  default:exit(NABLA_ERROR|fprintf(stderr, "\nPARTICLE job hookTurnTokenToVariableForParticleJob\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'Std Function' switch
 *****************************************************************************/
static void hookTurnTokenToVariableForStdFunction(nablaMain *arc,
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
    nprintf(arc, "/*NodeVar*/", "/*hookTurnTokenToVariableForStdFunction*/"); // NODE variable
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
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] StdJob hookTurnTokenToVariableForStdFunction\n"));
  }
}


/*****************************************************************************
 * Transformation de tokens en variables selon les contextes dans le cas d'un '[Cell|node]Enumerator'
 *****************************************************************************/
nablaVariable *hookTurnTokenToVariable(astNode * n,
                                       nablaMain *nabla,
                                       nablaJob *job){
  nablaVariable *var=nMiddleVariableFind(nabla->variables, n->token);
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL) return NULL;

  // On récupère la variable de ce job pour ces propriétés
  //nablaVariable *used=nMiddleVariableFind(job->used_variables, n->token);
  nablaVariable *used=nMiddleVariableFindWithSameJobItem(nabla,job,job->used_variables, n->token);
  assert(used);
  
  dbg("\n\t[hookTurnTokenToVariable] %s_%s token=%s", var->item, var->name, n->token);
  if (used->is_gathered)
    dbg("\n\t[hookTurnTokenToVariable] %s_%s will be GATHERED here!", var->item, var->name);

  // Si on est dans une expression d'Aleph, on garde la référence à la variable  telle-quelle
  if (job->parse.alephKeepExpression==true){
    nprintf(nabla, "/*hookTurnTokenToVariable*/", "/*aleph*/%s_%s", var->item, var->name);
    return var;
  }

  // Set good isDotXYZ
  if (job->parse.isDotXYZ==0 &&
      strcmp(var->type,"real3")==0 &&
      job->parse.left_of_assignment_operator==true){
//    #warning Diffracting OFF
    //nprintf(nabla, NULL, "/* DiffractingNOW */");
    //job->parse.diffracting=true;
    //job->parse.isDotXYZ=job->parse.diffractingXYZ=1;
  }
  //nprintf(nabla, NULL, "\n\t/*hookTurnTokenToVariable::isDotXYZ=%d, job->parse.diffractingXYZ=%d*/", job->parse.isDotXYZ, job->parse.diffractingXYZ);

  // Check whether this variable is being gathered
  if (hookTurnTokenToGatheredVariable(nabla,used,job)){
    dbg("\n\t[hookTurnTokenToVariable] hookTurnTokenToGatheredVariable!");
    return var;
  }
  
  // Check whether there's job for a cell job
  hookTurnTokenToVariableForCellJob(nabla,var,job);
  
  // Check whether there's job for a node job
  hookTurnTokenToVariableForNodeJob(nabla,var,job);
  
  // Check whether there's job for a face job
  hookTurnTokenToVariableForFaceJob(nabla,var,job);
  
  // Check whether there's job for a standard function
  hookTurnTokenToVariableForStdFunction(nabla,var,job);

  // Check whether there's job for a standard function
  hookTurnTokenToVariableForParticleJob(nabla,var,job);
  
  return var;
}


