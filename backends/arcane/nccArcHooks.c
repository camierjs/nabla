/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcaneVariable.c														  *
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
#include "nabla.tab.h"


//***************************************************************************** 
// * Traitement des tokens SYSTEM
// ****************************************************************************
void arcaneHookTurnBracketsToParentheses(nablaMain* nabla, nablaJob *job,
                                         nablaVariable *var, char cnfg){
  dbg("\n\t[actJobItemParse] primaryExpression hits Arcane variable");
  if ((  cnfg=='c' && (var->item[0]!='c'))
      ||(cnfg=='n' && (var->item[0]!='n'))           
      ||(cnfg=='f' && (var->item[0]!='f'))
      ||(cnfg=='e' && (var->item[0]!='e'))
      ||(cnfg=='m' && (var->item[0]!='m'))
      //||(cnfg=='c' && (var->item[0]!='f'))
      ){
    //nprintf(nabla, "/*tBktON*/", "/*tBktON:%c*/", var->item[0]);
    nprintf(nabla, "/*tBktON*/", NULL);
    job->parse.turnBracketsToParentheses=true;
  }else{
    nprintf(nabla, "/*tBktOFF*/", NULL);
    job->parse.turnBracketsToParentheses=false;
  }
}


// ***************************************************************************** 
// * Traitement des tokens SYSTEM
// *****************************************************************************
//#warning nablaSystem and test error (ddfv alpha)
void arcaneHookSystem(astNode * n,nablaMain *arc, const char cnf, char enum_enum){
  char *itm=(cnf=='c')?"cell":(cnf=='n')?"node":"face";
  char *etm=(enum_enum=='c')?"c":(enum_enum=='n')?"n":"f";
  //if (n->tokenid == DIESE)         nprintf(arc, "/*nablaSystem*/", "%s.index()",etm);//asInteger
  if (n->tokenid == LID)           nprintf(arc, "/*nablaSystem*/", "[%s->localId()]",itm);//asInteger
  if (n->tokenid == SID)           nprintf(arc, "/*nablaSystem*/", "[subDomain()->subDomainId()]");
  if (n->tokenid == THIS)          nprintf(arc, "/*nablaSystem THIS*/", NULL);
  if (n->tokenid == NBNODE)        nprintf(arc, "/*nablaSystem NBNODE*/", NULL);
  if (n->tokenid == NBCELL)        nprintf(arc, "/*nablaSystem NBCELL*/", NULL);
  //if (n->tokenid == INODE)         nprintf(arc, "/*nablaSystem INODE*/", NULL);
  if (n->tokenid == BOUNDARY_CELL) nprintf(arc, "/*nablaSystem BOUNDARY_CELL*/", NULL);
  if (n->tokenid == FATAL)         nprintf(arc, "/*nablaSystem*/", "throw FatalErrorException");
  if (n->tokenid == BACKCELL)      nprintf(arc, "/*nablaSystem*/", "[%s->backCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == BACKCELLUID)   nprintf(arc, "/*nablaSystem*/", "[%s->backCell().uniqueId()]",itm);
  if (n->tokenid == FRONTCELL)     nprintf(arc, "/*nablaSystem*/", "[%s->frontCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == FRONTCELLUID)  nprintf(arc, "/*nablaSystem*/", "[%s->frontCell().uniqueId()]",itm);
  if (n->tokenid == NEXTCELL)      nprintf(arc, "/*nablaSystem NEXTCELL*/", "[nextCell]");
  if (n->tokenid == PREVCELL)      nprintf(arc, "/*nablaSystem PREVCELL*/", "[prevCell]");
  if (n->tokenid == NEXTNODE)      nprintf(arc, "/*nablaSystem NEXTNODE*/", "[nextNode]");
  if (n->tokenid == PREVNODE)      nprintf(arc, "/*nablaSystem PREVNODE*/", "[prevNode]");
  if (n->tokenid == PREVLEFT)      nprintf(arc, "/*nablaSystem PREVLEFT*/", "[cn.previousLeft()]");
  if (n->tokenid == PREVRIGHT)     nprintf(arc, "/*nablaSystem PREVRIGHT*/", "[cn.previousRight()]");
  if (n->tokenid == NEXTLEFT)      nprintf(arc, "/*nablaSystem NEXTLEFT*/", "[cn.nextLeft()]");
  if (n->tokenid == NEXTRIGHT)     nprintf(arc, "/*nablaSystem NEXTRIGHT*/", "[cn.nextRight()]");
}



//*****************************************************************************
// * Transformation de tokens en variables Arcane
// * selon les contextes dans le cas d'un '[Cell|node]Enumerator'
// ****************************************************************************
nablaVariable *arcaneHookTurnTokenToVariable(astNode * n,
                                             nablaMain *arc,
                                             nablaJob *job){
  //dbg("\n\t\t[arcaneHookTurnTokenToVariable]");
  assert(job->item!=NULL);
  const char cnfg=job->item[0];
  //char enum_enum=job->parse.enum_enum;
  //bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  //int isPostfixed=job->parse.isPostfixed;
  //dbg("\n\t\t[arcaneHookTurnTokenToVariable] local variabled but var!");
  nablaVariable *var=nablaVariableFind(arc->variables, n->token);
  //dbg("\n\t\t[arcaneHookTurnTokenToVariable] local variabled!");
  
  // Si on est dans une expression d'Aleph, on garde la référence à la variable  telle-quelle
  if (job->parse.alephKeepExpression==true){
    if (var == NULL) return NULL;
    nprintf(arc, NULL, "m_%s_%s", var->item, var->name);
    return var;
  }
    
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL){
    //dbg("\n\t\t[arcaneHookTurnTokenToVariable] Pas une variable!");
    //nprintf(arc,NULL,"/*tt2a (isPostfixed=%d)(isLeft=%d)*/",isPostfixed,left_of_assignment_operator);
    return NULL;
  }
  //dbg("\n\t\t[arcaneHookTurnTokenToVariable] m_%s_%s token=%s", var->item, var->name, n->token);

  //nprintf(arc, NULL, "/*tt2a (isPostfixed=%d)(isLeft=%d)*/",isPostfixed,left_of_assignment_operator);
  if (var->gmpRank==-1){
    nprintf(arc, NULL, "m_%s_%s", var->item, var->name);
  }else{
    if (cnfg=='c')
      nprintf(arc, NULL, "m_gmp[%d][cell->localId()].get_mpz_t()", var->gmpRank);
    if (cnfg=='g')
      nprintf(arc, NULL, "m_gmp[%d].get_mpz_t()", var->gmpRank);
    return var;
  }
    
  // Lancement de tous les transformations connues au sein des Cells jobs
  nprintf(arc,NULL,cellJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobNodeVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobFaceVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobParticleVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobGlobalVar(arc,job,var),NULL);

  // Lancement de tous les transformations connues au sein des Nodes jobs
  nprintf(arc,NULL,nodeJobNodeVar(arc,job,var),NULL);
  nprintf(arc,NULL,nodeJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,nodeJobFaceVar(arc,job,var),NULL);
  nprintf(arc,NULL,nodeJobGlobalVar(arc,job,var),NULL);

  // Lancement de tous les transformations connues au sein des Faces jobs
  nprintf(arc,NULL,faceJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,faceJobNodeVar(arc,job,var),NULL);
  nprintf(arc,NULL,faceJobFaceVar(arc,job,var),NULL);
  nprintf(arc,NULL,faceJobGlobalVar(arc,job,var),NULL);

  // Lancement de tous les transformations connues au sein des Particles jobs
  nprintf(arc,NULL,particleJobParticleVar(arc,job,var),NULL);
  nprintf(arc,NULL,particleJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,particleJobGlobalVar(arc,job,var),NULL);


// Lancement de tous les transformations connues au sein des fonctions
  nprintf(arc,NULL,functionGlobalVar(arc,job,var),NULL);

  return var;
}

