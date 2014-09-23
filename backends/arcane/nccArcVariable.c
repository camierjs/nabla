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



/***************************************************************************** 
 * Traitement des tokens SYSTEM
 *****************************************************************************/
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
  if (n->tokenid == INODE)         nprintf(arc, "/*nablaSystem INODE*/", NULL);
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


// ****************************************************************************
// * cellJobCellVar
// ****************************************************************************
static char *cellJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';

  dbg("\n\t\t[cellJobCellVar] scalar=%d, resolve=%d, foreach_none=%d, foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);

  if (scalar && !resolve) return "";
  if (scalar) return "[cell]";
  
  if (!scalar && !resolve) return "[cell]";
  if (!scalar && foreach_node) return "[cell][n.index()]";
  if (!scalar && foreach_face) return "[cell][f.index()]";
  if (!scalar) return "[cell]";

  error(!0,0,"Could not switch in cellJobCellVar!");
  return NULL;
}


// ****************************************************************************
// * cellJobNodeVar
// * ATTENTION à l'ordre!
// ****************************************************************************
static char *cellJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[cellJobNodeVar] scalar=%d, resolve=%d, foreach_none=%d, foreach_node=%d, foreach_face=%d, foreach_cell=%d isPostfixed=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell,job->parse.isPostfixed);

  if (resolve && foreach_none) return "[cell->node";
  if (resolve && foreach_face) return "[f->node";
  if (resolve && foreach_node) return "[n]";

  if (!resolve && foreach_none) return "[cell->node";
  if (!resolve && foreach_face) return "[";
  if (!resolve && foreach_node) return "[cell->node";
  
  error(!0,0,"Could not switch in cellJobNodeVar!");
  return NULL;
}


// ****************************************************************************
// * nodeJobNodeVar
// * ATTENTION à l'ordre!
// ****************************************************************************
static char *nodeJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[nodeJobNodeVar] scalar=%d, resolve=%d, foreach_none=%d, foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);

  if (scalar && !resolve) return "";
  
  if (scalar && resolve && foreach_face) return "[node]";
  if (scalar && resolve && foreach_node) return "[n]";
  if (scalar && resolve && foreach_cell) return "[node]";
  if (scalar && resolve && foreach_none) return "[node]";

  // On laisse passer pour le dièse
  if (!scalar && !resolve && foreach_cell) return "[node]";

  if (!scalar && resolve && foreach_cell) return "[node][c.index()]";
  if (!scalar && resolve && foreach_face) return "[node][f.index()]";
  if (!scalar && !resolve && foreach_face) return "[node]";
  
  error(!0,0,"Could not switch in nodeJobNodeVar!");
  return NULL;
}


/*****************************************************************************
 * Transformation de tokens en variables Arcane selon les contextes dans le cas d'un '[Cell|node]Enumerator'
 *****************************************************************************/
nablaVariable *arcaneHookTurnTokenToVariable(astNode * n,
                                             nablaMain *arc,
                                             nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  int isPostfixed=job->parse.isPostfixed;
  nablaVariable *var=nablaVariableFind(arc->variables, n->token);
  
  // Si on est dans une expression d'Aleph, on garde la référence à la variable  telle-quelle
  if (job->parse.alephKeepExpression==true){
    if (var == NULL) return NULL;
    nprintf(arc, NULL, "m_%s_%s", var->item, var->name);
    return var;
  }
    
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL){
    //nprintf(arc,NULL,"/*tt2a (isPostfixed=%d)(isLeft=%d)*/",isPostfixed,left_of_assignment_operator);
    return NULL;
  }
  dbg("\n\t[turnTokenToArcaneVariable] m_%s_%s token=%s", var->item, var->name, n->token);

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
    
  
  switch (cnfg){
    ///////////////
    // CELLS job
    ///////////////
  case ('c'):{
    nprintf(arc, "/*CellJob*/", NULL);
    switch (var->item[0]){
    case ('c'):{// CELL variable
      nprintf(arc,NULL,cellJobCellVar(arc,job,var),NULL);
      break;
    }
    case ('n'):{
      nprintf(arc,"/*NodeVar*/",cellJobNodeVar(arc,job,var),NULL);
      break;
    }
    case ('f'):{
      nprintf(arc, "/*FaceVar*/", NULL); // FACE variable
      if (enum_enum=='f') nprintf(arc, NULL, "[f]");
      if (enum_enum=='\0') nprintf(arc, NULL, "[cell->face");
      break;
    }
    case ('p'):{
      nprintf(arc, "/*ParticleVar*/", NULL); // PARTICLE variable
      if (enum_enum=='p') nprintf(arc, NULL, "[p]");
      if (enum_enum=='\0') nprintf(arc, NULL, "[cell->particle");
      break;
    }
    case ('g'):{
      nprintf(arc, "/*GlobalVar*/", NULL);
      nprintf(arc, NULL, "%s",(left_of_assignment_operator)?"":"()"); // "()" permet de récupérer les m_global_...()
      break;      // GLOBAL variable
    }
    default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job turnTokenToArcaneVariable\n"));
    }
    break;
  }

    ////////////////
    // NODES job
    ////////////////
  case ('n'):{
    nprintf(arc, "/*NodeJob*/", NULL);
    switch (var->item[0]){
    case ('c'):{
      nprintf(arc, "/*CellVar*/", NULL); // CELL variable
      if (var->dim!=0) nprintf(arc, NULL, "[cell][node->cell");
      if (enum_enum=='f')  nprintf(arc, NULL, "[");
      if (enum_enum=='n')  nprintf(arc, NULL, "[n]");
      if (enum_enum=='c')  nprintf(arc, NULL, "[c]");
      if (enum_enum=='\0') nprintf(arc, NULL, "[cell->node");
      break;
    }
    case ('n'):{
      nprintf(arc,NULL,nodeJobNodeVar(arc,job,var),NULL);
      break;
    }
    case ('f'):{
      nprintf(arc, "/*FaceVar*/", NULL); // FACE variable
      if (enum_enum=='f')  nprintf(arc, NULL, "[f]");
      if (enum_enum=='\0') nprintf(arc, NULL, "[face]");
      break;
    }
    case ('g'):{ nprintf(arc, "/*GlobalVar*/", NULL);// GLOBAL variable
      nprintf(arc, NULL, "%s",(left_of_assignment_operator)?"":"()"); // "()" permet de récupérer les m_global_...()
      break;
    }
    default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] NODES job turnTokenToArcaneVariable\n"));
    }
    break;
  }

    ///////////////
    // FACES job
    ///////////////
  case ('f'):{
    nprintf(arc, "/*FaceJob*/", NULL);
    switch (var->item[0]){
    case ('c'):{ nprintf(arc, "/*CellVar*/", NULL);// CELL variable
      nprintf(arc, NULL, "%s",
              ((var->dim==0)?
               ((enum_enum=='\0')?
                (isPostfixed==2)?"[":"[face->cell]"
                :"[c")
               :"[cell][node->cell")); 
      break;
    }
    case ('n'):{
      nprintf(arc, "/*NodeVar*/", NULL); // NODE variable
      if (enum_enum=='n'){
        nprintf(arc, NULL, "[n]");
      }else{
        if (isPostfixed!=2) nprintf(arc, NULL, "[face->node");
        if (isPostfixed==2) nprintf(arc, NULL, "[");
      }
      break;
    }
    case ('f'):{ nprintf(arc, "/*FaceVar*/", NULL);// FACE variable
      nprintf(arc, NULL, "[face]");
      break;
    }
    case ('g'):{ nprintf(arc, "/*GlobalVar*/", NULL); // GLOBAL variable
      nprintf(arc, "/*fg*/", "%s",(left_of_assignment_operator)?"":"()"); // "()" permet de récupérer les m_global_...()
      break;
    }
    default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] FACES job turnTokenToArcaneVariable\n"));
    }
    break;
  }

    
    ///////////////
    // PARTICLES job
    ///////////////
  case ('p'):{
    nprintf(arc, "/*ParticleJob*/", NULL);
    switch (var->item[0]){
    case ('p'):{
      nprintf(arc, "/*ParticleVar*/", NULL); // Particle variable
      nprintf(arc, NULL, "[particle]");
      break;
    }
    case ('c'):{
      nprintf(arc, "/*CellVar*/", "[particle->cell()]");// CELL variable
      break;
    }
    case ('n'):{
      nprintf(arc, "/*NodeVar*/", NULL); // NODE variable
      nprintf(arc, NULL, "\n#error ParticleJob NodeVar");
      break;
    }
    case ('f'):{
      nprintf(arc, "/*FaceVar*/", NULL);// FACE variable
      nprintf(arc, NULL, "\n#error ParticleJob FaceVar");
      break;
    }
    case ('g'):{
      nprintf(arc, "/*GlobalVar*/", NULL); // GLOBAL variable
      nprintf(arc, NULL, "%s",(left_of_assignment_operator)?"":"()"); // "()" permet de récupérer les m_global_...()
      break;
    }
    default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] PARTICLES job turnTokenToArcaneVariable\n"));
    }
    break;
  }


    
    ///////////////
    // Fonction standard
    ///////////////
  case ('\0'):{
    switch (var->item[0]){
    case ('c'):{
      nprintf(arc, NULL, NULL);// CELL variable
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
    case ('p'):{
      nprintf(arc, "/*ParticleVar*/", NULL); // NODE variable
      break;
    }
    case('g'):{
      nprintf(arc, "/*GlobalVar*/", NULL); // GLOBAL variable
      nprintf(arc, "/*0*/", "%s",(left_of_assignment_operator)?"":"()"); // "()" permet de récupérer les m_global_...()
      break;
    }
    default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] StdJob turnTokenToArcaneVariable\n"));
    }
    break;
  }
  default:
    nprintf(arc, "/*OtherJob*/", NULL);
    // Ce cas arrive pour les fonctions sans support qui jouent avec des variables Arcane
    // Pas sûr que cela doit arriver à terme, mais pour l'instant c'est le cas
    break;
  }

  return var;
}


void arcaneHookTurnBracketsToParentheses(nablaMain* nabla, nablaJob *job, nablaVariable *var, char cnfg){
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
