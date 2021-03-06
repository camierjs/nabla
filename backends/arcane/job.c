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
#include "backends/arcane/arcane.h"

extern bool adrs_it;


// ****************************************************************************
// * Fonctions pour le PREFIX de l'ENUMERATE_*
// * Cas o� il n'y a pas de XYZ
// ****************************************************************************
static char* arcaneHookPrefixEnumerateNoXYZ(nablaJob *j, const char itm){
  if (itm=='c' && j->forall_item=='c')
    return "\n\t\tCellCellGroup cells_pairgroup(allCells(),allCells(),IK_Node);\n\t\t";

  if (isAnArcaneFamily(j->entity->main)) return "";
  
  char prefix[2048];
  snprintf(prefix,2048,"\
debug()<<\"\33[1;37m[%sEntity::%s]\33[0m\";\n\
\tARCANE_HYODA_SOFTBREAK(subDomain());\n\t",j->entity->name,j->name);
  return sdup(prefix);
}


// ****************************************************************************
// * Cas o� il y a des ARROWS pour un CELL job
// ****************************************************************************
static char* arcaneHookPrefixEnumerateCellsArrows(nablaJob *j){
  char prefix[2048];
  prefix[0]=0;
   if (j->direction[0]=='N'||
       j->direction[1]=='e'||
       j->direction[3]=='s'||
       j->direction[4]=='S'||
       j->direction[5]=='w'||
       j->direction[7]=='n')
     strcat(prefix,"\n\tCellDirectionMng cdy(m_cartesian_mesh->cellDirection(MD_DirY));\n\t");
   if (j->direction[1]=='e'||
       j->direction[2]=='E'||
       j->direction[3]=='s'||
       j->direction[5]=='w'||
       j->direction[6]=='W'||
       j->direction[7]=='n')
     strcat(prefix,"\n\tCellDirectionMng cdx(m_cartesian_mesh->cellDirection(MD_DirX));\n\t");
   if (j->direction[8]=='B'||j->direction[9]=='F')
     strcat(prefix,"\n\tCellDirectionMng cdz(m_cartesian_mesh->cellDirection(MD_DirZ));\n\t");
   return sdup(prefix);   
}

// ****************************************************************************
// * Cas o� il y a des XYX pour un CELL job
// ****************************************************************************
static char* arcaneHookPrefixEnumerateXYZCells(nablaJob *j, const char *dir){
  char str[1024];
  if (sprintf(str,"\n\
\tCellDirectionMng cdm(m_cartesian_mesh->cellDirection(%s));",dir)<0)
    return NULL;
  return sdup(str);
}

// ****************************************************************************
// * Cas o� il y a des XYX pour un NODE job
// ****************************************************************************
static char* arcaneHookPrefixEnumerateXYZNodes(nablaJob *j, const char *dir){
  char str[1024];
  if (sprintf(str,"\n\
\tNodeDirectionMng ndm(m_cartesian_mesh->nodeDirection(%s));",dir)<0)
    return NULL;
  return sdup(str);
}

// ****************************************************************************
// * Cas o� il y a des XYX pour une fonction
// ****************************************************************************
static char* arcaneHookPrefixEnumerateXYZFunction(nablaJob *j, const char *dir){
  char prefix[2048];
  snprintf(prefix,2048,"\n\
\tdebug()<<\"\33[2;37m[%sEntity::%s]\33[0m\";\n\
\tARCANE_HYODA_SOFTBREAK(subDomain());",j->entity->name,j->name);
  return sdup(prefix);
}

// ****************************************************************************
// * Fonctions pour le PREFIX de l'ENUMERATE_*
// ****************************************************************************
char* arcaneHookPrefixEnumerate(nablaJob *j){
  dbg("\n\t[arcaneHookPrefixEnumerate]");
  const char *grp=j->scope;   // OWN||ALL
  const char *rgn=j->region;  // INNER, OUTER
  const char *dir=j->direction; // Direction
  const char itm=j->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
 
  if (j->xyz==NULL)
    return arcaneHookPrefixEnumerateNoXYZ(j,itm);
  
  const bool arrows = strncmp(j->xyz,"arrows",6)==0;

  // A partir d'ici, il y a au moins une direction � g�rer

  if (itm=='c' && arrows)
    return arcaneHookPrefixEnumerateCellsArrows(j);

  if (itm=='c' && !arrows)
    return arcaneHookPrefixEnumerateXYZCells(j,dir);

  if (itm=='n' && !arrows)
    return arcaneHookPrefixEnumerateXYZNodes(j,dir);
  
  // Pour une fonction, on ne fait que le debug
  if (itm=='\0')
    return arcaneHookPrefixEnumerateXYZFunction(j,dir);

  // On ne devrait pas �tre ici
  dbg("\n\t[arcaneHookPrefixEnumerate] grp=%c rgn=%c itm=%c", grp[0], rgn[0], itm);
  nablaError("[arcaneHookPrefixEnumerate] Could not distinguish ENUMERATE!");
  assert(NULL);
  return NULL;
}


// *************************************************************
// * arcaneHookDumpAnyEnumerate
// *************************************************************
static char *arcaneHookDumpAnyEnumerate(nablaJob *job){
  char *str=calloc(NABLA_MAX_FILE_NAME,1);
  sprintf(str,"%s","\
AnyItem::Family family;\
\n\tfamily ");
  for(char *p=job->item_set;*p!=0;p+=5){
    char item[6];
    item[0]=p[0]-32; // Majuscule
    item[1]=p[1];    // recopie de l'item en cours
    item[2]=p[2];
    item[3]=p[3];
    item[4]=p[4];
    item[5]=0;
    strcat(str," << AnyItem::GroupBuilder(all");strcat(str,item);strcat(str,"())");
  }
  strcat(str,";");
  strcat(str,"\n\tAnyItem::Variable");
  
  for(nablaVariable *variable=job->used_variables;
      variable != NULL; variable = variable->next){
    if (variable->item[0]=='g') continue;
    if (variable->dim>0) strcat(str,"Array");
    break;
  }
  
  strcat(str,"<Real> anyone(family);");
  for(nablaVariable *variable=job->used_variables;
       variable != NULL; variable = variable->next){
    if (variable->item[0]=='g') continue;
    for(char *p=job->item_set;*p!=0;p+=5){
      char item[6];
      item[0]=p[0]-32;// Majuscule
      item[1]=p[1];   // recopie de l'item en cours
      item[2]=p[2];
      item[3]=p[3];
      item[4]=p[4];
      item[5]=0;
      strcat(str,"\n\tanyone[all");strcat(str,item);strcat(str,"()] << m_");
      item[0]=p[0];
      item[4]=0;      //on enl�ve le 's'
      strcat(str,item);
      strcat(str,"_");
      strcat(str,variable->name);
      strcat(str,";");
    }
  }  
  strcat(str,"\n\tENUMERATE_ANY_ITEM(iitem,family.allItems())");
  return str;
}


// *************************************************************
// * arcaneHookDumpEnumerateCell
// *************************************************************
static char* arcaneHookDumpEnumerateCell(nablaJob *job, const char *grp, const char *rgn, const char nesw){
  const char jfi = job->forall_item;
  if (           jfi=='c' && nesw=='\0') return "ENUMERATE_ITEMPAIR(Cell,Cell,cell,cells_pairgroup)";

  if (nesw!='\0'){
    dbg("\n\t[arcaneHookDumpEnumerateCell] nesw='%c'\n", nesw);
    //nprintf(job->entity->main, NULL, "/*arcaneHookDumpEnumerateCell nesw=%c*/",nesw);
    // On va construire le nom du groupe de mailles sur lequelle le job it�re
    char *cellGroupName=(char*)calloc(NABLA_MAX_FILE_NAME,sizeof(char));
    strcat(cellGroupName,"ENUMERATE_CELL(cell,defaultMesh()->findGroup(\"");
    assert(cellGroupName);
    if (grp) strncat(cellGroupName,grp,3); // own || all
    if (rgn) strncat(cellGroupName,rgn,5); // inner || outer
    if (nesw=='n') strncat(cellGroupName,"North",5);
    if (nesw=='e') strncat(cellGroupName,"East",4);
    if (nesw=='s') strncat(cellGroupName,"South",5);
    if (nesw=='w') strncat(cellGroupName,"West",4);
    if (nesw=='N') strncat(cellGroupName,"OtherThanNorth",14);
    if (nesw=='E') strncat(cellGroupName,"OtherThanEast",13);
    if (nesw=='S') strncat(cellGroupName,"OtherThanSouth",14);
    if (nesw=='W') strncat(cellGroupName,"OtherThanWest",13);
    strcat(cellGroupName,"Cells\"))");
    return cellGroupName;
  }
  
  if (!grp && !rgn        && nesw=='\0') return "ENUMERATE_CELL(cell,allCells())";
  if (!grp && rgn[0]=='i' && nesw=='\0') return "ENUMERATE_CELL(cell,defaultMesh()->findGroup(\"innerCells\"))";
  if (!grp && rgn[0]=='o' && nesw=='\0') return "ENUMERATE_CELL(cell,defaultMesh()->findGroup(\"outerCells\"))";
  if ( grp && !rgn        && nesw=='\0') return "ENUMERATE_CELL(cell,ownCells())";

  assert(NULL);
  nablaError("Could not switch CELL ENUMERATE!");
  return NULL;
}

// *************************************************************
// * arcaneHookDumpEnumerateNode
// *************************************************************
static char* arcaneHookDumpEnumerateNode(nablaJob *job, const char *grp, const char *rgn, const char nesw){
  if (!grp && !rgn)        return "ENUMERATE_NODE(node,allNodes())";
  if (!grp && rgn[0]=='i') return "ENUMERATE_NODE(node,allCells().innerFaceGroup().nodeGroup())";
  if (!grp && rgn[0]=='o') return "ENUMERATE_NODE(node,allCells().outerFaceGroup().nodeGroup())";
  if ( grp && !rgn)        return "ENUMERATE_NODE(node,ownNodes())";
  if (!grp && !rgn)        return "ENUMERATE_NODE(node,allNodes())";
  if ( grp && rgn[0]=='i') return "ENUMERATE_NODE(node,allCells().innerFaceGroup().nodeGroup().own())";
  if ( grp && rgn[0]=='o') return "ENUMERATE_NODE(node,allCells().outerFaceGroup().nodeGroup().own())";
  if ( grp && !rgn)        return "ENUMERATE_NODE(node,ownNodes())";
  assert(NULL);
  nablaError("Could not switch NODE ENUMERATE!");
  return NULL;
}

// *************************************************************
// * arcaneHookDumpEnumerateFace
// *************************************************************
static char* arcaneHookDumpEnumerateFace(nablaJob *job, const char *grp, const char *rgn, const char nesw){
  if (!grp && !rgn)        return "ENUMERATE_FACE(face,allFaces())";
  if ( grp && !rgn)        return "ENUMERATE_FACE(face,ownFaces())";
  if (!grp && !rgn)        return "ENUMERATE_FACE(face,allFaces())";
  if ( grp && rgn[0]=='o') return "ENUMERATE_FACE(face,allCells().outerFaceGroup().own())";
  if (!grp && rgn[0]=='o') return "ENUMERATE_FACE(face,allCells().outerFaceGroup())";
  if ( grp && rgn[0]=='i') return "ENUMERATE_FACE(face,allCells().innerFaceGroup().own())";
  if (!grp && rgn[0]=='i') return "ENUMERATE_FACE(face,allCells().innerFaceGroup())";
  assert(NULL);
  nablaError("Could not switch FACE ENUMERATE!");
  return NULL;
}

// *************************************************************
// * Fonction produisant l'ENUMERATE_*
// *************************************************************
char* arcaneHookDumpEnumerate(nablaJob *job){
  const char *grp=job->scope;   // OWN||ALL
  const char *rgn=job->region;  // INNER, OUTER
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  const char nesw=neswOrNot(job->entity->main,job->nesw);
  //const bool xyz = (job->entity->libraries&(1<<with_cartesian))!=0;
  // Gestion des AnyItems
  if (job->nb_in_item_set>0) return arcaneHookDumpAnyEnumerate(job);
  dbg("\n\t[arcaneHookDumpEnumerate] forall_item='%c'", job->forall_item);
  if (itm=='\0') return ""; // Pour une fonction, on fait rien ici
  if (itm=='c') return arcaneHookDumpEnumerateCell(job,grp,rgn,nesw);
  if (itm=='n') return arcaneHookDumpEnumerateNode(job,grp,rgn,nesw);
  if (itm=='f') return arcaneHookDumpEnumerateFace(job,grp,rgn,nesw);
  if (itm=='p' && !grp && !rgn) return "ENUMERATE_PARTICLE(particle,m_particle_family->allItems())";
  if (itm=='e' && !grp && !rgn) return "";//ENUMERATE_ENV(env,m_material_mng)";
  if (itm=='m' && !grp && !rgn) return "";//ENUMERATE_MAT(mat,m_material_mng)";
  nablaError("Could not switch ENUMERATE!");
  return NULL;
}


/*****************************************************************************
 * Fonction postfix � l'ENUMERATE_*
 *****************************************************************************/
char* arcaneHookPostfixEnumerate(nablaJob *job){
  char *grp=job->scope;  // OWN||ALL
  char *rgn=job->region; // INNER, OUTER
  char *itm=job->item;   // (c)ells|(f)aces|(n)odes|(g)lobal
  char *xyz=job->xyz;    // Direction
  
  if (xyz==NULL){
    dbg("\n\t[postfixEnumerate] no xyz, returning");
    return "";// void ENUMERATE postfix";
  }
  
  assert(itm!=NULL);
  if (itm[0]=='\0'){
    dbg("\n\t[postfixEnumerate] function, returning");
    return "";// Pour une fonction, on ne fait rien ici
  }
  
  const bool arrows = strncmp(job->xyz,"arrows",6)==0;

  if (itm[0]=='c' && arrows){
    char postfix[1024];
    postfix[0]=0;
    
    // D�s qu'on a besoin d'un acc�s en Y
    if (job->direction[0] == 'N' ||
        job->direction[1] == 'e' ||
        job->direction[3] == 's' ||
        job->direction[4] == 'S' ||
        job->direction[5] == 'w' ||
        job->direction[7] == 'n') strcat(postfix,"\n\
\t\tDirCell cy(cdy.cell(*cell));");

    // D�s qu'on a besoin d'un acc�s en X
    if (job->direction[1] == 'e' ||
        job->direction[2] == 'E' ||
        job->direction[3] == 's' ||
        job->direction[5] == 'w' ||
        job->direction[6] == 'W' ||
        job->direction[7] == 'n') strcat(postfix,"\n\
\t\tDirCell cx(cdx.cell(*cell));");
        
    if (job->direction[0] == 'N')
      strcat(postfix,"\n\t\tCell north=cy.next();\n\t\t");

    if (job->direction[1] == 'e')
      strcat(postfix,"\n\
\t\tDirCell cyx(cdx.cell(cy.next()));\n\
\t\tCell north_east=cyx.next();\n\t\t");

    if (job->direction[2] == 'E')
      strcat(postfix,"\n\t\tCell east=cx.next();\n\t\t");
    
    if (job->direction[3] == 's')
      strcat(postfix,"\n\
\t\tDirCell cyx(cdx.cell(cy.previous()));\n\
\t\tCell south_east=cyx.next();\n\t\t");

    if (job->direction[4] == 'S')
      strcat(postfix,"\n\t\tCell south=cy.previous();\n\t\t");

    if (job->direction[5] == 'w')
      strcat(postfix,"\n\
\t\tDirCell cyx(cdx.cell(cy.previous()));\n\
\t\tCell south_west=cyx.next();\n\t\t");

    if (job->direction[6] == 'W')
      strcat(postfix,"\n\t\tCell west=cx.previous();\n\t\t");
    
    if (job->direction[7] == 'n')
      strcat(postfix,"\n\
\t\tDirCell cyx(cdx.cell(cy.next()));\n\
\t\tCell north_west=cyx.previous();\n\t\t");

    // On retourne notre cha�ne construite
    return sdup(postfix);
  }
  
  if (itm[0]=='c' && !arrows) return "\n\
\t\tDirCell cc(cdm.cell(*cell));\n\
\t\t__attribute__((unused)) Cell nextCell=cc.next();\n\
\t\t__attribute__((unused)) Cell prevCell=cc.previous();\n\t\t";
  
  if (itm[0]=='n' && !arrows) return "\n\
\t\tDirNode cc(ndm.node(*node));\n\
\t\t__attribute__((unused)) Node rightNode=cc.next();\n\
\t\t__attribute__((unused)) Node leftNode=cc.previous();\n\t\t";
  
  dbg("\n\t[postfixEnumerate] grp=%c rgn=%c itm=%c", grp[0], rgn[0], itm[0]);
  nablaError("Could not distinguish ENUMERATE!");
  assert(NULL);
  return NULL;
}


/***************************************************************************** 
 * Traitement des tokens NABLA ITEMS
 *****************************************************************************/
char* arcaneHookItem(nablaJob *job,const char j, const char itm, char enum_enum){
  const int isPostfixed = job->parse.isPostfixed;
  if (j=='c' && enum_enum=='\0' && itm=='f') return "cell->";//"/*arcaneItem-c0c*/cell";
  if (j=='c' && enum_enum=='\0' && itm=='c') return "cell";//"/*arcaneItem-c0c*/cell";
  if (j=='c' && enum_enum=='\0' && itm=='n') return "cell->";//"/*arcaneItem-c0n*/cell->";
  if (j=='c' && enum_enum=='f'  && itm=='n') return "f->";//"/*arcaneItem-cfn*/f->";
  if (j=='c' && enum_enum=='f'  && itm=='c') return "f->";//"/*arcaneItem-cfc*/f->";
  if (j=='c' && enum_enum=='n'  && itm=='n') return "cell->";
  if (j=='c' && enum_enum=='p'  && itm=='\0') return "internal()";

  if (j=='n' && enum_enum=='f'  && itm=='n') return "f->";//"/*arcaneItem-nfn*/f->";
  if (j=='n' && enum_enum=='f'  && itm=='c') return "f->";//"/*arcaneItem-nfc*/f->";
  if (j=='n' && enum_enum=='\0' && itm=='n') return "node";//"/*arcaneItem-n0n*/node";
  if (j=='n' && enum_enum=='c'  && itm=='c') return "c->"; // !! to fix for aleph1D run (not just gen)
  if (j=='n' && enum_enum=='\0'  && itm=='c') return "c->"; // !! same warning !!

  if (j=='f' && enum_enum=='\0' && itm=='f') return "face";//"/*arcaneItem-f0f*/face";
  if (isPostfixed!=2 && j=='f' && enum_enum=='\0' && itm=='n') return "/*!2f0n*/face->";//"/*arcaneItem-f0n*/face->";
  if (isPostfixed==2 && j=='f' && enum_enum=='\0' && itm=='n') return "/*2f0n*/face->";//"/*arcaneItem-f0n*/face->";
  if (j=='f' && enum_enum=='\0' && itm=='c') return "face->";//"/*arcaneItem-f0c*/face->";
  //if (j=='f' && enum_enum=='n' && itm=='n') return "/**/";

  if (j=='p' && enum_enum=='\0' && itm=='c') return "particle->";
  printf("j=%c itm=%c enum_enum=%c",j,itm,enum_enum);
  nablaError("Could not distinguish item!");
  return NULL;
}


/*****************************************************************************
 * Diff�rentes actions pour un job Nabla
 *****************************************************************************/
void arcaneHookSwitchToken(node *n, nablaJob *job){
  nablaMain *arc=job->entity->main;
  const char support=job->item[0];
  const char forall=job->parse.enum_enum;
  
  //if (n->tokenid!=0) dbg("\n\t[arcaneHookSwitchToken] n->token=%s (id:%d)", n->token, n->tokenid);

  // Dump des tokens possibles
  switch(n->tokenid){

  case (ITERATION):{
    nprintf(arc, "/*ITERATION*/", "subDomain()->commonVariables().globalIteration()");
    break;
  }

    // 'is_test' est trait� dans le hook 'arcaneHookIsTest'
  case (IS): break;
  case (IS_OP_INI): break;
  case (IS_OP_END): break;
  
  case(PREPROCS):{
    nprintf(arc, "/*PREPROCS*/","/*PREPROCS*/");
    break;
  }

  case(K_OFFSET):{
    nprintf(arc, "/*DIESE*/", NULL);
    if (support=='c' && forall!='\0') nprintf(arc, NULL, "[%c.index()]", forall);
    if (support=='n' && forall!='\0') nprintf(arc, NULL, "[%c.index()]", forall);
    if (support=='f' && forall!='\0') nprintf(arc, NULL, "[%c.index()]", forall);
    break;
  }
    
  case(DIESE):{
    nprintf(arc, "/*DIESE*/", NULL);
    if (support=='c' && forall!='\0') nprintf(arc, NULL, "%c.index()", forall);
    if (support=='n' && forall!='\0') nprintf(arc, NULL, "%c.index()", forall);
    if (support=='f' && forall!='\0') nprintf(arc, NULL, "%c.index()", forall);
    break;
  }

  case(LIB_ALEPH):{
    nprintf(arc, "/*LIB_ALEPH*/",NULL);
    break;
  }
  case(ALEPH_RHS):{
    nprintf(arc, "/*ALEPH_RHS*/","rhs");
    // On utilise le 'alephKeepExpression' pour indiquer qu'on est sur des vecteurs
    job->parse.alephKeepExpression=true;
    break;
  }
  case(ALEPH_LHS):{
    nprintf(arc, "/*ALEPH_LHS*/","lhs");
    // On utilise le 'alephKeepExpression' pour indiquer qu'on est sur des vecteurs
    job->parse.alephKeepExpression=true;
    break;
  }
  case(ALEPH_MTX):{
    nprintf(arc, "/*ALEPH_MTX*/","mtx");
    break;
  }
  case(ALEPH_RESET):{ nprintf(arc, "/*ALEPH_RESET*/",".reset()"); break;}
  case(ALEPH_SOLVE):{ nprintf(arc, "/*ALEPH_SOLVE*/","alephSolve()"); break;}
  case(ALEPH_SET):{
    // Si c'est un vecteur et setValue, on le laisse
    if (job->parse.alephKeepExpression==true){
    }else{
      job->parse.alephKeepExpression=true;
    }
    nprintf(arc, "/*ALEPH_SET*/",".setValue");
    break;
  }
  case(ALEPH_GET):{
    // Si c'est un vecteur et getValue, on le laisse pas
    if (job->parse.alephKeepExpression==true){
      //job->parse.alephKeepExpression=false;
    }
    nprintf(arc, "/*ALEPH_GET*/",".getValue");
    break;
  }
  case(ALEPH_ADD_VALUE):{
    // Si c'est un vecteur et addValue, on le laisse pas
    if (job->parse.alephKeepExpression==true){
      job->parse.alephKeepExpression=true;
    }else{
      job->parse.alephKeepExpression=true;
    }
    nprintf(arc, "/*ALEPH_ADD_VALUE*/",".addValue");
    break;
  }
  case(ALEPH_NEW_VALUE):{
    job->parse.alephKeepExpression=false;
    nprintf(arc, "/*ALEPH_NEW_VALUE*/",".newValue");
    break;
  }
    
  case(IOR_OP): { fprintf(arc->entity->src, " || "); break; } 
  case(AND_OP): { fprintf(arc->entity->src, " && "); break; } 

  case(CELL):{
    /* permet par exemple en particle de choper la cell sous-jacente */
    fprintf(arc->entity->src, "cell ");
   break;
  } 

  case(INT64):{
    //fprintf(arc->entity->src, "uint64_t ");
    fprintf(arc->entity->src, "Int64 ");
    break;
  } 

  case(UIDTYPE):{
    fprintf(arc->entity->src, "Int64 ");
    break;
  }

  case(COMPOUND_JOB_INI):{
    nprintf(arc, "/*COMPOUND_JOB_INI:*/",NULL);
    break;
  }
  case(COMPOUND_JOB_END):{
    nprintf(arc, "/*COMPOUND_JOB_END*/",NULL);
    break;
  }
    
  case(FORALL_INI):{ // le FORALL �vite de passer par lui
    nprintf(arc, NULL,"/*FORALL_INI*/");
    printf("/*FORALL_INI*/");
    break;
  }
    
  case(FORALL_END):{
    //nprintf(arc, NULL,"} // FORALL_END\n\t\t");
    nprintf(arc, NULL,"}\n\t\t");
    job->parse.enum_enum='\0';
    break;
  }
    
  case(FORALL_NODE_INDEX):{
    nprintf(arc, "/*FORALL_NODE_INDEX*/","[n.index()]");
    break;
  }
    
  case(FORALL_CELL_INDEX):{
    nprintf(arc, NULL, "[c.index()]");
    break;
  }
    
  case(FORALL):{
    dbg("\n\t\t\t\t\t[arcaneHookSwitchToken] forall_range");
    node *forall_range = n->next;
    assert(forall_range);
    job->enum_enum_node=forall_range;
    
    const node *nesw = dfsHit(forall_range,ruleToId(rule_nabla_nesw));
    const node *region = dfsHit(forall_range->children,ruleToId(rule_nabla_region));
    char *isFaceBfr=(char*)calloc(NABLA_MAX_FILE_NAME,sizeof(char));
    strcat(isFaceBfr,"\n\t\t\tif (!faceIs");
    const char *face_region = (region && region->children->tokenid==OUTER)?"Outer":NULL;
    if (face_region) strcat(isFaceBfr,face_region);
    const char *face_nesw = (nesw && nesw->children->tokenid==NORTH)?"North":
      (nesw && nesw->children->tokenid==EAST)?"East":
      (nesw && nesw->children->tokenid==SOUTH)?"South":
      (nesw && nesw->children->tokenid==WEST)?"West":NULL;
    if (face_nesw) strcat(isFaceBfr,face_nesw);
    strcat(isFaceBfr,"(f)) continue;");
    if (!face_nesw && !face_region) isFaceBfr="";
   
    dbg("\n\t\t\t\t\t[arcaneHookSwitchToken] dfsHit switch");
    node *forall_switch = dfsHit(forall_range->children,ruleToId(rule_forall_switch));
    assert(forall_switch);
    
    // R�cup�ration du tokenid du forall_switch
    const int tokenid = forall_switch->children->tokenid;
    dbg("\n\t\t\t\t\t[arcaneHookSwitchToken] tokenid=%d",tokenid);

    char *iterator=NULL;

    switch(tokenid){
    case(CELL):
    case(CELLS):{
      job->parse.enum_enum='c';
      if (support=='c') nprintf(arc,NULL,"ENUMERATE_SUB_ITEM(Cell,cc,cell){\n\t\t\t");
      if (support=='n') nprintf(arc,NULL,"for(CellEnumerator c(node->cells()); c.hasNext(); ++c){\n\t\t\t");
      break;
    }
    case(FACE):
    case(FACES):{
     job->parse.enum_enum='f';
     if (support=='c') nprintf(arc,NULL,"for(FaceEnumerator f(cell->faces()); f.hasNext(); ++f){%s\n\t\t\t",isFaceBfr);
      if (support=='n') nprintf(arc,NULL,"for(FaceEnumerator f(node->faces()); f.hasNext(); ++f){\n\t\t\t");
      break;
    }
    case(NODE):
    case(NODES):{
      job->parse.enum_enum='n';
      if (support=='c') nprintf(arc,NULL,"for(NodeEnumerator n(cell->nodes()); n.hasNext(); ++n){\n\t\t\t");
      if (support=='f') nprintf(arc,NULL,"for(NodeEnumerator n(face->nodes()); n.hasNext(); ++n){\n\t\t\t");
      break;
    }
    case(PARTICLE):
    case(PARTICLES):{
      job->parse.enum_enum='p';
      if (iterator==NULL)
        nprintf(arc,NULL,"for(ParticleEnumerator p(cellParticles(cell->localId())); p.hasNext(); ++p){\n\t\t\t");
      else
        nprintf(arc,NULL,"for(ParticleEnumerator p(cellParticles(%s->localId())); p.hasNext(); ++p){\n\t\t\t",iterator);
      break;
    }
      //case (MATERIAL): break;
    case(SET):{break;}
    
    default: nablaError("[arcaneHookSwitchToken] Could not distinguish FORALL!");
    }
    // Attention au cas o� on a un @ au lieu d'un statement
    if (n->next->next->tokenid == AT)
      nprintf(arc, "/* Found AT */", "knAt");
//#warning Skip du 'forall_range' qui renseigne sur le type de forall
    *n=*n->next->next;
    break;
  }

  case('}'):{
    nprintf(arc, NULL, "}\n\t"); 
    break;
  }
    
  case('['):{
    if (job->parse.turnBracketsToParentheses==true)
      nprintf(arc, NULL, "(");
    else
      nprintf(arc, NULL, "[");
    break;
  }

  case(']'):{
    if (job->parse.turnBracketsToParentheses==true){
      nprintf(arc, "/*tBktOFF*/", "/*tBktOFF*/)]");
      job->parse.turnBracketsToParentheses=false;
    }else{
      nprintf(arc, NULL, "/*!tBktOFF*/]");
    }
    nprintf(arc, "/* FlushingIsPostfixed */", NULL);
    job->parse.isPostfixed=0;
    break;
  }

  case (FATAL):{
    nprintf(arc, NULL, "throw FatalErrorException");
    break;
  }
    
  case (BOUNDARY_CELL):{
    if (forall=='f' && support=='c') nprintf(arc, NULL, "f->boundaryCell()");
    if (forall=='\0' && support=='c') nprintf(arc, NULL, "face->boundaryCell()");
    if (forall=='\0' && support=='f') nprintf(arc, NULL, "face->boundaryCell()");
   break;
  }
    
  case (BACKCELL):{
    if (forall=='f' && support=='c') nprintf(arc, NULL, "f->backCell()");
    if (forall=='\0' && support=='c') nprintf(arc, NULL, "face->backCell()");
    if (forall=='\0' && support=='f') nprintf(arc, NULL, "face->backCell()");
    if (forall=='f' && support=='n') nprintf(arc, NULL, "f->backCell()");
    break;
  }
    
  case (BACKCELLUID):{
    if (support=='f') nprintf(arc, NULL, "face->backCell().uniqueId()");
    break;
  }
    
  case (FRONTCELL):{
    if (forall=='f' && support=='c') nprintf(arc, NULL, "f->frontCell()");
    if (forall=='\0' && support=='f') nprintf(arc, NULL, "face->frontCell()");
    if (forall=='f' && support=='n') nprintf(arc, NULL, "f->frontCell()");
    break;
  }
    
  case (FRONTCELLUID):{
    if (support=='f')
      nprintf(arc, NULL, "face->frontCell().uniqueId()");
    break;
  }
    
  case (NBCELL):{
    if (forall=='c'  && support=='n') nprintf(arc, NULL, "node->nbCell()");
    if (forall=='f'  && support=='c') nprintf(arc, NULL, "f->nbCell()");
    if (forall=='f'  && support=='n') nprintf(arc, NULL, "f->nbCell()");
    if (forall=='\0' && support=='c') nprintf(arc, NULL, "face->nbCell()");
    if (forall=='\0' && support=='f') nprintf(arc, NULL, "face->nbCell()");
    if (forall=='\0' && support=='n') nprintf(arc, NULL, "node->nbCell()");
    break;
  }
    
  case (NBNODE):{
    if (support=='c' && forall=='\0') nprintf(arc, NULL, "cell->nbNode()");
    if (support=='c' && forall=='f') nprintf(arc, NULL, "cell->nbNode()");
    if (support=='c' && forall=='n') nprintf(arc, NULL, "cell->nbNode()");
    if (support=='n' && forall=='c') nprintf(arc, NULL, "c->nbNode()");
    // ENUMERATE_NODES, forall face mais qd m�me un nbNode, c'est pe un backCell->nbNode()?
    if (support=='n' && forall=='f') nprintf(arc, NULL, "nbNode");
    if (support=='f') nprintf(arc, NULL, "face->nbNode()");
    break;
  }
    
    //case (INODE):{ if (support=='c') nprintf(arc, NULL, "cell->node"); break; }    

  case (XYZ):{ nprintf(arc, "/*XYZ*/", NULL); break;}
  case (NEXTCELL):{ nprintf(arc, "/*NEXTCELL*/", "nextCell"); break;}
  case (PREVCELL):{ nprintf(arc, "/*PREVCELL*/", "prevCell"); break;}
  case (NEXTNODE):{ nprintf(arc, "/*NEXTNODE*/", "nextNode"); break; }
  case (PREVNODE):{ nprintf(arc, "/*PREVTNODE*/", "prevNode"); break; }
  case (PREVLEFT):{ nprintf(arc, "/*PREVLEFT*/", "cn.previousLeft()"); break; }
  case (PREVRIGHT):{ nprintf(arc, "/*PREVRIGHT*/", "cn.previousRight()"); break; }
  case (NEXTLEFT):{ nprintf(arc, "/*NEXTLEFT*/", "cn.nextLeft()"); break; }
  case (NEXTRIGHT):{ nprintf(arc, "/*NEXTRIGHT*/", "cn.nextRight()"); break; }

    // Gestion du THIS
  case (THIS):{
    if (support=='c') nprintf(arc, "/*THIS+c*/", "cell");
    if (support=='n') nprintf(arc, "/*THIS+n*/", "node");
    if (support=='f') nprintf(arc, "/*THIS+f*/", "*face");
    if (support=='p') nprintf(arc, "/*THIS+p*/", "particle");
    break;
  }
    
  case (SID):{
    nprintf(arc, NULL, "subDomain()->subDomainId()");
    break;
  }
  case (LID):{
    if (support=='c') nprintf(arc, NULL, "cell->localId()");
    if (support=='n') nprintf(arc, NULL, "node->localId()");
    break;
  }
  case (UID):{
    if (forall=='\0' && support=='c') nprintf(arc, NULL, "(*cell)->uniqueId().asInteger()");
    if (forall=='\0' && support=='n') nprintf(arc, NULL, "node->uniqueId().asInteger()");
    if (forall=='\0' && support=='f') nprintf(arc, NULL, "face->uniqueId().asInteger()");
    if (forall=='\0' && support=='p') nprintf(arc, NULL, "particle->uniqueId().asInteger()");
    if (forall=='p' && support=='c') nprintf(arc, NULL, "cell->uniqueId().asInteger()");
    if (forall=='c' && support=='c') nprintf(arc, NULL, "cell->uniqueId().asInteger()");
    if (forall=='c' && support=='n') nprintf(arc, NULL, "c->uniqueId().asInteger()");
    if (forall=='c' && support=='f') nprintf(arc, NULL, "c->uniqueId().asInteger()");
    if (forall=='c' && support=='p') nprintf(arc, NULL, "c->uniqueId().asInteger()");
    if (forall=='n' && support=='c') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (forall=='n' && support=='n') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (forall=='n' && support=='f') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (forall=='n' && support=='p') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (forall=='f' && support=='c') nprintf(arc, NULL, "f->uniqueId().asInteger()");
    if (forall=='f' && support=='n') nprintf(arc, NULL, "f->uniqueId().asInteger()");
    if (forall=='f' && support=='f') nprintf(arc, NULL, "f->uniqueId().asInteger()");
    if (forall=='f' && support=='p') nprintf(arc, NULL, "f->uniqueId().asInteger()");
    nprintf(arc, "/*uid*/", NULL);
    break;
  }
  case (AT):{ nprintf(arc, NULL, "; knAt"); break; }
  case ('='):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "="); break; }
  case (RSH_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, ">>="); break; }
  case (LSH_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "<<="); break; }
  case (ADD_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "+="); break; }
  case (SUB_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "-="); break; }
  case (MUL_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "*="); break; }
  case (DIV_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "/="); break; }
  case (MOD_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "%%="); break; }
  case (AND_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "&="); break; }
  case (XOR_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "^="); break; }
  case (IOR_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(arc, NULL, "|="); break; }
  case (LSH_OP):{ job->parse.left_of_assignment_operator=true; nprintf(arc, NULL, "<<"); break; }
    
  case (RETURN):{
    nprintf(arc, NULL, "\n\t\t}/* des sources */\n\t}/* de l'ENUMERATE */\n\treturn ");
    job->parse.got_a_return=true;
    job->parse.got_a_return_and_the_semi_colon=false;
    break;
  }
    
  case ('{'):{nprintf(arc, NULL, "{\n\t\t"); break; }
    
  case (')'):{ nprintf(arc, NULL, ")"); break; }

  case(END_OF_CALL):{
    job->parse.function_call_arguments=false;          
    break;
  }
    
  case (';'):{
    job->parse.turnBracketsToParentheses=false;
    job->parse.alephKeepExpression=false;
    nprintf(arc, "/*;+tBktOFF*/", ";\n\t\t");
    if (job->parse.got_a_return)
      job->parse.got_a_return_and_the_semi_colon=true;
    break;
  }
    
  case (ADRS_IN):{
    nprintf(arc, "/*ADRS_IN*/", NULL);
    adrs_it=true;
    break;
  }
  case (ADRS_OUT):{
    nprintf(arc, "/*ADRS_OUT*/", NULL);
    adrs_it=false;
    break;
  }
    
  case ('&'):{
    if (adrs_it==true){
      nprintf(arc, "/*&+adrs_it*/", NULL);
    }else{
      nprintf(arc, "/*& std*/", "&");
    }
    break;
  }
    
  default:{
    if (n->token) nprintf(arc, NULL, "%s",n->token);
    nMiddleInsertSpace(arc,n);
  }
    break;
  }
}


/***************************************************************************** 
 * nccArcaneJob
 *****************************************************************************/
void arcaneJob(nablaMain *nabla, node *n){
  nablaJob *job = nMiddleJobNew(nabla->entity);
  nMiddleJobAdd(nabla->entity, job);
  nMiddleJobFill(nabla,job,n,nabla->name);
}
