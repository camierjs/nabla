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


/*****************************************************************************
 * Fonction prefix à l'ENUMERATE_*
 *****************************************************************************/
char* arcaneHookPrefixEnumerate(nablaJob *j){
  dbg("\n\t[arcaneHookPrefixEnumerate]");
  char *grp=j->scope;   // OWN||ALL
  char *rgn=j->region;  // INNER, OUTER
  char itm=j->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  char *direction=j->direction; // Direction
  if (j->xyz==NULL){
    if (itm=='c' && j->forall_item=='c'){
      return "CellCellGroup cells_pairgroup(allCells(),allCells(),IK_Node);";
    }else{
      if (isAnArcaneFamily(j->entity->main)) return "";
      char prefix[2048];
      snprintf(prefix,2048,"debug()<<\"\33[1;37m[%sEntity::%s]\33[0m\";\n\tARCANE_HYODA_SOFTBREAK(subDomain());\n\t",j->entity->name,j->name);
      return sdup(prefix);
    }
  }
  if (itm=='c'){
    char str[1024];
    if (sprintf(str,"CellDirectionMng cdm(m_cartesian_mesh->cellDirection(%s));",direction)<0)
      return NULL;
    return sdup(str);
  }
  if (itm=='n'){
    char str[1024];
    if (sprintf(str,"NodeDirectionMng ndm(m_cartesian_mesh->nodeDirection(%s));",direction)<0)
      return NULL;
    return sdup(str);
  }
  // Pour une fonction, on ne fait que le debug
  if (itm=='\0'){// && !isAnArcaneFamily(j->entity->main)){
    char prefix[2048];
    snprintf(prefix,2048,"\n\tdebug()<<\"\33[2;37m[%sEntity::%s]\33[0m\";\n\tARCANE_HYODA_SOFTBREAK(subDomain());",j->entity->name,j->name);
    return sdup(prefix);
  }

  dbg("\n\t[arcaneHookPrefixEnumerate] grp=%c rgn=%c itm=%c", grp[0], rgn[0], itm);
  nablaError("[arcaneHookPrefixEnumerate] Could not distinguish ENUMERATE!");
  return NULL;
}


// *************************************************************
// * Fonction produisant l'ENUMERATE_* avec XYZ
// *************************************************************
char *arcaneHookDumpEnumerateXYZ(nablaJob *job){
  char *grp=job->scope;   // OWN||ALL
  char *rgn=job->region;  // INNER, OUTER
  char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal|(p)articles
  // Pour une fonction, on fait rien ici
  if (itm=='\0') return "";
  if (itm=='p' && grp==NULL && rgn==NULL)
    return "ENUMERATE_PARTICLE(particle,m_particle_family->allItems())";
  if (itm=='c' && grp==NULL && rgn==NULL)   return "ENUMERATE_CELL(cell,cdm.allCells())";
  if (itm=='c' && grp==NULL && rgn[0]=='o') return "ENUMERATE_CELL(cell,cdm.outerCells())";
  if (itm=='c' && grp==NULL && rgn[0]=='i') return "ENUMERATE_CELL(cell,cdm.innerCells())";
  if (itm=='n' && grp==NULL && rgn==NULL)   return "ENUMERATE_NODE(node,ndm.allNodes())";
  if (itm=='n' && grp==NULL && rgn[0]=='o') return "ENUMERATE_NODE(node,ndm.outerNodes())";
  if (itm=='n' && grp==NULL && rgn[0]=='i') return "ENUMERATE_NODE(node,ndm.innerNodes())";
  nablaError("Could not distinguish ENUMERATE with XYZ!");
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
      item[4]=0;      //on enlève le 's'
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
// * Fonction produisant l'ENUMERATE_*
// *************************************************************
char* arcaneHookDumpEnumerate(nablaJob *job){
  char *grp=job->scope;   // OWN||ALL
  char *rgn=job->region;  // INNER, OUTER
  char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  char *xyz=job->xyz;// Direction
  char forall_item=job->forall_item;
  //#warning Should avoid testing NULL and [0]!
  if (xyz!=NULL) return arcaneHookDumpEnumerateXYZ(job);
  if (job->nb_in_item_set>0) return arcaneHookDumpAnyEnumerate(job);
  //if (funcRegion!=NULL) funcRegion[0]-=32; // Si on a une région, on inner|outer => Inner|Outer
  dbg("\n\t[arcaneHookDumpEnumerate] forall_item='%c'", forall_item);
  // Pour une fonction, on fait rien ici
  if (itm=='\0') return "";

  if (itm=='c' && forall_item=='c')           return "ENUMERATE_ITEMPAIR(Cell,Cell,cell,cells_pairgroup)";
  if (itm=='p' && grp==NULL && rgn==NULL)     return "ENUMERATE_PARTICLE(particle,m_particle_family->allItems())";
  if (itm=='c' && grp==NULL && rgn==NULL)     return "ENUMERATE_CELL(cell,allCells())";
  if (itm=='c' && grp==NULL && rgn[0]=='i')   return "ENUMERATE_CELL(cell,innerCells())";
  if (itm=='c' && grp==NULL && rgn[0]=='o')   return "ENUMERATE_CELL(cell,outerCells())";
  if (itm=='c' && grp!=NULL && rgn==NULL)     return "ENUMERATE_CELL(cell,ownCells())";
  if (itm=='n' && grp==NULL && rgn==NULL)     return "ENUMERATE_NODE(node,allNodes())";
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return "ENUMERATE_NODE(node,allCells().innerFaceGroup().nodeGroup())";
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return "ENUMERATE_NODE(node,allCells().outerFaceGroup().nodeGroup())";
  if (itm=='n' && grp!=NULL && rgn==NULL)     return "ENUMERATE_NODE(node,ownNodes())";
  if (itm=='n' && grp==NULL && rgn==NULL)     return "ENUMERATE_NODE(node,allNodes())";
  if (itm=='n' && grp!=NULL && rgn[0]=='i')   return "ENUMERATE_NODE(node,allCells().innerFaceGroup().nodeGroup().own())";
  if (itm=='n' && grp!=NULL && rgn[0]=='o')   return "ENUMERATE_NODE(node,allCells().outerFaceGroup().nodeGroup().own())";
  if (itm=='n' && grp!=NULL && rgn==NULL)     return "ENUMERATE_NODE(node,ownNodes())";
  if (itm=='f' && grp==NULL && rgn==NULL)     return "ENUMERATE_FACE(face,allFaces())";
  if (itm=='f' && grp!=NULL && rgn==NULL)     return "ENUMERATE_FACE(face,ownFaces())";
  if (itm=='f' && grp==NULL && rgn==NULL)     return "ENUMERATE_FACE(face,allFaces())";
  if (itm=='f' && grp!=NULL && rgn[0]=='o')   return "ENUMERATE_FACE(face,allCells().outerFaceGroup().own())";
  if (itm=='f' && grp==NULL && rgn[0]=='o')   return "ENUMERATE_FACE(face,allCells().outerFaceGroup())";
  if (itm=='f' && grp!=NULL && rgn[0]=='i')   return "ENUMERATE_FACE(face,allCells().innerFaceGroup().own())";
  if (itm=='f' && grp==NULL && rgn[0]=='i')   return "ENUMERATE_FACE(face,allCells().innerFaceGroup())";
  if (itm=='e' && grp==NULL && rgn==NULL)     return "";//ENUMERATE_ENV(env,m_material_mng)";
  if (itm=='m' && grp==NULL && rgn==NULL)     return "";//ENUMERATE_MAT(mat,m_material_mng)";
  nablaError("Could not distinguish ENUMERATE!");
  return NULL;
}


/*****************************************************************************
 * Fonction postfix à l'ENUMERATE_*
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
    return "";// Pour une fonction, on fait rien ici
  }
  dbg("\n\t[postfixEnumerate] job with direction, working!");  
  if (itm[0]=='c') return "\tDirCell cc(cdm.cell(*cell));\n\
\t\t__attribute__((unused)) Cell nextCell=cc.next();\n\
\t\t// Should test for !nextCell.null() to build ccn\n\
\t\t//                      DirCell ccn(cdm.cell(nextCell));\n\
\t\t//__attribute__((unused)) Cell nextNextCell=ccn.next();\n\
\t\t__attribute__((unused)) Cell prevCell=cc.previous();\n\
\t\t// Should test for !prevCell.null() to build ccp\n\
\t\t//                        DirCell ccp(cdm.cell(prevCell));\n\
\t\t//__attribute__((unused)) Cell prevPrevCell=ccp.previous();\n\
\t\t__attribute__((unused)) DirCellNode cn(cdm.cellNode(*cell));\n";  
  if (itm[0]=='n') return "\tDirNode cc(ndm.node(*node));\n\
\t\tNode rightNode=cc.next();\n\t\tNode leftNode=cc.previous();";
  dbg("\n\t[postfixEnumerate] grp=%c rgn=%c itm=%c", grp[0], rgn[0], itm[0]);
  nablaError("Could not distinguish ENUMERATE!");
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
 * Différentes actions pour un job Nabla
 *****************************************************************************/
void arcaneHookSwitchToken(astNode *n, nablaJob *job){
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

    // 'is_test' est traité dans le hook 'arcaneHookIsTest'
  case (IS): break;
  case (IS_OP_INI): break;
  case (IS_OP_END): break;
  
  case(PREPROCS):{
    nprintf(arc, "/*PREPROCS*/","/*PREPROCS*/");
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
    
  case(FORALL_INI):{ break; }
  case(FORALL_END):{
    nprintf(arc, "/*FORALL_END*/",NULL);
    //if (job->forall_item=='c' && job->item[0]=='c') nprintf(arc, "/*cell_forall_cell*/","}");
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
    int tokenid;
    char *iterator=NULL;
    if (n->next->children->tokenid==IDENTIFIER){
      dbg("\n\t[arcaneHookSwitchToken] iterator %s", n->next->children->token);
      dbg("\n\t[arcaneHookSwitchToken] n->next->next->token=%s", n->next->children->next->children->token);
      iterator=sdup(n->next->children->token);
      tokenid=n->next->children->next->children->tokenid;
    }else{
      tokenid=n->next->children->children->tokenid;
    }
    switch(tokenid){
    case(CELL):
    case(CELLS):{
      job->parse.enum_enum='c';
      if (support=='c') nprintf(arc,NULL,"ENUMERATE_SUB_ITEM(Cell,cc,cell)");
      if (support=='n') nprintf(arc,NULL,"for(CellEnumerator c(node->cells()); c.hasNext(); ++c)\n\t\t\t");
      break;
    }
    case(FACE):
    case(FACES):{
     job->parse.enum_enum='f';
      if (support=='c') nprintf(arc,NULL,"for(FaceEnumerator f(cell->faces()); f.hasNext(); ++f)\n\t\t\t");
      if (support=='n') nprintf(arc,NULL,"for(FaceEnumerator f(node->faces()); f.hasNext(); ++f)\n\t\t\t");
      break;
    }
    case(NODE):
    case(NODES):{
      job->parse.enum_enum='n';
      if (support=='c') nprintf(arc,NULL,"for(NodeEnumerator n(cell->nodes()); n.hasNext(); ++n)\n\t\t\t");
      if (support=='f') nprintf(arc,NULL,"for(NodeEnumerator n(face->nodes()); n.hasNext(); ++n)\n\t\t\t");
      break;
    }
    case(PARTICLE):
    case(PARTICLES):{
      job->parse.enum_enum='p';
      if (iterator==NULL)
        nprintf(arc,NULL,"for(ParticleEnumerator p(cellParticles(cell->localId())); p.hasNext(); ++p)");
      else
        nprintf(arc,NULL,"for(ParticleEnumerator p(cellParticles(%s->localId())); p.hasNext(); ++p)",iterator);
      break;
    }
      //case (MATERIAL): break;
    case(SET):{break;}
    
    default: nablaError("[arcaneHookSwitchToken] Could not distinguish FORALL!");
    }
    // Attention au cas où on a un @ au lieu d'un statement
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
    // ENUMERATE_NODES, forall face mais qd même un nbNode, c'est pe un backCell->nbNode()?
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
void arcaneJob(nablaMain *nabla, astNode *n){
  nablaJob *job = nMiddleJobNew(nabla->entity);
  nMiddleJobAdd(nabla->entity, job);
  nMiddleJobFill(nabla,job,n,nabla->name);
}
