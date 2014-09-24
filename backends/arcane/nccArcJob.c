/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcaneJob.c											       			  *
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
 * Fonction prefix à l'ENUMERATE_*
 *****************************************************************************/
char* arcaneHookPrefixEnumerate(nablaJob *j){
  char *grp=j->group;   // OWN||ALL
  char *rgn=j->region;  // INNER, OUTER
  char itm=j->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  char *drctn=j->drctn; // Direction
  if (j->xyz==NULL){
    if (itm=='c' && j->foreach_item=='c'){
      return "CellCellGroup cells_pairgroup(allCells(),allCells(),IK_Node);";
    }else{
      char prefix[2048];
      snprintf(prefix,2048,"debug()<<\"\33[1;37m[%sEntity::%s]\33[0m\";\n\tARCANE_HYODA_SOFTBREAK(subDomain());",j->entity->name,j->name);
      return strdup(prefix);
    }
  }
  if (itm=='c'){
    char str[1024];
    if (sprintf(str,"CellDirectionMng cdm(m_cartesian_mesh->cellDirection(%s));",drctn)<0)
      return NULL;
    return strdup(str);
  }
  if (itm=='n'){
    char str[1024];
    if (sprintf(str,"NodeDirectionMng ndm(m_cartesian_mesh->nodeDirection(%s));",drctn)<0)
      return NULL;
    return strdup(str);
  }
  // Pour une fonction, on ne fait que le debug
  if (itm=='\0'){
    char prefix[2048];
    snprintf(prefix,2048,"\n\tdebug()<<\"\33[2;37m[%sEntity::%s]\33[0m\";\n\tARCANE_HYODA_SOFTBREAK(subDomain());",j->entity->name,j->name);
    return strdup(prefix);
  }

  dbg("\n\t[arcaneHookPrefixEnumerate] grp=%c rgn=%c itm=%c", grp[0], rgn[0], itm);
  error(!0,0,"[arcaneHookPrefixEnumerate] Could not distinguish ENUMERATE!");
  return NULL;
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_* avec XYZ
 *****************************************************************************/
char *arcaneHookDumpEnumerateXYZ(nablaJob *job){
  char *grp=job->group;   // OWN||ALL
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
  error(!0,0,"Could not distinguish ENUMERATE with XYZ!");
  return NULL;
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_*
 *****************************************************************************/
char* arcaneHookDumpEnumerate(nablaJob *job){
  char *grp=job->group;   // OWN||ALL
  char *rgn=job->region;  // INNER, OUTER
  char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  char *xyz=job->xyz;// Direction
  char foreach_item=job->foreach_item;
//#warning Should avoid testing NULL and [0]!
  if (xyz!=NULL) return arcaneHookDumpEnumerateXYZ(job);
  //if (funcRegion!=NULL) funcRegion[0]-=32; // Si on a une région, on inner|outer => Inner|Outer
  dbg("\n\t[arcaneHookDumpEnumerate] foreach_item='%c'", foreach_item);
  // Pour une fonction, on fait rien ici
  if (itm=='\0') return "";

  if (itm=='c' && foreach_item=='c')          return "ENUMERATE_ITEMPAIR(Cell,Cell,cell,cells_pairgroup)";
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
  error(!0,0,"Could not distinguish ENUMERATE!");
  return NULL;
}


/*****************************************************************************
 * Fonction postfix à l'ENUMERATE_*
 *****************************************************************************/
char* arcaneHookPostfixEnumerate(nablaJob *job){
  char *grp=job->group; // OWN||ALL
  char *rgn=job->region; // INNER, OUTER
  char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  char *xyz=job->xyz;// Direction
  if (xyz==NULL) return "";// void ENUMERATE postfix";
// Pour une fonction, on fait rien ici
  if (itm=='\0') return "";
  if (itm=='c') return "\tDirCell cc(cdm.cell(*cell));\n\
\t\t__attribute__((unused)) Cell nextCell=cc.next();\n\
\t\t// Should test for !nextCell.null() to build ccn\n\
\t\t//                      DirCell ccn(cdm.cell(nextCell));\n\
\t\t//__attribute__((unused)) Cell nextNextCell=ccn.next();\n\
\t\t__attribute__((unused)) Cell prevCell=cc.previous();\n\
\t\t// Should test for !prevCell.null() to build ccp\n\
\t\t//                        DirCell ccp(cdm.cell(prevCell));\n\
\t\t//__attribute__((unused)) Cell prevPrevCell=ccp.previous();\n\
\t\t__attribute__((unused)) DirCellNode cn(cdm.cellNode(*cell));\n";  
  if (itm=='n') return "\tDirNode cc(ndm.node(*node));\n\t\tNode rightNode=cc.next();\n\t\tNode leftNode=cc.previous();";
  dbg("\n\t[postfixEnumerate] grp=%c rgn=%c itm=%c", grp[0], rgn[0], itm);
  error(!0,0,"Could not distinguish ENUMERATE!");
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

  if (j=='f' && enum_enum=='\0' && itm=='f') return "face";//"/*arcaneItem-f0f*/face";
  if (isPostfixed!=2 && j=='f' && enum_enum=='\0' && itm=='n') return "face->";//"/*arcaneItem-f0n*/face->";
  if (isPostfixed==2 && j=='f' && enum_enum=='\0' && itm=='n') return "face->";//"/*arcaneItem-f0n*/face->";
  if (j=='f' && enum_enum=='\0' && itm=='c') return "face->";//"/*arcaneItem-f0c*/face->";
  //if (j=='f' && enum_enum=='n' && itm=='n') return "/**/";

  if (j=='p' && enum_enum=='\0' && itm=='c') return "particle->";
  printf("j=%c itm=%c enum_enum=%c",j,itm,enum_enum);
  error(!0,0,"Could not distinguish arcaneItem!");
  return NULL;
}


/*****************************************************************************
 * Différentes actions pour un job Nabla
 *****************************************************************************/
void arcaneHookSwitchToken(astNode *n, nablaJob *job){
  nablaMain *arc=job->entity->main;
  const char support=job->item[0];
  const char foreach=job->parse.enum_enum;
  
  if (n->tokenid!=0)
    dbg("\n\t[arcaneHookSwitchToken] n->token=%s", n->token);

  // Dump des tokens possibles
  switch(n->tokenid){
    
  case(DIESE):{
    nprintf(arc, "/*DIESE*/", NULL);
    if (support=='c' && foreach!='\0') nprintf(arc, NULL, "%c.index()", foreach);
    if (support=='n' && foreach!='\0') nprintf(arc, NULL, "%c.index()", foreach);
    if (support=='f' && foreach!='\0') nprintf(arc, NULL, "%c.index()", foreach);
    break;
  }
  case(PREFIX_PRIMARY_CONSTANT):{
    nprintf(arc, "/*PREFIX_PRIMARY_CONSTANT*/", NULL);
    break;
  }
  case(POSTFIX_PRIMARY_CONSTANT):{
    nprintf(arc, "/*POSTFIX_PRIMARY_CONSTANT*/", NULL);
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
    
  case(FOREACH_INI):{ break; }
  case(FOREACH_END):{
    nprintf(arc, "/*FOREACH_END*/",NULL);
    //if (job->foreach_item=='c' && job->item[0]=='c') nprintf(arc, "/*cell_foreach_cell*/","}");
    job->parse.enum_enum='\0';
    break;
  }
    
  case(FOREACH_NODE_INDEX):{
    nprintf(arc, "/*FOREACH_NODE_INDEX*/","[n.index()]");
    break;
  }
    
  case(FOREACH_CELL_INDEX):{
    nprintf(arc, NULL, "[c.index()]");
    break;
  }
    
  case(FOREACH):{
    int tokenid;
    char *iterator=NULL;
    if (n->next->tokenid==IDENTIFIER){
      dbg("\n\t[arcaneHookSwitchToken] n->next->token is IDENTIFIER %s", n->next->token);
      dbg("\n\t[arcaneHookSwitchToken] n->next->next->token=%s", n->next->next->token);
      iterator=strdup(n->next->token);
      tokenid=n->next->next->tokenid;
    }else{
      tokenid=n->next->children->tokenid;
    }
    switch(tokenid){
    case(CELL):{
      job->parse.enum_enum='c';
      if (support=='c') nprintf(arc, NULL, "ENUMERATE_SUB_ITEM(Cell,cc,cell)");
      if (support=='n') nprintf(arc, NULL, "for(CellEnumerator c(node->cells()); c.hasNext(); ++c)\n\t\t\t");
      break;
    }
    case(FACE):{
      job->parse.enum_enum='f';
      if (support=='c') nprintf(arc, NULL, "for(FaceEnumerator f(cell->faces()); f.hasNext(); ++f)\n\t\t\t");
      if (support=='n') nprintf(arc, NULL, "for(FaceEnumerator f(node->faces()); f.hasNext(); ++f)\n\t\t\t");
      break;
    }
    case(NODE):{
      job->parse.enum_enum='n';
      if (support=='c') nprintf(arc, NULL, "for(NodeEnumerator n(cell->nodes()); n.hasNext(); ++n)\n\t\t\t");
      if (support=='f') nprintf(arc, NULL, "for(NodeEnumerator n(face->nodes()); n.hasNext(); ++n)\n\t\t\t");
      break;
    }
    case(PARTICLE):{
      job->parse.enum_enum='p';
      if (iterator==NULL)
        nprintf(arc, NULL, "for(ParticleEnumerator p(cellParticles(cell->localId())); p.hasNext(); ++p)");
      else
        nprintf(arc, NULL, "for(ParticleEnumerator p(cellParticles(%s->localId())); p.hasNext(); ++p)",iterator);
      break;
    }
    case (MATERIAL): break;
    
    default: error(!0,0,"[arcaneHookSwitchToken] Could not distinguish FOREACH!");
    }
    // Attention au cas où on a un @ au lieu d'un statement
    if (n->next->next->tokenid == AT)
      nprintf(arc, "/* Found AT */", "knAt");
    // On skip le 'nabla_item' qui nous a renseigné sur le type de foreach
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
      nprintf(arc, "/*tBktOFF*/", ")]");
      job->parse.turnBracketsToParentheses=false;
    }else{
      nprintf(arc, NULL, "]");
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
    if (foreach=='f' && support=='c') nprintf(arc, NULL, "f->boundaryCell()");
    if (foreach=='\0' && support=='c') nprintf(arc, NULL, "face->boundaryCell()");
    if (foreach=='\0' && support=='f') nprintf(arc, NULL, "face->boundaryCell()");
   break;
  }
    
  case (BACKCELL):{
    if (foreach=='f' && support=='c') nprintf(arc, NULL, "f->backCell()");
    if (foreach=='\0' && support=='c') nprintf(arc, NULL, "face->backCell()");
    if (foreach=='\0' && support=='f') nprintf(arc, NULL, "face->backCell()");
    if (foreach=='f' && support=='n') nprintf(arc, NULL, "f->backCell()");
    break;
  }
    
  case (BACKCELLUID):{
    if (support=='f') nprintf(arc, NULL, "face->backCell().uniqueId()");
    break;
  }
    
  case (FRONTCELL):{
    if (foreach=='f' && support=='c') nprintf(arc, NULL, "f->frontCell()");
    if (foreach=='\0' && support=='f') nprintf(arc, NULL, "face->frontCell()");
    if (foreach=='f' && support=='n') nprintf(arc, NULL, "f->frontCell()");
    break;
  }
    
  case (FRONTCELLUID):{
    if (support=='f')
      nprintf(arc, NULL, "face->frontCell().uniqueId()");
    break;
  }
    
  case (NBCELL):{
    if (foreach=='c'  && support=='n') nprintf(arc, NULL, "node->nbCell()");
    if (foreach=='f'  && support=='c') nprintf(arc, NULL, "f->nbCell()");
    if (foreach=='f'  && support=='n') nprintf(arc, NULL, "f->nbCell()");
    if (foreach=='\0' && support=='c') nprintf(arc, NULL, "face->nbCell()");
    if (foreach=='\0' && support=='f') nprintf(arc, NULL, "face->nbCell()");
    if (foreach=='\0' && support=='n') nprintf(arc, NULL, "node->nbCell()");
    break;
  }
    
  case (NBNODE):{
    if (support=='c' && foreach=='\0') nprintf(arc, NULL, "cell->nbNode()");
    if (support=='c' && foreach=='f') nprintf(arc, NULL, "cell->nbNode()");
    if (support=='c' && foreach=='n') nprintf(arc, NULL, "cell->nbNode()");
    if (support=='n' && foreach=='c') nprintf(arc, NULL, "c->nbNode()");
    // ENUMERATE_NODES, foreach face mais qd même un nbNode, c'est pe un backCell->nbNode()?
    if (support=='n' && foreach=='f') nprintf(arc, NULL, "nbNode");
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
    if (support=='f') nprintf(arc, "/*THIS+f*/", "face");
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
    if (foreach=='\0' && support=='c') nprintf(arc, NULL, "cell->uniqueId().asInteger()");
    if (foreach=='\0' && support=='n') nprintf(arc, NULL, "node->uniqueId().asInteger()");
    if (foreach=='\0' && support=='f') nprintf(arc, NULL, "face->uniqueId().asInteger()");
    if (foreach=='\0' && support=='p') nprintf(arc, NULL, "particle->uniqueId().asInteger()");
    if (foreach=='p' && support=='c') nprintf(arc, NULL, "cell->uniqueId().asInteger()");
    if (foreach=='c' && support=='c') nprintf(arc, NULL, "cell->uniqueId().asInteger()");
    if (foreach=='c' && support=='n') nprintf(arc, NULL, "c->uniqueId().asInteger()");
    if (foreach=='c' && support=='f') nprintf(arc, NULL, "c->uniqueId().asInteger()");
    if (foreach=='c' && support=='p') nprintf(arc, NULL, "c->uniqueId().asInteger()");
    if (foreach=='n' && support=='c') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (foreach=='n' && support=='n') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (foreach=='n' && support=='f') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (foreach=='n' && support=='p') nprintf(arc, NULL, "n->uniqueId().asInteger()");
    if (foreach=='f' && support=='c') nprintf(arc, NULL, "f->uniqueId().asInteger()");
    if (foreach=='f' && support=='n') nprintf(arc, NULL, "f->uniqueId().asInteger()");
    if (foreach=='f' && support=='f') nprintf(arc, NULL, "f->uniqueId().asInteger()");
    if (foreach=='f' && support=='p') nprintf(arc, NULL, "f->uniqueId().asInteger()");
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
    
  case (')'):{
    nprintf(arc, "/*function_call_arguments@false*/", ")");
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
    
  case ('&'):{
    if (job->parse.function_call_arguments==true){
      nprintf(arc, "/*&+call*/", "adrs");
    }else{
      nprintf(arc, "/*& std*/", "&");
    }
    break;
  }
    
  default:{
    if (n->token) nprintf(arc, NULL, "%s",n->token);
    nablaInsertSpace(arc,n);
  }
    break;
  }
}


/***************************************************************************** 
 * nccArcaneJob
 *****************************************************************************/
void arcaneJob(nablaMain *arc, astNode *n){
  nablaJob *job = nablaJobNew(arc->entity);
  nablaJobAdd(arc->entity, job);
  nablaJobFill(arc,job,n,arc->name);
}
