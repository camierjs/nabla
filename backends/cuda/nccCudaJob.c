/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nccCudaFunction.c
 * Author   : Camier Jean-Sylvain
 * Created  : 2012.12.14
 * Updated  : 2012.12.14
 *****************************************************************************
 * Description:
 *****************************************************************************
 * Date			Author	Description
 * 2012.12.14	camierjs	Creation
 *****************************************************************************/
#include "nabla.h"
#include "nabla.tab.h"


/*****************************************************************************
 * Diffraction
 *****************************************************************************/
void cudaHookJobDiffractStatement(nablaMain *nabla, nablaJob *job, astNode **n){
  // On backup les statements qu'on rencontre pour éventuellement les diffracter (Real3 => _x, _y & _z)
  // Et on amorce la diffraction
  if ((*n)->ruleid == rulenameToId("expression_statement")
      && (*n)->children->ruleid == rulenameToId("expression")
      //&& (*n)->children->children->ruleid == rulenameToId("expression")
      //&& job->parse.statementToDiffract==NULL
      //&& job->parse.diffractingXYZ==0
      ){
      dbg("\n[nablaJobParse] amorce la diffraction");
//#warning Diffracting is turned OFF
      job->parse.statementToDiffract=NULL;//*n;
      // We're juste READY, not diffracting yet!
      job->parse.diffractingXYZ=0;      
      nprintf(nabla, "/* DiffractingREADY */",NULL);
  }
  
  // On avance la diffraction
  if ((*n)->tokenid == ';'
      && job->parse.diffracting==true
      && job->parse.statementToDiffract!=NULL
      && job->parse.diffractingXYZ>0
      && job->parse.diffractingXYZ<3){
    dbg("\n[nablaJobParse] avance dans la diffraction");
    job->parse.isDotXYZ=job->parse.diffractingXYZ+=1;
    (*n)=job->parse.statementToDiffract;
    nprintf(nabla, NULL, ";\n\t");
    nprintf(nabla, "\t/*<REdiffracting>*/", "/*diffractingXYZ=%d*/", job->parse.diffractingXYZ);
  }

  // On flush la diffraction 
  if ((*n)->tokenid == ';' 
      && job->parse.diffracting==true
      && job->parse.statementToDiffract!=NULL
      && job->parse.diffractingXYZ>0
      && job->parse.diffractingXYZ==3){
    dbg("\n[nablaJobParse] Flush de la diffraction");
    job->parse.diffracting=false;
    job->parse.statementToDiffract=NULL;
    job->parse.isDotXYZ=job->parse.diffractingXYZ=0;
    nprintf(nabla, "/*<end of diffracting>*/",NULL);
  }
  dbg("\n[nablaJobParse] return from token %s", (*n)->token?(*n)->token:"Null");
}


/*****************************************************************************
 * Fonction prefix à l'ENUMERATE_*
 *****************************************************************************/
char* cudaHookPrefixEnumerate(nablaJob *job){
  const register char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  //if (j->xyz==NULL) return "// void ENUMERATE prefix";
  nprintf(job->entity->main, "\n\t/*cudaHookPrefixEnumerate*/", "/*itm=%c*/", itm);
  if (itm=='c'  && strcmp(job->rtntp,"void")==0) return "CUDA_INI_CELL_THREAD(tcid);";
  if (itm=='c'  && strcmp(job->rtntp,"Real")==0) return "CUDA_INI_CELL_THREAD_RETURN_REAL(tcid);";
  if (itm=='n') return "CUDA_INI_NODE_THREAD(tnid);";
  if (itm=='\0' && job->is_an_entry_point) return "CUDA_INI_FUNCTION_THREAD(tid);";
  if (itm=='\0' && !job->is_an_entry_point) return "/*std function*/";
  error(!0,0,"Could not distinguish PREFIX Enumerate!");
  return NULL;
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_* avec XYZ
 *****************************************************************************/
char* cudaHookDumpEnumerateXYZ(nablaJob *job){
  char *xyz=job->xyz;// Direction
  nprintf(job->entity->main, "\n\t/*cudaHookDumpEnumerateXYZ*/", "/*xyz=%s, drctn=%s*/", xyz, job->drctn);
  return "// cudaHookDumpEnumerateXYZ has xyz drctn";
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_*
 *****************************************************************************/
char* cudaHookDumpEnumerate(nablaJob *job){
  char *grp=job->scope;   // OWN||ALL
  char *rgn=job->region;  // INNER, OUTER
  char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  //if (job->xyz!=NULL) return cudaHookDumpEnumerateXYZ(job);
  if (itm=='\0') return "// function cudaHookDumpEnumerate\n";
  if (itm=='c' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_CELL_WARP(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER\n\tFOR_EACH_CELL_WARP(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='o')   return "#warning Should be OUTER\n\tFOR_EACH_CELL_WARP(c)";
  if (itm=='c' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_CELL_WARP(c)";
  if (itm=='n' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return "#warning Should be OUTER\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='a' && rgn==NULL)   return "#warning Should be ALL\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER OWN\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='o') return "#warning Should be OUTER OWN\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='f' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_FACE_WARP(f)";
  if (itm=='f' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_FACE_WARP(f)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='o') return "#warning Should be OUTER OWN\n\tFOR_EACH_FACE_WARP(f)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER OWN\n\tFOR_EACH_FACE_WARP(f)";
  if (itm=='e' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_ENV_WARP(e)";
  if (itm=='m' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_MAT_WARP(m)";
  error(!0,0,"Could not distinguish ENUMERATE!");
  return NULL;
}


/*****************************************************************************
 * Fonction postfix à l'ENUMERATE_*
 *****************************************************************************/
char* cudaHookPostfixEnumerate(nablaJob *job){
  if (job->item[0]=='\0') return "// functioncudaHookPostfixEnumerate\n";
  if (job->xyz==NULL) return "";//// void ENUMERATE postfix\n\t";
  if (job->xyz!=NULL) return "// Postfix ENUMERATE with xyz direction\n\
#warning wrongs nextCell prevCell\n\
\t\tint prevCell=tcid-1;\n\
\t\tint nextCell=tcid+1;\n";
  error(!0,0,"Could not switch in cudaHookPostfixEnumerate!");
  return NULL;
}


/***************************************************************************** 
 * Traitement des tokens NABLA ITEMS
 *****************************************************************************/
char* cudaHookItem(nablaJob* job, const char j, const char itm, char enum_enum){
  if (j=='c' && enum_enum=='\0' && itm=='c') return "/*chi-c0c*/c";
  if (j=='c' && enum_enum=='\0' && itm=='n') return "/*chi-c0n*/c->";
  if (j=='c' && enum_enum=='f'  && itm=='n') return "/*chi-cfn*/f->";
  if (j=='c' && enum_enum=='f'  && itm=='c') return "/*chi-cfc*/f->";
  if (j=='n' && enum_enum=='f'  && itm=='n') return "/*chi-nfn*/f->";
  if (j=='n' && enum_enum=='f'  && itm=='c') return "/*chi-nfc*/f->";
  if (j=='n' && enum_enum=='\0' && itm=='n') return "/*chi-n0n*/n";
  if (j=='f' && enum_enum=='\0' && itm=='f') return "/*chi-f0f*/f";
  if (j=='f' && enum_enum=='\0' && itm=='n') return "/*chi-f0n*/f->";
  if (j=='f' && enum_enum=='\0' && itm=='c') return "/*chi-f0c*/f->";
  error(!0,0,"Could not switch in cudaHookItem!");
  return NULL;
}


/*****************************************************************************
 * FOREACH token switch
 *****************************************************************************/
static void cudaHookSwitchForeach(astNode *n, nablaJob *job){
  // Preliminary pertinence test
  if (n->tokenid != FOREACH) return;
  // Now we're allowed to work
  switch(n->next->children->tokenid){
  case(CELL):{
    job->parse.enum_enum='c';
    nprintf(job->entity->main, "/*chsf c*/", "for(int i=0;i<8;++i)");
    break;
  }
  case(NODE):{
    job->parse.enum_enum='n';
    nprintf(job->entity->main, "/*chsf n*/", "for(int n=0;n<8;++n)");
    break;
  }
  case(FACE):{
    job->parse.enum_enum='f';
    if (job->item[0]=='c')
      nprintf(job->entity->main, "/*chsf fc*/", "for(cFACE)");
    if (job->item[0]=='n')
      nprintf(job->entity->main, "/*chsf fn*/", "for(nFACE)");
    break;
  }
  }
  // Attention au cas où on a un @ au lieu d'un statement
  if (n->next->next->tokenid == AT)
    nprintf(job->entity->main, "/* Found AT */", NULL);
  // On skip le 'nabla_item' qui nous a renseigné sur le type de foreach
  *n=*n->next->next;
}


/*****************************************************************************
 * Différentes actions pour un job Nabla
 *****************************************************************************/
void cudaHookSwitchToken(astNode *n, nablaJob *job){
  nablaMain *nabla=job->entity->main;
  const char cnfgem=job->item[0];

  //nprintf(nabla, "/*cudaHookSwitchToken*/", NULL);
  cudaHookSwitchForeach(n,job);
  
  //nprintf(nabla, "/*cudaHookSwitchToken switch*/", "%d",n->tokenid);
  //nprintf(nabla, "/*cudaHookSwitchToken switch*/", "%s",n->token);
  // Dump des tokens possibles
  switch(n->tokenid){
    
  case(INTEGER):{
    nprintf(nabla, "/*INTEGER*/", "int ");
    break;
  }
    
  case(POSTFIX_CONSTANT):{
     nprintf(nabla, "/*postfix_constant@true*/", NULL);
     job->parse.postfix_constant=true;
    break;
  }
  case(POSTFIX_CONSTANT_VALUE):{
     nprintf(nabla, "/*postfix_constant_value*/", NULL);
     job->parse.postfix_constant=false;
     job->parse.turnBracketsToParentheses=false;
    break;
  }
  case (FATAL):{
    nprintf(nabla, "/*fatal*/", "fatal");
    break;
  }    
    // On regarde si on hit un appel de fonction
  case(CALL):{
    nablaJob *foundJob;
    nprintf(nabla, "/*JOB_CALL*/", NULL);
    char *callName=n->next->children->children->token;
    nprintf(nabla, "/*got_call*/", NULL);
    if ((foundJob=nablaJobFind(job->entity->jobs,callName))!=NULL){
      if (foundJob->is_a_function!=true){
        nprintf(nabla, "/*isNablaJob*/", NULL);
      }else{
        nprintf(nabla, "/*isNablaFunction*/", NULL);
      }
    }else{
      nprintf(nabla, "/*has not been found*/", NULL);
    }
    break;
  }
  case(END_OF_CALL):{
    nprintf(nabla, "/*ARGS*/", NULL);
    nprintf(nabla, "/*got_args*/", NULL);
    break;
  }

  case(PREPROCS):{
    nprintf(nabla, "/*PREPROCS*/", "\n%s\n",n->token);
    break;
  }
    
  case(UIDTYPE):{
    nprintf(nabla, "UIDTYPE", "int64");
    break;
  }
    
  case(COMPOUND_JOB_INI):{
    nprintf(nabla, "/*COMPOUND_JOB_INI:*/",NULL);
    break;
  }
  case(COMPOUND_JOB_END):{
    nprintf(nabla, "/*COMPOUND_JOB_END*/",NULL);
    break;
  }

  case(FOREACH_INI):{ break; }
  case(FOREACH_END):{
    nprintf(nabla, "/*FOREACH_END*/",NULL);
    job->parse.enum_enum='\0';
    job->parse.turnBracketsToParentheses=false;
    break;
  }
    
  case('}'):{
    nprintf(nabla, NULL, "}"); 
    break;
  }
    
  case('['):{
    if (job->parse.postfix_constant==true
        && job->parse.variableIsArray==true) break;
    if (job->parse.turnBracketsToParentheses==true)
      nprintf(nabla, NULL, "");
    else
      nprintf(nabla, NULL, "[");
    break;
  }

  case(']'):{
    if (job->parse.turnBracketsToParentheses==true){
      if (job->item[0]=='c') nprintf(nabla, "/*tBktOFF*/", "[tcid]]");//tcid+c
      if (job->item[0]=='n') nprintf(nabla, "/*tBktOFF*/", "[tnid]]");//tnid+c
      job->parse.turnBracketsToParentheses=false;
    }else{
      nprintf(nabla, NULL, "]");
    }
    //nprintf(nabla, "/*FlushingIsPostfixed*/","/*isDotXYZ=%d*/",job->parse.isDotXYZ);
    if (job->parse.isDotXYZ==1) nprintf(nabla, "/*]+FlushingIsPostfixed*/", ".x");
    if (job->parse.isDotXYZ==2) nprintf(nabla, NULL, ".y");
    if (job->parse.isDotXYZ==3) nprintf(nabla, NULL, ".z");
    job->parse.isPostfixed=0;
    // On flush le isDotXYZ
    job->parse.isDotXYZ=0;
    break;
  }

  case (BOUNDARY_CELL):{
    if (job->parse.enum_enum=='f' && cnfgem=='c') nprintf(nabla, NULL, "f->boundaryCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='c') nprintf(nabla, NULL, "face->boundaryCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='f') nprintf(nabla, NULL, "face->boundaryCell()");
   break;
  }
  case (BACKCELL):{
    if (job->parse.enum_enum=='f' && cnfgem=='c') nprintf(nabla, NULL, "f->backCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='c') nprintf(nabla, NULL, "face->backCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='f') nprintf(nabla, NULL, "face->backCell()");
    break;
  }
  case (BACKCELLUID):{
    if (cnfgem=='f')
      nprintf(nabla, NULL, "face->backCell().uniqueId()");
    break;
  }
  case (FRONTCELL):{
    if (job->parse.enum_enum=='f' && cnfgem=='c') nprintf(nabla, NULL, "f->frontCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='f') nprintf(nabla, NULL, "face->frontCell()");
    break;
  }
  case (FRONTCELLUID):{
    if (cnfgem=='f')
      nprintf(nabla, "/*FRONTCELLUID*/", "face->frontCell().uniqueId()");
    break;
  }
  case (NBCELL):{
    if (job->parse.enum_enum=='f'  && cnfgem=='c') nprintf(nabla, NULL, "f->nbCell()");
    if (job->parse.enum_enum=='f'  && cnfgem=='n') nprintf(nabla, NULL, "f->nbCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='c') nprintf(nabla, NULL, "face->nbCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='f') nprintf(nabla, NULL, "face->nbCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='n') nprintf(nabla, NULL, "node->nbCell()");
    break;
  }    
  case (NBNODE):{ if (cnfgem=='c') nprintf(nabla, NULL, "cell->nbNode()"); break; }    
    //case (INODE):{ if (cnfgem=='c') nprintf(nabla, NULL, "cell->node"); break; }    

  case (XYZ):{ nprintf(nabla, "/*XYZ*/", NULL); break;}
  case (NEXTCELL):{ nprintf(nabla, "/*token NEXTCELL*/", "nextCell"); break;}
  case (PREVCELL):{ nprintf(nabla, "/*token PREVCELL*/", "prevCell"); break;}
  case (NEXTNODE):{ nprintf(nabla, "/*token NEXTNODE*/", "nextNode"); break; }
  case (PREVNODE):{ nprintf(nabla, "/*token PREVNODE*/", "prevNode"); break; }
  case (PREVLEFT):{ nprintf(nabla, "/*token PREVLEFT*/", "cn.previousLeft()"); break; }
  case (PREVRIGHT):{ nprintf(nabla, "/*token PREVRIGHT*/", "cn.previousRight()"); break; }
  case (NEXTLEFT):{ nprintf(nabla, "/*token NEXTLEFT*/", "cn.nextLeft()"); break; }
  case (NEXTRIGHT):{ nprintf(nabla, "/*token NEXTRIGHT*/", "cn.nextRight()"); break; }
    // Gestion du THIS
  case (THIS):{
    if (cnfgem=='c') nprintf(nabla, "/*token THIS+c*/", "c");
    if (cnfgem=='n') nprintf(nabla, "/*token THIS+n*/", "n");
    if (cnfgem=='f') nprintf(nabla, "/*token THIS+f*/", "f");
    break;
  }
    
  case (SID):{
    nprintf(nabla, NULL, "subDomain()->subDomainId()");
    break;
  }
  case (LID):{
    if (cnfgem=='c') nprintf(nabla, "/*localId c*/", "c->localId()");
    if (cnfgem=='n') nprintf(nabla, "/*localId n*/", "n->localId()");
    break;
  }
  case (UID):{
    if (cnfgem=='c') nprintf(nabla, "/*uniqueId c*/", "(tcid)");//+c
    if (cnfgem=='n') nprintf(nabla, "/*uniqueId n*/", "(tnid)");//+n
    break;
  }
  case (AT):{ nprintf(nabla, "/*knAt*/", "; knAt"); break; }
  case ('='):{
    nprintf(nabla, "/*'='->!isLeft*/", "=");
    job->parse.left_of_assignment_operator=false;
    job->parse.turnBracketsToParentheses=false;
    job->parse.variableIsArray=false;
    break;
  }
  case (RSH_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, ">>="); break; }
  case (LSH_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "<<="); break; }
  case (ADD_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "+="); break; }
  case (SUB_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "-="); break; }
  case (MUL_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "*="); break; }
  case (DIV_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "/="); break; }
  case (MOD_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "%%="); break; }
  case (AND_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "&="); break; }
  case (XOR_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "^="); break; }
  case (IOR_ASSIGN):{ job->parse.left_of_assignment_operator=false; nprintf(nabla, NULL, "|="); break; }
  case (LSH_OP):{ job->parse.left_of_assignment_operator=true; nprintf(nabla, NULL, "<<"); break; }
  case (RETURN):{
    nprintf(nabla, NULL, "\n\t\t}/* des sources */\n\t}/* de l'ENUMERATE */\n\treturn ");
    //nprintf(nabla, NULL, "};\n\t return ");
    job->parse.got_a_return=true;
    job->parse.got_a_return_and_the_semi_colon=false;
    break;
  }
  case ('{'):{nprintf(nabla, NULL, "{\n\t\t"); break; }    
  case ('&'):{nprintf(nabla, NULL, "/*adrs*/&"); break; }    
  case (';'):{
    job->parse.variableIsArray=false;
    job->parse.turnBracketsToParentheses=false;
    nprintf(nabla, NULL, ";\n\t\t");
    if (job->parse.function_call_arguments==true){
      job->parse.function_call_arguments=false;
      nprintf(nabla, "/*!function_call_arguments*/", NULL);
    }
    if (job->parse.got_a_return)
      job->parse.got_a_return_and_the_semi_colon=true;
    break;
  }
  default:{
    if (n->token!=NULL) nprintf(nabla, NULL, "%s ", n->token);
    break;
  }
  }
}


/*****************************************************************************
  * Dump d'extra paramètres
 *****************************************************************************/
void cudaHookAddExtraParameters(nablaMain *nabla, nablaJob *job, int *numParams){
  nablaVariable *var;
  if (*numParams!=0) nprintf(nabla, NULL, ",");
  if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA){
    nprintf(nabla, NULL, "\n\t\treal3 *node_coord");
    *numParams+=1;
  }else{  nprintf(nabla, NULL, "\n\t\tReal *node_coordx,");
    *numParams+=1;
    nprintf(nabla, NULL, "\n\t\tReal *node_coordy,");
    *numParams+=1;
    nprintf(nabla, NULL, "\n\t\tReal *node_coordz");
    *numParams+=1;
  }
  // Et on rajoute les variables globales
  for(var=nabla->variables;var!=NULL;var=var->next){
    //if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->item, "global")!=0) continue;
    nprintf(nabla, NULL, ",\n\t\t%s *global_%s",
            //(*numParams!=0)?",":"", 
            (var->type[0]=='r')?"Real":(var->type[0]=='i')?"int":"/*Unknown type*/",
            var->name);
    *numParams+=1;
  }
  // Rajout pour l'instant systématiquement des connectivités
  if (job->item[0]=='c')
    cudaAddExtraConnectivitiesParameters(nabla, numParams);
}



// *****************************************************************************
// * Dump d'extra connectivity
// ****************************************************************************
void cudaAddExtraConnectivitiesArguments(nablaMain *nabla, int *numParams){
  const char* tabs="\t\t\t\t\t\t\t";
  nprintf(nabla, NULL, ",\n%scell_node",tabs);
  *numParams+=1;
  //nprintf(nabla, NULL, ",\n%scell_node_0",tabs);
  //nprintf(nabla, NULL, ",\n%scell_node_1",tabs);
  //nprintf(nabla, NULL, ",\n%scell_node_2",tabs);
  //nprintf(nabla, NULL, ",\n%scell_node_3",tabs);
  //nprintf(nabla, NULL, ",\n%scell_node_4",tabs);
  //nprintf(nabla, NULL, ",\n%scell_node_5",tabs);
  //nprintf(nabla, NULL, ",\n%scell_node_6",tabs);
  //nprintf(nabla, NULL, ",\n%scell_node_7",tabs);
  //*numParams+=8;
}

void cudaAddExtraConnectivitiesParameters(nablaMain *nabla, int *numParams){
  nprintf(nabla, NULL, ",\n\t\tint *cell_node");
  *numParams+=1;
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_0");
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_1");
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_2");
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_3");
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_4");
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_5");
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_6");
  //nprintf(nabla, NULL, ",\n\t\tint *cell_node_7");
  //*numParams+=8;
}


/*****************************************************************************
  * Dump d'extra arguments
 *****************************************************************************/
void cudaAddExtraArguments(nablaMain *nabla, nablaJob *job, int *numParams){
  const char* tabs="\t\t\t\t\t\t\t";
  
  { // Rajout pour l'instant systématiquement des node_coords et du global_deltat
    nablaVariable *var;
    if (*numParams!=0) nprintf(nabla, NULL, ",");
    if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)!=BACKEND_COLOR_OKINA_SOA){
      nprintf(nabla, NULL, "\n%snode_coord",tabs);
      *numParams+=1;
    }else{
      nprintf(nabla, NULL, "\n%snode_coordx,",tabs);
      *numParams+=1;
      nprintf(nabla, NULL, "\n%snode_coordy,",tabs);
      *numParams+=1;
      nprintf(nabla, NULL, "\n%snode_coordz",tabs);
      *numParams+=1;
    }
    // Et on rajoute les variables globales
    for(var=nabla->variables;var!=NULL;var=var->next){
      //if (strcmp(var->name, "time")==0) continue;
      if (strcmp(var->item, "global")!=0) continue;
      nprintf(nabla, NULL, ",global_%s", var->name);
      *numParams+=1;
   }
  }
  
  // Rajout pour l'instant systématiquement des connectivités
  if (job->item[0]=='c')
    cudaAddExtraConnectivitiesArguments(nabla, numParams);
}


/*****************************************************************************
  * Dump dans le src des parametres nabla en in comme en out
 *****************************************************************************/
void cudaHookDumpNablaParameterList(nablaMain *nabla,
                                    nablaJob *job,
                                    astNode *n,
                                    int *numParams){
  dbg("\n\t[cudaHookDumpNablaParameterList]");
  if (n==NULL) return;
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")){
    dbg("\n\t[cudaHookDumpNablaParameterList] compound_statement, returning");
    return;
  }
  if (n->tokenid=='@'){
    dbg("\n\t[cudaHookDumpNablaParameterList] @, returning");
    return;
  }
  
  //if (n->ruleid==rulenameToId("nabla_parameter_declaration"))    if (*numParams!=0) nprintf(nabla, NULL, ",");
  
  if (n->ruleid==rulenameToId("direct_declarator")){
    nablaVariable *var=nablaVariableFind(nabla->variables, n->children->token);
    //nprintf(nabla, NULL, "\n\t\t/*[cudaHookDumpNablaParameterList] looking for %s*/", n->children->token);
    *numParams+=1;
    // Si on ne trouve pas de variable, on a rien à faire
    if (var == NULL) return exit(NABLA_ERROR|fprintf(stderr, "\n[cudaHookDumpNablaParameterList] Variable error\n"));
    if (strcmp(var->type, "real3")!=0){
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
      }else{
        nprintf(nabla, NULL, ",\n\t\t%s *%s_%s", var->type, var->item, n->children->token);
      }
    }else{
      //exit(NABLA_ERROR|fprintf(stderr, "\n[cudaHookDumpNablaParameterList] Variable Real3 error\n"));
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
        nprintf(nabla, NULL, NULL);
      }else{
        if (var->dim==0){
          nprintf(nabla, NULL, ",\n\t\tReal3 *%s_%s", var->item, n->children->token);
        }else{
          nprintf(nabla, NULL, ",\n\t\treal3 *%s_%s", var->item, n->children->token);
        }
      }
    }
  }
  if (n->children != NULL) cudaHookDumpNablaParameterList(nabla, job, n->children, numParams);
  if (n->next != NULL) cudaHookDumpNablaParameterList(nabla, job, n->next, numParams);
}



// *****************************************************************************
// * Ajout des variables d'un job trouvé depuis une fonction @ée
// *****************************************************************************
void cudaAddNablaVariableList(nablaMain *nabla, astNode *n, nablaVariable **variables){
  //dbg("\n\t[cudaAddNablaVariableList]");
  if (n==NULL) return;
  
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) return;
  
  if (n->tokenid=='@') return;
    
  if (n->ruleid==rulenameToId("direct_declarator")){
    nablaVariable *hit=nablaVariableFind(nabla->variables, n->children->token);
    dbg("\n\t[cudaAddNablaVariableList] direct_declarator %s %s", hit->item, hit->name);
    // Si on ne trouve pas de variable, c'est pas normal
    if (hit == NULL) return exit(NABLA_ERROR|fprintf(stderr, "\n[cudaAddNablaVariableList] Variable error\n"));
    nablaVariable *allready_here=nablaVariableFind(*variables, hit->name);
    if (allready_here!=NULL){
      dbg("\n\t[cudaAddNablaVariableList] allready_here!");
    }else{
      // Création d'une nouvelle called_variable
      nablaVariable *new = nablaVariableNew(NULL);
      new->name=strdup(hit->name);
      new->item=strdup(hit->item);
      new->type=strdup(hit->type);
      new->dim=hit->dim;
      new->size=hit->size;
      // Rajout à notre liste
      if (*variables==NULL){
        dbg("\n\t[cudaAddNablaVariableList] first hit");
        *variables=new;
      }else{
        dbg("\n\t[cudaAddNablaVariableList] last hit");
        nablaVariableLast(*variables)->next=new;
      }
    }
  }
  if (n->children != NULL) cudaAddNablaVariableList(nabla, n->children, variables);
  if (n->next != NULL) cudaAddNablaVariableList(nabla, n->next, variables);
}


/*****************************************************************************
  * Dump dans le src des arguments nabla en in comme en out
 *****************************************************************************/
void cudaDumpNablaArgumentList(nablaMain *nabla, astNode *n, int *numParams){
  //nprintf(nabla,"\n\t[cudaDumpNablaArgumentList]",NULL);
  if (n==NULL) return;
  
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) return;
  
  if (n->tokenid=='@') return;
  
  //if (n->ruleid==rulenameToId("nabla_parameter_declaration"))    if (*numParams!=0) nprintf(nabla, NULL, ",");
  
  if (n->ruleid==rulenameToId("direct_declarator")){
    nablaVariable *var=nablaVariableFind(nabla->variables, n->children->token);
    //nprintf(nabla, NULL, "\n\t\t/*[cudaDumpNablaArgumentList] looking for %s*/", n->children->token);
    *numParams+=1;
    // Si on ne trouve pas de variable, on a rien à faire
    if (var == NULL) return exit(NABLA_ERROR|fprintf(stderr, "\n[cudaHookDumpNablaArgumentList] Variable error\n"));
    if (strcmp(var->type, "real3")!=0){
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
      }else{
        nprintf(nabla, NULL, ",\n\t\t\t\t\t\t\t%s_%s", var->item, n->children->token);
      }
    }else{
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0)
        nprintf(nabla, NULL, NULL);
      else
        nprintf(nabla, NULL,  ",\n\t\t\t\t\t\t\t%s_%s", var->item, n->children->token);
    }
  }
  if (n->children != NULL) cudaDumpNablaArgumentList(nabla, n->children, numParams);
  if (n->next != NULL) cudaDumpNablaArgumentList(nabla, n->next, numParams);
}


/*****************************************************************************
  * Dump dans le src l'appel des fonction de debug des arguments nabla  en out
 *****************************************************************************/
void cudaDumpNablaDebugFunctionFromOutArguments(nablaMain *nabla, astNode *n, bool in_or_out){
  //nprintf(nabla,"\n\t[cudaHookDumpNablaParameterList]",NULL);
  if (n==NULL) return;
  
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) return;
  if (n->tokenid=='@') return;

  if (n->tokenid==OUT) in_or_out=false;
  if (n->tokenid==INOUT) in_or_out=false;
    
  if (n->ruleid==rulenameToId("direct_declarator")){
    nablaVariable *var=nablaVariableFind(nabla->variables, n->children->token);
    // Si on ne trouve pas de variable, on a rien à faire
    if (var == NULL)
      return exit(NABLA_ERROR|fprintf(stderr, "\n[cudaDumpNablaDebugFunctionFromOutArguments] Variable error\n"));
    if (!in_or_out){
      nprintf(nabla,NULL,"\n\t\t//printf(\"\\n%sVariable%sDim%s_%s:\");",
              (var->item[0]=='n')?"Node":"Cell",
              (strcmp(var->type,"real3")==0)?"XYZ":"",
              (var->dim==0)?"0":"1",
              var->name);
      nprintf(nabla,NULL,"//dbg%sVariable%sDim%s_%s();",
              (var->item[0]=='n')?"Node":"Cell",
              (strcmp(var->type,"real3")==0)?"XYZ":"",
              (var->dim==0)?"0":"1",
              var->name);
    }
  }
  cudaDumpNablaDebugFunctionFromOutArguments(nabla, n->children, in_or_out);
  cudaDumpNablaDebugFunctionFromOutArguments(nabla, n->next, in_or_out);
}


/*****************************************************************************
 * Génération d'un kernel associé à un support
 *****************************************************************************/
void cudaHookJob(nablaMain *nabla, astNode *n){
  nablaJob *job = nablaJobNew(nabla->entity);
  nablaJobAdd(nabla->entity, job);
  nablaJobFill(nabla,job,n,NULL);
  
  // On teste *ou pas* que le job retourne bien 'void' dans le cas de CUDA
  if ((strcmp(job->rtntp,"void")!=0) && (job->is_an_entry_point==true))
    exit(NABLA_ERROR|fprintf(stderr, "\n[cudaHookJob] Error with return type which is not void\n"));
}


