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
#include "nabla.tab.h"

static void lambdaHookIsTestIni(nablaMain *nabla, nablaJob *job, astNode *n){
  const astNode* isNode = dfsFetchTokenId(n->next,IS);
  assert(isNode);
  const char *token2function = isNode->next->token;
  assert(token2function);
  if (isNode->next->tokenid==OWN)
    nprintf(nabla, "/*IS_OP_INI*/", "_isOwn_(");
  else
    nprintf(nabla, "/*IS_OP_INI*/", "_%s_(", token2function);
  // Et on purge le token pour pas qu'il soit parsé
  isNode->next->token[0]=0;
}
//static void lambdaHookIsTestIs(nablaMain *nabla, nablaJob *job, astNode *n){}
static void lambdaHookIsTestEnd(nablaMain *nabla, nablaJob *job, astNode *n){
  nprintf(nabla, "/*IS_OP_END*/", ")");
}

// *****************************************************************************
// *
// *****************************************************************************
void lambdaHookIsTest(nablaMain *nabla, nablaJob *job, astNode *n, int token){
  assert(token==IS || token==IS_OP_INI || token==IS_OP_END);
  if (token==IS_OP_INI) lambdaHookIsTestIni(nabla,job,n);
  if (token==IS) return;
  if (token==IS_OP_END) lambdaHookIsTestEnd(nabla,job,n);
}


// ****************************************************************************
// * lambdaHookTokenPrefix
// ****************************************************************************
char* lambdaHookTokenPrefix(struct nablaMainStruct *nabla){return strdup("");}

// ****************************************************************************
// * lambdaHookTokenPostfix
// ****************************************************************************
char* lambdaHookTokenPostfix(struct nablaMainStruct *nabla){return strdup("");}


// ****************************************************************************
// * lambdaHookTurnTokenToOption
// ****************************************************************************
void lambdaHookTurnTokenToOption(struct nablaMainStruct *nabla,nablaOption *opt){
  nprintf(nabla, "/*tt2o lambda*/", "%s", opt->name);
}

  
// ****************************************************************************
// * FORALL token switch
// ****************************************************************************
static void lambdaHookSwitchForall(astNode *n, nablaJob *job){
  const char cnfg=job->item[0];

// Preliminary pertinence test
  if (n->tokenid != FORALL) return;
  // Now we're allowed to work
  switch(n->next->children->tokenid){
  case(CELL):{
    job->parse.enum_enum='c';
    nprintf(job->entity->main, "/*chsf c*/", "FOR_EACH_NODE_CELL(c)");
    break;
  }
  case(NODE):{
    job->parse.enum_enum='n';
    if ((job->entity->libraries&(1<<with_real))!=0) // Iteration 1D
      nprintf(job->entity->main, "/*chsf n*/", "for(int n=0;n<2;++n)");
    else{
      assert(cnfg=='c'||cnfg=='f');
      if (cnfg=='c') nprintf(job->entity->main, "/*chsf n*/", "for(int n=0;n<NABLA_NODE_PER_CELL;++n)");
      if (cnfg=='f') nprintf(job->entity->main, "/*chsf n*/", "for(int n=0;n<NABLA_NODE_PER_FACE;++n)");
    }
    break;
  }
  case(FACE):{
    job->parse.enum_enum='f';
    if (job->item[0]=='c')
      nprintf(job->entity->main, "/*chsf fc*/", "for(int f=0;f<4;++f)");
    if (job->item[0]=='n')
      nprintf(job->entity->main, "/*chsf fn*/", "for(nFACE)");
    break;
  }
  }
  // Attention au cas où on a un @ au lieu d'un statement
  if (n->next->next->tokenid == AT)
    nprintf(job->entity->main, "/* Found AT */", NULL);
  // On skip le 'nabla_item' qui nous a renseigné sur le type de forall
  *n=*n->next->next;
}


// *****************************************************************************
// * lambdaHookSwitchAleph
// *****************************************************************************
static bool lambdaHookSwitchAleph(astNode *n, nablaJob *job){
  const nablaMain *nabla=job->entity->main;

  //nprintf(nabla, "/*lambdaHookSwitchAleph*/","/*lambdaHookSwitchAleph*/");

  switch(n->tokenid){
  case(LIB_ALEPH):{
    nprintf(nabla, "/*LIB_ALEPH*/","/*LIB_ALEPH*/");
    return true;
  }
  case(ALEPH_RHS):{
    nprintf(nabla, "/*ALEPH_RHS*/","rhs");
    // On utilise le 'alephKeepExpression' pour indiquer qu'on est sur des vecteurs
    job->parse.alephKeepExpression=true;
    return true;
  }
  case(ALEPH_LHS):{
    nprintf(nabla, "/*ALEPH_LHS*/","lhs");
    // On utilise le 'alephKeepExpression' pour indiquer qu'on est sur des vecteurs
    job->parse.alephKeepExpression=true;
    return true;
  }
  case(ALEPH_MTX):{
    nprintf(nabla, "/*ALEPH_MTX*/","mtx");
    job->parse.alephKeepExpression=true;
    return true;
  }
  case(ALEPH_RESET):{ nprintf(nabla, "/*ALEPH_RESET*/",".reset()"); break;}
  case(ALEPH_SOLVE):{ nprintf(nabla, "/*ALEPH_SOLVE*/","alephSolve()"); break;}
  case(ALEPH_SET):{
    // Si c'est un vecteur et setValue, on le laisse
    if (job->parse.alephKeepExpression==true){
      /**/
    }else{
      job->parse.alephKeepExpression=true;
    }
    nprintf(nabla, "/*ALEPH_SET*/",".setValue");
    return true;
  }
  case(ALEPH_GET):{
    // Si c'est un vecteur et getValue, on le laisse pas
    if (job->parse.alephKeepExpression==true){
      //job->parse.alephKeepExpression=false;
    }
    nprintf(nabla, "/*ALEPH_GET*/",".getValue");
    return true;
  }
  case(ALEPH_ADD_VALUE):{
    nprintf(nabla, "/*ALEPH_ADD_VALUE*/","/*ALEPH_ADD_VALUE*/");
    // Si c'est un vecteur et addValue, on le laisse pas
    if (job->parse.alephKeepExpression==true){
      job->parse.alephKeepExpression=true;
    }else{
      job->parse.alephKeepExpression=true;
    }
    nprintf(nabla, "/*ALEPH_ADD_VALUE*/",".addValue");
    return true;
  }
  case(ALEPH_NEW_VALUE):{
    job->parse.alephKeepExpression=false;
    nprintf(nabla, "/*ALEPH_NEW_VALUE*/",".newValue");
    return true;
  }
  }
  return false;
}

/*****************************************************************************
 * Différentes actions pour un job Nabla
 *****************************************************************************/
void lambdaHookSwitchToken(astNode *n, nablaJob *job){
  nablaMain *nabla=job->entity->main;
  const char cnfgem=job->item[0];
  const char forall=job->parse.enum_enum;

  //if (n->token) nprintf(nabla, NULL, "\n/*token=%s*/",n->token);
  if (n->token)
    dbg("\n\t[lambdaHookSwitchToken] token: '%s'?", n->token);
 
  // On tests si c'est un token Aleph
  // Si c'est le cas, on a fini
  if (lambdaHookSwitchAleph(n,job)) return;
  
  lambdaHookSwitchForall(n,job);
  
  switch(n->tokenid){
    
    // 'is_test' est traité dans le hook 'lambdaHookIsTest'
  case (IS): break;
  case (IS_OP_INI): break;
  case (IS_OP_END): break;

  case (DIESE):{
    //nprintf(nabla, "/*DIESE*/", "/*DIESE*/");
    if (cnfgem=='c' && forall!='\0') nprintf(nabla, NULL, "%c", forall);
    if (cnfgem=='n' && forall!='\0') nprintf(nabla, NULL, "%c", forall);
    if (cnfgem=='f' && forall!='\0') nprintf(nabla, NULL, "%c", forall);
    break;
  }

  case (MATERIAL):{
    nprintf(nabla, "/*MATERIAL*/", "/*MATERIAL*/");
    break;
  }

  case (MIN_ASSIGN):{
    job->parse.left_of_assignment_operator=false;
    nprintf(nabla, "/*MIN_ASSIGN*/", "/*MIN_ASSIGN*/=ReduceMinToDouble");
    break;
  }
  case (MAX_ASSIGN):{
    job->parse.left_of_assignment_operator=false;
    nprintf(nabla, "/*MAX_ASSIGN*/", "/*MAX_ASSIGN*/=ReduceMaxToDouble");
    break;
  }

  case(CONST):{
    nprintf(nabla, "/*CONST*/", "%sconst ", job->entity->main->hook->pragma->align());
    break;
  }
  case(ALIGNED):{
    nprintf(nabla, "/*ALIGNED*/", "%s", job->entity->main->hook->pragma->align());
    break;
  }
    
  case(REAL):{
    nprintf(nabla, "/*Real*/", "real ");
    break;
  }
  case(REAL3):{
    if ((job->entity->libraries&(1<<with_real))!=0)
      exit(NABLA_ERROR|
           fprintf(stderr,
                   "[nLambdaHookSwitchToken] Real3 can't be used with R library!\n"));
    
    assert((job->entity->libraries&(1<<with_real))==0);
    nprintf(nabla, "/*Real3*/", "real3 ");
    break;
  }
  case(INTEGER):{
    nprintf(nabla, "/*INTEGER*/", "int ");
    break;
  }
    //case(RESTRICT):{nprintf(nabla, "/*RESTRICT*/", "__restrict__ ");break;}
    
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
    dbg("\n\t[lambdaHookSwitchToken] CALL?!");
    // S'il y a des appels Aleph derrière, on ne déclenche pas la suite
    if (n->next)
      if (n->next->children)
        if (n->next->children->tokenid==LIB_ALEPH) break;
    dbg("\n\t[lambdaHookSwitchToken] JOB_CALL");
    nablaJob *foundJob;
    nprintf(nabla, "/*JOB_CALL*/", NULL);
    if ( n->next->children->children->token){
      dbg("\n\t[lambdaHookSwitchToken] JOB_CALL next children children");
      if (n->next->children->children->token){
        dbg("\n\t[lambdaHookSwitchToken] JOB_CALL next children children token");
        char *callName=n->next->children->children->token;
        nprintf(nabla, "/*got_call*/", NULL);
        if ((foundJob=nMiddleJobFind(job->entity->jobs,callName))!=NULL){
          if (foundJob->is_a_function!=true){
            nprintf(nabla, "/*isNablaJob*/", NULL);
          }else{
            nprintf(nabla, "/*isNablaFunction*/", NULL);
          }
        }else{
          nprintf(nabla, "/*has not been found*/", NULL);
        }
      }
    }
    dbg("\n\t[lambdaHookSwitchToken] JOB_CALL done");
    break;
  }

  case(END_OF_CALL):{
    nprintf(nabla, "/*ARGS*/", NULL);
    nprintf(nabla, "/*got_args*/", NULL);
    break;
  }

  case(PREPROCS):{
    //nprintf(nabla, "/*PREPROCS*/", "\n%s\n",n->token);
    break;
  }
    
  case(UIDTYPE):{
    nprintf(nabla, "UIDTYPE", "int ");
    break;
  }
    
  case(FORALL_INI):{
    nprintf(nabla, "/*FORALL_INI*/", "{\n\t\t\t");//FORALL_INI
    nprintf(nabla, "/*lambdaFilterGather*/", "%s",lambdaHookFilterGather(job));
    break;
  }
  case(FORALL_END):{
    nprintf(nabla, "/*lambdaFilterScatter*/", lambdaHookFilterScatter(job));
    nprintf(nabla, "/*FORALL_END*/", "\n\t\t}\n\t");//FORALL_END
    job->parse.enum_enum='\0';
    job->parse.turnBracketsToParentheses=false;
    break;
  }

  case(COMPOUND_JOB_INI):{
    if (job->parse.returnFromArgument &&
        ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP))
      nprintf(nabla, NULL, "int tid = omp_get_thread_num();");
    //nprintf(nabla, NULL, "/*COMPOUND_JOB_INI:*/");
    break;
  }
    
  case(COMPOUND_JOB_END):{
    //nprintf(nabla, NULL, "/*:COMPOUND_JOB_END*/");
    break;
  }
     
  case('}'):{
    nprintf(nabla, NULL, "}"); 
    break;
  }
    
  case('['):{
    if (job->parse.postfix_constant==true && job->parse.variableIsArray==true) break;
    if (job->parse.turnBracketsToParentheses==true)
      nprintf(nabla, NULL, "");
    else
      nprintf(nabla, NULL, "[");
    break;
  }

  case(']'):{
    //nprintf(nabla, NULL, "/*swTkn ']'*/");
    if (job->parse.turnBracketsToParentheses==true){
      switch  (job->item[0]){
      case('c'):{nprintf(nabla, NULL, "]]"); break;}
      case('n'):{nprintf(nabla, NULL, "]]"); break;}
      case('f'):{nprintf(nabla, NULL, "]]"); break;}
      default:{nprintf(nabla, NULL, "/*]?*/)]");}
      }
      job->parse.turnBracketsToParentheses=false;
    }else{
      nprintf(nabla, NULL, "]");
    }
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
    if (job->parse.enum_enum=='\0' && cnfgem=='f' && job->parse.alephKeepExpression==true)
      nprintf(nabla, NULL, "face_cell[f+NABLA_NB_FACES*0]");     
    if (job->parse.enum_enum=='\0' && cnfgem=='f' && job->parse.alephKeepExpression==false)
      nprintf(nabla, NULL, "face_cell[f+NABLA_NB_FACES*0]");
  break;
  }
  case (BACKCELLUID):{
    if (cnfgem=='f')
      nprintf(nabla, NULL, "face->backCell().uniqueId()");
    break;
  }
  case (FRONTCELL):{
    if (job->parse.enum_enum=='f' && cnfgem=='c') nprintf(nabla, NULL, "f->frontCell()");
    if (job->parse.enum_enum=='\0' && cnfgem=='f' && job->parse.alephKeepExpression==false)
      nprintf(nabla, NULL, "face_cell[f+NABLA_NB_FACES*1]");
    if (job->parse.enum_enum=='\0' && cnfgem=='f' && job->parse.alephKeepExpression==true)
      nprintf(nabla, NULL, "face_cell[f+NABLA_NB_FACES*1]");
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
  case (NBNODE):{
    assert(cnfgem=='c' || cnfgem=='f');
    if (cnfgem=='c') nprintf(nabla, NULL, "NABLA_NODE_PER_CELL");
    if (cnfgem=='f') nprintf(nabla, NULL, "NABLA_NODE_PER_FACE");
    break;
  }    
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
    if (job->parse.enum_enum=='n'  && cnfgem=='f'){
      nprintf(nabla, NULL, "face_node[n*NABLA_NB_FACES+f]");
      break;
    }
    nprintf(nabla, "/*uniqueId c*/", "%c",cnfgem);
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
    if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP){
      char mnx[4]={'M','x','x','\0'};
      const char *var=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
      astNode *min,*max,*compound_statement=dfsFetch(job->nblParamsNode,rulenameToId("compound_statement"));
      compound_statement=compound_statement->parent;
      assert(compound_statement!=NULL);
      //printf("compound_statement->rule=%s",compound_statement->rule);
      assert(compound_statement->ruleid==rulenameToId("compound_statement"));
      /////////////////////////////////////
      // A *little* bit too cavalier here!
      /////////////////////////////////////
      min=max=NULL;
      min=dfsFetchToken(compound_statement,"min");
      max=dfsFetchToken(compound_statement,"max");
      assert(min!=NULL || max !=NULL);
      if (min!=NULL) {mnx[1]='i';mnx[2]='n'; nprintf(nabla,"/*MIN*/","/*OpenMP REDUCE MIN*/");}
      if (max!=NULL) {mnx[1]='a';mnx[2]='x'; nprintf(nabla,"/*MAX*/","/*OpenMP REDUCE MMAXIN*/");}
      //printf("mnx=%s\n",mnx); fflush(stdout);
      nprintf(nabla, NULL, "\n\t\t}/* des sources */\n\t}/* de l'ENUMERATE */\
\n\tfor (int i=0; i<threads; i+=1){\n\
//var=%s, mnx=%s\n\
\t\t%s=(Reduce%sToDouble(%s_per_thread[i])<Reduce%sToDouble(%s))?Reduce%sToDouble(%s_per_thread[i]):Reduce%sToDouble(%s); \
\n\t\t//info()<<\"%s=\"<<%s;\
  \n\t}\n\treturn ",var,mnx,var,mnx,var,mnx,var,mnx,var,mnx,var,var,var);
      job->parse.returnFromArgument=false;
    }else{
      nprintf(nabla, NULL, "\n\t\t}/* des sources */\n\t}/* de l'ENUMERATE */\n\treturn ");
    }
    job->parse.got_a_return=true;
    job->parse.got_a_return_and_the_semi_colon=false;
    break;
  }
  case ('{'):{nprintf(nabla, NULL, "{\n\t\t"); break; }    
  case ('&'):{nprintf(nabla, NULL, "&"); break; }    
  case (';'):{
    job->parse.variableIsArray=false;
    job->parse.alephKeepExpression=false;
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
    //if (n->token!=NULL) nprintf(nabla, NULL, "%s/*?*/ ", n->token);
    if (n->token!=NULL) nprintf(nabla, NULL, "%s ", n->token);
    break;
  }
  }
}
