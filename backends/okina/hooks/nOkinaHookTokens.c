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


// ****************************************************************************
// * FORALL token switch
// ****************************************************************************
static void nOkinaHookTokenSwitchForall(astNode *n, nablaJob *job){
  // Preliminary pertinence test
  if (n->tokenid != FORALL) return;
  // Now we're allowed to work
  switch(n->next->children->tokenid){
  case(CELL):{
    job->parse.enum_enum='c';
    nprintf(job->entity->main, "/*chsf c*/", "FOR_EACH_NODE_WARP_CELL(c)");
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
  // On skip le 'nabla_item' qui nous a renseigné sur le type de forall
  *n=*n->next->next;
}


// ****************************************************************************
// * Différentes actions pour un job Nabla
// ****************************************************************************
void nOkinaHookTokenSwitch(astNode *n, nablaJob *job){
  nablaMain *nabla=job->entity->main;
  const char cnfgem=job->item[0];
  
  //if (n->token) nprintf(nabla, NULL, "\n/*token=%s*/",n->token);
  if (n->token)
    dbg("\n\t[nOkinaHookSwitchToken] token: '%s'?", n->token);
 
  nOkinaHookTokenSwitchForall(n,job);
  
  switch(n->tokenid){

  case (FILETYPE):{
    nprintf(nabla, "/*FILETYPE*/", "/*FILETYPE*/");
    break;
  }
  case (FILECALL):{
    nprintf(nabla, "/*FILECALL*/", "/*FILECALL*/");
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
    nprintf(nabla, "/*CONST*/", "%sconst ", job->entity->main->pragma->align());
    break;
  }
  case(ALIGNED):{
    nprintf(nabla, "/*ALIGNED*/", "%s", job->entity->main->pragma->align());
    break;
  }
    
  case(REAL):{
    nprintf(nabla, "/*Real*/", "real ");
    break;
  }
  case(REAL3):{
    if ((job->entity->libraries&(1<<real))!=0)
      exit(NABLA_ERROR|
           fprintf(stderr,
                   "[nOkinaHookSwitchToken] Real3 can't be used with R library!\n"));
    
    assert((job->entity->libraries&(1<<real))==0);
    nprintf(nabla, "/*Real3*/", "real3 ");
    break;
  }
  case(INTEGER):{
    nprintf(nabla, "/*INTEGER*/", "integer ");
    break;
  }
  case(NATURAL):{
    nprintf(nabla, "/*NATURAL*/", "integer ");
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
    nprintf(nabla, "UIDTYPE", "int ");
    break;
  }
    
  case(FORALL_INI):{
    nprintf(nabla, "/*FORALL_INI*/", "{\n\t\t\t");//FORALL_INI
    nprintf(nabla, "/*okinaGather*/", "%s",nOkinaHookGather(job));
    break;
  }
  case(FORALL_END):{
    nprintf(nabla, "/*okinaScatter*/", nOkinaHookScatter(job));
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
      if (job->item[0]=='c') nprintf(nabla, "/*tBktOFF*/", "[c]]");
      if (job->item[0]=='n') nprintf(nabla, "/*tBktOFF*/", "[c]]");
      job->parse.turnBracketsToParentheses=false;
    }else{
      nprintf(nabla, NULL, "]");
    }
    //nprintf(nabla, "/*FlushingIsPostfixed*/","/*isDotXYZ=%d*/",job->parse.isDotXYZ);
    //if (job->parse.isDotXYZ==1) nprintf(nabla, NULL, "[c]]/*]+FlushingIsPostfixed*/");
                                        //"[((c>>WARP_BIT)*((1+1+1)<<WARP_BIT))+(c&((1<<WARP_BIT)-1))]]/*]+FlushingIsPostfixed*/");
    //if (job->parse.isDotXYZ==1) nprintf(nabla, NULL, NULL);
    //if (job->parse.isDotXYZ==2) nprintf(nabla, NULL, NULL);
    //if (job->parse.isDotXYZ==3) nprintf(nabla, NULL, NULL);
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
  case (NBNODE):{ if (cnfgem=='c') nprintf(nabla, NULL, "8/*cell->nbNode()*/"); break; }    
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
    if (cnfgem=='c' && (nabla->colors&BACKEND_COLOR_OKINA_STD)==BACKEND_COLOR_OKINA_STD)
      nprintf(nabla, "/*uniqueId c*/", "(WARP_SIZE*c)");
    if (cnfgem=='c' && (nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
      nprintf(nabla, "/*uniqueId c*/", "integer(WARP_SIZE*c+0,WARP_SIZE*c+1)");
    if (cnfgem=='c' && (nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
      nprintf(nabla, "/*uniqueId c*/", "integer(WARP_SIZE*c+0,WARP_SIZE*c+1,WARP_SIZE*c+2,WARP_SIZE*c+3)");
    if (cnfgem=='c' && (nabla->colors&BACKEND_COLOR_OKINA_AVX2)==BACKEND_COLOR_OKINA_AVX2)
      nprintf(nabla, "/*uniqueId c*/", "integer(WARP_SIZE*c+0,WARP_SIZE*c+1,WARP_SIZE*c+2,WARP_SIZE*c+3)");
    if (cnfgem=='c' && (nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
      nprintf(nabla, "/*uniqueId c*/", "integer(WARP_SIZE*c+0,WARP_SIZE*c+1,WARP_SIZE*c+2,WARP_SIZE*c+3,WARP_SIZE*c+4,WARP_SIZE*c+5,WARP_SIZE*c+6,WARP_SIZE*c+7)");
    
    if (cnfgem=='n'&& (nabla->colors&BACKEND_COLOR_OKINA_STD)==BACKEND_COLOR_OKINA_STD)
      nprintf(nabla, "/*uniqueId n*/", "(n)");
    if (cnfgem=='n'&& (nabla->colors&BACKEND_COLOR_OKINA_SSE)==BACKEND_COLOR_OKINA_SSE)
      nprintf(nabla, "/*uniqueId n*/", "integer(WARP_SIZE*n+0,WARP_SIZE*n+1)");
    if (cnfgem=='n'&& (nabla->colors&BACKEND_COLOR_OKINA_AVX)==BACKEND_COLOR_OKINA_AVX)
      nprintf(nabla, "/*uniqueId n*/", "integer(WARP_SIZE*n+0,WARP_SIZE*n+1,WARP_SIZE*n+2,WARP_SIZE*n+3)");
    if (cnfgem=='n'&& (nabla->colors&BACKEND_COLOR_OKINA_MIC)==BACKEND_COLOR_OKINA_MIC)
      nprintf(nabla, "/*uniqueId n*/", "integer(WARP_SIZE*n+0,WARP_SIZE*n+1,WARP_SIZE*n+2,WARP_SIZE*n+3,WARP_SIZE*n+4,WARP_SIZE*n+5,WARP_SIZE*n+6,WARP_SIZE*n+7)");
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
    //if (n->token!=NULL) nprintf(nabla, NULL, "/*default*/%s ", n->token);
    break;
  }
  }
}

