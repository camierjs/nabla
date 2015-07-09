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


/***************************************************************************** 
 * Backend Generic for JOBS - New, Add, Last functions
 *****************************************************************************/
nablaJob *nMiddleJobNew(nablaEntity *entity){
  int i;
  nablaJob *job;
  job = (nablaJob *)malloc(sizeof(nablaJob));
  assert(job != NULL);
  job->is_an_entry_point=false;
  job->is_a_function=false;
  job->scope=job->region=job->item=job->name=job->xyz=job->drctn=NULL;
  job->at[0]=0;
  job->whenx=0;
  for(i=0;i<32;++i)
    job->whens[i]=HUGE_VAL;
  job->returnTypeNode=NULL;
  job->stdParamsNode=NULL;
  job->nblParamsNode=NULL;
  job->ifAfterAt=NULL;
  job->called_variables=NULL;
  job->in_out_variables=NULL;
  job->variables_to_gather_scatter=NULL;
  job->forall_item='\0';
  job->reduction=false;
   {// Outils de parsing
    job->parse.left_of_assignment_operator=false;
    job->parse.turnBracketsToParentheses=false;
    job->parse.enum_enum='\0';
    job->parse.got_a_return=false;
    job->parse.got_a_return_and_the_semi_colon=false;
    job->parse.imbricated_scopes=0;
    job->parse.isPostfixed=0;
    job->parse.isDotXYZ=0;
    job->parse.statementToDiffract=NULL;
    job->parse.diffracting=false;
    job->parse.diffractingXYZ=0;
    job->parse.entityScopeToDump=true;
    job->parse.selection_statement_in_compound_statement=false;
    job->parse.function_call_name=NULL;                                             
    job->parse.function_call_arguments=false;
    job->parse.postfix_constant=false;
    job->parse.variableIsArray=false;
    job->parse.alephKeepExpression=false;
    job->parse.inout=enum_in_variable;
    job->parse.iGather=0;
    job->parse.iScatter=0;
    job->parse.returnFromArgument=false;
  }
  job->entity=entity;
  job->next=NULL;
  return job; 
}

nablaJob *nMiddleJobAdd(nablaEntity *entity, nablaJob *job) {
  assert(job != NULL);
  if (entity->jobs == NULL)
    entity->jobs=job;
  else
    nMiddleJobLast(entity->jobs)->next=job;
  return job;
}

nablaJob *nMiddleJobLast(nablaJob *jobs) {
  while(jobs->next != NULL)
    jobs = jobs->next;
  return jobs;
}

nablaJob *nMiddleJobFind(nablaJob *jobs,char *name){
  nablaJob *job=jobs;
  while(job != NULL) {
    if(strcmp(job->name, name) == 0)
      return job;
    job = job->next;
  }
  return NULL;
}



/***************************************************************************** 
 * Backend Generic for JOBS - New, Add, Last functions
 *****************************************************************************/
static void actNablaJobParameterItem(astNode * n, void *current_item){
  current_item=n->children->token;
  dbg("\n\titem set to '%s' ", n->children->token);
}
static void actNablaJobParameterDirectDeclarator(astNode * n, void *current_item){
  dbg("'%s'", n->children->token);
}
void nMiddleScanForNablaJobParameter(astNode * n, int ruleid, nablaMain *arc){
  char *current_item=NULL;
  RuleAction tokact[]={
    {rulenameToId("nabla_item"),actNablaJobParameterItem},
    {rulenameToId("direct_declarator"),actNablaJobParameterDirectDeclarator},
    {0,NULL}};
  
  //dbg("\n\t[scanForNablaJobParameter] %s", n->token);
  if (n->tokenid=='@') return;
  if (n->ruleid==rulenameToId("compound_statement")) return;
  
  if (ruleid ==  n->ruleid){
    //dbg("\n\t[scanForNablaJobParameter] %s", n->token);
    // On peut en profiter pour générer les IN, OUT & INOUT pour ce job
    //getInOutPutsNodes(fOut, n->children->children->next, "CCCCCC");
    scanTokensForActions(n, tokact, (void*)&current_item);
  }
  if(n->children != NULL) nMiddleScanForNablaJobParameter(n->children, ruleid, arc);
  if(n->next != NULL) nMiddleScanForNablaJobParameter(n->next, ruleid, arc);
}


// ****************************************************************************
// * scanForNablaJobAtConstant
// ****************************************************************************
void nMiddleScanForNablaJobAtConstant(astNode *n, nablaMain *arc){
  for(;n not_eq NULL;n=n->next){
    if (n->tokenid!=AT) continue;
    dbg("\n\t[scanForNablaJobAtConstant] %s ", n->token);
    // Si ce Nabla Job a un 'AT', c'est qu'il faut renseigner les .config et .axl
    nablaJob *entry_point=nMiddleJobLast(arc->entity->jobs);
    entry_point->is_an_entry_point=true;      
    nMiddleAtConstantParse(n->next->children,arc,entry_point->at);
    nMiddleStoreWhen(arc,entry_point->at);
    return;
  }
}


void nMiddleScanForIfAfterAt(astNode *n, nablaJob *entry_point, nablaMain *nabla){
  for(;n not_eq NULL;n=n->next){
    if (n->tokenid!=IF) continue;
    dbg("\n\t[scanForIfAfterAt] %s ", n->token);
    // Si ce Nabla Job a un IF après le 'AT', c'est qu'il faudra l'insérer lors de la génération
    entry_point->ifAfterAt=n->next->next->children;
    return;
  }
}

// *****************************************************************************
// * dumpIfAfterAt
// ****************************************************************************
void nMiddleDumpIfAfterAt(astNode *n, nablaMain *nabla){
  //if ((n->ruleid == rulenameToId("primary_expression")) && (n->children->token!=NULL))
  if (n->token!=NULL){
    if (nMiddleOptionFindName(nabla->options, n->token)!=NULL){
      hprintf(nabla, "/*dumpIfAfterAt+Option*/", "options()->%s()", n->token);
    }else{
      hprintf(nabla, "/*dumpIfAfterAt*/", " %s ", n->token);
    }
  }
  if(n->children != NULL) nMiddleDumpIfAfterAt(n->children,nabla);
  if(n->next != NULL) nMiddleDumpIfAfterAt(n->next,nabla);
}


char nMiddleScanForNablaJobForallItem(astNode * n){
  char it;
  if (n->tokenid==FORALL)
    return n->next->children->token[0];
  if(n->children != NULL) if ((it=nMiddleScanForNablaJobForallItem(n->children))!='\0') return it;
  if(n->next != NULL) if ((it=nMiddleScanForNablaJobForallItem(n->next))!='\0') return it;
  return '\0';
}




/*****************************************************************************
 * Différentes actions pour un job Nabla
 *****************************************************************************/
void nMiddleJobParse(astNode *n, nablaJob *job){
  nablaMain *nabla=job->entity->main;
  const char cnfgem=job->item[0];
  
  //if (n->token) nprintf(nabla, NULL, "\n/*[nablaJobParse] token: '%s'*/", n->token);
  if (n->token)
    dbg("\n[nablaJobParse] token: '%s'?", n->token);

  if (job->parse.got_a_return && job->parse.got_a_return_and_the_semi_colon) return;
  
  if (nabla->hook->diffractStatement)
    nabla->hook->diffractStatement(nabla,job,&n);
 
  // On regarde si on a un appel de fonction avec l'argument_expression_list
  if ((n->ruleid == rulenameToId("argument_expression_list"))
      && (job->parse.function_call_arguments==false)){
    nprintf(nabla, "/*function_call_arguments*/", NULL);
    job->parse.function_call_arguments=true;
  }

    // On regarde de quel coté d'un assignment nous sommes
  if (n->ruleid == rulenameToId("assignment_expression")){
    if (n->children!=NULL){
      if (n->children->next!=NULL){
        if (n->children->next->ruleid == rulenameToId("assignment_operator")){
          job->parse.left_of_assignment_operator=true;
          nprintf(nabla, "/*isLeft*/", NULL);
        }
      }
    }
  }
  
  // On cherche les doubles postfix_expression suivies d'un '[' pour gestion des variables
  // nablaVariables retourne
  //             0 pour continuer (pas une variable nabla),
  //            '1' s'il faut return'er car on y a trouvé un nNablaSystem,
  //             2 pour continuer mais en informant turnTokenToVariable (hit but unknown)
  if (n->ruleid == rulenameToId("postfix_expression"))
    if (n->children->ruleid == rulenameToId("postfix_expression"))
      if (n->children->next != NULL)
        if (n->children->next->tokenid == '[')
          if ((job->parse.isPostfixed=nMiddleVariables(nabla,n,cnfgem,job->parse.enum_enum))==1){
            //nprintf(nabla, NULL, "/*isPostfixed=1, but returning*/",NULL);
            return;
          }

  // On on va chercher les .[x|y|z]
  if (n->ruleid == rulenameToId("postfix_expression"))
    if (n->children->ruleid == rulenameToId("postfix_expression"))
      if (n->children->next != NULL)
        if (n->children->next->tokenid == '.')
          if (n->children->next->next != NULL){
            if ((n->children->next->next->token[0] == 'x') ||
                (n->children->next->next->token[0] == 'y') ||
                (n->children->next->next->token[0] == 'z')
                ){
              //nprintf(nabla, NULL, "/*.%c*/", n->children->next->next->token[0]);
              //nprintf(nabla, "/*SettingIsDotXYZ*/", NULL);
              job->parse.isDotXYZ=n->children->next->next->token[0]-'w';
              // On flush cette postfix_expression pour la masquer de la génération
//#warning Should be a hook, but let it be for now
//              if (nabla->backend==BACKEND_CUDA) n->children->next=NULL;
              //nprintf(nabla, NULL, "/*nJob:isDotXYZ=%d*/", job->parse.isDotXYZ);
            }
          }
  
  // C'est le cas primary_expression suivi d'un nabla_item
  if ((n->ruleid == rulenameToId("primary_expression"))
      && (n->children->ruleid == rulenameToId("nabla_item"))){
    dbg("\n\t[nablaJobParse] C'est le cas primary_expression suivi d'un nabla_item");
    nprintf(nabla, "/*nabla_item*/", "\t%s",
            nabla->hook->item(job,cnfgem,n->children->children->token[0],job->parse.enum_enum));
  }
  
  // Dés qu'on a une primary_expression, on teste pour voir si ce n'est pas une option
  if ((n->ruleid == rulenameToId("primary_expression")) && (n->children->token!=NULL)){
    if (nMiddleTurnTokenToOption(n->children,nabla)!=NULL){
      dbg("\n\t[nablaJobParse] primaryExpression hits option");
      return;
    }
  }

  // Dés qu'on a une primary_expression,
  // on teste pour voir si ce n'est pas un argument que l'on return
  if ((n->ruleid == rulenameToId("primary_expression"))
      && (n->children->token!=NULL)
      && (job->parse.returnFromArgument)){
    dbg("\n\t[nablaJobParse] primary_expression test for return");
    if (nabla->hook->primary_expression_to_return){
      dbg("\n\t\t[nablaJobParse] primary_expression_to_return");
      if (nabla->hook->primary_expression_to_return(nabla,job,n)==true)
        return;
    }else{
       dbg("\n\t\t[nablaJobParse] ELSE primary_expression_to_return");
   }
  }
  
  // Dés qu'on a une primary_expression, on teste pour voir si ce n'est pas une variable
  if ((n->ruleid == rulenameToId("primary_expression")) && (n->children->token!=NULL)){
    nablaVariable *var;
    dbg("\n\t[nablaJobParse] primaryExpression=%s child->token=%s",n->rule,n->children->token);
    if ((var=nabla->hook->turnTokenToVariable(n->children, nabla,job))!=NULL){
      if (nabla->hook->turnBracketsToParentheses)
        nabla->hook->turnBracketsToParentheses(nabla,job,var,cnfgem);
      return;
    }
  }
  
  // On fait le switch des tokens
  nabla->hook->switchTokens(n, job);

  // On continue en s'enfonçant
  if (n->children != NULL)
    nMiddleJobParse(n->children, job);
   
  // On continue en allant à droite
  if (n->next != NULL)
    nMiddleJobParse(n->next, job);

}



/*****************************************************************************
 * Dump pour le header
 *****************************************************************************/
int nMiddleDumpParameterTypeList(FILE *file, astNode * n){
  int number_of_parameters_here=0;
  //if (n->token != NULL) fprintf(file, "/*dumpParameterTypeList %s:%d*/",n->token,number_of_parameters_here);
  
  if ((n->token != NULL )&&(strncmp(n->token,"xyz",3)==0)){// hit 'xyz'
    //fprintf(file, "/*xyz hit!*/");
    number_of_parameters_here+=1;
  }

  if ((n->token != NULL )&&(strncmp(n->token,"void",4)==0)){
    //fprintf(file, "/*void hit!*/");
    number_of_parameters_here-=1;
  }
  
  if ((n->token != NULL )&&(strncmp(n->token,"void",4)!=0)){// avoid 'void'
    if (strncmp(n->token,"restrict",8)==0){
      fprintf(file, "__restrict__ ");
    }else if (strncmp(n->token,"aligned",7)==0){
      fprintf(file, "/*aligned*/");
    }else{
      //fprintf(file, "/*%s*/", n->token);
      fprintf(file, "%s ", n->token);
      dbg("\n\t\t[dumpParameterTypeList] %s", n->token);
    }
  }
  // A chaque parameter_declaration, on incrémente le compteur de paramètre
  if (n->ruleid==rulenameToId("parameter_declaration")){
    //fprintf(file, "/*number_of_parameters_here+=1*/");
    dbg("\n\t\t[dumpParameterTypeList] number_of_parameters_here+=1");
    number_of_parameters_here+=1;
  }
  if (n->children != NULL)
    number_of_parameters_here+=nMiddleDumpParameterTypeList(file, n->children);
  if (n->next != NULL)
    number_of_parameters_here+=nMiddleDumpParameterTypeList(file, n->next);
  
  //fprintf(file, "/*return %d*/",number_of_parameters_here);
  return number_of_parameters_here;
}



/*****************************************************************************
 * Remplissage de la structure 'job'
 * Dump dans le src de la déclaration de ce job en fonction du backend
 *****************************************************************************/
void nMiddleJobFill(nablaMain *nabla,
                    nablaJob *job,
                    astNode *n,
                    const char *namespace){
  int numParams;
  astNode *nd;
  job->is_a_function=false;
  assert(job != NULL);
  job->scope  = dfsFetchFirst(n->children,rulenameToId("nabla_scope"));
  job->region = dfsFetchFirst(n->children,rulenameToId("nabla_region"));
  job->item   = dfsFetchFirst(n->children,rulenameToId("nabla_items"));
    
  // Si on a pas trouvé avec les items, cela doit être un matenvs
  if (job->item==NULL)
    job->item = dfsFetchFirst(n->children,rulenameToId("nabla_matenvs"));
  assert(job->item);
  job->rtntp = dfsFetchFirst(n->children,rulenameToId("type_specifier"));
  
  // On va chercher le premier identifiant qui est le nom du job
  nd=dfsFetchTokenId(n->children,IDENTIFIER);
  assert(nd);
  job->name=strdup(nd->token);
  //assert(n->children->next->next->token!=NULL);
  //job->name = strdup(n->children->next->next->token);
  job->name_utf8 = strdup(nd->token_utf8);
  //nprintf(nabla, NULL, "/*name=%s*/", job->name);
  dbg("\n\n\t[nablaJobFill] named '%s'", job->name);

  // On va chercher s'il y a des xyz dans les parameter_type_list
  //assert(n->children->next->next->next->children!=NULL);
  nd=dfsFetch(n->children,rulenameToId("parameter_type_list"));
  job->xyz = dfsFetchFirst(nd,rulenameToId("nabla_xyz_declaration"));
  job->drctn = dfsFetchFirst(n->children,
                              rulenameToId("nabla_xyz_direction"));
  // Vérification si l'on a des 'directions' dans les paramètres
  if (job->xyz!=NULL){
    dbg("\n\t[nablaJobFill] direction=%s, xyz=%s",
        job->drctn?job->drctn:"NULL",
        job->xyz?job->xyz:"NULL");
    nprintf(nabla, NULL, "\n\n/*For next job: xyz=%s*/", job->xyz);
  }
  // Récupération du type de retour
  assert(n->children->next->children->ruleid==rulenameToId("type_specifier"));
  job->returnTypeNode=n->children->next->children;
  // Récupération de la liste des paramètres
  nd=dfsFetch(n->children,rulenameToId("parameter_type_list"));
  job->stdParamsNode=nd->children;
  dbg("\n\t[nablaJobFill] scope=%s region=%s item=%s type_de_retour=%s name=%s",
      (job->scope!=NULL)?job->scope:"", (job->region!=NULL)?job->region:"",
      job->item, job->rtntp, job->name);
  // nMiddleScanForNablaJobParameter ne fait que dumper
  //nMiddleScanForNablaJobParameter(n->children, rulenameToId("nabla_parameter_list"), nabla);
  nMiddleScanForNablaJobAtConstant(n->children, nabla);
  nMiddleScanForIfAfterAt(n->children, job, nabla);
  // On remplit la ligne du fichier SRC
  nprintf(nabla, NULL, "\n\n\n\
// ********************************************************\n\
// * %s job\n\
// ********************************************************\n\
%s %s %s%s%s(", job->name,
          nabla->hook->entryPointPrefix(nabla,job),
          job->rtntp,
          namespace?namespace:"",
          namespace?(isAnArcaneModule(nabla)==true)?"Module::":"Service::":"",
          job->name);
  // On va chercher les paramètres standards
  numParams=nMiddleDumpParameterTypeList(nabla->entity->src, job->stdParamsNode);
  //nprintf(nabla, NULL,"/*numParams=%d*/",numParams);
  dbg("\n\t[nablaJobFill] numParams=%d", numParams);
  
  // On va chercher les paramètres nabla in/out/inout
  //nd=dfsFetch(n->children,rulenameToId("nabla_parameter"));
  job->nblParamsNode=n->children->next->next->next->next->next->next;
  //job->nblParamsNode=dfsFetch(n->children,rulenameToId("nabla_inout_parameter"));

  // Si on a un type de retour et des arguments
  if (numParams!=0 && strncmp(job->rtntp,"void",4)!=0){
    dbg("\n\t[nablaJobFill] Returning perhaps from an argument!");
    job->parse.returnFromArgument=true;
  }
  
  // On s'autorise un endroit pour insérer des paramètres
  dbg("\n\t[nablaJobFill] On s'autorise un endroit pour insérer des paramètres");
  if (nabla->hook->addExtraParameters!=NULL) 
    nabla->hook->addExtraParameters(nabla, job,&numParams);
  
  // Et on dump les in et les out
  dbg("\n\t[nablaJobFill] Et on dump les in et les out");
  if (nabla->hook->dumpNablaParameterList!=NULL){
    dbg("\n\t\t[nablaJobFill] job->nblParamsNode->ruleid is '%s'",job->nblParamsNode->rule);
    // On enlève ce teste tant qu'on autorise des jobs sans déclaration
    // Mais il faudra le remettre quand on l'obligera!
    // assert(job->nblParamsNode->ruleid==rulenameToId("nabla_parameter_list"));
    nabla->hook->dumpNablaParameterList(nabla,job,job->nblParamsNode,&numParams);
  }

  // On ferme la parenthèse des paramètres que l'on avait pas pris dans les tokens
  nprintf(nabla, NULL, "){");// du job

  if (job->parse.returnFromArgument)
    if (nabla->hook->returnFromArgument)
      nabla->hook->returnFromArgument(nabla,job);

  // On prépare le bon ENUMERATE suivant le forall interne
  // On saute l'éventuel forall du début
  dbg("\n\t[nablaJobFill] On prépare le bon ENUMERATE");
  if ((job->forall_item=nMiddleScanForNablaJobForallItem(n->children->next))!='\0')
    dbg("\n\t[nablaJobFill] scanForNablaJobForallItem found '%c'", job->forall_item);

  dbg("\n\t[nablaJobFill] On avance jusqu'au COMPOUND_JOB_INI afin de sauter les listes de paramètres");
  for(n=n->children->next; n->tokenid!=COMPOUND_JOB_INI; n=n->next);
  //n=n->next; // On saute le COMPOUND_JOB_INI *ou pas*
  
  dbg("\n\t[nablaJobFill] On cherche s'il y a un selection statement");
  if (dfsFetch(n,rulenameToId("selection_statement"))!=NULL){
    dbg("\n\t[nablaJobFill] Found a selection statement in this job!");
    job->parse.selection_statement_in_compound_statement=true;
  }else{
     dbg("\n\t[nablaJobFill] No selection statement in this job!");
 }
  
  dbg("\n\t[nablaJobFill] prefixEnumerate");
  nprintf(nabla, NULL, "\n\t%s", nabla->hook->prefixEnumerate(job));
  
  dbg("\n\t[nablaJobFill] dumpEnumerate");
  nprintf(nabla, NULL, "\n\t%s{", nabla->hook->dumpEnumerate(job));// de l'ENUMERATE_
  
  dbg("\n\t[nablaJobFill] postfixEnumerate");
  nprintf(nabla, NULL, "\n\t\t%s", nabla->hook->postfixEnumerate(job));
  dbg("\n\t[nablaJobFill] postfixEnumerate done");
  
  // Et on dump les tokens dans ce job
  dbg("\n\t[nablaJobFill] Now parsing...");
  nMiddleJobParse(n,job);

  if (!job->parse.got_a_return)
    nprintf(nabla, NULL, "}");// de l'ENUMERATE
  nprintf(nabla, NULL, "\n}");// du job
  dbg("\n\t[nablaJobFill] done");
}


