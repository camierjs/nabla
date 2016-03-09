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
#include "backends/arcane/arcane.h"


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
  job->nb_in_item_set=0;
  job->scope=job->region=job->item=job->item_set=job->name=job->xyz=job->direction=NULL;
  job->at[0]=0;
  job->when_sign=1.0;// permet de gérer les '-' pour les '@'
  job->when_index=0; // permet de gérer les ',' pour les '@'
  job->when_depth=0; // permet de spécifier la hiérarchie des '@'
  for(i=0;i<NABLA_JOB_WHEN_MAX;++i)
    job->whens[i]=HUGE_VAL;
  job->jobNode=NULL;
  job->returnTypeNode=NULL;
  job->stdParamsNode=NULL;
  job->nblParamsNode=NULL;
  job->ifAfterAt=NULL;
  job->called_variables=NULL;
  job->variables_to_gather_scatter=NULL;
  job->forall_item='\0';
  job->reduction=false;
  job->swirl_index=0;
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
void nMiddleScanForNablaJobAtConstant(astNode *n, nablaMain *nabla){
  for(;n not_eq NULL;n=n->next){
    if (n->tokenid!=AT) continue;
    // Si ce Nabla Job a un 'AT', c'est qu'il faut renseigner les .config et .axl
    nablaJob *entry_point=nMiddleJobLast(nabla->entity->jobs);
    dbg("\n\t[scanForNablaJobAtConstant] entry_point '%s', token: '%s'", entry_point->name, n->token);
    entry_point->is_an_entry_point=true;      
    dbg("\n\t[scanForNablaJobAtConstant] dfsDumpToken:"); dfsDumpToken(n->next->children);
    entry_point->whens[entry_point->when_index]=0.0;
    nMiddleAtConstantParse(entry_point,n->next->children,nabla);
    nMiddleStoreWhen(entry_point, nabla);
    return;
  }
}


// *****************************************************************************
// * nMiddleScanForIfAfterAt
// *****************************************************************************
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
// * static dumpIfAfterAt functions
// * A cleaner!
// ****************************************************************************
static void dumpIfAfterAtFormat(astNode *n, nablaMain *nabla, bool dump_in_header,char *info, char *format){
  if (dump_in_header)
    hprintf(nabla, info, format, n->token);
  else
    nprintf(nabla, info, format, n->token);
}
static void dumpIfAfterAtToken(astNode *n, nablaMain *nabla, bool dump_in_header){
  if (dump_in_header)
    hprintf(nabla, "/*h dumpIfAfterAtToken*/", " %s ", n->token);
  else
    nprintf(nabla, "/*n dumpIfAfterAtToken*/", " %s ", n->token);
}
static void dumpIfAfterAt(astNode *n, nablaMain *nabla, bool dump_in_header){
// #warning LAMBDA vs ARCANE in middlend!
  if (!dump_in_header){ // LAMBDA ici
    nablaVariable *var=NULL;
    if ((var=nMiddleVariableFind(nabla->variables, n->token))!=NULL){
      nprintf(nabla, NULL, " global_%s[0] ", n->token);
    }else nprintf(nabla, NULL, " %s ", n->token);
    return;
  }
  // ARCANE ici
  nablaVariable *var=NULL;
  char *info=NULL;
  char *format=NULL;
  if (n->token) dbg("\n\t[dumpIfAfterAt] token='%s' ", n->token);
  // Si on hit une option
  if (nMiddleOptionFindName(nabla->options, n->token)!=NULL){
    info="/*dumpIfAfterAt+Option*/";
//#warning HOOK NEEDED HERE!
    format="options()->%s()";
  }
  // Si on hit une variable
  if ((var=nMiddleVariableFind(nabla->variables, n->token))!=NULL){
    info="/*dumpIfAfterAt+Variable*/";
    format="m_global_%s()";
  }
  // On choisit le format ou pas
  if (info!=NULL)
    dumpIfAfterAtFormat(n,nabla,dump_in_header,info,format);
  else dumpIfAfterAtToken(n,nabla,dump_in_header);
}


// *****************************************************************************
// * dumpIfAfterAt
// ****************************************************************************
void nMiddleDumpIfAfterAt(astNode *n, nablaMain *nabla, bool dump_in_header){
  if (n->token!=NULL) dumpIfAfterAt(n,nabla,dump_in_header);
  if(n->children != NULL) nMiddleDumpIfAfterAt(n->children,nabla,dump_in_header);
  if(n->next != NULL) nMiddleDumpIfAfterAt(n->next,nabla,dump_in_header);
}


// *****************************************************************************
// * nMiddleScanForNablaJobForallItem
// *****************************************************************************
char nMiddleScanForNablaJobForallItem(astNode *n){
  char it;
  if (n->tokenid==FORALL)
    return n->next->children->token[0];
  if(n->children != NULL)
    if ((it=nMiddleScanForNablaJobForallItem(n->children))!='\0')
      return it;
  if(n->next != NULL)
    if ((it=nMiddleScanForNablaJobForallItem(n->next))!='\0')
      return it;
  return '\0';
}


// ****************************************************************************
// * Différentes actions pour un job Nabla
// ****************************************************************************
void nMiddleJobParse(astNode *n, nablaJob *job){
  nablaMain *nabla=job->entity->main;
  const char cnfgem=job->item[0];
  
  //if (n->token) dbg("\n\t\t\t[nablaJobParse] token '%s'?", n->token);

  if (job->parse.got_a_return &&
      job->parse.got_a_return_and_the_semi_colon) return;
   
  // On regarde si on a un appel de fonction avec l'argument_expression_list
  if ((n->ruleid == rulenameToId("argument_expression_list"))
      && (job->parse.function_call_arguments==false))
    job->parse.function_call_arguments=true;

    // On regarde de quel coté d'un assignment nous sommes
  if (n->ruleid == rulenameToId("assignment_expression"))
    if (n->children!=NULL)
      if (n->children->next!=NULL)
        if (n->children->next->ruleid == rulenameToId("assignment_operator"))
          job->parse.left_of_assignment_operator=true;
  
  // On cherche les doubles postfix_expression suivies d'un '[' pour gestion des variables
  // nablaVariables retourne
  //             0 pour continuer (pas une variable nabla),
  //            '1' s'il faut return'er car on y a trouvé un nNablaSystem,
  //             2 pour continuer mais en informant turnTokenToVariable (hit but unknown)
  if (n->ruleid == rulenameToId("postfix_expression"))
    if (n->children->ruleid == rulenameToId("postfix_expression"))
      if (n->children->next != NULL)
        if (n->children->next->tokenid == '[')
          if ((job->parse.isPostfixed=nMiddleVariables(nabla,n,cnfgem,job->parse.enum_enum))==1)
            return;
          

  // On on va chercher les .[x|y|z]
  if (n->ruleid == rulenameToId("postfix_expression"))
    if (n->children->ruleid == rulenameToId("postfix_expression"))
      if (n->children->next != NULL)
        if (n->children->next->tokenid == '.')
          if (n->children->next->next != NULL)
            if ((n->children->next->next->token[0] == 'x') ||
                (n->children->next->next->token[0] == 'y') ||
                (n->children->next->next->token[0] == 'z'))
              job->parse.isDotXYZ=n->children->next->next->token[0]-'w';
  
  // C'est le cas primary_expression suivi d'un nabla_item
  if ((n->ruleid == rulenameToId("primary_expression"))
      && (n->children->ruleid == rulenameToId("nabla_item")))
    nprintf(nabla, "/*nabla_item*/", "\t%s",
            nabla->hook->forall->item(job,
                                      cnfgem,
                                      n->children->children->token[0],
                                      job->parse.enum_enum));
    
  // Dés qu'on a une primary_expression, on teste pour voir si ce n'est pas une option
  if ((n->ruleid == rulenameToId("primary_expression")) && (n->children->token!=NULL))
    if (nMiddleTurnTokenToOption(n->children,nabla)!=NULL)
      return;
  
  // Dés qu'on a une primary_expression,
  // on teste pour voir si ce n'est pas un argument que l'on return
  // A confirmer si cela est toujours utile
  if ((n->ruleid == rulenameToId("primary_expression"))
      && (n->children->token!=NULL)
      && (job->parse.returnFromArgument)){
    dbg("\n\t[nablaJobParse] primary_expression test for return");
    if (nabla->hook->grammar->primary_expression_to_return){
      dbg("\n\t\t[nablaJobParse] primary_expression_to_return");
      if (nabla->hook->grammar->primary_expression_to_return(nabla,job,n)==true)
        return;
    }else{
       dbg("\n\t\t[nablaJobParse] ELSE primary_expression_to_return");
    }
  }
  
  // Dés qu'on a une primary_expression, on teste pour voir si ce n'est pas une variable
  if ((n->ruleid == rulenameToId("primary_expression")) && (n->children->token!=NULL)){
    nablaVariable *var;
    //dbg("\n\t[nablaJobParse] primaryExpression=%s child->token=%s",n->rule,n->children->token);
    if ((var=nabla->hook->token->variable(n->children, nabla,job))!=NULL){
      if (nabla->hook->token->turnBracketsToParentheses)
        nabla->hook->token->turnBracketsToParentheses(nabla,job,var,cnfgem);
      return;
    }
  }

  // Gestion du 'is_test', IS_OP_INI & IS_OP_END
  if (n->tokenid == IS_OP_INI) nabla->hook->token->isTest(nabla,job,n,IS_OP_INI);
  if (n->tokenid == IS_OP_END) nabla->hook->token->isTest(nabla,job,n,IS_OP_END);
  if (n->ruleid == rulenameToId("is_test")) nabla->hook->token->isTest(nabla,job,n,IS);

  // On fait le switch du token
  nabla->hook->token->svvitch(n, job);

  if (n->children != NULL) nMiddleJobParse(n->children, job);
  if (n->next != NULL) nMiddleJobParse(n->next, job);
}



// ****************************************************************************
// * Remplissage de la structure 'job
// * Dump dans le src de la déclaration de ce job en fonction du backend
// ****************************************************************************
void nMiddleJobFill(nablaMain *nabla,
                    nablaJob *job,
                    astNode *n,
                    const char *namespace){
  char *set=calloc(NABLA_MAX_FILE_NAME,1);
  int numParams=0;
  astNode *nd;
  job->is_a_function=false;
  assert(job != NULL);
  
  // Récupération scope/region/items
  job->scope  = dfsFetchFirst(n->children,rulenameToId("nabla_scope"));
  job->region = dfsFetchFirst(n->children,rulenameToId("nabla_region"));
  job->item   = dfsFetchFirst(n->children,rulenameToId("nabla_items"));
  // On va chercher tous les items de l'ensemble, on stop au token SET_END
  job->item_set = dfsFetchAll(n->children->children,rulenameToId("nabla_items"),
                              &job->nb_in_item_set,set);
  assert(job->item);
  // On test en DFS dans le nabla_job_decl pour voir s'il y a un type retour
  job->return_type = dfsFetchFirst(n->children->children,rulenameToId("type_specifier"));
  
  // Nom du job:
  // On va chercher le premier identifiant qui est le nom du job *ou pas*
  //dbg("\n\n\t// **********************************************************************");
  if (!job->return_type){ // Pas de 'void', pas de nom, on en créé un
    const char* kName=mkktemp("kernel");
    job->has_to_be_unlinked=true;
    job->name=strdup(kName);
    job->name_utf8=strdup(kName);
    job->return_type=strdup("void");
  }else{
    nd=dfsFetchTokenId(n->children,IDENTIFIER);
    assert(nd);
    job->name=strdup(nd->token);
    job->name_utf8 = strdup(nd->token_utf8);
  }
  dbg("\n\n* Nabla Job: %s", job->name); // org-mode job name
  dbg("\n\t[nablaJobFill] Kernel named '%s', on item '%s', item_set=%s",
      job->name,job->item,job->item_set);
  dbg("\n\t[nablaJobFill] Kernel item set idx=%d, job->item_set=%s, set=%s",
      job->nb_in_item_set,
      job->item_set,
      set);
  //dbg("\n\t// **********************************************************************");
  
  // Scan DFS pour récuérer les in/inout/out
  // Et on dump dans le log les tokens de ce job
  dbg("\n\t[nablaJobFill] Now dfsVariables:");
  dfsVariables(nabla,job,n,false);
  dfsVariablesDump(nabla,job,n);
  
  // On va chercher s'il y a des xyz dans les parameter_type_list
  // Ne devrait plus être utilisé
  job->xyz = dfsFetchFirst(n,rulenameToId("nabla_xyz_declaration"));
  job->direction = dfsFetchFirst(n->children,
                              rulenameToId("nabla_xyz_direction"));
  
  // Vérification si l'on a des 'directions' dans les paramètres
  if (job->xyz!=NULL){
    dbg("\n\t[nablaJobFill] direction=%s, xyz=%s",
        job->direction?job->direction:"NULL",
        job->xyz?job->xyz:"NULL");
    nprintf(nabla, NULL, "\n\n/*For next job: xyz=%s*/", job->xyz);
  }
  // Récupération du type de retour
  job->returnTypeNode=dfsFetch(n->children->children,rulenameToId("type_specifier"));
  
  // Récupération de la liste des paramètres
  nd=dfsFetch(n->children,rulenameToId("parameter_type_list"));
  if (nd) job->stdParamsNode=nd->children;
  dbg("\n\t[nablaJobFill] scope=%s region=%s item=%s type_de_retour=%s name=%s",
      (job->scope!=NULL)?job->scope:"", (job->region!=NULL)?job->region:"",
      job->item, job->return_type, job->name);

  // Gestion du '@' de ce job
  nMiddleScanForNablaJobAtConstant(n->children, nabla);

  // Gestion du 'if' de ce job
  nMiddleScanForIfAfterAt(n->children, job, nabla);

  // Dump dans les fichiers
  
  // On remplit la ligne du fichier SRC
  nprintf(nabla, NULL, "\n\n\n\
// ********************************************************\n\
// * %s job\n\
// ********************************************************\n\
%s %s %s%s%s(", job->name,
          nabla->hook->call->entryPointPrefix(nabla,job),
          job->return_type,
          namespace?namespace:"",
          namespace?(isAnArcaneModule(nabla)==true)?"Module::":"Service::":"",
          job->name);
  
  // On va chercher les paramètres standards
  // Si used_options et used_variables ont été utilisées
  numParams=nMiddleDumpParameterTypeList(nabla,nabla->entity->src, job->stdParamsNode);
  nprintf(nabla, NULL,"/*job numParams=%d*/",numParams);
  if (nabla->hook->grammar->dfsVariable()) // Okina still is false here
    nMiddleParamsDumpFromDFS(nabla,job,numParams);
    
  // On va chercher les paramètres nabla in/out/inout
  job->nblParamsNode=dfsFetch(n->children,rulenameToId("nabla_parameter_list"));

  // Si on a un type de retour et des arguments
  // Ne devrait plus y en avoir
  if (numParams!=0 && strncmp(job->return_type,"void",4)!=0){
    dbg("\n\t[nablaJobFill] Returning perhaps from an argument!");
    job->parse.returnFromArgument=true;
  }
  
  // On s'autorise un endroit pour insérer des paramètres
  dbg("\n\t[nablaJobFill] On s'autorise un endroit pour insérer des paramètres");
  if (nabla->hook->call->addExtraParameters!=NULL) 
    nabla->hook->call->addExtraParameters(nabla, job,&numParams);
  
  // Et on dump les in et les out
  dbg("\n\t[nablaJobFill] Et on dump les in et les out");
  if (nabla->hook->call->dumpNablaParameterList!=NULL)
    nabla->hook->call->dumpNablaParameterList(nabla,job,job->nblParamsNode,&numParams);
  
  // On ferme la parenthèse des paramètres que l'on avait pas pris dans les tokens
  nprintf(nabla, NULL, "){");// du job

  if (job->parse.returnFromArgument)
    if (nabla->hook->grammar->returnFromArgument)
      nabla->hook->grammar->returnFromArgument(nabla,job);

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
  nprintf(nabla, NULL, "\n\t%s",  cHOOKj(nabla,forall,prefix,job));
  
  dbg("\n\t[nablaJobFill] dumpEnumerate");
  nprintf(nabla, NULL, "%s{", nabla->hook->forall->dump(job));// de l'ENUMERATE_
  
  dbg("\n\t[nablaJobFill] postfixEnumerate");
  nprintf(nabla, NULL, "\n\t\t%s", nabla->hook->forall->postfix(job));
  dbg("\n\t[nablaJobFill] postfixEnumerate done");
  
  dbg("\n\t[nablaJobFill] NOW PARSING...");
  nMiddleJobParse(n,job);

  if (!job->parse.got_a_return)
    nprintf(nabla, NULL, "}");// de l'ENUMERATE
  
  nprintf(nabla, NULL, "\n}\n");// du job, on rajoute un \n pour les preproc possibles
  dbg("\n\t[nablaJobFill] done");
}


