/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nablaJobs.c
 * Author   : Camier Jean-Sylvain
 * Created  : 2012.12.13
 * Updated  : 2012.12.13
 *****************************************************************************
 * Description:
 *****************************************************************************
 * Date			Author	Description
 * 2012.12.13	camierjs	Creation
 *****************************************************************************/
#include "nabla.h"
#include "nabla.tab.h"


/***************************************************************************** 
 * Backend Generic for JOBS - New, Add, Last functions
 *****************************************************************************/
nablaJob *nablaJobNew(nablaEntity *entity){
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
  job->foreach_item='\0';
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

nablaJob *nablaJobAdd(nablaEntity *entity, nablaJob *job) {
  assert(job != NULL);
  if (entity->jobs == NULL)
    entity->jobs=job;
  else
    nablaJobLast(entity->jobs)->next=job;
  return job;
}

nablaJob *nablaJobLast(nablaJob *jobs) {
  while(jobs->next != NULL)
    jobs = jobs->next;
  return jobs;
}

nablaJob *nablaJobFind(nablaJob *jobs,char *name){
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
static void actNablaJobParameterItem(astNode * n, void *generic_arg){
  dbg(" (%s,", n->children->token);
}
static void actNablaJobParameterDirectDeclarator(astNode * n, void *generic_arg){
  dbg("%s)", n->children->token);
}
void scanForNablaJobParameter(astNode * n, int ruleid, nablaMain *arc){
  RuleAction tokact[]={
    {rulenameToId("nabla_item"),actNablaJobParameterItem},
    {rulenameToId("direct_declarator"),actNablaJobParameterDirectDeclarator},
    {0,NULL}};
  if (ruleid ==  n->ruleid){
    dbg("\n\t[scanForNablaJobParameter] %s", n->children->children->token);
    // On peut en profiter pour générer les IN, OUT & INOUT pour ce job
    //getInOutPutsNodes(fOut, n->children->children->next, "CCCCCC");
    scanTokensForActions(n, tokact, (void*)arc);
  }
  if(n->children != NULL) scanForNablaJobParameter(n->children, ruleid, arc);
  if(n->next != NULL) scanForNablaJobParameter(n->next, ruleid, arc);
}

void scanForNablaJobAtConstant(astNode * n, int ruleid, nablaMain *arc){
  for(;n not_eq NULL;n=n->next){
    if (n->tokenid!=AT) continue;
    dbg("\n\t[scanForNablaJobAtConstant] %s ", n->token);
    // Si ce Nabla Job a un 'AT', c'est qu'il faut renseigner les .config et .axl
    nablaJob *entry_point=nablaJobLast(arc->entity->jobs);
    entry_point->is_an_entry_point=true;      
    nablaAtConstantParse(n->next->children,arc,entry_point->at);
    nablaStoreWhen(arc,entry_point->at);
    return;
  }
}

void scanForIfAfterAt(astNode *n, nablaJob *entry_point, nablaMain *nabla){
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
void dumpIfAfterAt(astNode *n, nablaMain *nabla){
  //if ((n->ruleid == rulenameToId("primary_expression")) && (n->children->token!=NULL))
  if (n->token!=NULL){
    if (findOptionName(nabla->options, n->token)!=NULL){
      hprintf(nabla, "/*dumpIfAfterAt+Option*/", "options()->%s()", n->token);
    }else{
      hprintf(nabla, "/*dumpIfAfterAt*/", " %s ", n->token);
    }
  }
  if(n->children != NULL) dumpIfAfterAt(n->children,nabla);
  if(n->next != NULL) dumpIfAfterAt(n->next,nabla);
}


char scanForNablaJobForeachItem(astNode * n){
  char it;
  if (n->tokenid==FOREACH)
    return n->next->children->token[0];
  if(n->children != NULL) if ((it=scanForNablaJobForeachItem(n->children))!='\0') return it;
  if(n->next != NULL) if ((it=scanForNablaJobForeachItem(n->next))!='\0') return it;
  return '\0';
}




/*****************************************************************************
 * Différentes actions pour un job Nabla
 *****************************************************************************/
void nablaJobParse(astNode *n, nablaJob *job){
  nablaMain *nabla=job->entity->main;
  const char cnfgem=job->item[0];

  if (job->parse.got_a_return && job->parse.got_a_return_and_the_semi_colon) return;
  
  if (nabla->hook->diffractStatement)
    nabla->hook->diffractStatement(nabla,job,&n);
 
  // On regarde si on a un appel de fonction avec l'argument_expression_list
  if ((n->ruleid == rulenameToId("argument_expression_list")) && (job->parse.function_call_arguments==false)){
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
          if ((job->parse.isPostfixed=nablaVariables(nabla,n,cnfgem,job->parse.enum_enum))==1){
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
              nprintf(nabla, "/*SettingIsDotXYZ*/", NULL);
              job->parse.isDotXYZ=n->children->next->next->token[0]-'w';
              // On flush cette postfix_expression pour la masquer de la génération
//#warning Should be a hook, but let it be for now
              if (nabla->backend==BACKEND_CUDA) n->children->next=NULL;
              //nprintf(nabla, NULL, "/*isDotXYZ=%d*/", job->parse.isDotXYZ);
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
    if (turnTokenToOption(n->children,nabla)!=NULL){
      dbg("\n\t[nablaJobParse] primaryExpression hits option");
      return;
    }
  }

  // Dés qu'on a une primary_expression, on teste pour voir si ce n'est pas un argument que l'on return
  if ((n->ruleid == rulenameToId("primary_expression"))
      && (n->children->token!=NULL)
      && (job->parse.returnFromArgument)){
    if (nabla->hook->primary_expression_to_return)
      nabla->hook->primary_expression_to_return(nabla,job,n);
    return;
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
    nablaJobParse(n->children, job);
   
  // On continue en allant à droite
  if (n->next != NULL)
    nablaJobParse(n->next, job);

}



/*****************************************************************************
 * Dump pour le header
 *****************************************************************************/
int dumpParameterTypeList(FILE *file, astNode * n){
  int number_of_parameters_here=0;
  if ((n->token != NULL )&&(strncmp(n->token,"void",4)!=0)){// avoid 'void'
    if (strncmp(n->token,"restrict",8)==0){
      fprintf(file, "__restrict__ ");
    }else if (strncmp(n->token,"aligned",7)==0){
      fprintf(file, "/*aligned*/");
    }else{
      fprintf(file, "%s ", n->token);
      dbg("\n\t\t[dumpParameterTypeList] %s", n->token);
    }
  }
  // A chaque parameter_declaration, on incrémente le compteur de paramètre
  if (n->ruleid==rulenameToId("parameter_declaration")){
    dbg("\n\t\t[dumpParameterTypeList] number_of_parameters_here+=1");
    number_of_parameters_here+=1;
  }
  if (n->children != NULL)
    number_of_parameters_here+=dumpParameterTypeList(file, n->children);
  if (n->next != NULL)
    number_of_parameters_here+=dumpParameterTypeList(file, n->next);
  return number_of_parameters_here;
}



/*****************************************************************************
 * Remplissage de la structure 'job'
 * Dump dans le src de la déclaration de ce job en fonction du backend
 *****************************************************************************/
void nablaJobFill(nablaMain *nabla,
                  nablaJob *job,
                  astNode *n,
                  const char *namespace){
  int numParams;
  job->is_a_function=false;
  assert(job != NULL);
  /*if (nabla->optionDumpTree==true)
    fprintf(nabla->dot, "\n\t%sJob_%s;", job->item, job->name);
  */
  job->scope  = dfsFetchFirst(n->children,rulenameToId("nabla_scope"));
  job->region = dfsFetchFirst(n->children,rulenameToId("nabla_region"));
  job->item   = dfsFetchFirst(n->children,rulenameToId("nabla_items"));
  // Si on a pas trouvé avec les items, cela doit être un matenvs
  if (job->item==NULL)
    job->item = dfsFetchFirst(n->children,rulenameToId("nabla_matenvs"));
  assert(job->item!=NULL);
  job->rtntp  = dfsFetchFirst(n->children,rulenameToId("type_specifier"));
  assert(n->children->next->next->token!=NULL);
  job->name   = strdup(n->children->next->next->token);
  job->name_utf8   = strdup(n->children->next->next->token_utf8);
  //nprintf(nabla, NULL, "/*name=%s*/", job->name);
  assert(n->children->next->next->next->children!=NULL);
  dbg("\n\n\t[nablaJobFill] named '%s'", job->name);
  job->xyz    = dfsFetchFirst(n->children->next->next->next->children,
                              rulenameToId("nabla_xyz_declaration"));
  //nprintf(nabla, NULL, "/*xyz=%s*/", job->xyz);
  job->drctn  = dfsFetchFirst(n->children->next->next->next->children,
                              rulenameToId("nabla_xyz_direction"));
  // Vérification si l'on a des 'directions' dans les paramètres
  if (job->xyz!=NULL){
    dbg("\n\t[nablaJobFill] direction=%s, xyz=%s",
        job->drctn?job->drctn:"NULL",
        job->xyz?job->xyz:"NULL");
    nprintf(nabla, NULL, "/*xyz=%s*/", job->xyz);
  }
  // Récupération du type de retour
  assert(n->children->next->children->ruleid==rulenameToId("type_specifier"));
  job->returnTypeNode=n->children->next->children;
  //dbg("\n\t[nablaJobFill] Type de retour: %s", job->returnTypeNode->token);
  // Récupération de la liste des paramètres
  assert(n->children->next->next->next->ruleid==rulenameToId("parameter_type_list"));
  job->stdParamsNode=n->children->next->next->next->children;
  dbg("\n\t[nablaJobFill] scope=%s region=%s item=%s type_de_retour=%s name=%s",
      (job->scope!=NULL)?job->scope:"", (job->region!=NULL)?job->region:"",
      job->item, job->rtntp, job->name);
  scanForNablaJobParameter(n->children, rulenameToId("nabla_parameter"), nabla);
  scanForNablaJobAtConstant(n->children, rulenameToId("at_constant"), nabla);
  scanForIfAfterAt(n->children, job, nabla);
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
  numParams=dumpParameterTypeList(nabla->entity->src, job->stdParamsNode);
  dbg("\n\t[nablaJobFill] numParams=%d", numParams);
  job->nblParamsNode=n->children->next->next->next->next;

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
  if (nabla->hook->dumpNablaParameterList!=NULL)
    nabla->hook->dumpNablaParameterList(nabla,job,job->nblParamsNode,&numParams);

  // On ferme la parenthèse des paramètres que l'on avait pas pris dans les tokens
  nprintf(nabla, NULL, "){// du job");

  if (job->parse.returnFromArgument)
    if (nabla->hook->returnFromArgument)
      nabla->hook->returnFromArgument(nabla,job);

  
  // On prépare le bon ENUMERATE
  dbg("\n\t[nablaJobFill] On prépare le bon ENUMERATE");
  if ((job->foreach_item=scanForNablaJobForeachItem(n))!='\0')
    dbg("\n\t[nablaJobFill] scanForNablaJobForeachItem found '%c'", job->foreach_item);

  // On avance jusqu'au COMPOUND_JOB_INI afin de sauter les listes de paramètres
  for(n=n->children->next; n->tokenid!=COMPOUND_JOB_INI; n=n->next);
  //n=n->next; // On saute le COMPOUND_JOB_INI *ou pas*
  
  // On cherche s'il y a un selection statement
  if (dfsFetch(n,rulenameToId("selection_statement"))!=NULL){
    dbg("\n\t[nablaJobFill] Found a selection statement in this job!");
    job->parse.selection_statement_in_compound_statement=true;
  }else{
     dbg("\n\t[nablaJobFill] No selection statement in this job!");
 }
  
  dbg("\n\t[nablaJobFill] prefixEnumerate");
  nprintf(nabla, NULL, "\n\t%s", nabla->hook->prefixEnumerate(job));
  
  dbg("\n\t[nablaJobFill] dumpEnumerate");
  nprintf(nabla, NULL, "\n\t%s{// de l'ENUMERATE_", nabla->hook->dumpEnumerate(job));
  
  dbg("\n\t[nablaJobFill] postfixEnumerate");
  nprintf(nabla, NULL, "\n\t\t%s", nabla->hook->postfixEnumerate(job));
  dbg("\n\t[nablaJobFill] postfixEnumerate done");
  
  // Et on dump les tokens dans ce job
  dbg("\n\t[nablaJobFill] Now parsing...");
  nablaJobParse(n,job);

  if (!job->parse.got_a_return)
    nprintf(nabla, NULL, "}// de l'ENUMERATE");
  //if (nabla->backend==BACKEND_CUDA)    nprintf(nabla, NULL, "\n}// du tid test");
  nprintf(nabla, NULL, "\n}// du job");
  dbg("\n\t[nablaJobFill] done");
}


