// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
  job->forall_item='\0';
  job->min_assignment=false;
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
    // On peut en profiter pour g�n�rer les IN, OUT & INOUT pour ce job
    //getInOutPutsNodes(fOut, n->children->children->next, "CCCCCC");
    scanTokensForActions(n, tokact, (void*)arc);
  }
  if(n->children != NULL) scanForNablaJobParameter(n->children, ruleid, arc);
  if(n->next != NULL) scanForNablaJobParameter(n->next, ruleid, arc);
}


// ****************************************************************************
// * scanForNablaJobAtConstant
// ****************************************************************************
void scanForNablaJobAtConstant(astNode *n, nablaMain *arc){
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
    // Si ce Nabla Job a un IF apr�s le 'AT', c'est qu'il faudra l'ins�rer lors de la g�n�ration
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


char scanForNablaJobForallItem(astNode * n){
  char it;
  if (n->tokenid==FORALL)
    return n->next->children->token[0];
  if(n->children != NULL) if ((it=scanForNablaJobForallItem(n->children))!='\0') return it;
  if(n->next != NULL) if ((it=scanForNablaJobForallItem(n->next))!='\0') return it;
  return '\0';
}




/*****************************************************************************
 * Diff�rentes actions pour un job Nabla
 *****************************************************************************/
void nablaJobParse(astNode *n, nablaJob *job){
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

    // On regarde de quel cot� d'un assignment nous sommes
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
  //            '1' s'il faut return'er car on y a trouv� un nNablaSystem,
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
              //nprintf(nabla, "/*SettingIsDotXYZ*/", NULL);
              job->parse.isDotXYZ=n->children->next->next->token[0]-'w';
              // On flush cette postfix_expression pour la masquer de la g�n�ration
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
  
  // D�s qu'on a une primary_expression, on teste pour voir si ce n'est pas une option
  if ((n->ruleid == rulenameToId("primary_expression")) && (n->children->token!=NULL)){
    if (turnTokenToOption(n->children,nabla)!=NULL){
      dbg("\n\t[nablaJobParse] primaryExpression hits option");
      return;
    }
  }

  // D�s qu'on a une primary_expression,
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
  
  // D�s qu'on a une primary_expression, on teste pour voir si ce n'est pas une variable
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

  // On continue en s'enfon�ant
  if (n->children != NULL)
    nablaJobParse(n->children, job);
   
  // On continue en allant � droite
  if (n->next != NULL)
    nablaJobParse(n->next, job);

}



/*****************************************************************************
 * Dump pour le header
 *****************************************************************************/
int dumpParameterTypeList(FILE *file, astNode * n){
  int number_of_parameters_here=0;
  //if (n->token != NULL) fprintf(file, "/*dumpParameterTypeList %s:%d*/",n->token,number_of_parameters_here);
  
  if ((n->token != NULL )&&(strncmp(n->token,"xyz",3)==0)){// hit 'xyz'
    //fprintf(file, "/*xyz here!*/");
    number_of_parameters_here+=1;
  }

  if ((n->token != NULL )&&(strncmp(n->token,"void",4)==0)){
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
  // A chaque parameter_declaration, on incr�mente le compteur de param�tre
  if (n->ruleid==rulenameToId("parameter_declaration")){
    //fprintf(file, "/*number_of_parameters_here+=1*/");
    dbg("\n\t\t[dumpParameterTypeList] number_of_parameters_here+=1");
    number_of_parameters_here+=1;
  }
  if (n->children != NULL)
    number_of_parameters_here+=dumpParameterTypeList(file, n->children);
  if (n->next != NULL)
    number_of_parameters_here+=dumpParameterTypeList(file, n->next);
  
  //fprintf(file, "/*return %d*/",number_of_parameters_here);
  return number_of_parameters_here;
}



/*****************************************************************************
 * Remplissage de la structure 'job'
 * Dump dans le src de la d�claration de ce job en fonction du backend
 *****************************************************************************/
void nablaJobFill(nablaMain *nabla,
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
    
  // Si on a pas trouv� avec les items, cela doit �tre un matenvs
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
  // V�rification si l'on a des 'directions' dans les param�tres
  if (job->xyz!=NULL){
    dbg("\n\t[nablaJobFill] direction=%s, xyz=%s",
        job->drctn?job->drctn:"NULL",
        job->xyz?job->xyz:"NULL");
    nprintf(nabla, NULL, "\n\n/*For next job: xyz=%s*/", job->xyz);
  }
  // R�cup�ration du type de retour
  assert(n->children->next->children->ruleid==rulenameToId("type_specifier"));
  job->returnTypeNode=n->children->next->children;
  // R�cup�ration de la liste des param�tres
  nd=dfsFetch(n->children,rulenameToId("parameter_type_list"));
  job->stdParamsNode=nd->children;
  dbg("\n\t[nablaJobFill] scope=%s region=%s item=%s type_de_retour=%s name=%s",
      (job->scope!=NULL)?job->scope:"", (job->region!=NULL)?job->region:"",
      job->item, job->rtntp, job->name);
  // Remplissage des 
  scanForNablaJobParameter(n->children, rulenameToId("nabla_parameter"), nabla);
  scanForNablaJobAtConstant(n->children, nabla);
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
  // On va chercher les param�tres standards
  numParams=dumpParameterTypeList(nabla->entity->src, job->stdParamsNode);
  //nprintf(nabla, NULL,"/*numParams=%d*/",numParams);
  dbg("\n\t[nablaJobFill] numParams=%d", numParams);
  
  // On va chercher les param�tres nabla in/out/inout
  //nd=dfsFetch(n->children,rulenameToId("nabla_parameter"));
  job->nblParamsNode=n->children->next->next->next->next->next->next;
  //job->nblParamsNode=dfsFetch(n->children,rulenameToId("nabla_parameter"));

  // Si on a un type de retour et des arguments
  if (numParams!=0 && strncmp(job->rtntp,"void",4)!=0){
    dbg("\n\t[nablaJobFill] Returning perhaps from an argument!");
    job->parse.returnFromArgument=true;
  }
  
  // On s'autorise un endroit pour ins�rer des param�tres
  dbg("\n\t[nablaJobFill] On s'autorise un endroit pour ins�rer des param�tres");
  if (nabla->hook->addExtraParameters!=NULL) 
    nabla->hook->addExtraParameters(nabla, job,&numParams);
  
  // Et on dump les in et les out
  dbg("\n\t[nablaJobFill] Et on dump les in et les out");
  if (nabla->hook->dumpNablaParameterList!=NULL)
    nabla->hook->dumpNablaParameterList(nabla,job,job->nblParamsNode,&numParams);

  // On ferme la parenth�se des param�tres que l'on avait pas pris dans les tokens
  nprintf(nabla, NULL, "){");// du job

  if (job->parse.returnFromArgument)
    if (nabla->hook->returnFromArgument)
      nabla->hook->returnFromArgument(nabla,job);

  // On pr�pare le bon ENUMERATE suivant le forall interne
  // On saute l'�ventuel forall du d�but
  dbg("\n\t[nablaJobFill] On pr�pare le bon ENUMERATE");
  if ((job->forall_item=scanForNablaJobForallItem(n->children->next))!='\0')
    dbg("\n\t[nablaJobFill] scanForNablaJobForallItem found '%c'", job->forall_item);

  dbg("\n\t[nablaJobFill] On avance jusqu'au COMPOUND_JOB_INI afin de sauter les listes de param�tres");
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
  nablaJobParse(n,job);

  if (!job->parse.got_a_return)
    nprintf(nabla, NULL, "}");// de l'ENUMERATE
  nprintf(nabla, NULL, "\n}");// du job
  dbg("\n\t[nablaJobFill] done");
}


