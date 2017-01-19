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


// ****************************************************************************
// * nMiddleFunctionDumpHeader
// ****************************************************************************
void nMiddleFunctionDumpHeader(FILE *file, astNode *n){
  if (n->rule!=NULL)
    if (n->ruleid == ruleToId(rule_compound_statement))
      return;
  for(;n->token != NULL;){
    if (n->tokenid == AT) return; // Pas besoin des @ dans le header
    fprintf(file,"%s ",n->token);
    break;
  }
  if(n->children != NULL) nMiddleFunctionDumpHeader(file, n->children);
  if(n->next != NULL) nMiddleFunctionDumpHeader(file, n->next);
}

 
// ****************************************************************************
// * nMiddleFunctionParse
// * Action de parsing d'une fonction
// ****************************************************************************
static void nMiddleFunctionParse(astNode * n, nablaJob *fct){
  nablaMain *nabla=fct->entity->main;
  // On utilise hook->grammar->skipBody
  const bool dump = (nabla->hook->grammar->skipBody?
                     nabla->hook->grammar->skipBody(nabla):true);

  // On regarde si on est 'à gauche' d'un 'assignment_expression',
  // dans quel cas il faut rajouter ou pas de '()' aux variables globales
  if (n->ruleid == ruleToId(rule_assignment_expression))
    if (n->children!=NULL)
      if (n->children->next!=NULL)
        if (n->children->next->ruleid == ruleToId(rule_assignment_operator))
          fct->parse.left_of_assignment_operator=true;

  // On ne traite qu'un TOKEN ici, on break systématiquement
  for(;n->token != NULL;){
    //dbg("\n\t\t[nablaFunctionParse] TOKEN '%s'", n->token);
    
    if(dump && n->tokenid==CONST){
      dprintf(dump,nabla,
              "/*CONST*/", "%s const ",
              cHOOK(fct->entity->main,pragma,align));
      break;
    }
        
    if(n->tokenid==FORALL_END){
      dprintf(dump,nabla, "/*FORALL_END*/",NULL);
      fct->parse.enum_enum='\0';
      break;
    }
    
    if (n->tokenid==FORALL_INI){ break;}
    
    if (n->tokenid==FORALL){
      int tokenid;
      char *support=NULL;
      if (n->next->children->tokenid==IDENTIFIER){
        dbg("\n\t[nablaFunctionParse] IDENTIFIER=%s",n->next->children->token);
        dbg("\n\t[nablaFunctionParse] token=%s",n->next->children->next->token);
        support=sdup(n->next->children->token);
        tokenid=n->next->children->next->children->tokenid;
      }else{
        tokenid=n->next->children->children->tokenid;
      }
      switch(tokenid){
      case(CELL):{
        dprintf(dump,nabla, NULL, "/*FORALL CELL*/");
        // On annonce que l'on va travailler sur un forall cell
        fct->parse.enum_enum='c';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          dprintf(dump,nabla, NULL,
                  "for(CellEnumerator c%s(%s->cells()); c%s.hasNext(); ++c%s)",
                  support,support,support,support);
        break;
      }
      case(FACE):{
        dprintf(dump,nabla, NULL, "/*FORALL FACE*/");
        fct->parse.enum_enum='f';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          dprintf(dump,nabla, NULL,
                  "for(FaceEnumerator f%s(%s->faces()); f%s.hasNext(); ++f%s)",
                  support,support,support,support);
        break;
      }
      case(NODE):{
        dprintf(dump,nabla, NULL, "/*FORALL NODE*/");
        // On annonce que l'on va travailler sur un forall node
        fct->parse.enum_enum='n';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          dprintf(dump,nabla, NULL,
                  "for(NodeEnumerator n%s(%s->nodes()); n%s.hasNext(); ++n%s)",
                  support,support,support,support);
        break;
      }
      case(PARTICLE):{
        dprintf(dump,nabla, NULL, "/*FORALL PARTICLE*/");
        fct->parse.enum_enum='p';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          dprintf(dump,nabla, NULL,
                  "for(ParticleEnumerator p%s(cellParticles(%s->localId())); p%s.hasNext(); ++p%s)",
                  support,support,support,support);
        break;
      }
        //case(SET):{        break;      }
      default: nablaError("[nablaFunctionParse] Could not distinguish FORALL!");
      }
//#warning On skip le nabla_item qui renseigne sur le type de forall
      *n=*n->next->next;
      break;
    }

    if (dump && n->tokenid == FATAL){
      nabla->hook->token->fatal(nabla);
      break;
    }
    
    if(dump && n->tokenid == INT64){
      dprintf(dump,nabla, NULL, "Int64 ");
      break;
    }
    
    if (dump && n->tokenid == CALL){
      dbg("\n\t[nablaFunctionParse] CALL");
      nabla->hook->call->addCallNames(nabla,fct,n);
      dbg("\n\t[nablaFunctionParse] CALL done");
      break;
    }
    
    if (n->tokenid == END_OF_CALL){
      //nabla->hook->call->addArguments(nabla,fct);
      break;
    }

    if (n->tokenid == RETURN){
      if (dump) nprintf(nabla,NULL,"return ");
      if (!dump && strncmp(fct->return_type,"void",4)!=0){
        nprintf(nabla,NULL,"return 0;");
      }
      break;
    }
    
    if (dump && n->tokenid == TIME){
      nabla->hook->token->time(nabla);
      break;
    }

    if (dump && n->tokenid == EXIT){
      nabla->hook->token->exit(nabla,fct);
      break;
    }
    
    if (dump && n->tokenid == ERROR){
      nabla->hook->token->error(nabla,fct);
      break;
    }

    if (dump && n->tokenid == ITERATION){
      nabla->hook->token->iteration(nabla);
      break;
    }

    if (n->tokenid == PREPROCS){
      dbg("\n\t[nablaFunctionParse] PREPROCS");
      dprintf(dump,nabla, "/*Preprocs*/", "\n");
      break;
    }
    
    /*if (n->tokenid == AT){
      dbg("\n\t[nablaFunctionParse] knAt");
      fprintf(nabla->entity->src, "; knAt");
      break;
      }*/
    
    if (n->tokenid == LIB_ALEPH){
      dprintf(dump,nabla, "/*LIB_ALEPH*/", NULL);
      break;
    }
    
    if (n->tokenid == ALEPH_RESET){
      dprintf(dump,nabla, "/*ALEPH_RESET*/", ".reset()");
      break;
    }
    
    if (n->tokenid == ALEPH_SOLVE){
      dprintf(dump,nabla, "/*ALEPH_SOLVE*/", "alephSolve()");
      break;
    }
    
    if (dump && nMiddleTurnTokenToOption(n,nabla)!=NULL){
      dbg("\n\t[nablaFunctionParse] OPTION hit!");
      break;
    }

    // dbg("\n\t[nablaFunctionParse] Trying turnTokenToVariable hook!");
    if (dump && nabla->hook->token->variable(n, nabla, fct)!=NULL) break;
    if (n->tokenid == '{'){ nprintf(nabla, NULL,"{\n"); break; }
    if (n->tokenid == '}'){ nprintf(nabla, NULL,"}\n"); break; }
    if (n->tokenid == ';'){ dprintf(dump,nabla, NULL,";\n"); break; }
    
    if (n->tokenid==MIN_ASSIGN){
      dprintf(dump,nabla,"/*MIN_ASSIGN*/","=ReduceMinToDouble");
      break;
    }
    if (n->tokenid==MAX_ASSIGN){
      dprintf(dump,nabla, "/*MAX_ASSIGN*/","=ReduceMaxToDouble");
      break;
    }
    
    // Dernière action possible: on dump
    //dbg("\n\t[nablaFunctionParse]  Dernière action possible: on dump ('%s')",n->token);
    fct->parse.left_of_assignment_operator=false;
    dprintf(dump,nabla,NULL,"%s",n->token);
    if (dump) nMiddleInsertSpace(nabla,n);
    break;
  }
  if(n->children != NULL) nMiddleFunctionParse(n->children, fct);
  if(n->next != NULL) nMiddleFunctionParse(n->next, fct);
}


/*****************************************************************************
 * Remplissage de la structure 'fct'
 * Dump dans le src de la déclaration de ce fct en fonction du backend
 *****************************************************************************/
void nMiddleFunctionFill(nablaMain *nabla,
                         nablaJob *fct,
                         astNode *n,
                         const char *namespace){
  int numParams;
  fct->jobNode=n;
  fct->called_variables=NULL;
  astNode *nFctName = dfsFetch(n->children,ruleToId(rule_direct_declarator));
  assert(nFctName->children->tokenid==IDENTIFIER);
  dbg("\n* Fonction '%s'", nFctName->children->token); // org-mode function item
  dbg("\n\t// * [nablaFctFill] Fonction '%s'", nFctName->children->token);
  fct->name=sdup(nFctName->children->token);
  dbg("\n\t[nablaFctFill] Coté UTF-8, on a: '%s'", nFctName->children->token_utf8);
  fct->name_utf8=sdup(nFctName->children->token_utf8);
  fct->is_a_function=true;
  assert(fct != NULL);
  if (fct->xyz!=NULL)
    dbg("\n\t[nablaFctFill] direction=%s, xyz=%s",
        fct->direction?fct->direction:"NULL", fct->xyz?fct->xyz:"NULL");
  fct->scope  = sdup("NoGroup");
  fct->region = sdup("NoRegion");
  fct->item   = sdup("\0function\0");fct->item[0]=0; 
  dbg("\n\t[nablaFctFill] Looking for fct->rtntp:");
  fct->return_type  = dfsFetchFirst(n->children,ruleToId(rule_type_specifier));
  dbg("\n\t[nablaFctFill] fct->rtntp=%s", fct->return_type);
  fct->xyz    = sdup("NoXYZ");
  fct->direction  = sdup("NoDirection");
  dbg("\n\t[nablaFctFill] On refait (sic) pour le noeud");
  fct->returnTypeNode=dfsFetch(n->children,ruleToId(rule_type_specifier));
  dbg("\n\t[nablaFctFill] On va chercher le nom de la fonction");
  
  //dbg("\n\t[nablaFctFill] fct->name=%s", fct->name);
 
  // Scan DFS pour récuérer les in/inout/out
  // Et on dump dans le log les tokens de cette fct
  dbg("\n\t[nablaJobFill] Now dfsVariables...");
  dfsVariables(nabla,fct,n,false);
  dfsVariablesDump(nabla,fct,n);
  // Scan DFS pour récupérer les exit
  dfsExit(nabla,fct,n);

  // Récupération de la liste des paramètres
  dbg("\n\t[nablaFctFill] On va chercher la list des paramètres");
  astNode *nParams=dfsFetch(n->children,ruleToId(rule_parameter_type_list));
  fct->stdParamsNode=nParams->children;
  
  dbg("\n\t[nablaFctFill] scope=%s region=%s item=%s type=%s name=%s",
      (fct->scope!=NULL)?fct->scope:"Null",
      (fct->region!=NULL)?fct->region:"Null",
      fct->item,//2+
      fct->return_type, fct->name);
  nMiddleScanForNablaJobAtConstant(n->children, nabla);
  dbg("\n\t[nablaFctFill] Now fillinf SRC file");
  nprintf(nabla, NULL, "\n\
// ********************************************************\n\
// * %s fct\n\
// ********************************************************\n\
%s %s %s%s%s(", fct->name, 
          (nabla->hook->call->entryPointPrefix)?
          nabla->hook->call->entryPointPrefix(nabla,fct):"",
          fct->return_type,
          namespace?namespace:"",
          namespace?(isAnArcaneModule(nabla))?"Module::":
          isAnArcaneService(nabla)?"Service::":"::":"",
          fct->name);
  dbg("\n\t[nablaFctFill] On va chercher les paramètres standards pour le src");
  
  // On va chercher les paramètres standards
  // Si used_options et used_variables ont été utilisées
  numParams=nMiddleDumpParameterTypeList(nabla,nabla->entity->src, nParams);
  //nprintf(nabla, NULL,"/*fct numParams=%d*/",numParams);
  if (nabla->hook->grammar->dfsVariable?
      nabla->hook->grammar->dfsVariable(nabla):false)
    nMiddleParamsDumpFromDFS(nabla,fct,numParams);
      
  // On s'autorise un endroit pour insérer des paramètres
  //dbg("\n\t[nablaFctFill] adding ExtraParameters");
  //nprintf(nabla, NULL,"/*ExtraParameters*/");
  if (nabla->hook->call->addExtraParameters!=NULL && fct->is_an_entry_point)
    nabla->hook->call->addExtraParameters(nabla, fct, &numParams);
  
  //dbg("\n\t[nablaFctFill] launching dfsForCalls");
  //nprintf(nabla, NULL,"/*dfsForCalls*/");
  if (nabla->hook->call->dfsForCalls)
    nabla->hook->call->dfsForCalls(nabla,fct,n,namespace,nParams);
    
  // On avance jusqu'au compound_statement afin de sauter les listes de paramètres
  dbg("\n\t[nablaFctFill] On avance jusqu'au compound_statement");
  for(n=n->children->next;
      n->ruleid!=ruleToId(rule_compound_statement);
      n=n->next) {
    //dbg("\n\t[nablaFctFill] n rule is '%s'", n->rule);
  }
  //dbg("\n\t[nablaFctFill] n rule is finally '%s'", n->rule);
  // On saute le premier '{'
  n=n->children->next;
  nprintf(nabla, NULL, "){\n\t");

  // Et on dump les tokens dans ce fct
  dbg("\n\t[nablaFctFill] Now dumping function tokens");
  nMiddleFunctionParse(n,fct);
  dbg("\n\t[nablaFctFill] done");
}


