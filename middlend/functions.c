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


// ****************************************************************************
// * nMiddleFunctionDumpHeader
// ****************************************************************************
void nMiddleFunctionDumpHeader(FILE *file, astNode * n){
  if (n->rule!=NULL)
    if (n->ruleid == rulenameToId("compound_statement")) return;
  for(;n->token != NULL;){
    if (n->tokenid == AT) return; // Pas besoin des @ dans le header
    fprintf(file,"%s ",n->token); break;
  }
  if(n->children != NULL) nMiddleFunctionDumpHeader(file, n->children);
  if(n->next != NULL) nMiddleFunctionDumpHeader(file, n->next);
}


/*****************************************************************************
 * nMiddleFunctionDeclarationReal3
 *****************************************************************************/
__attribute__((unused)) static void nMiddleFunctionDeclarationReal3(astNode * n){
  if (n->ruleid != rulenameToId("declaration")
      || n->children->token!=NULL) // Cela peut être le cas de PREPROCS
    return;
  // On va chercher le type
  char *type = dfsFetchFirst(n->children,rulenameToId("type_specifier"));
  assert(type!=NULL);
  dbg("\n\t[nablaFunctionDeclarationReal3] declaration with type '%s'",type);
  if (strcmp(type,"Real3")!=0) return;
  // On va chercher le nom de la variable locale
  char *name = dfsFetchFirst(n->children,rulenameToId("direct_declarator"));
  assert(name!=NULL);
  dbg("\n\t[nablaFunctionDeclarationReal3] direct_declarator name is '%s'",name);
  // On regarde si dans notre cas on a un '='
  astNode *egal=dfsFetchTokenId(n,'=');
  // On vérifie qu'on a bien hité
  assert(egal!=NULL);
  if (egal==NULL){
    dbg("\n\t[nablaFunctionDeclarationReal3] NULL egal, returning");
    return;
  }
  dbg("\n\t[nablaFunctionDeclarationReal3] egal id is '%d'",egal->tokenid);
  // On regarde si dans notre cas on a un 'cross'
  bool cross3=false;
  astNode *nCross=dfsFetchRule(n->children,rulenameToId("primary_expression"));
  // On vérifie qu'on a bien hité
  if ((strcmp(nCross->children->token,"cross")==0) && nCross!=NULL){
    cross3=true;
    nCross->children->token=strdup("cross");
    dbg("\n\t[nablaFunctionDeclarationReal3] nCross, work todo");
  }
  // Et on transforme ceci en un cpy3
  dbg("\n\t[nablaFunctionDeclarationReal3] Et on transforme ceci en un cpy3");
  egal->token=strdup(cross3?";":"; cpy3(");
  astNode *point_virgule=dfsFetchTokenId(n->children,';');
  assert(point_virgule!=NULL);
  char end_of_line[1024];
  snprintf(end_of_line, sizeof(end_of_line),(cross3==true)?";\n\t":", &%s);\n\t",name);
  point_virgule->token=strdup(end_of_line);
  // On flush l'ancien ';'
  point_virgule->tokenid=0;
}


/*****************************************************************************
 * nMiddleFunctionDeclarationDouble
 *****************************************************************************/
__attribute__((unused)) static void nMiddleFunctionDeclarationDouble(astNode * n){
  if (n->ruleid != rulenameToId("declaration")
      || n->children->token!=NULL) // Cela peut être le cas de PREPROCS
    return;
  // On va chercher le type
  astNode *nType = dfsFetchRule(n->children,rulenameToId("type_specifier"));
  dbg("\n\t[nablaFunctionDeclarationDouble] declaration with type '%s'",nType->children->token);
  if (strcmp(nType->children->token,"double")!=0) return;
  // On va chercher le nom de la variable locale
  astNode *nName = dfsFetchRule(n->children,rulenameToId("direct_declarator"));
  dbg("\n\t[nablaFunctionDeclarationDouble] direct_declarator name is '%s'",nName->children->token);
  // On regarde si dans notre cas on a un '='
  astNode *egal=dfsFetchTokenId(n,'=');
  // On vérifie qu'on a bien hité
  assert(egal!=NULL);
  if (egal==NULL){
    dbg("\n\t[nablaFunctionDeclarationDouble] NULL egal, returning");
    return;
  }
  dbg("\n\t[nablaFunctionDeclarationDouble] egal id is '%d'",egal->tokenid);
  astNode *nPointVirgule=dfsFetchTokenId(n,';');
  if (nPointVirgule==NULL){
    dbg("\n\t[nablaFunctionDeclarationDouble] NULL nPointVirgule, returning");
    return;
  }
  // Et on transforme ceci en un Real3
  nType->children->token=strdup("Real3");
  egal->token=strdup("; doubletoReal3(");
  char end_of_line[1024];
  snprintf(end_of_line, sizeof(end_of_line),", &%s);\n\t",nName->children->token);
  nPointVirgule->token=strdup(end_of_line);
  nPointVirgule->tokenid=0;
}

  
// ****************************************************************************
// * nMiddleFunctionParse
// *Action de parsing d'une fonction
// ****************************************************************************
void nMiddleFunctionParse(astNode * n, nablaJob *fct){
  nablaMain *nabla=fct->entity->main;
 
  // On regarde si on est 'à gauche' d'un 'assignment_expression',
  // dans quel cas il faut rajouter ou pas de '()' aux variables globales
  if (n->ruleid == rulenameToId("assignment_expression"))
    if (n->children!=NULL)
      if (n->children->next!=NULL)
        if (n->children->next->ruleid == rulenameToId("assignment_operator"))
          fct->parse.left_of_assignment_operator=true;

  // On ne traite qu'un TOKEN ici, on break systématiquement
  for(;n->token != NULL;){
    //dbg("\n\t\t[nablaFunctionParse] TOKEN '%s'", n->token);
    
    if(n->tokenid==CONST){
      nprintf(nabla, "/*CONST*/", "%s const ", fct->entity->main->hook->pragma->align());
      break;
    }

    if(n->tokenid==FORALL_END){
      nprintf(nabla, "/*FORALL_END*/",NULL);
      fct->parse.enum_enum='\0';
      break;
    }
    
    if (n->tokenid==FORALL_INI){ break;}
    
    if (n->tokenid==FORALL){
      int tokenid;
      char *support=NULL;
      if (n->next->tokenid==IDENTIFIER){
        dbg("\n\t[nablaFunctionParse] n->next->token is IDENTIFIER %s", n->next->token);
        dbg("\n\t[nablaFunctionParse] n->next->next->token=%s", n->next->next->token);
        support=strdup(n->next->token);
        tokenid=n->next->next->tokenid;
      }else{
        tokenid=n->next->children->tokenid;
      }
      switch(tokenid){
      case(CELL):{
        nprintf(nabla, NULL, "/*FORALL CELL*/");
        // On annonce que l'on va travailler sur un forall cell
        fct->parse.enum_enum='c';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          nprintf(nabla, NULL, "for(CellEnumerator c%s(%s->cells()); c%s.hasNext(); ++c%s)",
                  support,support,support,support);
        break;
      }
      case(FACE):{
        nprintf(nabla, NULL, "/*FORALL FACE*/");
        fct->parse.enum_enum='f';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          nprintf(nabla, NULL, "for(FaceEnumerator f%s(%s->faces()); f%s.hasNext(); ++f%s)",
                  support,support,support,support);
        break;
      }
      case(NODE):{
        nprintf(nabla, NULL, "/*FORALL NODE*/");
        // On annonce que l'on va travailler sur un forall node
        fct->parse.enum_enum='n';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          nprintf(nabla, NULL, "for(NodeEnumerator n%s(%s->nodes()); n%s.hasNext(); ++n%s)",
                  support,support,support,support);
        break;
      }
      case(PARTICLE):{
        nprintf(nabla, NULL, "/*FORALL PARTICLE*/");
        fct->parse.enum_enum='p';
        if (support==NULL)
          nablaError("[nablaFunctionParse] No support for this FORALL!");
        else
          nprintf(nabla, NULL, "for(ParticleEnumerator p%s(cellParticles(%s->localId())); p%s.hasNext(); ++p%s)",support,support,support,support);
        break;
      }
      default: nablaError("[nablaFunctionParse] Could not distinguish FORALL!");
      }
      // On skip le 'nabla_item' qui nous a renseigné sur le type de forall
      *n=*n->next->next;
      break;
    }

    if (n->tokenid == FATAL){
      nabla->hook->token->fatal(nabla);
      break;
    }
    
    if(n->tokenid == INT64){
      nprintf(nabla, NULL, "Int64 ");
      break;
    }
    
    if (n->tokenid == CALL){
      dbg("\n\t[nablaFunctionParse] CALL");
      nabla->hook->call->addCallNames(nabla,fct,n);
      dbg("\n\t[nablaFunctionParse] CALL done");
      break;
    }
    
    if (n->tokenid == END_OF_CALL){
      nabla->hook->call->addArguments(nabla,fct);
      break;
    }

    if (n->tokenid == TIME){
      nabla->hook->token->time(nabla);
      break;
    }

    if (n->tokenid == EXIT){
      nabla->hook->token->exit(nabla,fct);
      break;
    }

    if (n->tokenid == ITERATION){
      nabla->hook->token->iteration(nabla);
      break;
    }

    if (n->tokenid == PREPROCS){
      dbg("\n\t[nablaFunctionParse] PREPROCS");
      nprintf(nabla, "/*Preprocs*/", "\n");
      break;
    }
    
    if (n->tokenid == AT){
      dbg("\n\t[nablaFunctionParse] knAt");
      fprintf(nabla->entity->src, "; knAt");
      break;
    }
    
    if (n->tokenid == LIB_ALEPH){
      nprintf(nabla, "/*LIB_ALEPH*/", NULL);
      break;
    }
    
    if (n->tokenid == ALEPH_RESET){
      nprintf(nabla, "/*ALEPH_RESET*/", ".reset()");
      break;
    }
    
    if (n->tokenid == ALEPH_SOLVE){
      nprintf(nabla, "/*ALEPH_SOLVE*/", "alephSolve()");
      break;
    }
    
    if (nMiddleTurnTokenToOption(n,nabla)!=NULL){
      dbg("\n\t[nablaFunctionParse] OPTION hit!");
      break;
    }

    // dbg("\n\t[nablaFunctionParse] Trying turnTokenToVariable hook!");
    if (nabla->hook->token->variable(n, nabla, fct)!=NULL) break;
    if (n->tokenid == '{'){ fprintf(nabla->entity->src, "{\n"); break; }
    if (n->tokenid == '}'){ fprintf(nabla->entity->src, "}\n"); break; }
    if (n->tokenid == ';'){ fprintf(nabla->entity->src, ";\n\t"); break; }
    
    if (n->tokenid==MIN_ASSIGN){
      //fct->min_assignment=true;
      fprintf(nabla->entity->src, "/*MIN_ASSIGN*/=ReduceMinToDouble");
      break;
    }
    if (n->tokenid==MAX_ASSIGN){
      fprintf(nabla->entity->src,  "/*MAX_ASSIGN*/=ReduceMaxToDouble");
      break;
    }
    
    // Dernière action possible: on dump
    //dbg("\n\t[nablaFunctionParse]  Dernière action possible: on dump ('%s')",n->token);
    fct->parse.left_of_assignment_operator=false;
    fprintf(nabla->entity->src,"%s",n->token);
    nMiddleInsertSpace(nabla,n);
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
  astNode *nFctName;
  astNode *nParams;
  fct->jobNode=n;
  fct->called_variables=NULL;
  nFctName = dfsFetch(n->children,rulenameToId("direct_declarator"));
  assert(nFctName->children->tokenid==IDENTIFIER);
  //dbg("\n\n\t// **********************************************************************");
  dbg("\n* Fonction '%s'", nFctName->children->token); // org-mode function item
  dbg("\n\t// * [nablaFctFill] Fonction '%s'", nFctName->children->token);
  //dbg("\n\t// **********************************************************************");
  fct->name=strdup(nFctName->children->token);
  dbg("\n\t[nablaFctFill] Coté UTF-8, on a: '%s'", nFctName->children->token_utf8);
  fct->name_utf8=strdup(nFctName->children->token_utf8);

  //dbg("\n\n\t[nablaFctFill] ");
  fct->is_a_function=true;
  assert(fct != NULL);
  if (fct->xyz!=NULL)
    dbg("\n\t[nablaFctFill] direction=%s, xyz=%s",
        fct->direction?fct->direction:"NULL", fct->xyz?fct->xyz:"NULL");
  fct->scope  = strdup("NoGroup");
  fct->region = strdup("NoRegion");
  fct->item   = strdup("\0function\0");fct->item[0]=0; 
  dbg("\n\t[nablaFctFill] Looking for fct->rtntp:");
  fct->return_type  = dfsFetchFirst(n->children,rulenameToId("type_specifier"));
  dbg("\n\t[nablaFctFill] fct->rtntp=%s", fct->return_type);
  fct->xyz    = strdup("NoXYZ");
  fct->direction  = strdup("NoDirection");
  dbg("\n\t[nablaFctFill] On refait (sic) pour le noeud");
  fct->returnTypeNode=dfsFetch(n->children,rulenameToId("type_specifier"));
  dbg("\n\t[nablaFctFill] On va chercher le nom de la fonction");
  
  //dbg("\n\t[nablaFctFill] fct->name=%s", fct->name);
 
  // Scan DFS pour récuérer les in/inout/out
  // Et on dump dans le log les tokens de cette fct
  dbg("\n\t[nablaJobFill] Now dfsVariables...");
  dfsVariables(nabla,fct,n,false);
  dfsVariablesDump(nabla,fct,n);
  
  // Récupération de la liste des paramètres
  dbg("\n\t[nablaFctFill] On va chercher la list des paramètres");
  nParams=dfsFetch(n->children,rulenameToId("parameter_type_list"));
  fct->stdParamsNode=nParams->children;
  
  dbg("\n\t[nablaFctFill] scope=%s region=%s item=%s type=%s name=%s",
      (fct->scope!=NULL)?fct->scope:"Null",
      (fct->region!=NULL)?fct->region:"Null",
      fct->item,//2+
      fct->return_type, fct->name);
  nMiddleScanForNablaJobAtConstant(n->children, nabla);
  dbg("\n\t[nablaFctFill] Now fillinf SRC file");
  nprintf(nabla, NULL, "\n\n\
// ********************************************************\n\
// * %s fct\n\
// ********************************************************\n\
%s %s %s%s%s(", fct->name, 
          nabla->hook->call->entryPointPrefix(nabla,fct),
          fct->return_type,
          namespace?namespace:"",
          namespace?(isAnArcaneModule(nabla)==true)?"Module::":"Service::":"",
          fct->name);
  dbg("\n\t[nablaFctFill] On va chercher les paramètres standards pour le src");

  // On va chercher les paramètres standards
  // Si used_options et used_variables ont été utilisées
  if (fct->used_options==NULL && fct->used_variables==NULL){
    numParams=nMiddleDumpParameterTypeList(nabla,nabla->entity->src, nParams);
    nprintf(nabla, NULL,"/*fct nMiddleDumpParameterTypeList numParams=%d*/",numParams);
  }else{
    numParams=nMiddleDumpParameterTypeList(nabla,nabla->entity->src, nParams);
    nprintf(nabla, NULL,"/*numParams=%d*/",numParams);
    if (nabla->hook->grammar->dfsVariable())
      nMiddleParamsDumpFromDFS(nabla,fct,numParams);
  }
    
  // On s'autorise un endroit pour insérer des paramètres
  //dbg("\n\t[nablaFctFill] adding ExtraParameters");
  if (nabla->hook->call->addExtraParameters!=NULL && fct->is_an_entry_point)
    nabla->hook->call->addExtraParameters(nabla, fct, &numParams);
  
  //dbg("\n\t[nablaFctFill] launching dfsForCalls");
  if (nabla->hook->call->dfsForCalls)
    nabla->hook->call->dfsForCalls(nabla,fct,n,namespace,nParams);
    
  // On avance jusqu'au compound_statement afin de sauter les listes de paramètres
  dbg("\n\t[nablaFctFill] On avance jusqu'au compound_statement");
  for(n=n->children->next;
      n->ruleid!=rulenameToId("compound_statement");
      n=n->next) {
    //dbg("\n\t[nablaFctFill] n rule is '%s'", n->rule);
  }
  //dbg("\n\t[nablaFctFill] n rule is finally '%s'", n->rule);
  // On saute le premier '{'
  n=n->children->next;
  nprintf(nabla, NULL, "){\n");
  // On prépare le bon ENUMERATE
  dbg("\n\t[nablaFctFill] prefixEnumerate");
  nprintf(nabla, NULL, "\t%s", nabla->hook->forall->prefix(fct));
  dbg("\n\t[nablaFctFill] dumpEnumerate");
  nprintf(nabla, NULL, "\n\t%s", nabla->hook->forall->dump(fct));
  dbg("\n\t[nablaFctFill] postfixEnumerate");
  nprintf(nabla, NULL, "\t%s", nabla->hook->forall->postfix(fct));
  // Et on dump les tokens dans ce fct
  dbg("\n\t[nablaFctFill] Now dumping function tokens");
  nMiddleFunctionParse(n,fct);
  dbg("\n\t[nablaFctFill] done");
}


