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

// ****************************************************************************
// * Generic DFS scan for specifics actions
// ****************************************************************************
void scanTokensForActions(astNode * n, RuleAction *tokact, void *arc){
  unsigned int nId=tokenidToRuleid(n->tokenid);
  for(int i=0;tokact[i].action!=NULL;++i)
    if ((nId==tokact[i].ruleid)||(n->ruleid==tokact[i].ruleid)) tokact[i].action(n,arc);
  if(n->children != NULL) scanTokensForActions(n->children, tokact, arc);
  if(n->next != NULL) scanTokensForActions(n->next, tokact, arc);
}


// ****************************************************************************
// * DFS scan until a first hit is found
// ****************************************************************************
char *dfsFetchFirst(astNode *n, int ruleid){
  char *rtn;
  //if (n->ruleid!=0) dbg("\n\t[dfsFetchFirst] n->rule=%s",n->rule);
  if (n->ruleid==ruleid) {
    //dbg("\n\t\t[dfsFetchFirst] returning token '%s'",n->children->token);
    return sdup(n->children->token);
  }
  if (n->children != NULL){
    if ((rtn=dfsFetchFirst(n->children, ruleid))!=NULL){
      //dbg("\n\t[dfsFetchFirst] return");
      return rtn;
    }
  }
  // Si on a un voisin token non null, on évite de parser tout la branche pour rien
  // Ou pas, car on a mtn des ','
  //if (n->next!=NULL) if (n->next->token!=0) return NULL;
  if (n->next != NULL){
    if ((rtn=dfsFetchFirst(n->next, ruleid))!=NULL){
      //dbg("\n\t[dfsFetchFirst] return");
      return rtn;
    }
  }
  //dbg("\n\t[dfsFetchFirst] return NULL");
  return NULL;
}



// ****************************************************************************
// * DFS scan until a all hit is found
// ****************************************************************************
char *dfsFetchAll(astNode *n, const int ruleid,
                int *idx,
                char *rtn){
  //if (n->ruleid) dbg("\n\t[dfsFetchAll] n->rule=%s",n->rule);
  if (n->ruleid==ruleid){
    //dbg("\n\t[dfsFetchAll] HIT rtn=%s",rtn);
    if (rtn[0]!=0){
      *idx+=1; // On incrémente idx que si on a déjà un item!
      //dbg("INC idx!");
    }
    strcat(rtn,n->children->token);
    return NULL;
  }
  if (n->children != NULL) dfsFetchAll(n->children, ruleid,idx,rtn);
  if (n->next != NULL) dfsFetchAll(n->next, ruleid,idx,rtn);
  //dbg("\n\t\t[dfsFetchAll] returning rtn='%s'", rtn);
  return rtn;
}


// ****************************************************************************
// On lance une recherche dfs jusqu'à un token en next 
// ****************************************************************************
astNode *dfsFetch(astNode *n, int ruleid){
  //if (n==NULL) return;
  //if (n->ruleid!=0) dbg("\n\t\t[dfsFetch] n->rule=%s",n->rule?rtn->rule:"xNULL");
  if (n->ruleid==ruleid) return n->children;
  if (n->children != NULL){
    astNode *rtn=dfsFetch(n->children, ruleid);
    if (rtn!=NULL){
      //dbg("\n\t\t[dfsFetch] return %s",rtn->rule?rtn->rule:"yNULL");
      return rtn;
    }
  }
  // Si on a un voisin token non null, on évite de parser tout la branche pour rien
  //if (n->next!=NULL)    if (n->next->token!=0)      return NULL;
  if (n->next != NULL){
    astNode *rtn=dfsFetch(n->next, ruleid);
    if (rtn!=NULL){
      //dbg("\n\t\t[dfsFetch] return %s", rtn->rule?rtn->rule:"zNULL");
      return rtn;
    }
  }
  //dbg("\n\t\t[dfsFetch] return NULL");
  return NULL;
}


// ****************************************************************************
// 
// ****************************************************************************
void dfsDumpToken(astNode *n){
  if (n->token!=NULL) dbg("%s",n->token);
  if (n->children != NULL) dfsDumpToken(n->children);
  if (n->next != NULL) dfsDumpToken(n->next);
}


// ****************************************************************************
// * DFS scan until tokenid is found
// ****************************************************************************
astNode *dfsFetchTokenId(astNode *n, int tokenid){
  if (n==NULL){
    //dbg("\n\t\t[dfsFetchToken] return first NULL");
    return NULL;
  }
  //if (n->tokenid!=0) dbg("\n\t\t[dfsFetchToken] token is '%s'",n->token);
  if (n->tokenid==tokenid){
    //dbg("\n\t\t[dfsFetchToken] hit what we are looking for: token '%s'",n->token);
    return n;
  }
  if (n->children != NULL){
    astNode* dfs_id=dfsFetchTokenId(n->children, tokenid);
    if (dfs_id!=NULL){
      //dbg("\n\t\t[dfsFetchToken] cFound return %s", rtn->token);
      return dfs_id;
    }
  }
  if (n->next != NULL){
    astNode* dfs_id=dfsFetchTokenId(n->next, tokenid);
    if (dfs_id!=NULL){
      //dbg("\n\t\t[dfsFetchToken] nFound return %s", rtn->token);
      return dfs_id;
    }
  }
  //dbg("\n\t\t[dfsFetchToken] return last NULL");
  return NULL;
}


// ****************************************************************************
// * DFS scan until token is found
// ****************************************************************************
astNode *dfsFetchToken(astNode *n, const char *token){
  dbg("\n\t\t[dfsFetchToken] looking for token '%s'", token);
  if (n==NULL){
    dbg("\n\t\t[dfsFetchToken] n==NULL, returning");
    return NULL;
  }
  if (n->token!=NULL){
    dbg("\n\t\t[dfsFetchToken] dumping token");
    dbg("\n\t\t[dfsFetchToken] token is '%s'",n->token);
    if (strcmp(n->token,token)==0){
      dbg("\n\t\t[dfsFetchToken] hit what we are looking for: token '%s'",n->token);
      return n;
    }
  }
  if (n->children != NULL){
    astNode* rtn=dfsFetchToken(n->children, token);
    if (rtn!=NULL){
      dbg("\n\t\t[dfsFetchToken] cFound return %s", rtn->token);
      return rtn;
    }
  }
  if (n->next != NULL){
    astNode *rtn=dfsFetchToken(n->next, token);
    if (rtn!=NULL){
      dbg("\n\t\t[dfsFetchToken] nFound return %s", rtn->token);
      return rtn;
    }
  }
  dbg("\n\t\t[dfsFetchToken] return last NULL");
  return NULL;
}



// ****************************************************************************
// * dfsFetchRule
// ****************************************************************************
astNode *dfsFetchRule(astNode *n, int ruleid){
  if (n==NULL){
    dbg("\n\t\t[dfsFetchRule] return first NULL");
    return NULL;
  }
  if (n->ruleid!=0) dbg("\n\t\t[dfsFetchRule] rule is '%s'",n->rule);
  if (n->ruleid==ruleid){
    dbg("\n\t\t[dfsFetchRule] hit what we are looking for: rule '%s'",n->rule);
    return n;
  }
  if (n->children != NULL){
    astNode *rtn=dfsFetchRule(n->children, ruleid);
    if (rtn!=NULL){
      dbg("\n\t\t[dfsFetchRule] cFound return %s", rtn->rule);
      return rtn;
    }
  }
  if (n->next != NULL){
    astNode *rtn=dfsFetchRule(n->next, ruleid);
    if (rtn!=NULL){
      dbg("\n\t\t[dfsFetchRule] nFound return %s", rtn->rule);
      return rtn;
    }
  }
  dbg("\n\t\t[dfsFetchRule] return last NULL");
  return NULL;
}


// *****************************************************************************
// * Ajout des variables d'un job trouvé depuis une fonction @ée
// *****************************************************************************
static void addNablaVariableList(nablaMain *nabla,
                                 astNode *n,
                                 nablaVariable **variables){
  if (n==NULL) return;
  if (n->tokenid!=0)
    dbg("\n\t\t\t[addNablaVariableList] token is '%s'",n->token);

  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==ruleToId(rule_compound_statement)) {
    dbg("\n\t\t\t[addNablaVariableList] '{', returning");
    return;
  }
  
  if (n->tokenid=='@'){
    dbg("\n\t\t\t[addNablaVariableList] '@', returning");
    return;
  }
    
  if (n->ruleid==ruleToId(rule_direct_declarator)){
    dbg("\n\t\t\t[addNablaVariableList] Found a direct_declarator!");
    dbg("\n\t\t\t[addNablaVariableList] Now looking for: '%s'",
        n->children->token);
    nablaVariable *hit=nMiddleVariableFind(nabla->variables,
                                           n->children->token);
    dbg("\n\t\t\t[addNablaVariableList] Got the direct_declarator '%s' on %ss",
        hit->name, hit->item);
    // Si on ne trouve pas de variable, c'est pas normal
    if (hit == NULL)
      return exit(NABLA_ERROR|
                  fprintf(stderr,
                          "\n\t\t[addNablaVariableList] Variable error\n"));
    dbg("\n\t\t\t[addNablaVariableList] Now testing if its allready in our growing variables list");
    nablaVariable *allready_here=nMiddleVariableFind(*variables, hit->name);
    if (allready_here!=NULL){
      dbg("\n\t\t\t[addNablaVariableList] allready_here!");
    }else{
      // Création d'une nouvelle called_variable
      nablaVariable *new = nMiddleVariableNew(NULL);
      new->name=sdup(hit->name);
      new->item=sdup(hit->item);
      new->type=sdup(hit->type);
      new->dim=hit->dim;
      new->size=hit->size;
      // Rajout à notre liste
      if (*variables==NULL){
        dbg("\n\t\t\t[addNablaVariableList] first hit");
        *variables=new;
      }else{
        dbg("\n\t\t\t[addNablaVariableList] last hit");
        nMiddleVariableLast(*variables)->next=new;
      }
    }
  }
  if (n->children != NULL) addNablaVariableList(nabla, n->children, variables);
  if (n->next != NULL) addNablaVariableList(nabla, n->next, variables);
}


// ****************************************************************************
// * DFS scan to get all job calls
// ****************************************************************************
//#warning Seules les fonctions @ées peuvent lancer des jobs!
int dfsScanJobsCalls(void *vars, void *main, astNode * n){
  nablaMain *nabla=(nablaMain*)main;
  nablaVariable **variables=(nablaVariable**)vars;
  int nb_called=0;
  if (n==NULL){
    //dbg("\n\t\t[dfsScanJobsCalls] return first NULL");
    return 0;
  }
  if (n->tokenid!=0) dbg("\n\t\t[dfsScanJobsCalls] token is '%s'",n->token);
  if (n->tokenid==CALL){
    nablaJob *foundJob;
    const char *callName=n->next->children->children->token;
    dbg("\n\t\t[dfsScanJobsCalls] hit what we are looking for: token '%s', calling '%s'",
        n->token, callName);
    if ((foundJob=nMiddleJobFind(nabla->entity->jobs,callName))!=NULL){
      if (foundJob->is_a_function!=true){
        dbg("\n\t\t[dfsScanJobsCalls] which is a job!");
        nb_called+=1;
        // On va maintenant rajouter à notre liste de variables celles du job trouvé
        if (foundJob->nblParamsNode!=NULL){
          dbg("\n\t\t[dfsScanJobsCalls] Looking its Nabla Variables List:");
          addNablaVariableList(nabla,foundJob->nblParamsNode,variables);
          dbg("\n\t\t[dfsScanJobsCalls] done!");
        }
      }else{
        dbg("\n\t\t[dfsScanJobsCalls] which is a function!");
      }
    }else{
      dbg("\n\t\t[dfsScanJobsCalls] which is a function!");
   }
  }
  nb_called+=dfsScanJobsCalls(vars, main, n->children);
  nb_called+=dfsScanJobsCalls(vars, main, n->next);
  return nb_called;
}

