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


// *****************************************************************************
// * UTF8 DFS
// *****************************************************************************
void dfsUtf8(astNode * n){
  if (n==NULL) return;
  if (n->token!=NULL){
    //dbg("\n\t\t[dfsUtf8] n->token=%s",n->token);
    n->token_utf8=strdup(n->token);
    nUtf8SupThree(&n->token_utf8);
  }
  nUtf8(&n->token);
  dfsUtf8(n->children);
  dfsUtf8(n->next);
}


/*****************************************************************************
 * Generic DFS scan for specifics actions
 *****************************************************************************/
void scanTokensForActions(astNode * n, RuleAction *tokact, void *arc){
  register unsigned int nId=tokenidToRuleid(n->tokenid);
  register int i;
  for(i=0;tokact[i].action!=NULL;++i)
    if ((nId==tokact[i].ruleid)||(n->ruleid==tokact[i].ruleid)) tokact[i].action(n,arc);
  if(n->children != NULL) scanTokensForActions(n->children, tokact, arc);
  if(n->next != NULL) scanTokensForActions(n->next, tokact, arc);
}




char *dfsFetchFirst(astNode *n, int ruleid){
  char *rtn;
  //if (n->ruleid!=0) dbg("\n\t[dfsFetchFirst] n->rule=%s",n->rule);
  if (n->ruleid==ruleid) {
    //dbg("\n\t\t[dfsFetchFirst] returning token '%s'",n->children->token);
    return strdup(n->children->token);
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


// On lance une recherche dfs jusqu'à un token en next 
astNode *dfsFetch(astNode *n, int ruleid){
  astNode *rtn;
  //if (n==NULL) return;
  //if (n->ruleid!=0) dbg("\n\t\t[dfsFetch] n->rule=%s",n->rule?rtn->rule:"xNULL");
  if (n->ruleid==ruleid) return n->children;
  
  if (n->children != NULL){
    if ((rtn=dfsFetch(n->children, ruleid))!=NULL){
      //dbg("\n\t\t[dfsFetch] return %s",rtn->rule?rtn->rule:"yNULL");
      return rtn;
    }
  }
  // Si on a un voisin token non null, on évite de parser tout la branche pour rien
  //if (n->next!=NULL)    if (n->next->token!=0)      return NULL;
  if (n->next != NULL){
    if ((rtn=dfsFetch(n->next, ruleid))!=NULL){
      //dbg("\n\t\t[dfsFetch] return %s", rtn->rule?rtn->rule:"zNULL");
      return rtn;
    }
  }
  //dbg("\n\t\t[dfsFetch] return NULL");
  return NULL;
}


/*****************************************************************************
 * dfsFetchToken
 *****************************************************************************/
astNode *dfsFetchTokenId(astNode *n, int tokenid){
  register astNode *rtn;
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
    if ((rtn=dfsFetchTokenId(n->children, tokenid))!=NULL){
      //dbg("\n\t\t[dfsFetchToken] cFound return %s", rtn->token);
      return rtn;
    }
  }
    
  if (n->next != NULL){
    if ((rtn=dfsFetchTokenId(n->next, tokenid))!=NULL){
      //dbg("\n\t\t[dfsFetchToken] nFound return %s", rtn->token);
      return rtn;
    }
  }
  
  //dbg("\n\t\t[dfsFetchToken] return last NULL");
  return NULL;
}


/*****************************************************************************
 * dfsFetchToken
 *****************************************************************************/
astNode *dfsFetchToken(astNode *n, const char *token){
  astNode *rtn;
  
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
    if ((rtn=dfsFetchToken(n->children, token))!=NULL){
      dbg("\n\t\t[dfsFetchToken] cFound return %s", rtn->token);
      return rtn;
    }
  }
    
  if (n->next != NULL){
    if ((rtn=dfsFetchToken(n->next, token))!=NULL){
      dbg("\n\t\t[dfsFetchToken] nFound return %s", rtn->token);
      return rtn;
    }
  }
  
  dbg("\n\t\t[dfsFetchToken] return last NULL");
  return NULL;
}



/*****************************************************************************
 * dfsFetchRule
 *****************************************************************************/
astNode *dfsFetchRule(astNode *n, int ruleid){
  register astNode *rtn;
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
    if ((rtn=dfsFetchRule(n->children, ruleid))!=NULL){
      dbg("\n\t\t[dfsFetchRule] cFound return %s", rtn->rule);
      return rtn;
    }
  }
    
  if (n->next != NULL){
    if ((rtn=dfsFetchRule(n->next, ruleid))!=NULL){
      dbg("\n\t\t[dfsFetchRule] nFound return %s", rtn->rule);
      return rtn;
    }
  }
  
  dbg("\n\t\t[dfsFetchRule] return last NULL");
  return NULL;
}



// ****************************************************************************
// * dfsScanJobsCalls
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
    char *callName=n->next->children->children->token;
    dbg("\n\t\t[dfsScanJobsCalls] hit what we are looking for: token '%s', calling '%s'",
        n->token, callName);
    if ((foundJob=nablaJobFind(nabla->entity->jobs,callName))!=NULL){
      if (foundJob->is_a_function!=true){
        dbg("\n\t\t[dfsScanJobsCalls] which is a job!");
        nb_called+=1;
        // On va maintenant rajouter à notre liste de variables celles du job trouvé
        if (foundJob->nblParamsNode!=NULL){
          dbg("\n\t\t[dfsScanJobsCalls] Looking its Nabla Variables List:");
          cudaAddNablaVariableList(nabla,foundJob->nblParamsNode,variables);
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

