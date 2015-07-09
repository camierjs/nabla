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






// *****************************************************************************
// * Dump d'extra connectivity
// ****************************************************************************
void cudaAddExtraConnectivitiesArguments(nablaMain *nabla, int *numParams){
  const char* tabs="\t\t\t\t\t\t\t";
  nprintf(nabla, NULL, ",\n%scell_node",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%snode_cell",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%snode_cell_corner",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%scell_prev",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%scell_next",tabs);
  *numParams+=1;
}

void cudaAddExtraConnectivitiesParameters(nablaMain *nabla, int *numParams){
  nprintf(nabla, NULL, ",\n\t\tint *cell_node");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *node_cell");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *node_cell_corner");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *cell_prev");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *cell_next");
  *numParams+=1;
}


/*****************************************************************************
  * Dump d'extra arguments
 *****************************************************************************/
void cudaAddExtraArguments(nablaMain *nabla, nablaJob *job, int *numParams){
  const char* tabs="\t\t\t\t\t\t\t";
  { // Rajout pour l'instant systématiquement des node_coords et du global_deltat
    nablaVariable *var;
    if (*numParams!=0) nprintf(nabla, NULL, "/*cudaAddExtraArguments*/,");
    nprintf(nabla, NULL, "\n%snode_coord",tabs);
    *numParams+=1;
    // Et on rajoute les variables globales
    for(var=nabla->variables;var!=NULL;var=var->next){
      //if (strcmp(var->name, "time")==0) continue;
      if (strcmp(var->item, "global")!=0) continue;
      nprintf(nabla, NULL, ",global_%s", var->name);
      *numParams+=1;
   }
  }
  // Rajout pour l'instant systématiquement des connectivités
  if (job->item[0]=='c'||job->item[0]=='n')
    cudaAddExtraConnectivitiesArguments(nabla, numParams);
}


// *****************************************************************************
// * Ajout des variables d'un job trouvé depuis une fonction @ée
// *****************************************************************************
void cudaAddNablaVariableList(nablaMain *nabla, astNode *n, nablaVariable **variables){
  if (n==NULL) return;
  if (n->tokenid!=0) dbg("\n\t\t\t[cudaAddNablaVariableList] token is '%s'",n->token);

  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) {
    dbg("\n\t\t\t[cudaAddNablaVariableList] '{', returning");
    return;
  }
  
  if (n->tokenid=='@'){
    return;
    dbg("\n\t\t\t[cudaAddNablaVariableList] '@', returning");
  }
    
  if (n->ruleid==rulenameToId("direct_declarator")){
    dbg("\n\t\t\t[cudaAddNablaVariableList] Found a direct_declarator!");
    dbg("\n\t\t\t[cudaAddNablaVariableList] Now looking for: '%s'",n->children->token);
    nablaVariable *hit=nMiddleVariableFind(nabla->variables, n->children->token);
    dbg("\n\t\t\t[cudaAddNablaVariableList] Got the direct_declarator '%s' on %ss", hit->name, hit->item);
    // Si on ne trouve pas de variable, c'est pas normal
    if (hit == NULL)
      return exit(NABLA_ERROR|fprintf(stderr, "\n\t\t[cudaAddNablaVariableList] Variable error\n"));
    dbg("\n\t\t\t[cudaAddNablaVariableList] Now testing if its allready in our growing variables list");
    nablaVariable *allready_here=nMiddleVariableFind(*variables, hit->name);
    if (allready_here!=NULL){
      dbg("\n\t\t\t[cudaAddNablaVariableList] allready_here!");
    }else{
      // Création d'une nouvelle called_variable
      nablaVariable *new = nMiddleVariableNew(NULL);
      new->name=strdup(hit->name);
      new->item=strdup(hit->item);
      new->type=strdup(hit->type);
      new->dim=hit->dim;
      new->size=hit->size;
      // Rajout à notre liste
      if (*variables==NULL){
        dbg("\n\t\t\t[cudaAddNablaVariableList] first hit");
        *variables=new;
      }else{
        dbg("\n\t\t\t[cudaAddNablaVariableList] last hit");
        nMiddleVariableLast(*variables)->next=new;
      }
    }
  }
  if (n->children != NULL) cudaAddNablaVariableList(nabla, n->children, variables);
  if (n->next != NULL) cudaAddNablaVariableList(nabla, n->next, variables);
}


/*****************************************************************************
  * Dump dans le src des arguments nabla en in comme en out
 *****************************************************************************/
void cudaDumpNablaArgumentList(nablaMain *nabla, astNode *n, int *numParams){
  //nprintf(nabla,"\n\t[cudaDumpNablaArgumentList]",NULL);
  if (n==NULL) return;
  
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) return;
  
  if (n->tokenid=='@') return;
  
  //if (n->ruleid==rulenameToId("nabla_parameter_declaration"))    if (*numParams!=0) nprintf(nabla, NULL, ",");
  
  if (n->ruleid==rulenameToId("direct_declarator")){
    nablaVariable *var=nMiddleVariableFind(nabla->variables, n->children->token);
    nprintf(nabla, NULL, "\n\t\t/*[cudaDumpNablaArgumentList] looking for %s*/", n->children->token);
    *numParams+=1;
    // Si on ne trouve pas de variable, on a rien à faire
    if (var == NULL) return exit(NABLA_ERROR|fprintf(stderr, "\n[cudaHookDumpNablaArgumentList] Variable error\n"));
    if (strcmp(var->type, "real3")!=0){
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
      }else{
        nprintf(nabla, NULL, ",\n\t\t\t\t\t\t\t%s_%s", var->item, n->children->token);
      }
    }else{
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0)
        nprintf(nabla, NULL, NULL);
      else
        nprintf(nabla, NULL,  ",\n\t\t\t\t\t\t\t%s_%s", var->item, n->children->token);
    }
  }
  if (n->children != NULL) cudaDumpNablaArgumentList(nabla, n->children, numParams);
  if (n->next != NULL) cudaDumpNablaArgumentList(nabla, n->next, numParams);
}


/*****************************************************************************
  * Dump dans le src l'appel des fonction de debug des arguments nabla  en out
 *****************************************************************************/
void cudaDumpNablaDebugFunctionFromOutArguments(nablaMain *nabla, astNode *n, bool in_or_out){
  //nprintf(nabla,"\n\t[cudaHookDumpNablaParameterList]",NULL);
  if (n==NULL) return;
  
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) return;
  if (n->tokenid=='@') return;

  if (n->tokenid==OUT) in_or_out=false;
  if (n->tokenid==INOUT) in_or_out=false;
    
  if (n->ruleid==rulenameToId("direct_declarator")){
    nablaVariable *var=nMiddleVariableFind(nabla->variables, n->children->token);
    // Si on ne trouve pas de variable, on a rien à faire
    if (var == NULL)
      return exit(NABLA_ERROR|fprintf(stderr, "\n[cudaDumpNablaDebugFunctionFromOutArguments] Variable error\n"));
    if (!in_or_out){
      nprintf(nabla,NULL,"\n\t\t//printf(\"\\n%sVariable%sDim%s_%s:\");",
              (var->item[0]=='n')?"Node":"Cell",
              (strcmp(var->type,"real3")==0)?"XYZ":"",
              (var->dim==0)?"0":"1",
              var->name);
      nprintf(nabla,NULL,"//dbg%sVariable%sDim%s_%s();",
              (var->item[0]=='n')?"Node":"Cell",
              (strcmp(var->type,"real3")==0)?"XYZ":"",
              (var->dim==0)?"0":"1",
              var->name);
    }
  }
  cudaDumpNablaDebugFunctionFromOutArguments(nabla, n->children, in_or_out);
  cudaDumpNablaDebugFunctionFromOutArguments(nabla, n->next, in_or_out);
}
