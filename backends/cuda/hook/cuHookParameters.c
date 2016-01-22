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


/*****************************************************************************
  * Dump dans le src des parametres nabla en in comme en out
 *****************************************************************************/
void cuHookDumpNablaParameterList(nablaMain *nabla,
                                    nablaJob *job,
                                    astNode *n,
                                    int *numParams){
  if (n==NULL){
    dbg("\n\t\t[cuHookDumpNablaParameterList] NULL node, returning");
    return;
  }
  //dbg("\n\t\t[cuHookDumpNablaParameterList]");
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")){
    dbg("\n\t[cuHookDumpNablaParameterList] compound_statement, returning");
    return;
  }
  if (n->tokenid=='@'){
    dbg("\n\t[cuHookDumpNablaParameterList] @, returning");
    return;
  }
  
  //if (n->ruleid==rulenameToId("nabla_parameter_declaration"))    if (*numParams!=0) nprintf(nabla, NULL, ",");
  if (n->rule) dbg("\n\t\t[cuHookDumpNablaParameterList] rule '%s'", n->rule);
  if (n->token) dbg("\n\t\t[cuHookDumpNablaParameterList] token '%s'", n->token);
  
  if (n->ruleid==rulenameToId("direct_declarator")){
    dbg("\n\t\t[cuHookDumpNablaParameterList] Looking for '%s':", n->children->token);
    nablaVariable *var=nMiddleVariableFind(nabla->variables, n->children->token);
    *numParams+=1;
    // Si on ne trouve pas de variable, on a rien à faire
    if (var == NULL)
      return exit(NABLA_ERROR|fprintf(stderr, "\n[cuHookDumpNablaParameterList] Variable error\n"));

    dbg("\n\t\t\t[cuHookDumpNablaParameterList] Working with '%s %s':", var->item, var->name);
    
    if (strcmp(var->type, "real3")!=0){
      dbg("\n\t\t\t[cuHookDumpNablaParameterList] Non real3 variable!\n");
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
      }else{
        nprintf(nabla, NULL, ",\n\t\t%s *%s_%s", var->type, var->item, n->children->token);
      }
    }else{
      //dbg("\n\t\t[cuHookDumpNablaParameterList] Working with '%s':", var->name);
      //exit(NABLA_ERROR|fprintf(stderr, "\n[cuHookDumpNablaParameterList] Variable real3 error\n"));
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
        //nprintf(nabla, NULL, NULL);
        dbg("\n\t\t\t[cuHookDumpNablaParameterList] Found 'node coord', nothing to do!");
      }else{
        dbg("\n\t\t\t[cuHookDumpNablaParameterList] Found %s %s!", var->item, n->children->token);
        if (var->dim==0){
          nprintf(nabla, NULL, ",\n\t\treal3 *%s_%s", var->item, n->children->token);
        }else{
          nprintf(nabla, NULL, ",\n\t\treal3 *%s_%s", var->item, n->children->token);
        }
      }
    }
    // Si elles n'ont pas le même support, c'est qu'il va falloir insérer un gather/scatter
    if (var->item[0] != job->item[0]){
      // Création d'une nouvelle in_out_variable
      nablaVariable *new = nMiddleVariableNew(NULL);
      new->name=strdup(var->name);
      new->item=strdup(var->item);
      new->type=strdup(var->type);
      new->dim=var->dim;
      new->size=var->size;
      new->inout=job->parse.inout;
      // Rajout à notre liste
      if (job->variables_to_gather_scatter==NULL)
        job->variables_to_gather_scatter=new;
      else
        nMiddleVariableLast(job->variables_to_gather_scatter)->next=new;
    }
  }
  if (n->children != NULL) cuHookDumpNablaParameterList(nabla, job, n->children, numParams);
  if (n->next != NULL) cuHookDumpNablaParameterList(nabla, job, n->next, numParams);
  dbg("\n\t\t\t[cuHookDumpNablaParameterList] done!");
}


// ****************************************************************************
// * Dump d'extra paramètres
// ****************************************************************************
void cuHookAddExtraParameters(nablaMain *nabla, nablaJob *job, int *numParams){
  nablaVariable *var;
  if (*numParams!=0) nprintf(nabla, NULL, ",");
  nprintf(nabla, NULL, "\n\t\treal3 *node_coord");
  *numParams+=1;
  // Et on rajoute les variables globales
  for(var=nabla->variables;var!=NULL;var=var->next){
    //if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->item, "global")!=0) continue;
    nprintf(nabla, NULL, ",\n\t\t%s *global_%s",
            //(*numParams!=0)?",":"", 
            (var->type[0]=='r')?"real":(var->type[0]=='i')?"int":"/*Unknown type*/",
            var->name);
    *numParams+=1;
  }
  // Rajout pour l'instant systématiquement des connectivités
  if (job->item[0]=='c' || job->item[0]=='n')
    nMiddleParamsAddExtra(nabla, numParams);
}

