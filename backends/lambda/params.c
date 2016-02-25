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
// * Dump dans le src des parametres nabla en in comme en out
// * On va surtout remplir les variables 'in' utilisées de support différent
// * pour préparer les GATHER/SCATTER
// ****************************************************************************
void lambdaHookDumpNablaParameterList(nablaMain *nabla,
                                     nablaJob *job,
                                     astNode *n,
                                     int *numParams){
  //nprintf(nabla, NULL, "/*lambdaHookDumpNablaParameterList*/");
  // S'il n'y a pas de in ni de out, on a rien à faire
  if (n==NULL) return;
  // Aux premier COMPOUND_JOB_INI ou '@', on a terminé
  if (n->tokenid==COMPOUND_JOB_INI) return;
  if (n->tokenid=='@') return;
  // Si on trouve un token 'OUT', c'est qu'on passe des 'in' aux 'out'
  if (n->tokenid==OUT) job->parse.inout=enum_out_variable;
  // Si on trouve un token 'INOUT', c'est qu'on passe des 'out' aux 'inout'
  if (n->tokenid==INOUT) job->parse.inout=enum_inout_variable;
  
  if (n->rule) dbg("\n\t\t[lambdaHookDumpNablaParameterList] rule '%s'", n->rule);
  if (n->token) dbg("\n\t\t[lambdaHookDumpNablaParameterList] token '%s'", n->token);

  // Dés qu'on hit une déclaration, c'est qu'on a une variable candidate
  if (n->ruleid==rulenameToId("direct_declarator")){
    // On la récupère
    nablaVariable *var=nMiddleVariableFind(nabla->variables, n->children->token);
    *numParams+=1;
    dbg("\n\t[lambdaHookDumpNablaParameterList] Looking for variable '%s'", n->children->token);
    // Si elle n'existe pas, c'est pas normal à ce stade: c'est une erreur de nom
    if (var == NULL)
      return exit(NABLA_ERROR|fprintf(stderr,
                                      "\n[lambdaHookDumpNablaParameterList] Cannot find variable '%s'!\n",
                                      n->children->token));
    dbg("\n\t\t[lambdaHookDumpNablaParameterList] Working with '%s %s':", var->item, var->name);

    // Dump des parametres des variables
    if (strcmp(var->type, "real3")!=0){
      dbg("\n\t\t[cudaHookDumpNablaParameterList] Non Real3 variable!\n");
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
      }else{
        nprintf(nabla, NULL, ",\n\t\t%s *%s_%s", var->type, var->item, n->children->token);
      }
    }else{
      //dbg("\n\t\t[cudaHookDumpNablaParameterList] Working with '%s':", var->name);
      //exit(NABLA_ERROR|fprintf(stderr, "\n[cudaHookDumpNablaParameterList] Variable Real3 error\n"));
      if (strncmp(var->item, "node", 4)==0 && strncmp(n->children->token, "coord", 5)==0){
        //nprintf(nabla, NULL, NULL);
        dbg("\n\t\t\t[cudaHookDumpNablaParameterList] Found 'node coord', nothing to do!\n");
      }else{
        dbg("\n\t\t\t[cudaHookDumpNablaParameterList] Found %s %s!\n", var->item, n->children->token);
        if (var->dim==0){
          nprintf(nabla, NULL, ",\n\t\tReal3 *%s_%s", var->item, n->children->token);
        }else{
          nprintf(nabla, NULL, ",\n\t\treal3 *%s_%s", var->item, n->children->token);
        }
      }
    }

    // Si elles n'ont pas le même support, c'est qu'il va falloir insérer un gather/scatter
    if (var->item[0] != job->item[0]){
      nprintf(nabla, NULL, "\n\t\t/* gather/scatter for %s_%s*/", var->item, var->name);
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
  
  if (n->children != NULL) lambdaHookDumpNablaParameterList(nabla,job,n->children,numParams);
  if (n->next != NULL) lambdaHookDumpNablaParameterList(nabla,job,n->next, numParams);

}



/*****************************************************************************
  * Dump d'extra paramètres
 *****************************************************************************/
void lambdaHookAddExtraParameters(nablaMain *nabla, nablaJob *job, int *numParams){
  // Rajout pour l'instant systématiquement des node_coords et du global_deltat
  nablaVariable *var;
  if (*numParams!=0) nprintf(nabla, NULL, ",");
  // Si on est dans le cas 1D
  if ((nabla->entity->libraries&(1<<with_real))!=0)
    nprintf(nabla, NULL, "\n\t\treal *node_coord");
  else if ((nabla->entity->libraries&(1<<with_real2))!=0)
    nprintf(nabla, NULL, "\n\t\treal3 *node_coord");
  else // Sinon c'est le 3D
    nprintf(nabla, NULL, "\n\t\treal3 *node_coord");
  *numParams+=1;
  // Et on rajoute les variables globales
  for(var=nabla->variables;var!=NULL;var=var->next){
    //if (strcmp(var->name, "time")==0) continue;
    if (strcmp(var->item, "global")!=0) continue;
    nprintf(nabla, NULL, ",\n\t\t%s *global_%s",
            (var->type[0]=='r')?"real":(var->type[0]=='i')?"int":"/*Unknown type*/",
            var->name);
    *numParams+=1;
  }
  // Rajout pour l'instant systématiquement des connectivités
  if (job->item[0]=='c' || job->item[0]=='n')
    nMiddleParamsAddExtra(nabla, numParams);
  //nprintf(nabla, NULL, "/*eolambdaHookAddExtraParameters*/");
}


// ****************************************************************************
// *
// ****************************************************************************
void lambdaHookAddExtraParametersDFS(nablaMain *nabla,
                                     nablaJob *job,
                                     int *numParams){
  nprintf(nabla, NULL, "/*nothing in lambdaHookAddExtraParametersDFS*/");
}


// ****************************************************************************
// * Dump dans le src des parametres nabla en in comme en out
// * On va surtout remplir les variables 'in' utilisées de support différent
// * pour préparer les GATHER/SCATTER
// ****************************************************************************
void lambdaHookDumpNablaParameterListDFS(nablaMain *nabla,
                                     nablaJob *job,
                                     astNode *n,
                                     int *numParams){
  nprintf(nabla, NULL, "/*nothing in lambdaHookDumpNablaParameterListDFS*/");
}
