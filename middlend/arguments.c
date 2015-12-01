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


/*****************************************************************************
 * Dump pour le header
 *****************************************************************************/
int nMiddleDumpParameterTypeList(nablaMain *nabla, FILE *file, astNode * n){
  int number_of_parameters_here=0;
  
  // Si on  a pas eu de parameter_type_list, on a rien à faire
  if (n==NULL) return -1;
  
  if ((n->token != NULL )&&(strncmp(n->token,"xyz",3)==0)){// hit 'xyz'
    //fprintf(file, "/*xyz hit!*/");
    number_of_parameters_here+=1;
  }
  if ((n->token != NULL )&&(strncmp(n->token,"void",4)==0)){
    //fprintf(file, "/*void hit!*/");
    number_of_parameters_here-=1;
  }
  if ((n->token != NULL )&&(strncmp(n->token,"void",4)!=0)){// avoid 'void'
    if (strncmp(n->token,"restrict",8)==0){
      fprintf(file, "__restrict__ ");
    }else if (strncmp(n->token,"aligned",7)==0){
      fprintf(file, "/*aligned*/");
    }else{
      fprintf(file, "%s ", n->token);
    }
  }
  // A chaque parameter_declaration, on incrémente le compteur de paramètre
  if (n->ruleid==rulenameToId("parameter_declaration")){
    dbg("\n\t\t[nMiddleDumpParameterTypeList] number_of_parameters_here+=1");
    number_of_parameters_here+=1;
  }
  if (n->children != NULL)
    number_of_parameters_here+=nMiddleDumpParameterTypeList(nabla,file, n->children);
  if (n->next != NULL)
    number_of_parameters_here+=nMiddleDumpParameterTypeList(nabla,file, n->next);
  return number_of_parameters_here;
}



// ****************************************************************************
// * nMiddleDfsForCalls
// ****************************************************************************
void nMiddleDfsForCalls(nablaMain *nabla,
                        nablaJob *job, astNode *n,
                        const char *namespace,
                        astNode *nParams){
  int nb_called;
  nablaVariable *var;
  // On scan en dfs pour chercher ce que cette fonction va appeler
  dbg("\n\t[cudaDfsForCalls] On scan en DFS pour chercher ce que cette fonction va appeler");
  nb_called=dfsScanJobsCalls(&job->called_variables,nabla,n);
  dbg("\n\t[cudaDfsForCalls] nb_called = %d", nb_called);
  if (nb_called!=0){
    int numParams=1;
    nMiddleParamsAddExtra(nabla,&numParams);
    dbg("\n\t[cudaDfsForCalls] dumping variables found:");
    for(var=job->called_variables;var!=NULL;var=var->next){
      dbg("\n\t\t[cudaDfsForCalls] variable %s %s %s", var->type, var->item, var->name);
      nprintf(nabla, NULL, ",\n\t\t/*used_called_variable*/%s *%s_%s",var->type, var->item, var->name);
    }
  }

  // Si le job is_an_entry_point, il sera placé avant le main
  // donc pas besoin de le déclarer
  if (job->is_an_entry_point) return;

  // Sinon, on remplit la ligne du hdr
  hprintf(nabla, NULL, "\n%s %s %s%s(",
          nabla->hook->call->entryPointPrefix(nabla,job),
          job->return_type,
          namespace?"Entity::":"",
          job->name);
  // On va chercher les paramètres standards pour le hdr
  nMiddleDumpParameterTypeList(nabla,nabla->entity->hdr, nParams);
  hprintf(nabla, NULL, ");");
}


// *****************************************************************************
// * 
// *****************************************************************************
void nMiddleParamsAddExtra(nablaMain *nabla, int *numParams){
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
  nprintf(nabla, NULL, ",\n\t\tint *face_cell");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *face_node");
  *numParams+=1;
}



// *****************************************************************************
// * Dump d'extra connectivity
// ****************************************************************************
void nMiddleArgsAddExtra(nablaMain *nabla, int *numParams){
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
  nprintf(nabla, NULL, ",\n%sface_cell",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%sface_node",tabs);
  *numParams+=1;
}


// ****************************************************************************
// * Dump d'extra arguments
// ****************************************************************************
void nMiddleArgsAddGlobal(nablaMain *nabla, nablaJob *job, int *numParams){
  const char* tabs="\t\t\t\t\t\t\t";
  { // Rajout pour l'instant systématiquement des node_coords et du global_deltat
    nablaVariable *var;
    if (*numParams!=0) nprintf(nabla, NULL, "/*nMiddleAddExtraArguments*/,");
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
    nMiddleArgsAddExtra(nabla, numParams);
}


// ****************************************************************************
// * Dump dans le src des arguments nabla en in comme en out
// ****************************************************************************
void nMiddleArgsDump(nablaMain *nabla, astNode *n, int *numParams){
  //nprintf(nabla,"\n\t[nMiddleArgsDump]",NULL);
  if (n==NULL) return;
  // Si on tombe sur la '{', on arrête; idem si on tombe sur le token '@'
  if (n->ruleid==rulenameToId("compound_statement")) return;
  if (n->tokenid=='@') return;
  //if (n->ruleid==rulenameToId("nabla_parameter_declaration"))
  //   if (*numParams!=0) nprintf(nabla, NULL, ",");
  if (n->ruleid==rulenameToId("direct_declarator")){
    nablaVariable *var=nMiddleVariableFind(nabla->variables, n->children->token);
    //nprintf(nabla, NULL, "\n\t\t/*[cudaDumpNablaArgumentList] looking for %s*/", n->children->token);
    *numParams+=1;
    // Si on ne trouve pas de variable, on a rien à faire
    if (var == NULL)
      return exit(NABLA_ERROR|fprintf(stderr, "\n[nMiddleArgsDump] Variable error\n"));
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
  if (n->children != NULL) nMiddleArgsDump(nabla, n->children, numParams);
  if (n->next != NULL) nMiddleArgsDump(nabla, n->next, numParams);
}
