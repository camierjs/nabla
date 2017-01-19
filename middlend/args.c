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

// ****************************************************************************
// * Dump pour le header
// ****************************************************************************
int nMiddleDumpParameterTypeList(nablaMain *nabla, FILE *file, astNode *n){
  int number_of_parameters_here=0;
  
  // Si on  a pas eu de parameter_type_list, on a rien à faire
  if (n==NULL) return 0;
  
  if ((n->token != NULL )&&(strncmp(n->token,"xyz",3)==0)){// hit 'xyz'
    //fprintf(file, "/*xyz hit!*/");
    number_of_parameters_here+=1;
  }
  if ((n->token != NULL )&&(strncmp(n->token,"void",4)==0)){
    //fprintf(file, "/*void hit!*/");
    number_of_parameters_here-=1;
  }
  ///////////////////////////////////////
  // Le DUMP ne devrait pas se faire ici!
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
  if (n->ruleid==ruleToId(rule_parameter_declaration)){
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
}


// *****************************************************************************
// * nMiddleFunctionDumpFwdDeclaration
// * On revient des hooks pour faire ceci
// *****************************************************************************
void nMiddleFunctionDumpFwdDeclaration(nablaMain *nabla,
                                       nablaJob *fct,
                                       astNode *n,
                                       const char *namespace){
  int i=0;
  
  // On remplit la ligne du hdr
  hprintf(nabla, NULL, "\n%s %s %s%s(",
          nabla->hook->call->entryPointPrefix?
          nabla->hook->call->entryPointPrefix(nabla,fct):"",
          fct->return_type,
          namespace?"Entity::":"",
          fct->name);
   
  // On va chercher les paramètres standards pour le hdr
  i=nMiddleDumpParameterTypeList(nabla,nabla->entity->hdr,n);

  // Dunp des variables du job dans le header
  nablaVariable *var=fct->used_variables;
  for(;var!=NULL;var=var->next,i+=1)
    hprintf(nabla, NULL, "%s%s%s%s %s_%s",
            (i==0)?"":",",
            cHOOKn(nabla,vars,idecl),
            //(var->in&&!var->out)?"/*in*/":"",
            var->type,
            cHOOKn(nabla,vars,odecl),
            var->item, var->name);
  
  //printf("[1;34m[dfsExit] job %s exits? %s[m\n",fct->name, fct->exists?"yes":"no");

  if (fct->exists)
    hprintf(nabla, NULL,"%sconst int hlt_level, bool* hlt_exit",
            i>0?",":"");
   
  hprintf(nabla, NULL, ");");
}


// *****************************************************************************
// * 
// *****************************************************************************
void nMiddleParamsAddExtra(nablaMain *nabla, int *numParams){
  nprintf(nabla, NULL, ",\n\t\tint *xs_cell_node");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *xs_node_cell");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *xs_node_cell_corner");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *xs_cell_prev");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *xs_cell_next");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *xs_face_cell");
  *numParams+=1;
  nprintf(nabla, NULL, ",\n\t\tint *xs_face_node");
  *numParams+=1;
}



// *****************************************************************************
// * Dump d'extra connectivity
// ****************************************************************************
void nMiddleArgsAddExtra(nablaMain *nabla, int *numParams){
  const char* tabs="\t\t\t\t\t\t\t";
  nprintf(nabla, NULL, ",\n%sxs_cell_node",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%sxs_node_cell",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%sxs_node_cell_corner",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%sxs_cell_prev",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%sxs_cell_next",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%sxs_face_cell",tabs);
  *numParams+=1;
  nprintf(nabla, NULL, ",\n%sxs_face_node",tabs);
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
  if (n->ruleid==ruleToId(rule_compound_statement)) return;
  if (n->tokenid=='@') return;
  //if (n->ruleid==ruleToId(rule_nabla_parameter_declaration))
  //   if (*numParams!=0) nprintf(nabla, NULL, ",");
  if (n->ruleid==ruleToId(rule_direct_declarator)){
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


// ****************************************************************************
// * xsArgs
// ****************************************************************************
static char* xsArgs(nablaMain *nabla, const char j,const char v,const char d){
  if (j=='c' and v=='n' and d=='0') return ", xs_cell_node";
  if (j=='c' and v=='f' and d=='0') return ", xs_cell_face";
  if (j=='c' and v=='x' and d=='0') return ", xs_cell_prev";
  if (j=='c' and v=='x' and d=='1') return ", xs_cell_next";

  if (j=='n' and v=='c' and d=='0') return ", xs_node_cell";
  if (j=='n' and v=='f' and d=='0') return ", xs_node_face";
  if (j=='n' and v=='c' and d=='1') return ", xs_node_cell, xs_node_cell_corner";
  
  if (j=='f' and v=='n' and d=='0') return ", xs_face_node";
  if (j=='f' and v=='c' and d=='0') return ", xs_face_cell";

  
  nprintf(nabla, NULL,"/*xsArgs, error!*/");
  assert(NULL);
  return NULL;
}


// ****************************************************************************
// * iRTNo
// ****************************************************************************
static char* iRTNo(nablaMain *nabla,
                   const char *type, const char *var){
  char* dest=(char*)calloc(NABLA_MAX_FILE_NAME,sizeof(char));
  sprintf(dest,", const %s%s%s %s",
          cHOOKn(nabla,vars,idecl),
          type,
          cHOOKn(nabla,vars,odecl),
          var);
  char* rtn=sdup(dest);
  free(dest);
  return rtn;
}
// ****************************************************************************
// * iRTN2o
// ****************************************************************************
static char* iRTN2o(nablaMain *nabla,
                    const char *type, const char *var,
                    const char *type2, const char *var2){
  char* dest=(char*)calloc(NABLA_MAX_FILE_NAME,sizeof(char));
  sprintf(dest,", const %s%s%s %s, %s%s%s %s",
          cHOOKn(nabla,vars,idecl),type,
          cHOOKn(nabla,vars,odecl),var,
          cHOOKn(nabla,vars,idecl),type2,
          cHOOKn(nabla,vars,odecl),var2);
  char* rtn=sdup(dest);
  free(dest);
  return rtn;
}


// ****************************************************************************
// * xsParam
// ****************************************************************************
static char* xsParam(nablaMain *nabla, const char j,const char v,const char d){
  if (j=='c' and v=='n' and d=='0') return iRTNo(nabla,"int","xs_cell_node");
  if (j=='c' and v=='f' and d=='0') return iRTNo(nabla,"int","xs_cell_face");
  if (j=='c' and v=='x' and d=='0') return iRTNo(nabla,"int","xs_cell_prev");
  if (j=='c' and v=='x' and d=='1') return iRTNo(nabla,"int","xs_cell_next");

  if (j=='n' and v=='c' and d=='0') return iRTNo(nabla,"int","xs_node_cell");
  if (j=='n' and v=='f' and d=='0') return iRTNo(nabla,"int","xs_node_face");
  if (j=='n' and v=='c' and d=='1') return iRTN2o(nabla,
                                                  "int","xs_node_cell",
                                                  "int","xs_node_cell_corner");
  
  if (j=='f' and v=='n' and d=='0') return iRTNo(nabla,"int","xs_face_node");
  if (j=='f' and v=='c' and d=='0') return iRTNo(nabla,"int","xs_face_cell");
  
  nprintf(nabla, NULL,"/*xsParam, error!*/");
  assert(NULL);
  return NULL;
}

// ****************************************************************************
// * isInXS
// ****************************************************************************
static bool isInXS(const char j,const char v,const char d,
                   const char *XS, const int m){
  for(int i=0;i<3*m;i+=3)
    if (j==XS[i] and v==XS[i+1] and d==XS[i+2]) return true;
  return false;
}

// ****************************************************************************
// * Dump dans le src des arguments depuis le scan DFS
// ****************************************************************************
void nMiddleParamsDumpFromDFS(nablaMain *nabla, nablaJob *job, int numParams){
  int i;
  nablaVariable *var;
  int nXS=0;
  char *XS=(char*)calloc(3*NABLA_JOB_WHEN_MAX,sizeof(char));
  const bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  const bool dim2D = (job->entity->libraries&(1<<with_real2))!=0;
  
  if ((!nabla->hook->grammar->dfsExtra) ||
      (!nabla->hook->grammar->dfsExtra(nabla,job,true))){
    const char j=job->item[0];
    const char e=job->enum_enum;
    //printf("job %s, e=%c",job->name,e);
    if (j=='c') {
      nprintf(nabla, NULL,"%sconst int NABLA_NB_CELLS_WARP,const int NABLA_NB_CELLS",numParams==0?"":",");
      numParams+=2;
    }
    if (j=='c' && e=='n'){
      nprintf(nabla, NULL,",const int NABLA_NODE_PER_CELL");
      numParams+=1;
    }
    if (j=='n'){
      nprintf(nabla, NULL,"%sconst int NABLA_NB_NODES_WARP, const int NABLA_NB_NODES",numParams==0?"":",");
      numParams+=2;
    }
    if (j=='n' && e=='c'){
      nprintf(nabla, NULL,",const int NABLA_NODE_PER_CELL");
      numParams+=1;
    }
    if (j=='f'){
      nprintf(nabla, NULL,"%sconst int NABLA_NB_FACES,\
 const int NABLA_NB_FACES_INNER,\
 const int NABLA_NB_FACES_OUTER",numParams==0?"":",");
      numParams+=3;
    }
    if (j=='f' && e=='n'){
      nprintf(nabla, NULL,",const int NABLA_NODE_PER_FACE");
      numParams+=1;
    }
    if (j=='p'){
      nprintf(nabla, NULL,"%sconst int NABLA_NB_PARTICLES",numParams==0?"":",");
      numParams+=1;
    }
  }
  i=numParams;

  // On va chercher le prefix des variables de notre backend
  const char *prefix=
    nabla->hook->token->prefix?
    nabla->hook->token->prefix(nabla):"";
  // Dump des variables du job
  for(var=job->used_variables;var!=NULL;var=var->next,i+=1)
    nprintf(nabla, NULL, "%s%s%s%s %s%s_%s",//__restrict__
            (i==0)?"":",",
            //(var->in&&!var->out)?"":"",//const
            cHOOKn(nabla,vars,idecl),
            // Si on est en 1D ou 2D et qu'on a un real3,
            // c'est qu'on joue peut-être avec les coords? => on force à real
            (strncmp(var->type,"real3",5)==0&&dim1D)?"real":
            (strncmp(var->type,"real3x3",7)==0&&dim2D)?"real3x3":
            (strncmp(var->type,"real3",5)==0&&dim2D)?"real3":
            (nabla->hook->grammar->dfsArgType)?
            nabla->hook->grammar->dfsArgType(nabla,var):var->type,
            cHOOKn(nabla,vars,odecl),
            prefix,var->item,var->name);

  // Dump des XS des variables du job
  for(var=job->used_variables;var!=NULL;var=var->next,i+=1){
    if (!var->is_gathered) continue;
    const char j=job->item[0];
    const char v=var->item[0];
    const char d='0'+var->dim;
    nprintf(nabla, NULL,"/*isInXS(%c,%c,%c,\"%s\",%d)?*/",j,v,d,XS,nXS);
    if (isInXS(j,v,d,XS,nXS)){
      nprintf(nabla, NULL,"/*isInXS, continuing!*/");
      continue;
    }
    nprintf(nabla, NULL, "/*new XS:%c->%c%c*/",j,v,d);
    nprintf(nabla, NULL, xsParam(nabla,j,v,d));
    // Et on rajoute cette connectivité
    XS[3*nXS+0]=j;
    XS[3*nXS+1]=v;
    XS[3*nXS+2]=d;
    nXS+=1;
  }
  free(XS);

  if (job->is_a_function && job->exists){
    nprintf(nabla, NULL,"%sconst int hlt_level, bool* hlt_exit",i==0?"":",");
    numParams+=2;
  }

}


// ****************************************************************************
// * nMiddleArgsDumpFromDFS
// ****************************************************************************
void nMiddleArgsDumpFromDFS(nablaMain *nabla, nablaJob *job){
  int i=0;
  nablaVariable *var;
  int nXS=0;
  char *XS=(char*)calloc(3*NABLA_JOB_WHEN_MAX,sizeof(char));

  if ((!job->is_a_function) &&
      ((!nabla->hook->grammar->dfsExtra) ||
       (!nabla->hook->grammar->dfsExtra(nabla,job,false)))){
    const char j=job->item[0];
    const char e=job->enum_enum;
    dbg("[1;34m[nMiddleArgsDumpFromDFS] job %s, j=%c, enum_enum=%c[0m\n",job->name,j,e?e:'0');
    if (j=='c'){
      nprintf(nabla, NULL,"NABLA_NB_CELLS_WARP,NABLA_NB_CELLS");
      i+=2;
    }
    if (j=='c' && e=='n'){
      nprintf(nabla, NULL,",NABLA_NODE_PER_CELL");
      i+=1;
    }
    if (j=='n'){
      nprintf(nabla, NULL,"NABLA_NB_NODES_WARP,NABLA_NB_NODES");
      i+=2;
    }
    if (j=='n' && e=='c'){
      nprintf(nabla, NULL,",NABLA_NODE_PER_CELL");
      i+=1;
    }
    if (j=='f'){
      nprintf(nabla, NULL,"NABLA_NB_FACES,NABLA_NB_FACES_INNER,NABLA_NB_FACES_OUTER");
      i+=3;
    }
    if (j=='f' && e=='n'){
      nprintf(nabla, NULL,",NABLA_NODE_PER_FACE");
      i+=1;
    }
    if (j=='p'){
      nprintf(nabla, NULL,"NABLA_NB_PARTICLES");
      i+=1;
    }
  }

// Dump des options du job
  //nablaOption *opt=job->used_options;
  //for(i=0;opt!=NULL;opt=opt->next,i+=1)
  //  nprintf(nabla, NULL, "%s%s", (i==0)?"":",", opt->name);
  
  // Dump des variables du job
  for(var=job->used_variables;var!=NULL;var=var->next,i+=1)
    nprintf(nabla, NULL, "%s%s_%s",
            (i==0)?"":",",
            var->item,var->name);
  
  // Dump des XS des variables du job
  for(var=job->used_variables;var!=NULL;var=var->next,i+=1){
    if (!var->is_gathered) continue;
    const char j=job->item[0];
    const char v=var->item[0];
    const char d='0'+var->dim;
    if (isInXS(j,v,d,XS,nXS)) continue;
    nprintf(nabla, NULL, xsArgs(nabla,j,v,d));
    XS[3*nXS+0]=j;
    XS[3*nXS+1]=v;
    XS[3*nXS+2]=d;
    nXS+=1;
  }
  free(XS);

  if (job->exists)
    nprintf(nabla, NULL,"%shlt_level,hlt_exit",i==0?"":",");
}
