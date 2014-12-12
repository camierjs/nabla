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
 * Traitement des transformations '[', '(' & ''
 *****************************************************************************/
void cudaHookTurnBracketsToParentheses(nablaMain* nabla, nablaJob *job, nablaVariable *var, char cnfg){
  dbg("\n\t[actJobItemParse] primaryExpression hits Cuda variable");
  if ((cnfg=='c' && var->item[0]=='n')
      ||(cnfg=='c' && var->item[0]=='f')
      ||(cnfg=='n' && var->item[0]!='n')            
      ||(cnfg=='f' && var->item[0]!='f')
      ||(cnfg=='e' && var->item[0]!='e')
      ||(cnfg=='m' && var->item[0]!='m')
      ){
    nprintf(nabla, "/*turnBracketsToParentheses@true*/", "/*%c %c*/", cnfg, var->item[0]);
    job->parse.turnBracketsToParentheses=true;
  }else{
    //nprintf(nabla, NULL, "/*.xyz?*/");
    if (job->parse.postfix_constant==true && job->parse.variableIsArray==true) return;
    //if (job->parse.postfix_constant==true) return;
    if (job->parse.isDotXYZ==1) nprintf(nabla, "/*cudaHookTurnBracketsToParentheses_X*/", ".x");
    if (job->parse.isDotXYZ==2) nprintf(nabla, "/*cudaHookTurnBracketsToParentheses_Y*/", ".y");
    if (job->parse.isDotXYZ==3) nprintf(nabla, "/*cudaHookTurnBracketsToParentheses_Z*/", ".z");
    //nprintf(nabla, NULL, "/*.xyz flushing isDotXYZ*/");
    //job->parse.isDotXYZ=0;
    job->parse.turnBracketsToParentheses=false;
  }
}


/***************************************************************************** 
 * Traitement des tokens SYSTEM
 *****************************************************************************/
void cudaHookSystem(astNode * n,nablaMain *arc, const char cnf, char enum_enum){
  char *itm=(cnf=='c')?"cell":(cnf=='n')?"node":"face";
  char *etm=(enum_enum=='c')?"c":(enum_enum=='n')?"n":"f";
  if (n->tokenid == LID)           nprintf(arc, "/*chs*/", "[%s->localId()]",itm);//asInteger
  if (n->tokenid == SID)           nprintf(arc, "/*chs*/", "[subDomain()->subDomainId()]");
  if (n->tokenid == THIS)          nprintf(arc, "/*chs THIS*/", NULL);
  if (n->tokenid == NBNODE)        nprintf(arc, "/*chs NBNODE*/", NULL);
  if (n->tokenid == NBCELL)        nprintf(arc, "/*chs NBCELL*/", NULL);
  //if (n->tokenid == INODE)         nprintf(arc, "/*chs INODE*/", NULL);
  if (n->tokenid == BOUNDARY_CELL) nprintf(arc, "/*chs BOUNDARY_CELL*/", NULL);
  if (n->tokenid == FATAL)         nprintf(arc, "/*chs*/", "throw FatalErrorException]");
  if (n->tokenid == BACKCELL)      nprintf(arc, "/*chs*/", "[%s->backCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == BACKCELLUID)   nprintf(arc, "/*chs*/", "[%s->backCell().uniqueId()]",itm);
  if (n->tokenid == FRONTCELL)     nprintf(arc, "/*chs*/", "[%s->frontCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == FRONTCELLUID)  nprintf(arc, "/*chs*/", "[%s->frontCell().uniqueId()]",itm);
  if (n->tokenid == NEXTCELL)      nprintf(arc, "/*chs NEXTCELL*/", "[nextCell]");
  if (n->tokenid == PREVCELL)      nprintf(arc, "/*chs PREVCELL*/", "[prevCell]");
  if (n->tokenid == NEXTNODE)      nprintf(arc, "/*chs NEXTNODE*/", "[nextNode]");
  if (n->tokenid == PREVNODE)      nprintf(arc, "/*chs PREVNODE*/", "[prevNode]");
  if (n->tokenid == PREVLEFT)      nprintf(arc, "/*chs PREVLEFT*/", "[cn.previousLeft()]");
  if (n->tokenid == PREVRIGHT)     nprintf(arc, "/*chs PREVRIGHT*/", "[cn.previousRight()]");
  if (n->tokenid == NEXTLEFT)      nprintf(arc, "/*chs NEXTLEFT*/", "[cn.nextLeft()]");
  if (n->tokenid == NEXTRIGHT)     nprintf(arc, "/*chs NEXTRIGHT*/", "[cn.nextRight()]");
  //error(!0,0,"Could not switch Cuda Hook System!");
}


/*****************************************************************************
 * Prépare le nom de la fonction
 *****************************************************************************/
static void nvar(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  nprintf(nabla, "/*tt2a*/", "%s_%s", var->item, var->name);
  if (strcmp(var->type,"real3")!=0){
    nprintf(nabla, "/*nvar no diffraction possible here*/",NULL);
    return;
  }
  return;
}


/*****************************************************************************
 * Postfix d'un .x|y|z selon le isDotXYZ
 *****************************************************************************/
static void setDotXYZ(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  //nprintf(nabla,NULL,"/*setDotXYZ*/");
  switch (job->parse.isDotXYZ){
  case(0): break;
  case(1): {nprintf(nabla, "/*setDotX+flush*/", ".x"); break;}
  case(2): {nprintf(nabla, "/*setDotY+flush*/", ".y"); break;}
  case(3): {nprintf(nabla, "/*setDotZ+flush*/", ".z"); break;}
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[nvar] Switch isDotXYZ error\n"));
  }
  //nprintf(nabla,NULL,"/*setDotXYZ: Flushing isDotXYZ*/");
  job->parse.isDotXYZ=0;
  job->parse.turnBracketsToParentheses=false;
}


/*****************************************************************************
 * Tokens to variables 'CELL Job' switch
 *****************************************************************************/
static void cudaHookTurnTokenToVariableForCellJob(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'c') return;
  
  nprintf(arc, "/*CellJob*/",NULL);
  
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if ((job->parse.function_call_arguments==true)&&(var->dim==1)){
    //nprintf(arc, "/*function_call_arguments,*/","&");
  }
  switch (var->item[0]){
  case ('c'):{
    nvar(arc,var,job);
    nprintf(arc, "/*CellVar*/",
            "%s",
            ((var->dim==0)? (isPostfixed==2)?"":"[tcid":
             //(enum_enum!='\0')?"[n+8*(tcid)"://"[n*NABLA_NB_CELLS+tcid+c]":
             (enum_enum!='\0')?"[tcid+n*NABLA_NB_CELLS"://"[n*NABLA_NB_CELLS+tcid+c]":
             (var->dim==1)?"[8*(tcid)":"[tcid")); // [(cell_xs_node[n])[tcid+c]] [n.index()]
    job->parse.variableIsArray=(var->dim==1)?true:false;
    if (job->parse.postfix_constant==true
        && job->parse.variableIsArray==true)
      nprintf(arc, NULL,"+");
    else
      nprintf(arc, NULL,"]");
    break;
  }
  case ('n'):{
    nvar(arc,var,job);
    if (enum_enum=='f') nprintf(arc, "/*f*/", "[");
    if (enum_enum=='n') nprintf(arc, "/*n*/", "[cell_node[tcid+n*NABLA_NB_CELLS]]");
    if (isPostfixed!=2 && enum_enum=='\0'){
      if (job->parse.postfix_constant==true)
        nprintf(arc, "/*NodeVar + postfix_constant*/", "[");
      else
        nprintf(arc, "/*NodeVar 0*/", "[cell_node_");
    }
    if (isPostfixed==2 && enum_enum=='\0') nprintf(arc, "/*NodeVar 2&0*/", "[cell_node_");
    if (job->parse.postfix_constant!=true) setDotXYZ(arc,var,job);
    break;
  }
  case ('f'):{
    nvar(arc,var,job);
    if (enum_enum=='f') nprintf(arc, "/*FaceVar*/", "[f]");
    if (enum_enum=='\0') nprintf(arc, "/*FaceVar*/", "[cell->face");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "*%s_%s", var->item, var->name);
    break;      // GLOBAL variable
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job turnTokenToCudaVariable\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'NODE Job' switch
 *****************************************************************************/
static void cudaHookTurnTokenToVariableForNodeJob(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'n') return;
  nprintf(arc, "/*NodeJob*/", NULL);

  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);

  switch (var->item[0]){
  case ('c'):{
    if (var->dim!=0)     nprintf(arc, "/*CellVar dim!0*/", "[8*tnid+i]");//tcid][c");
    if (var->dim==0 && enum_enum=='f')  nprintf(arc, "/*CellVar f*/", "[");
    if (var->dim==0 && enum_enum=='n')  nprintf(arc, "/*CellVar n*/", "[n]");
    if (var->dim==0 && enum_enum=='c')  nprintf(arc, "/*CellVar c*/", "[8*tnid+i]");
    if (var->dim==0 && enum_enum=='\0') nprintf(arc, "/*CellVar 0*/", "[cell->node");
    break;
  }
  case ('n'):{
    if ((isPostfixed!=2) && enum_enum=='f')  nprintf(arc, "/*NodeVar !2f*/", "[tnid]");
    if ((isPostfixed==2) && enum_enum=='f')  ;//nprintf(arc, NULL);
    if ((isPostfixed==2) && enum_enum=='\0') nprintf(arc, "/*NodeVar 20*/", NULL);
    if ((isPostfixed!=2) && enum_enum=='n')  nprintf(arc, "/*NodeVar !2n*/", "[n]");
    if ((isPostfixed!=2) && enum_enum=='c')  nprintf(arc, "/*NodeVar !2c*/", "[tnid]");
    if ((isPostfixed!=2) && enum_enum=='\0') nprintf(arc, "/*NodeVar !20*/", "[tnid]");
    break;
  }
  case ('f'):{
    if (enum_enum=='f')  nprintf(arc, "/*FaceVar f*/", "[f]");
    if (enum_enum=='\0') nprintf(arc, "/*FaceVar 0*/", "[face]");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "*%s_%s", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] NODES job turnTokenToCudaVariable\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'FACE Job' switch
 *****************************************************************************/
static void cudaHookTurnTokenToVariableForFaceJob(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'f') return;
  nprintf(arc, "/*FaceJob*/", NULL);
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);
  switch (var->item[0]){
  case ('c'):{
    nprintf(arc, "/*CellVar*/",
            "%s",
            ((var->dim==0)?
             ((enum_enum=='\0')?
              (isPostfixed==2)?"[":"[face->cell"
              :"[c")
             :"[cell][node->cell")); 
    break;
  }
  case ('n'):{
    nprintf(arc, "/*NodeVar*/", "[face->node");
    break;
  }
  case ('f'):{
    nprintf(arc, "/*FaceVar*/", "[face]");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "*%s_%s", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job turnTokenToCudaVariable\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'Std Function' switch
 *****************************************************************************/
static void cudaHookTurnTokenToVariableForStdFunction(nablaMain *arc,
                                                      nablaVariable *var,
                                                      nablaJob *job){
  const char cnfg=job->item[0];
 
  // Preliminary pertinence test
  if (cnfg != '\0') return;
  nprintf(arc, "/*StdJob*/", NULL);// Fonction standard
  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);
  switch (var->item[0]){
  case ('c'):{
    nprintf(arc, "/*CellVar*/", NULL);// CELL variable
    break;
  }
  case ('n'):{
    nprintf(arc, "/*NodeVar*/", NULL); // NODE variable
    break;
  }
  case ('f'):{
    nprintf(arc, "/*FaceVar*/", NULL);// FACE variable
    break;
  }
  case('g'):{
    nprintf(arc, "/*GlobalVar*/", "*%s_%s", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] StdJob turnTokenToCudaVariable\n"));
  }
}


/*****************************************************************************
 * Transformation de tokens en variables Cuda selon les contextes dans le cas d'un '[Cell|node]Enumerator'
 *****************************************************************************/
nablaVariable *cudaHookTurnTokenToVariable(astNode * n,
                                           nablaMain *arc,
                                           nablaJob *job){
  nablaVariable *var=nablaVariableFind(arc->variables, n->token);
  
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL) return NULL;
  dbg("\n\t[cudaHookTurnTokenToVariable] %s_%s token=%s", var->item, var->name, n->token);

  // Set good isDotXYZ
  if (job->parse.isDotXYZ==0 && strcmp(var->type,"real3")==0 && job->parse.left_of_assignment_operator==true){
//    #warning Diffracting OFF
    //nprintf(arc, NULL, "/* DiffractingNOW */");
    //job->parse.diffracting=true;
    //job->parse.isDotXYZ=job->parse.diffractingXYZ=1;
  }
  //nprintf(arc, NULL, "\n\t/*cudaHookTurnTokenToVariable::isDotXYZ=%d*/", job->parse.isDotXYZ);
 
  // Check whether there's job for a cell job
  cudaHookTurnTokenToVariableForCellJob(arc,var,job);
  // Check whether there's job for a node job
  cudaHookTurnTokenToVariableForNodeJob(arc,var,job);
  // Check whether there's job for a face job
  cudaHookTurnTokenToVariableForFaceJob(arc,var,job);
  // Check whether there's job for a face job
  cudaHookTurnTokenToVariableForStdFunction(arc,var,job);
  return var;
}


/***************************************************************************** 
 * Upcase de la chaîne donnée en argument
 *****************************************************************************/
static inline char *itemUPCASE(const char *itm){
  if (itm[0]=='c') return "CELLS";
  if (itm[0]=='n') return "NODES";
  if (itm[0]=='g') return "GLOBAL";
  dbg("\n\t[itemUPCASE] itm=%s", itm);
  exit(NABLA_ERROR|fprintf(stderr, "\n[itemUPCASE] Error with given item\n"));
  return NULL;
}



/***************************************************************************** 
 * enums pour les différents dumps à faire: déclaration, malloc et free
 *****************************************************************************/
typedef enum {
  CUDA_VARIABLES_DECLARATION=0,
  CUDA_VARIABLES_MALLOC,
  CUDA_VARIABLES_FREE
} CUDA_VARIABLES_SWITCH;


// Pointeur de fonction vers une qui dump ce que l'on souhaite
typedef NABLA_STATUS (*pFunDump)(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth);


/***************************************************************************** 
 * Dump d'un MALLOC d'une variables dans le fichier source
 *****************************************************************************/
static NABLA_STATUS cudaGenerateSingleVariableMalloc(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth){
  if (var->dim==0){
    fprintf(nabla->entity->src,"\n\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&%s_%s%s%s, NABLA_NB_%s*sizeof(%s)));",
            var->item, var->name, postfix?postfix:"", depth?depth:"",
            itemUPCASE(var->item), postfix?"real":var->type);
  }else{
    fprintf(nabla->entity->src,
            "\n\tCUDA_HANDLE_ERROR(cudaCalloc((void**)&%s_%s, NABLA_NB_%s*8*sizeof(%s)));",
            var->item, var->name, 
            itemUPCASE(var->item), postfix?"real":var->type);
  }
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'un FREE d'une variables dans le fichier source
 *****************************************************************************/
static NABLA_STATUS cudaGenerateSingleVariableFree(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth){  
  if (var->dim==0)
    fprintf(nabla->entity->src,"\n\tCUDA_HANDLE_ERROR(cudaFree(%s_%s%s%s));",
            var->item, var->name, postfix?postfix:"", depth?depth:"");
  else
    fprintf(nabla->entity->src,"\n\tCUDA_HANDLE_ERROR(cudaFree(%s_%s));",
            var->item, var->name);
  
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'une variables dans le fichier
 *****************************************************************************/
static NABLA_STATUS cudaGenerateSingleVariable(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth){  
  if (var->dim==0)
    fprintf(nabla->entity->hdr,"\n__builtin_align__(8) %s *%s_%s%s%s; //%s host_%s_%s%s%s[NABLA_NB_%s];",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"", depth?depth:"",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"", depth?depth:"",
            itemUPCASE(var->item));
  if (var->dim==1)
    fprintf(nabla->entity->hdr,"\n__builtin_align__(8) %s *%s_%s%s; //%s host_%s_%s%s[NABLA_NB_%s][%ld];",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"",
            itemUPCASE(var->item),
            var->size);
  return NABLA_OK;
}


/***************************************************************************** 
 * Retourne quelle fonction selon l'enum donné
 *****************************************************************************/
pFunDump witch2func(CUDA_VARIABLES_SWITCH witch){
  switch (witch){
  case (CUDA_VARIABLES_DECLARATION): return cudaGenerateSingleVariable;
  case (CUDA_VARIABLES_MALLOC): return cudaGenerateSingleVariableMalloc;
  case (CUDA_VARIABLES_FREE): return cudaGenerateSingleVariableFree;
  default: exit(NABLA_ERROR|fprintf(stderr, "\n[witch2switch] Error with witch\n"));
  }
}






/***************************************************************************** 
 * Dump d'une variables de dimension 1
 *****************************************************************************/
static NABLA_STATUS cudaGenericVariableDim1(nablaMain *nabla, nablaVariable *var, pFunDump fDump){
  //int i;
  //char depth[]="[0]";
  dbg("\n[cudaGenerateVariableDim1] variable %s", var->name);
  //for(i=0;i<NABLA_HARDCODED_VARIABLE_DIM_1_DEPTH;++i,depth[1]+=1) fDump(nabla, var, NULL, depth);
  fDump(nabla, var, NULL, "/*8*/");
  return NABLA_OK;
}

/***************************************************************************** 
 * Dump d'une variables de dimension 0
 *****************************************************************************/
static NABLA_STATUS cudaGenericVariableDim0(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  dbg("\n[cudaGenerateVariableDim0] variable %s", var->name);
  if (strcmp(var->type,"real3")!=0)
    return fDump(nabla, var, NULL, NULL);
  else
    return fDump(nabla, var, NULL, NULL);
  /*return fDump(nabla, var, "_x", NULL)|
           fDump(nabla, var, "_y", NULL)|
           fDump(nabla, var, "_z", NULL);*/
  return NABLA_ERROR;
}

/***************************************************************************** 
 * Dump d'une variables
 *****************************************************************************/
static NABLA_STATUS cudaGenericVariable(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  if (!var->axl_it) return NABLA_OK;
  if (var->item==NULL) return NABLA_ERROR;
  if (var->name==NULL) return NABLA_ERROR;
  if (var->type==NULL) return NABLA_ERROR;
  if (var->dim==0) return cudaGenericVariableDim0(nabla,var,fDump);
  if (var->dim==1) return cudaGenericVariableDim1(nabla,var,fDump);
  dbg("\n[cudaGenericVariable] variable dim error: %d", var->dim);
  exit(NABLA_ERROR|fprintf(stderr, "\n[cudaGenericVariable] Error with given variable\n"));
}


/***************************************************************************** 
 * Dump des options
 *****************************************************************************/
static void cudaOptions(nablaMain *nabla){
  nablaOption *opt;

  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Options\n\
// ********************************************************");
  for(opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->hdr,
            "\n#define %s %s",
            opt->name,
            opt->dflt);
}

/***************************************************************************** 
 * Dump des globals
 *****************************************************************************/
static void cudaGlobals(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Globals, coté CUDA\n\
// ********************************************************\n\
double *global_deltat;\n\
int *global_iteration;\n\
double *global_time; double host_time;\n\
double *global_min_array; //double host_min_array[CUDA_NB_BLOCKS_PER_GRID];\n");
}


/***************************************************************************** 
 * Dump des variables
 *****************************************************************************/
void cudaVariablesPrefix(nablaMain *nabla){
  nablaVariable *var;

  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Variables\n\
// ********************************************************");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (cudaGenericVariable(nabla, var, witch2func(CUDA_VARIABLES_DECLARATION))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[cudaVariables] Error with variable %s\n", var->name));
    if (cudaGenericVariable(nabla, var, witch2func(CUDA_VARIABLES_MALLOC))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[cudaVariables] Error with variable %s\n", var->name));
  }
  cudaOptions(nabla);
  cudaGlobals(nabla);
}



void cudaVariablesPostfix(nablaMain *nabla){
  nablaVariable *var;
  for(var=nabla->variables;var!=NULL;var=var->next)
    if (cudaGenericVariable(nabla, var, witch2func(CUDA_VARIABLES_FREE))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[cudaVariables] Error with variable %s\n", var->name));
}



