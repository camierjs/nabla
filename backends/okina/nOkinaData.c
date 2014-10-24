/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nccOkinaVariables.c
 * Author   : Camier Jean-Sylvain
 * Created  : 2012.12.13
 * Updated  : 2012.12.13
 *****************************************************************************
 * Description:
 *****************************************************************************
 * Date			Author	Description
 * 2012.12.13	camierjs	Creation
 *****************************************************************************/
#include "nabla.h"
#include "nabla.tab.h"

/***************************************************************************** 
 * Traitement des transformations '[', '(' & ''
 *****************************************************************************/
void okinaHookTurnBracketsToParentheses(nablaMain* nabla, nablaJob *job, nablaVariable *var, char cnfg){
  dbg("\n\t[actJobItemParse] primaryExpression hits variable");
  if (  (cnfg=='c' && var->item[0]=='n')
      ||(cnfg=='c' && var->item[0]=='f')
      ||(cnfg=='n' && var->item[0]!='n')            
      ||(cnfg=='f' && var->item[0]!='f')
      ||(cnfg=='e' && var->item[0]!='e')
      ||(cnfg=='m' && var->item[0]!='m')
      ){
    if (!job->parse.selection_statement_in_compound_statement)
      nprintf(nabla, "/*turnBracketsToParentheses@true*/", "/*%c %c*/", cnfg, var->item[0]);
    else
      nprintf(nabla, "/*turnBracketsToParentheses+if@true*/", "/*%c %c*/cell_node[", cnfg, var->item[0]);      
    job->parse.turnBracketsToParentheses=true;
  }else{
    if (job->parse.postfix_constant==true
        && job->parse.variableIsArray==true) return;
    if (job->parse.isDotXYZ==1) nprintf(nabla, "/*okinaHookTurnBracketsToParentheses_X*/", NULL);
    if (job->parse.isDotXYZ==2) nprintf(nabla, "/*okinaHookTurnBracketsToParentheses_Y*/", NULL);
    if (job->parse.isDotXYZ==3) nprintf(nabla, "/*okinaHookTurnBracketsToParentheses_Z*/", NULL);
    job->parse.isDotXYZ=0;
    job->parse.turnBracketsToParentheses=false;
  }
}


/***************************************************************************** 
 * Traitement des tokens SYSTEM
 *****************************************************************************/
void okinaHookSystem(astNode * n,nablaMain *arc, const char cnf, char enum_enum){
  char *itm=(cnf=='c')?"cell":(cnf=='n')?"node":"face";
  char *etm=(enum_enum=='c')?"c":(enum_enum=='n')?"n":"f";
  if (n->tokenid == LID)           nprintf(arc, "/*chs*/", "[%s->localId()]",itm);//asInteger
  if (n->tokenid == SID)           nprintf(arc, "/*chs*/", "[subDomain()->subDomainId()]");
  if (n->tokenid == THIS)          nprintf(arc, "/*chs THIS*/", NULL);
  if (n->tokenid == NBNODE)        nprintf(arc, "/*chs NBNODE*/", NULL);
  if (n->tokenid == NBCELL)        nprintf(arc, "/*chs NBCELL*/", NULL);
  //if (n->tokenid == INODE)         nprintf(arc, "/*chs INODE*/", NULL);
  if (n->tokenid == BOUNDARY_CELL) nprintf(arc, "/*chs BOUNDARY_CELL*/", NULL);
  if (n->tokenid == FATAL)         nprintf(arc, "/*chs*/", "throw FatalErrorException");
  if (n->tokenid == BACKCELL)      nprintf(arc, "/*chs*/", "[%s->backCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == BACKCELLUID)   nprintf(arc, "/*chs*/", "[%s->backCell().uniqueId()]",itm);
  if (n->tokenid == FRONTCELL)     nprintf(arc, "/*chs*/", "[%s->frontCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == FRONTCELLUID)  nprintf(arc, "/*chs*/", "[%s->frontCell().uniqueId()]",itm);
  if (n->tokenid == NEXTCELL)      nprintf(arc, NULL, ")");
  if (n->tokenid == PREVCELL)      nprintf(arc, NULL, ")");
  if (n->tokenid == NEXTNODE)      nprintf(arc, NULL, "[n])+nextNode))");
  if (n->tokenid == PREVNODE)      nprintf(arc, NULL, "[n])-prevNode))");
  if (n->tokenid == PREVLEFT)      nprintf(arc, "/*chs PREVLEFT*/", "[cn.previousLeft()]");
  if (n->tokenid == PREVRIGHT)     nprintf(arc, "/*chs PREVRIGHT*/", "[cn.previousRight()]");
  if (n->tokenid == NEXTLEFT)      nprintf(arc, "/*chs NEXTLEFT*/", "[cn.nextLeft()]");
  if (n->tokenid == NEXTRIGHT)     nprintf(arc, "/*chs NEXTRIGHT*/", "[cn.nextRight()]");
  //error(!0,0,"Could not switch Okina Hook System!");
}


/*****************************************************************************
 * Prépare le nom de la variable
 *****************************************************************************/
static void nvar(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  if (!job->parse.selection_statement_in_compound_statement){
    nprintf(nabla, "/*tt2a*/", "%s_%s", var->item, var->name);
  }else{
    nprintf(nabla,NULL,"/*%s*/",var->type);
    if (strcmp(var->type,"real")==0)
      nprintf(nabla, "/*tt2a(if+real)*/", "((double*)%s_%s)", var->item, var->name);
    if (strcmp(var->type,"integer")==0)
      nprintf(nabla, "/*tt2a(if+int)*/", "((int*)%s_%s)", var->item, var->name);
    if (strcmp(var->type,"real3")==0)
      nprintf(nabla, "/*tt2a(if+real3)*/", "/*if+real3 still in real3 vs double3*/%s_%s", var->item, var->name);
    //nprintf(nabla, "/*tt2a(if+real3)*/", "((double3*)%s_%s)", var->item, var->name);
  }    
  if (strcmp(var->type,"real3")!=0){
    nprintf(nabla, "/*nvar no diffraction possible here*/",NULL);
    return;
  }
  return;
}


/*****************************************************************************
 * Postfix d'un .x|y|z slon le isDotXYZ
 *****************************************************************************/
static void setDotXYZ(nablaMain *nabla, nablaVariable *var, nablaJob *job){
  switch (job->parse.isDotXYZ){
  case(0): break;
  case(1): {nprintf(nabla, "/*setDotX+flush*/", ""); break;}
  case(2): {nprintf(nabla, "/*setDotY+flush*/", ""); break;}
  case(3): {nprintf(nabla, "/*setDotZ+flush*/", ""); break;}
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[nvar] Switch isDotXYZ error\n"));
  }
  // Flush isDotXYZ
  job->parse.isDotXYZ=0;
  job->parse.turnBracketsToParentheses=false;
}


/*****************************************************************************
 * Tokens to gathered  variables
 *****************************************************************************/
static bool okinaHookTurnTokenToGatheredVariable(nablaMain *arc,
                                                 nablaVariable *var,
                                                 nablaJob *job){
  //nprintf(arc, NULL, "/*gathered variable?*/");
  if (!var->is_gathered) return false;
  nprintf(arc, NULL, "/*gathered variable!*/gathered_%s_%s",var->item,var->name);
  return true;
}


/*****************************************************************************
 * Tokens to variables 'CELL Job' switch
 *****************************************************************************/
static void okinaHookTurnTokenToVariableForCellJob(nablaMain *arc,
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
            ((var->dim==0)? (isPostfixed==2)?"":"[c":
             (enum_enum!='\0')?"/*ee*/[n+8*c":
             (var->dim==1)?"/*c1*/[8*c":"[c"));
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
    if (enum_enum=='n') nprintf(arc, "/*n*/", "[cell_node[n*NABLA_NB_CELLS+c]]");
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
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;      // GLOBAL variable
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job okinaHookTurnTokenToVariableForCellJob\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'NODE Job' switch
 *****************************************************************************/
static void okinaHookTurnTokenToVariableForNodeJob(nablaMain *arc,
                                                  nablaVariable *var,
                                                  nablaJob *job){
  const char cnfg=job->item[0];
  char enum_enum=job->parse.enum_enum;
  int isPostfixed=job->parse.isPostfixed;

  // Preliminary pertinence test
  if (cnfg != 'n') return;
  nprintf(arc, "/*NodeJob*/",NULL);

  // On dump le nom de la variable trouvée, sauf pour les globals qu'on doit faire précédé d'un '*'
  if (var->item[0]!='g') nvar(arc,var,job);

  switch (var->item[0]){
  case ('c'):{
    if (var->dim!=0)     nprintf(arc, "/*CellVar dim!0*/", "[c][c");
    if (enum_enum=='f')  nprintf(arc, "/*CellVar f*/", "[");
    if (enum_enum=='n')  nprintf(arc, "/*CellVar n*/", "[n]");
    if (enum_enum=='c')  nprintf(arc, "/*CellVar c*/", "[c]");
    if (enum_enum=='\0') nprintf(arc, "/*CellVar 0*/", "[cell->node");
    break;
  }
  case ('n'):{
    if ((isPostfixed!=2) && enum_enum=='f')  nprintf(arc, "/*NodeVar !2f*/", "/*!2f*/[n]");
    if ((isPostfixed==2) && enum_enum=='f')  ;//nprintf(arc, NULL);
    if ((isPostfixed==2) && enum_enum=='\0') nprintf(arc, "/*NodeVar 20*/", NULL);
    if ((isPostfixed!=2) && enum_enum=='n')  nprintf(arc, "/*NodeVar !2n*/", "/*!2n*/[n]");
    if ((isPostfixed!=2) && enum_enum=='c')  nprintf(arc, "/*NodeVar !2c*/", "/*!2c*/[n]");
    if ((isPostfixed!=2) && enum_enum=='\0') nprintf(arc, "/*NodeVar !20*/", "/*!20*/[n]");
    break;
  }
  case ('f'):{
    if (enum_enum=='f')  nprintf(arc, "/*FaceVar f*/", "[f]");
    if (enum_enum=='\0') nprintf(arc, "/*FaceVar 0*/", "[face]");
    break;
  }
  case ('g'):{
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] NODES job okinaHookTurnTokenToVariableForNodeJob\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'FACE Job' switch
 *****************************************************************************/
static void okinaHookTurnTokenToVariableForFaceJob(nablaMain *arc,
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
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] CELLS job okinaHookTurnTokenToVariableForFaceJob\n"));
  }
}


/*****************************************************************************
 * Tokens to variables 'Std Function' switch
 *****************************************************************************/
static void okinaHookTurnTokenToVariableForStdFunction(nablaMain *arc,
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
    nprintf(arc, "/*GlobalVar*/", "%s_%s[0]", var->item, var->name);
    break;
  }
  default:exit(NABLA_ERROR|fprintf(stderr, "\n[ncc] StdJob okinaHookTurnTokenToVariableForStdFunction\n"));
  }
}


/*****************************************************************************
 * Transformation de tokens en variables selon les contextes dans le cas d'un '[Cell|node]Enumerator'
 *****************************************************************************/
nablaVariable *okinaHookTurnTokenToVariable(astNode * n,
                                            nablaMain *arc,
                                            nablaJob *job){
  nablaVariable *var=nablaVariableFind(arc->variables, n->token);
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL) return NULL;
  dbg("\n\t[okinaHookTurnTokenToVariable] %s_%s token=%s", var->item, var->name, n->token);

  // Set good isDotXYZ
  if (job->parse.isDotXYZ==0 && strcmp(var->type,"real3")==0 && job->parse.left_of_assignment_operator==true){
//    #warning Diffracting OFF
    //nprintf(arc, NULL, "/* DiffractingNOW */");
    //job->parse.diffracting=true;
    //job->parse.isDotXYZ=job->parse.diffractingXYZ=1;
  }
  //nprintf(arc, NULL, "\n\t/*okinaHookTurnTokenToVariable::isDotXYZ=%d, job->parse.diffractingXYZ=%d*/", job->parse.isDotXYZ, job->parse.diffractingXYZ);

  // Check whether this variable is being gathered
  if (okinaHookTurnTokenToGatheredVariable(arc,var,job)){
    return var;
  }
  
  // Check whether there's job for a cell job
  okinaHookTurnTokenToVariableForCellJob(arc,var,job);
  
  // Check whether there's job for a node job
  okinaHookTurnTokenToVariableForNodeJob(arc,var,job);
  
  // Check whether there's job for a face job
  okinaHookTurnTokenToVariableForFaceJob(arc,var,job);
  
  // Check whether there's job for a face job
  okinaHookTurnTokenToVariableForStdFunction(arc,var,job);
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
  OKINA_VARIABLES_DECLARATION=0,
  OKINA_VARIABLES_MALLOC,
  OKINA_VARIABLES_FREE
} OKINA_VARIABLES_SWITCH;


// Pointeur de fonction vers une qui dump ce que l'on souhaite
typedef NABLA_STATUS (*pFunDump)(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth);


/***************************************************************************** 
 * Dump d'un MALLOC d'une variables dans le fichier source
 *****************************************************************************/
static NABLA_STATUS okinaGenerateSingleVariableMalloc(nablaMain *nabla,
                                                    nablaVariable *var,
                                                    char *postfix,
                                                    char *depth){
  nprintf(nabla,"\n\t// okinaGenerateSingleVariableMalloc",NULL);
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'un FREE d'une variables dans le fichier source
 *****************************************************************************/
static NABLA_STATUS okinaGenerateSingleVariableFree(nablaMain *nabla,
                                                  nablaVariable *var,
                                                  char *postfix,
                                                  char *depth){  
  nprintf(nabla,"\n\t// okinaGenerateSingleVariableFree",NULL);
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'une variables dans le fichier
 *****************************************************************************/
static NABLA_STATUS okinaGenerateSingleVariable(nablaMain *nabla,
                                              nablaVariable *var,
                                              char *postfix,
                                              char *depth){  
  nprintf(nabla,"\n\t// okinaGenerateSingleVariable",NULL);
  if (var->dim==0)
    fprintf(nabla->entity->hdr,"\n%s %s_%s%s%s[NABLA_NB_%s/WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));",
            postfix?"real":var->type, var->item, var->name, postfix?postfix:"", depth?depth:"",
            itemUPCASE(var->item));
  if (var->dim==1)
    fprintf(nabla->entity->hdr,"\n%s %s_%s%s[%ld*NABLA_NB_%s/WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));;",
            postfix?"real":var->type,
            var->item,var->name,
            postfix?postfix:"",
            var->size,
            itemUPCASE(var->item));
  return NABLA_OK;
}


/***************************************************************************** 
 * Retourne quelle fonction selon l'enum donné
 *****************************************************************************/
static pFunDump witch2func(OKINA_VARIABLES_SWITCH witch){
  switch (witch){
  case (OKINA_VARIABLES_DECLARATION): return okinaGenerateSingleVariable;
  case (OKINA_VARIABLES_MALLOC): return okinaGenerateSingleVariableMalloc;
  case (OKINA_VARIABLES_FREE): return okinaGenerateSingleVariableFree;
  default: exit(NABLA_ERROR|fprintf(stderr, "\n[witch2switch] Error with witch\n"));
  }
}


/***************************************************************************** 
 * Dump d'une variables de dimension 1
 *****************************************************************************/
static NABLA_STATUS okinaGenericVariableDim1(nablaMain *nabla, nablaVariable *var, pFunDump fDump){
  //int i;
  //char depth[]="[0]";
  dbg("\n[okinaGenerateVariableDim1] variable %s", var->name);
  //for(i=0;i<NABLA_HARDCODED_VARIABLE_DIM_1_DEPTH;++i,depth[1]+=1) fDump(nabla, var, NULL, depth);
  fDump(nabla, var, NULL, "/*8*/");
  return NABLA_OK;
}

/***************************************************************************** 
 * Dump d'une variables de dimension 0
 *****************************************************************************/
static NABLA_STATUS okinaGenericVariableDim0(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  dbg("\n[okinaGenerateVariableDim0] variable %s", var->name);
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
static NABLA_STATUS okinaGenericVariable(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  if (!var->axl_it) return NABLA_OK;
  if (var->item==NULL) return NABLA_ERROR;
  if (var->name==NULL) return NABLA_ERROR;
  if (var->type==NULL) return NABLA_ERROR;
  if (var->dim==0) return okinaGenericVariableDim0(nabla,var,fDump);
  if (var->dim==1) return okinaGenericVariableDim1(nabla,var,fDump);
  dbg("\n[okinaGenericVariable] variable dim error: %d", var->dim);
  exit(NABLA_ERROR|fprintf(stderr, "\n[okinaGenericVariable] Error with given variable\n"));
}


/***************************************************************************** 
 * Dump des options
 *****************************************************************************/
static void okinaOptions(nablaMain *nabla){
  nablaOption *opt;
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * Options\n\
// ********************************************************");
  for(opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->hdr,
            "\n#define %s %s",
            opt->name, opt->dflt);
}

/***************************************************************************** 
 * Dump des globals
 *****************************************************************************/
static void okinaGlobals(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * Temps de la simulation\n\
// ********************************************************\n\
Real global_deltat[1];\n\
int global_iteration;\n\
double global_time;\n");
}


/***************************************************************************** 
 * Dump des variables
 *****************************************************************************/
void okinaVariablesPrefix(nablaMain *nabla){
  nablaVariable *var;

  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Variables\n\
// ********************************************************");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (okinaGenericVariable(nabla, var, witch2func(OKINA_VARIABLES_DECLARATION))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[okinaVariables] Error with variable %s\n", var->name));
    if (okinaGenericVariable(nabla, var, witch2func(OKINA_VARIABLES_MALLOC))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[okinaVariables] Error with variable %s\n", var->name));
  }
  okinaOptions(nabla);
  okinaGlobals(nabla);
}



void okinaVariablesPostfix(nablaMain *nabla){
  nablaVariable *var;
  for(var=nabla->variables;var!=NULL;var=var->next)
    if (okinaGenericVariable(nabla, var, witch2func(OKINA_VARIABLES_FREE))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[okinaVariables] Error with variable %s\n", var->name));
}



