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
#include "ccHook.h"
#include "nabla.tab.h"
#include "frontend/nablaAst.h"


// ****************************************************************************
// * Std or Mic DEFINES
// ****************************************************************************
nablaDefine ccDefines[]={
  {"integer", "Integer"},
  {"real", "Real"},
  {"WARP_SIZE", "(1<<WARP_BIT)"},
  {"WARP_ALIGN", "(8<<WARP_BIT)"},    
  {"NABLA_NB_GLOBAL_WARP","WARP_SIZE"},
  {"reducemin(a)","0.0"},
  {"rabs(a)","fabs(a)"},
  {"set(a)", "a"},
  {"set1(cst)", "cst"},
  {"square_root(u)", "sqrt(u)"},
  {"cube_root(u)", "cbrt(u)"},
  {"store(u,_u)", "(*u=_u)"},
  {"load(u)", "(*u)"},
  {"zero()", "0.0"},
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_INI)"},
  {"DBG_OFF", "0x0000ul"},
  {"DBG_CELL_VOLUME", "0x0001ul"},
  {"DBG_CELL_CQS", "0x0002ul"},
  {"DBG_GTH", "0x0004ul"},
  {"DBG_NODE_FORCE", "0x0008ul"},
  {"DBG_INI_EOS", "0x0010ul"},
  {"DBG_EOS", "0x0020ul"},
  {"DBG_DENSITY", "0x0040ul"},
  {"DBG_MOVE_NODE", "0x0080ul"},
  {"DBG_INI", "0x0100ul"},
  {"DBG_INI_CELL", "0x0200ul"},
  {"DBG_INI_NODE", "0x0400ul"},
  {"DBG_LOOP", "0x0800ul"},
  {"DBG_FUNC_IN", "0x1000ul"},
  {"DBG_FUNC_OUT", "0x2000ul"},
  {"DBG_VELOCITY", "0x4000ul"},
  {"DBG_BOUNDARIES", "0x8000ul"},
  {"DBG_ALL", "0xFFFFul"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opMod(u,v)", "(u%v)"},
  {"opScaMul(u,v)","dot3(u,v)"},
  {"opVecMul(u,v)","cross(u,v)"},    
  {"dot", "dot3"},
  {"knAt(a)",""},
  {"fatal(a,b)","exit(-1)"},
  {"synchronize(a)","_Pragma(\"omp barrier\")"},
  {"mpi_reduce(how,what)","how##ToDouble(what)"},
  {"reduce(how,what)","how##ToDouble(what)"},
  {"xyz","int"},
  {"GlobalIteration", "global_iteration"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {NULL,NULL}
};


// ****************************************************************************
// * Forward Declarations
// ****************************************************************************
char* ccForwards[]={
  "inline std::ostream& info(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline std::ostream& debug(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline int WARP_BASE(int a){ return (a>>WARP_BIT);}",
  "inline int WARP_OFFSET(int a){ return (a&(WARP_SIZE-1));}",
  "inline int WARP_NFFSET(int a){ return ((WARP_SIZE-1)-WARP_OFFSET(a));}",
  "static void nabla_ini_node_coords(void);",
  "static void verifCoords(void);",
  NULL
};




// ****************************************************************************
// * IVDEP Pragma
// ****************************************************************************
char *ccHookPragmaIccIvdep(void){ return "\\\n_Pragma(\"ivdep\")"; }
char *ccHookPragmaGccIvdep(void){ return "__declspec(align(64)) "; }


// ****************************************************************************
// * ALIGN hooks
// ****************************************************************************
char *ccHookPragmaIccAlign(void){ return "\\\n_Pragma(\"ivdep\")"; }
char *ccHookPragmaGccAlign(void){ return "__attribute__ ((aligned(WARP_ALIGN))) "; }


// ****************************************************************************
// * INCLUDES
// ****************************************************************************
char* ccHookIncludes(void){return "";}
char *ccHookBits(void){return "64";}


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* ccHookPrevCell(void){
  return "gatherk_and_zero_neg_ones(cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+0],";
}


// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* ccHookNextCell(void){
  return "gatherk_and_zero_neg_ones(cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+0],";
}


// ****************************************************************************
// * Gather for Cells
// ****************************************************************************
static char* ccHookGatherCells(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration)
    return strdup("register int __attribute__((unused)) cw,ia;");
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
cw=(c<<WARP_BIT);\n\t\t\t\
gather%sk(ia=cell_node[n*NABLA_NB_CELLS+cw+0],\n\t\t\t\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":"real3",
           strcmp(var->type,"real")==0?"":"3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item,
           var->name);
  return strdup(gather);
}


// ****************************************************************************
// * Gather for Nodes
// * En STD, le gather aux nodes est le même qu'aux cells
// ****************************************************************************
static char* ccHookGatherNodes(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration){
    return strdup("int nw;");
  }
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
nw=(n<<WARP_BIT);\n\t\t\t\
//#warning continue node_cell_corner\n\
//if (node_cell_corner[8*nw+c]==-1) continue;\n\
gatherFromNode_%sk%s(node_cell[8*nw+c],\n\
%s\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":"real3",
           strcmp(var->type,"real")==0?"":"3",
           var->dim==0?"":"Array8",
           var->dim==0?"":"\t\t\t\t\t\tnode_cell_corner[8*nw+c],\n\t\t\t",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item,
           var->name);
  return strdup(gather);
}


// ****************************************************************************
// * Gather switch
// ****************************************************************************
char* ccHookGather(nablaJob *job,nablaVariable* var, enum_phase phase){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return ccHookGatherCells(job,var,phase);
  if (itm=='n') return ccHookGatherNodes(job,var,phase);
  error(!0,0,"Could not distinguish job item in ccStdGather!");
  return NULL;
}


// ****************************************************************************
// * Scatter
// ****************************************************************************
char* ccHookScatter(nablaVariable* var){
  char scatter[1024];
  snprintf(scatter, 1024, "\tscatter%sk(ia, &gathered_%s_%s, %s_%s);",
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           var->item, var->name);
  return strdup(scatter);
}


// ****************************************************************************
// * Std or Mic TYPEDEFS
// ****************************************************************************
nablaTypedef ccTypedef[]={
  {"struct real3","Real3"},
  {NULL,NULL}
};

// ****************************************************************************
// * PARALLEL OpenMP
// ****************************************************************************
char *ccHookParallelOpenMPSync(void){
  return "";//#pragma omp barrier\n";
}
char *ccHookParallelOpenMPSpawn(void){
  return "";//#pragma omp spawn ";
}
char *ccHookParallelOpenMPLoop(struct nablaMainStruct *n){
  return "\\\n_Pragma(\"omp parallel for firstprivate(NABLA_NB_CELLS,NABLA_NB_CELLS_WARP,NABLA_NB_NODES)\")\\\n";
  //return "\\\n_Pragma(\"ivdep\")\\\n_Pragma(\"vector aligned\")\\\n_Pragma(\"omp parallel for firstprivate(NABLA_NB_CELLS,NABLA_NB_CELLS_WARP,NABLA_NB_NODES)\")\\\n";
}
char *ccHookParallelOpenMPIncludes(void){
  return "#include <omp.h>\n";
}


// ****************************************************************************
// * PARALLEL Void
// ****************************************************************************
char *ccHookParallelVoidSync(void){
  return "";
}
char *ccHookParallelVoidSpawn(void){
  return "";
}
char *ccHookParallelVoidLoop(void){
  return "";
}
char *ccHookParallelVoidIncludes(void){
  return "";
}


// ****************************************************************************
// * PARALLEL Cilk
// ****************************************************************************
char *ccHookParallelCilkSync(void){
  return "cilk_sync;\n";
}
char *ccHookParallelCilkSpawn(void){
  return "cilk_spawn ";
}
char *ccHookParallelCilkLoop(struct nablaMainStruct *n){
  return "cilk_";
}
char *ccHookParallelCilkIncludes(void){
  return "#include <cilk/cilk.h>\n";
}


// ****************************************************************************
// * Function Hooks
// ****************************************************************************
void ccHookFunctionName(nablaMain *arc){
  nprintf(arc, NULL, "%s", arc->name);
}
void ccHookFunction(nablaMain *nabla, astNode *n){
  nablaJob *fct=nablaJobNew(nabla->entity);
  nablaJobAdd(nabla->entity, fct);
  nablaFctFill(nabla,fct,n,NULL);
}


// ****************************************************************************
// * ENUMERATES Hooks
// ****************************************************************************
void ccDefineEnumerates(nablaMain *nabla){
  const char *parallel_prefix_for_loop=nabla->parallel->loop(nabla);
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define FOR_EACH_CELL(c) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<8;n+=1)\n\
\n\
#define FOR_EACH_CELL_WARP(c) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
#define FOR_EACH_CELL_WARP_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
\n\
#define FOR_EACH_CELL_WARP_NODE(n)\\\n\
  %sfor(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\\\n\
    for(int n=8-1;n>=0;--n)\n\
\n\
#define FOR_EACH_NODE(n) /*%s*/for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=8*n;c<8;c+=1,nc+=1)\n\
\n\
#define FOR_EACH_NODE_WARP(n) %sfor(int n=0;n<NABLA_NB_NODES_WARP;n+=1)\n\
\n\
#define FOR_EACH_NODE_WARP_CELL(c)\\\n\
    for(int c=0;c<8;c+=1)\n",
          parallel_prefix_for_loop, // FOR_EACH_CELL
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_SHARED
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_NODE
          parallel_prefix_for_loop, // FOR_EACH_NODE
          parallel_prefix_for_loop  // FOR_EACH_NODE_WARP
          );
}



// ****************************************************************************
// * ccHookReduction
// ****************************************************************************
static void ccHookReduction(struct nablaMainStruct *nabla, astNode *n){
  const astNode *item_node = n->children->next->children;
  const astNode *global_var_node = n->children->next->next;
  const astNode *reduction_operation_node = global_var_node->next;
  const astNode *item_var_node = reduction_operation_node->next;
  const astNode *at_single_cst_node = item_var_node->next->next->children->next->children;
  char *global_var_name = global_var_node->token;
  char *item_var_name = item_var_node->token;
  // Préparation du nom du job
  char job_name[NABLA_MAX_FILE_NAME];
  job_name[0]=0;
  strcat(job_name,"ccReduction_");
  strcat(job_name,global_var_name);
  // Rajout du job de reduction
  nablaJob *redjob = nablaJobNew(nabla->entity);
  redjob->is_an_entry_point=true;
  redjob->is_a_function=false;
  redjob->scope  = strdup("NoGroup");
  redjob->region = strdup("NoRegion");
  redjob->item   = strdup(item_node->token);
  redjob->rtntp  = strdup("void");
  redjob->name   = strdup(job_name);
  redjob->name_utf8 = strdup(job_name);
  redjob->xyz    = strdup("NoXYZ");
  redjob->drctn  = strdup("NoDirection");
  assert(at_single_cst_node->parent->ruleid==rulenameToId("at_single_constant"));
  dbg("\n\t[ccHookReduction] @ %s",at_single_cst_node->token);
  sprintf(&redjob->at[0],at_single_cst_node->token);
  redjob->whenx  = 1;
  redjob->whens[0] = atof(at_single_cst_node->token);
  nablaJobAdd(nabla->entity, redjob);
  const double reduction_init = (reduction_operation_node->tokenid==MIN_ASSIGN)?1.0e20:0.0;
  // Génération de code associé à ce job de réduction
  nprintf(nabla, NULL, "\n\
// ******************************************************************************\n\
// * Kernel de reduction de la variable '%s' vers la globale '%s'\n\
// ******************************************************************************\n\
void %s(void){ // @ %s\n\
\tconst double reduction_init=%e;\n\
\tconst int threads = omp_get_max_threads();\n\
\tReal %s_per_thread[threads];\n\
\tdbgFuncIn();\n\
\tfor (int i=0; i<threads;i+=1) %s_per_thread[i] = reduction_init;\n\
\tFOR_EACH_%s_WARP_SHARED(%s,reduction_init){\n\
\t\tconst int tid = omp_get_thread_num();\n\
\t\t%s_per_thread[tid] = min(%s_%s[%s],%s_per_thread[tid]);\n\
\t}\n\
\tglobal_%s[0]=reduction_init;\n\
\tfor (int i=0; i<threads; i+=1){\n\
\t\tconst Real real_global_%s=global_%s[0];\n\
\t\tglobal_%s[0]=(ReduceMinToDouble(%s_per_thread[i])<ReduceMinToDouble(real_global_%s))?\n\
\t\t\t\t\t\t\t\t\tReduceMinToDouble(%s_per_thread[i]):ReduceMinToDouble(real_global_%s);\n\
\t}\n\
}\n\n",   item_var_name,global_var_name,
          job_name,
          at_single_cst_node->token,
          reduction_init,
          global_var_name,
          global_var_name,
          (item_node->token[0]=='c')?"CELL":(item_node->token[0]=='n')?"NODE":"NULL",
          (item_node->token[0]=='c')?"c":(item_node->token[0]=='n')?"n":"?",
          global_var_name,
          (item_node->token[0]=='c')?"cell":(item_node->token[0]=='n')?"node":"?",
          item_var_name,
          (item_node->token[0]=='c')?"c":(item_node->token[0]=='n')?"n":"?",
          global_var_name,
          global_var_name,global_var_name,
          global_var_name,global_var_name,global_var_name,
          global_var_name,global_var_name,global_var_name
          );  
}


// ****************************************************************************
// * Dump des variables appelées
// ****************************************************************************
static void ccHookDfsForCalls(struct nablaMainStruct *nabla,
                             nablaJob *fct,
                             astNode *n,
                             const char *namespace,
                             astNode *nParams){
  // Maintenant qu'on a tous les called_variables potentielles, on remplit aussi le hdr
  // On remplit la ligne du hdr
  hprintf(nabla, NULL, "\n%s %s %s%s(",
          nabla->hook->entryPointPrefix(nabla,fct),
          fct->rtntp,
          namespace?"Entity::":"",
          fct->name);
  // On va chercher les paramètres standards pour le hdr
  dumpParameterTypeList(nabla->entity->hdr, nParams);
  hprintf(nabla, NULL, ");");
}


// ****************************************************************************
// * Dump du préfix des points d'entrées: inline ou pas
// ****************************************************************************
static char* ccHookEntryPointPrefix(struct nablaMainStruct *nabla, nablaJob *entry_point){
  //return "";
  return "static inline";
}

static void ccHookIteration(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*ITERATION*/", "cc_iteration()");
}
static void ccHookExit(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*EXIT*/", "exit(0.0)");
}
static void ccHookTime(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*TIME*/", "global_time");
}
static void ccHookFatal(struct nablaMainStruct *nabla){
  nprintf(nabla, NULL, "fatal");
}
static void ccHookAddCallNames(struct nablaMainStruct *nabla,nablaJob *fct,astNode *n){
  nablaJob *foundJob;
  char *callName=n->next->children->children->token;
  nprintf(nabla, "/*function_got_call*/", "/*%s*/",callName);
  fct->parse.function_call_name=NULL;
  if ((foundJob=nablaJobFind(fct->entity->jobs,callName))!=NULL){
    if (foundJob->is_a_function!=true){
      nprintf(nabla, "/*isNablaJob*/", NULL);
      fct->parse.function_call_name=strdup(callName);
    }else{
      nprintf(nabla, "/*isNablaFunction*/", NULL);
    }
  }else{
    nprintf(nabla, "/*has not been found*/", NULL);
  }
}
static void ccHookAddArguments(struct nablaMainStruct *nabla,nablaJob *fct){
  if (fct->parse.function_call_name!=NULL){
    //nprintf(nabla, "/*ShouldDumpParamsInCc*/", "/*Arg*/");
    int numParams=1;
    nablaJob *called=nablaJobFind(fct->entity->jobs,fct->parse.function_call_name);
    ccAddExtraArguments(nabla, called, &numParams);
    if (called->nblParamsNode != NULL)
      ccDumpNablaArgumentList(nabla,called->nblParamsNode,&numParams);
  }
}

static void ccHookTurnTokenToOption(struct nablaMainStruct *nabla,nablaOption *opt){
  nprintf(nabla, "/*tt2o cc*/", "%s", opt->name);
}



/*****************************************************************************
 * Cc libraries
 *****************************************************************************/
void ccHookLibraries(astNode * n, nablaEntity *entity){
  fprintf(entity->src, "\n/*lib %s*/",n->children->token);
}


// ****************************************************************************
// * ccInclude
// ****************************************************************************
void ccInclude(nablaMain *nabla){
  fprintf(nabla->entity->src,"#include \"%s.h\"\n", nabla->entity->name);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void ccHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __CC_%s_H__\n#define __CC_%s_H__",
          nabla->entity->name,
          nabla->entity->name);
}

/***************************************************************************** 
 * 
 *****************************************************************************/
static void ccHeaderIncludes(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,"\n\n\n\
// *****************************************************************************\n\
// * Cc includes\n\
// *****************************************************************************\n\
%s // from nabla->simd->includes\n\
#include <sys/time.h>\n\
#include <stdlib.h>\n\
#include <stdio.h>\n\
#include <string.h>\n\
#include <vector>\n\
#include <math.h>\n\
#include <assert.h>\n\
#include <stdarg.h>\n\
#include <iostream>\n\
%s // from nabla->parallel->includes()",
          nabla->simd->includes(),
          nabla->parallel->includes());
}


// ****************************************************************************
// * ccHeader for Std, Avx or Mic
// ****************************************************************************
extern char knStdReal_h[];
extern char knStdReal3_h[];
extern char knStdInteger_h[];
extern char knStdGather_h[];
extern char knStdScatter_h[];
extern char knStdOStream_h[];
extern char knStdTernary_h[];

static char *dumpExternalFile(char *file){
  return file+NABLA_LICENSE_HEADER;
}
static void ccHeaderSimd(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knStdInteger_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(knStdReal_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(knStdReal3_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(knStdTernary_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(knStdGather_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(knStdScatter_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(knStdOStream_h));
}


// ****************************************************************************
// * ccHeader for Dbg
// ****************************************************************************
extern char knDbg_h[];
static void ccHeaderDbg(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knDbg_h));
}


// ****************************************************************************
// * ccHeader for Maths
// ****************************************************************************
extern char knMth_h[];
static void ccHeaderMth(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knMth_h));
}


/***************************************************************************** 
 * 
 *****************************************************************************/
static void ccHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __CC_%s_H__\n",nabla->entity->name);
}


// ****************************************************************************
// * ccHookPrimaryExpressionToReturn
// ****************************************************************************
static bool ccHookPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
  const char* var=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  dbg("\n\t[ccHookPrimaryExpressionToReturn] ?");
  if (var!=NULL && strcmp(n->children->token,var)==0){
    dbg("\n\t[ccHookPrimaryExpressionToReturn] primaryExpression hits returned argument");
    nprintf(nabla, NULL, "%s_per_thread[tid]",var);
    return true;
  }else{
    dbg("\n\t[ccHookPrimaryExpressionToReturn] ELSE");
    //nprintf(nabla, NULL, "%s",n->children->token);
  }
  return false;
}


// ****************************************************************************
// * ccHookReturnFromArgument
// ****************************************************************************
static void ccHookReturnFromArgument(nablaMain *nabla, nablaJob *job){
  const char *rtnVariable=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nprintf(nabla, NULL, "\
\n\tint threads = omp_get_max_threads();\
\n\tReal %s_per_thread[threads];", rtnVariable);
}



/*****************************************************************************
 * cc
 *****************************************************************************/
NABLA_STATUS cc(nablaMain *nabla,
                astNode *root,
                const char *nabla_entity_name){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char hdrFileName[NABLA_MAX_FILE_NAME];
  // Définition des hooks pour l'AVX ou le MIC
  nablaBackendSimdHooks nablaCcSimdStdHooks={
    ccStdBits,
    ccStdGather,
    ccStdScatter,
    ccStdTypedef,
    ccStdDefines,
    ccStdForwards,
    ccStdPrevCell,
    ccStdNextCell,
    ccStdIncludes
  };
  nabla->simd=&nablaCcSimdStdHooks;
    
  // Définition des hooks pour Cilk+ *ou pas*
  nablaBackendParallelHooks ccCilkHooks={
    ccHookParallelCilkSync,
    ccHookParallelCilkSpawn,
    ccHookParallelCilkLoop,
    ccHookParallelCilkIncludes
  };
  nablaBackendParallelHooks ccOpenMPHooks={
    ccHookParallelOpenMPSync,
    ccHookParallelOpenMPSpawn,
    ccHookParallelOpenMPLoop,
    ccHookParallelOpenMPIncludes
  };
  nablaBackendParallelHooks ccVoidHooks={
    ccHookParallelVoidSync,
    ccHookParallelVoidSpawn,
    ccHookParallelVoidLoop,
    ccHookParallelVoidIncludes
  };
  nabla->parallel=&ccVoidHooks;
  if ((nabla->colors&BACKEND_COLOR_CILK)==BACKEND_COLOR_CILK)
    nabla->parallel=&ccCilkHooks;
  if ((nabla->colors&BACKEND_COLOR_OpenMP)==BACKEND_COLOR_OpenMP)
    nabla->parallel=&ccOpenMPHooks;

  
  nablaBackendPragmaHooks ccPragmaICCHooks ={
    ccHookPragmaIccIvdep,
    ccHookPragmaIccAlign
  };
  nablaBackendPragmaHooks ccPragmaGCCHooks={
    ccHookPragmaGccIvdep,
    ccHookPragmaGccAlign
  };
  // Par defaut, on met GCC
  nabla->pragma=&ccPragmaGCCHooks;
  if ((nabla->colors&BACKEND_COLOR_ICC)==BACKEND_COLOR_ICC)
    nabla->pragma=&ccPragmaICCHooks;
  
  static nablaBackendHooks ccBackendHooks={
    // Jobs stuff
    ccHookPrefixEnumerate,
    ccHookDumpEnumerateXYZ,
    ccHookDumpEnumerate,
    ccHookPostfixEnumerate,
    ccHookItem,
    ccHookSwitchToken,
    ccHookTurnTokenToVariable,
    ccHookSystem,
    ccHookAddExtraParameters,
    ccHookDumpNablaParameterList,
    ccHookTurnBracketsToParentheses,
    ccHookJobDiffractStatement,
    // Other hooks
    ccHookFunctionName,
    ccHookFunction,
    ccHookJob,
    ccHookReduction,
    ccHookIteration,
    ccHookExit,
    ccHookTime,
    ccHookFatal,
    ccHookAddCallNames,
    ccHookAddArguments,
    ccHookTurnTokenToOption,
    ccHookEntryPointPrefix,
    ccHookDfsForCalls,
    ccHookPrimaryExpressionToReturn,
    ccHookReturnFromArgument
  };
  nabla->hook=&ccBackendHooks;

  // Rajout de la variable globale 'iteration'
  nablaVariable *iteration = nablaVariableNew(nabla);
  nablaVariableAdd(nabla, iteration);
  iteration->axl_it=false;
  iteration->item=strdup("global");
  iteration->type=strdup("integer");
  iteration->name=strdup("iteration");
 
  // Ouverture du fichier source du entity
  sprintf(srcFileName, "%s.cc", nabla->name);
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  // Ouverture du fichier header du entity
  sprintf(hdrFileName, "%s.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  
  // Dump dans le HEADER des includes, typedefs, defines, debug, maths & errors stuff
  ccHeaderPrefix(nabla);
  ccHeaderIncludes(nabla);
  nablaDefines(nabla,nabla->simd->defines);
  nablaTypedefs(nabla,nabla->simd->typedefs);
  nablaForwards(nabla,nabla->simd->forwards);

  // On inclue les fichiers kn'SIMD'
  ccHeaderSimd(nabla);
  ccHeaderDbg(nabla);
  ccHeaderMth(nabla);
  ccMesh(nabla);
  ccDefineEnumerates(nabla);

  // Dump dans le fichier SOURCE
  ccInclude(nabla);
  
  // Parse du code préprocessé et lance les hooks associés
  nablaMiddlendParseAndHook(root,nabla);
  ccMainVarInitKernel(nabla);

  // Partie PREFIX
  ccMainPrefix(nabla);
  ccVariablesPrefix(nabla);
  ccMainMeshPrefix(nabla);
  
  // Partie Pré Init
  ccMainPreInit(nabla);
  ccMainVarInitCall(nabla);
      
  // Dump des entry points dans le main
  ccMain(nabla);

  // Partie Post Init
  ccMainPostInit(nabla);
  
  // Partie POSTFIX
  ccHeaderPostfix(nabla); 
  ccMainMeshPostfix(nabla);
  ccVariablesPostfix(nabla);
  ccMainPostfix(nabla);
  return NABLA_OK;
}
