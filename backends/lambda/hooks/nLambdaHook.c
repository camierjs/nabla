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
#include "frontend/nablaAst.h"


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
// * Defines
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
// * Typedefs
// ****************************************************************************
nablaTypedef ccTypedef[]={
  {"struct real3","Real3"},
  {NULL,NULL}
};


// ****************************************************************************
// * INCLUDES
// ****************************************************************************
char *ccHookBits(void){return "64";}
char* ccHookIncludes(void){return "";}


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
char *ccHookParallelVoidLoop(struct nablaMainStruct *n){
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
  nablaJob *fct=nMiddleJobNew(nabla->entity);
  nMiddleJobAdd(nabla->entity, fct);
  nMiddleFunctionFill(nabla,fct,n,NULL);
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


/*****************************************************************************
 * Fonction postfix à l'ENUMERATE_*
 *****************************************************************************/
char* ccHookPostfixEnumerate(nablaJob *job){
  if (job->is_a_function) return "";
  if (job->item[0]=='\0') return "// job ccHookPostfixEnumerate\n";
  if (job->xyz==NULL) return ccFilterGather(job);
  if (job->xyz!=NULL) return "// Postfix ENUMERATE with xyz direction\n\
\t\tconst int __attribute__((unused)) max_x = NABLA_NB_CELLS_X_AXIS;\n\
\t\tconst int __attribute__((unused)) max_y = NABLA_NB_CELLS_Y_AXIS;\n\
\t\tconst int __attribute__((unused)) max_z = NABLA_NB_CELLS_Z_AXIS;\n\
\t\tconst int delta_x = NABLA_NB_CELLS_Y_AXIS*NABLA_NB_CELLS_Z_AXIS;\n\
\t\tconst int delta_y = 1;\n\
\t\tconst int delta_z = NABLA_NB_CELLS_Y_AXIS;\n\
\t\tconst int delta = (direction==MD_DirX)?delta_x:(direction==MD_DirY)?delta_y:delta_z;\n\
\t\tconst int __attribute__((unused)) prevCell=delta;\n\
\t\tconst int __attribute__((unused)) nextCell=delta;\n";
  nablaError("Could not switch in ccHookPostfixEnumerate!");
  return NULL;
}


/***************************************************************************** 
 * Traitement des tokens NABLA ITEMS
 *****************************************************************************/
char* ccHookItem(nablaJob *j, const char job, const char itm, char enum_enum){
  if (job=='c' && enum_enum=='\0' && itm=='c') return "/*chi-c0c*/c";
  if (job=='c' && enum_enum=='\0' && itm=='n') return "/*chi-c0n*/c->";
  if (job=='c' && enum_enum=='f'  && itm=='n') return "/*chi-cfn*/f->";
  if (job=='c' && enum_enum=='f'  && itm=='c') return "/*chi-cfc*/f->";
  if (job=='n' && enum_enum=='f'  && itm=='n') return "/*chi-nfn*/f->";
  if (job=='n' && enum_enum=='f'  && itm=='c') return "/*chi-nfc*/f->";
  if (job=='n' && enum_enum=='\0' && itm=='n') return "/*chi-n0n*/n";
  if (job=='f' && enum_enum=='\0' && itm=='f') return "/*chi-f0f*/f";
  if (job=='f' && enum_enum=='\0' && itm=='n') return "/*chi-f0n*/f->";
  if (job=='f' && enum_enum=='\0' && itm=='c') return "/*chi-f0c*/f->";
  nablaError("Could not switch in ccHookItem!");
  return NULL;
}


// ****************************************************************************
// * Dump des variables appelées
// ****************************************************************************
void ccHookDfsForCalls(struct nablaMainStruct *nabla,
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
  nMiddleDumpParameterTypeList(nabla->entity->hdr, nParams);
  hprintf(nabla, NULL, ");");
}


// ****************************************************************************
// * Dump du préfix des points d'entrées: inline ou pas
// ****************************************************************************
char* ccHookEntryPointPrefix(struct nablaMainStruct *nabla, nablaJob *entry_point){
  //return "";
  return "static inline";
}

void ccHookIteration(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*ITERATION*/", "cc_iteration()");
}
void ccHookExit(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*EXIT*/", "exit(0.0)");
}
void ccHookTime(struct nablaMainStruct *nabla){
  nprintf(nabla, "/*TIME*/", "global_time");
}
void ccHookFatal(struct nablaMainStruct *nabla){
  nprintf(nabla, NULL, "fatal");
}
void ccHookAddCallNames(struct nablaMainStruct *nabla,nablaJob *fct,astNode *n){
  nablaJob *foundJob;
  char *callName=n->next->children->children->token;
  nprintf(nabla, "/*function_got_call*/", "/*%s*/",callName);
  fct->parse.function_call_name=NULL;
  if ((foundJob=nMiddleJobFind(fct->entity->jobs,callName))!=NULL){
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
void ccHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,
          "#ifndef __CC_%s_H__\n#define __CC_%s_H__",
          nabla->entity->name,
          nabla->entity->name);
}


/***************************************************************************** 
 * 
 *****************************************************************************/
void ccHeaderIncludes(nablaMain *nabla){
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

char *dumpExternalFile(char *file){
  return file+NABLA_LICENSE_HEADER;
}
void ccHeaderSimd(nablaMain *nabla){
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
void ccHeaderDbg(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knDbg_h));
}


// ****************************************************************************
// * ccHeader for Maths
// ****************************************************************************
extern char knMth_h[];
void ccHeaderMth(nablaMain *nabla){
  assert(nabla->entity->name!=NULL);
  fprintf(nabla->entity->hdr,dumpExternalFile(knMth_h));
}


/***************************************************************************** 
 * 
 *****************************************************************************/
void ccHeaderPostfix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n#endif // __CC_%s_H__\n",nabla->entity->name);
}


// ****************************************************************************
// * ccHookPrimaryExpressionToReturn
// ****************************************************************************
bool ccHookPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
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



/*****************************************************************************
 * Génération d'un kernel associé à un support
 *****************************************************************************/
void ccHookJob(nablaMain *nabla, astNode *n){
  nablaJob *job = nMiddleJobNew(nabla->entity);
  nMiddleJobAdd(nabla->entity, job);
  nMiddleJobFill(nabla,job,n,NULL);
  
  // On teste *ou pas* que le job retourne bien 'void' dans le cas de CC
  //if ((strcmp(job->rtntp,"void")!=0) && (job->is_an_entry_point==true))
  //  exit(NABLA_ERROR|fprintf(stderr, "\n[ccHookJob] Error with return type which is not void\n"));
}


