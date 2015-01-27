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



/*****************************************************************************
 * Fonction prefix à l'ENUMERATE_*
 *****************************************************************************/
char* ccHookPrefixEnumerate(nablaJob *job){
  char prefix[NABLA_MAX_FILE_NAME];
  //const nablaMain* nabla=job->entity->main;
              
  if (job->parse.returnFromArgument){
    const char *var=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
    if (sprintf(prefix,"dbgFuncIn();\n\tfor (int i=0; i<threads;i+=1) %s_per_thread[i] = %s;",var,var)<=0){
      error(!0,0,"Error in ccHookPrefixEnumerate!");
    }
  }else{
    if (sprintf(prefix,"dbgFuncIn();")<=0)
      error(!0,0,"Error in ccHookPrefixEnumerate!");
  }
      
  //const register char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  //nprintf(job->entity->main, "\n\t/*ccHookPrefixEnumerate*/", "/*itm=%c*/", itm);
  return strdup(prefix);
}



// ****************************************************************************
// *
// ****************************************************************************
/*static char * ccReturnVariableNameForOpenMP(nablaJob *job){
  char str[NABLA_MAX_FILE_NAME];
  if (job->is_a_function) return "";
  if (sprintf(str,"%s_per_thread",
              dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator")))<=0)
    error(!0,0,"Could not patch format!");
  return strdup(str);
  }*/
static char * ccReturnVariableNameForOpenMPWitoutPerThread(nablaJob *job){
  char str[NABLA_MAX_FILE_NAME];
  if (job->is_a_function) return "";
  if (sprintf(str,"%s",
              dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator")))<=0)
    error(!0,0,"Could not patch format!");
  return strdup(str);
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_*
 *****************************************************************************/
static char* ccSelectEnumerate(nablaJob *job){
  const char *grp=job->scope;   // OWN||ALL
  const char *rgn=job->region;  // INNER, OUTER
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  //if (job->xyz!=NULL) return ccHookDumpEnumerateXYZ(job);
  if (itm=='\0') return "\n";// function ccHookDumpEnumerate\n";
  if (itm=='c' && grp==NULL && rgn==NULL)     return "FOR_EACH_CELL%s%s(c";
  if (itm=='c' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER\n\tFOR_EACH_CELL%s%s(c";
  if (itm=='c' && grp==NULL && rgn[0]=='o')   return "//#warning Should be OUTER\n\tFOR_EACH_CELL%s%s(c";
  if (itm=='c' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_CELL%s%s(c";
  if (itm=='n' && grp==NULL && rgn==NULL)     return "FOR_EACH_NODE%s%s(n";
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER\n\tFOR_EACH_NODE%s%s(n";
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return "//#warning Should be OUTER\n\tFOR_EACH_NODE%s%s(n";
  if (itm=='n' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_NODE%s%s(n";
  if (itm=='n' && grp[0]=='a' && rgn==NULL)   return "#warning Should be ALL\n\tFOR_EACH_NODE%s%s(n";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER OWN\n\tFOR_EACH_NODE%s%s(n";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='o') return "#warning Should be OUTER OWN\n\tFOR_EACH_NODE%s%s(n";
  if (itm=='f' && grp==NULL && rgn==NULL)     return "FOR_EACH_FACE%s%s(f";
  if (itm=='f' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_FACE%s%s(f";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='o') return "#warning Should be OUTER OWN\n\tFOR_EACH_FACE%s%s(f";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER OWN\n\tFOR_EACH_FACE%s%s(f";
  if (itm=='e' && grp==NULL && rgn==NULL)     return "FOR_EACH_ENV%s%s(e";
  if (itm=='m' && grp==NULL && rgn==NULL)     return "FOR_EACH_MAT%s%s(m";
  
  error(!0,0,"Could not distinguish ENUMERATE!");
  return NULL;
}


char* ccHookDumpEnumerate(nablaJob *job){
  const char *forall=strdup(ccSelectEnumerate(job));
  const char *warping=job->parse.selection_statement_in_compound_statement?"":"_WARP";
  char format[NABLA_MAX_FILE_NAME];
  char str[NABLA_MAX_FILE_NAME];
  dbg("\n\t[ccHookDumpEnumerate] Preparing:");
  dbg("\n\t[ccHookDumpEnumerate]\t\tforall=%s",forall);
  dbg("\n\t[ccHookDumpEnumerate]\t\twarping=%s",warping);

  // On prépare le format grace à la partie du forall,
  // on rajoute l'extension suivant si on a une returnVariable
  if (job->parse.returnFromArgument){
    const char *ompCcLocal=job->parse.returnFromArgument?"_SHARED":"";
    //const char *ompCcReturnVariable=ccReturnVariableNameForOpenMP(job);
    const char *ompCcReturnVariableWitoutPerThread=ccReturnVariableNameForOpenMPWitoutPerThread(job);
    //const char *ompCcLocalVariableComa=",";//job->parse.returnFromArgument?",":"";
    //const char *ompCcLocalVariableName=job->parse.returnFromArgument?ompCcReturnVariable:"";
    if (sprintf(format,"%s%%s%%s)",forall)<=0) error(!0,0,"Could not patch format!");
    if (sprintf(str,format,    // FOR_EACH_XXX%s%s(
                warping,       // _WARP or not
                ompCcLocal, // _SHARED or not
                ",",           //ompCcLocalVariableComa,
                ompCcReturnVariableWitoutPerThread)<=0) error(!0,0,"Could not patch warping within ENUMERATE!");
  }else{
    dbg("\n\t[ccHookDumpEnumerate] No returnFromArgument");
    if (sprintf(format,"%s%s",  // FOR_EACH_XXX%s%s(x + ')'
                forall,
                job->is_a_function?"":")")<=0)
      error(!0,0,"Could not patch format!");
    dbg("\n[ccHookDumpEnumerate] format=%s",format);
    if (sprintf(str,format,
                warping,
                "",
                "")<=0)
      error(!0,0,"Could not patch warping within ENUMERATE!");
  }
  return strdup(str);
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_* avec XYZ
 *****************************************************************************/
char* ccHookDumpEnumerateXYZ(nablaJob *job){
  //char *xyz=job->xyz;// Direction
  //nprintf(job->entity->main, "\n\t/*ccHookDumpEnumerateXYZ*/", "/*xyz=%s, drctn=%s*/", xyz, job->drctn);
  return "// ccHookDumpEnumerateXYZ has xyz drctn";
}

