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
 * Fonction prefix à l'ENUMERATE_*
 *****************************************************************************/
char* nCudaHookPrefixEnumerate(nablaJob *job){
  const register char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  //if (j->xyz==NULL) return "// void ENUMERATE prefix";
  //nprintf(job->entity->main, "\n\t/*cudaHookPrefixEnumerate*/", "/*itm=%c*/", itm);
  if (itm=='c'  && strcmp(job->return_type,"void")==0) return "CUDA_INI_CELL_THREAD(tcid);";
  if (itm=='c'  && strcmp(job->return_type,"Real")==0) return "CUDA_INI_CELL_THREAD_RETURN_REAL(tcid);";
  if (itm=='n') return "CUDA_INI_NODE_THREAD(tnid);";
  if (itm=='\0' && job->is_an_entry_point
      && job->called_variables!=NULL) return "CUDA_LAUNCHER_FUNCTION_THREAD(tid);";
  if (itm=='\0' && job->is_an_entry_point
      && job->called_variables==NULL) return "CUDA_INI_FUNCTION_THREAD(tid);";
  if (itm=='\0' && job->is_an_entry_point) return "CUDA_INI_FUNCTION_THREAD(tid);";
  if (itm=='\0' && !job->is_an_entry_point) return "/*std function*/";
  nablaError("Could not distinguish PREFIX Enumerate!");
  return NULL;
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_* avec XYZ
 *****************************************************************************/
char* nCudaHookDumpEnumerateXYZ(nablaJob *job){
  char *xyz=job->xyz;// Direction
  nprintf(job->entity->main, NULL, "/*xyz=%s, drctn=%s*/", xyz, job->direction);
  return "// cudaHookDumpEnumerateXYZ has xyz direction";
}


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_*
 *****************************************************************************/
char* nCudaHookDumpEnumerate(nablaJob *job){
  char *grp=job->scope;   // OWN||ALL
  char *rgn=job->region;  // INNER, OUTER
  char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  //if (job->xyz!=NULL) return cudaHookDumpEnumerateXYZ(job);
  if (itm=='\0') return "// function cudaHookDumpEnumerate\n";
  if (itm=='c' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_CELL_WARP(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER\n\tFOR_EACH_CELL_WARP(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='o')   return "//#warning Should be OUTER\n\tFOR_EACH_CELL_WARP(c)";
  if (itm=='c' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_CELL_WARP(c)";
  if (itm=='n' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return "//#warning Should be OUTER\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='a' && rgn==NULL)   return "#warning Should be ALL\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER OWN\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='o') return "#warning Should be OUTER OWN\n\tFOR_EACH_NODE_WARP(n)";
  if (itm=='f' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_FACE_WARP(f)";
  if (itm=='f' && grp[0]=='o' && rgn==NULL)   return "#warning Should be OWN\n\tFOR_EACH_FACE_WARP(f)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='o') return "//#warning Should be OUTER OWN\n\tFOR_EACH_FACE_WARP(f)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER OWN\n\tFOR_EACH_FACE_WARP(f)";
  if (itm=='e' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_ENV_WARP(e)";
  if (itm=='m' && grp==NULL && rgn==NULL)     return "";//FOR_EACH_MAT_WARP(m)";
  nablaError("Could not distinguish ENUMERATE!");
  return NULL;
}


/*****************************************************************************
 * Fonction postfix à l'ENUMERATE_*
 *****************************************************************************/
char* nCudaHookPostfixEnumerate(nablaJob *job){
  if (job->item[0]=='\0') return "// functioncudaHookPostfixEnumerate\n";
  if (job->xyz==NULL) return "";//// void ENUMERATE postfix\n\t";
  if (job->xyz!=NULL) return "// Postfix ENUMERATE with xyz direction\n\
\t\t//int prevCell=tcid-1;\n\
\t\t//int nextCell=tcid+1;\n";
  nablaError("Could not switch in cudaHookPostfixEnumerate!");
  return NULL;
}
