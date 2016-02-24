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
#include "backends/kokkos/hook.h"


/*****************************************************************************
 * Fonction produisant l'ENUMERATE_*
 *****************************************************************************/
static char* hookSelectEnumerate(nablaJob *job){
  const char *grp=job->scope;   // OWN||ALL
  const char *rgn=job->region;  // INNER, OUTER
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  dbg("\n\t\t[hookSelectEnumerate] function?");
  //if (job->xyz!=NULL) return hookDumpEnumerateXYZ(job);
  if (itm=='\0') return "\n";// function hookDumpEnumerate\n";
  dbg("\n\t\t[hookSelectEnumerate] cell?");
  if (itm=='p' && grp==NULL && rgn==NULL)     return "FOR_EACH_PARTICLE(p";
  if (itm=='c' && grp==NULL && rgn==NULL)     return "FOR_EACH_CELL(c";
  if (itm=='c' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER cells\n\tFOR_EACH_CELL(c";
  if (itm=='c' && grp==NULL && rgn[0]=='o')   return "#warning Should be OUTER cells\n\tFOR_EACH_CELL(c";
  if (itm=='c' && grp[0]=='o' && rgn==NULL)   return "\n\tFOR_EACH_CELL(c";
  dbg("\n\t\t[hookSelectEnumerate] node?");
  if (itm=='n' && grp==NULL && rgn==NULL)     return "FOR_EACH_NODE(n";
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER nodes\n\tFOR_EACH_NODE(n";
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return "#warning Should be OUTER nodes\n\tFOR_EACH_NODE(n";
  if (itm=='n' && grp[0]=='o' && rgn==NULL)   return "\n\tFOR_EACH_NODE(n";
  if (itm=='n' && grp[0]=='a' && rgn==NULL)   return "\n\tFOR_EACH_NODE(n";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER nodes\n\tFOR_EACH_NODE(n";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='o') return "#warning Should be OUTER nodes\n\tFOR_EACH_NODE(n";
  dbg("\n\t\t[hookSelectEnumerate] face? (itm=%c, grp='%s', rgn='%s')", itm, grp, rgn);
  if (itm=='f' && grp==NULL && rgn==NULL)     return "FOR_EACH_FACE(f";
  if (itm=='f' && grp==NULL && rgn[0]=='i')   return "\n\tFOR_EACH_INNER_FACE(f";
  if (itm=='f' && grp==NULL && rgn[0]=='o')   return "\n\tFOR_EACH_OUTER_FACE(f";
  // ! Tester grp==NULL avant ces prochains:
  if (itm=='f' && grp[0]=='o' && rgn==NULL)   return "\n\tFOR_EACH_FACE(f";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='o') return "\n\tFOR_EACH_OWN_OUTER_FACE(f";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='i') return "\n\tFOR_EACH_OWN_INNER_FACE(f";
  dbg("\n\t\t[hookSelectEnumerate] env?");
  if (itm=='e' && grp==NULL && rgn==NULL)     return "FOR_EACH_ENV(e";
  dbg("\n\t\t[hookSelectEnumerate] mat?");
  if (itm=='m' && grp==NULL && rgn==NULL)     return "FOR_EACH_MAT(m";
  dbg("\n\t\t[hookSelectEnumerate] Could not distinguish ENUMERATE!");
  nablaError("Could not distinguish ENUMERATE!");
  return NULL;
}


// ****************************************************************************
// * Fonction produisant l'ENUMERATE_*
// ****************************************************************************
char* hookForAllDump(nablaJob *job){
  dbg("\n\t[hookDumpEnumerate]");
  const char *forall=strdup(hookSelectEnumerate(job));
  char str[NABLA_MAX_FILE_NAME];
  dbg("\n\t[hookDumpEnumerate] No returnFromArgument");
  if (sprintf(str,"%s%s",forall,job->is_a_function?"":")")<=0)
    nablaError("Could not patch format!");  
    return strdup(str);
}


// **************************************************************************** 
// * Traitement des tokens NABLA ITEMS
// ****************************************************************************
char* hookForAllItem(nablaJob *j,
                     const char job,
                     const char itm,
                     char enum_enum){
  nprintf(j->entity->main, "/*hookItem*/", "/*hookItem*/");
  if (job=='c' && enum_enum=='\0' && itm=='c') return "/*chi-c0c*/c";
  if (job=='c' && enum_enum=='\0' && itm=='n') return "/*chi-c0n*/c->";
  if (job=='c' && enum_enum=='f'  && itm=='n') return "/*chi-cfn*/f->";
  if (job=='c' && enum_enum=='f'  && itm=='c') return "/*chi-cfc*/f->";
  if (job=='n' && enum_enum=='f'  && itm=='n') return "/*chi-nfn*/f->";
  if (job=='n' && enum_enum=='f'  && itm=='c') return "/*chi-nfc*/f->";
  if (job=='n' && enum_enum=='c'  && itm=='c') return "/*chi-ncc*/xs_node_";
  if (job=='n' && enum_enum=='\0' && itm=='c') return "/*chi-n0c*/xs_node_";
  if (job=='n' && enum_enum=='\0' && itm=='n') return "/*chi-n0n*/n";
  if (job=='f' && enum_enum=='\0' && itm=='f') return "/*chi-f0f*/f";
  if (job=='f' && enum_enum=='\0' && itm=='n') return "/*chi-f0n*/xs_face_";
  if (job=='f' && enum_enum=='\0' && itm=='c' &&
      j->parse.alephKeepExpression==false) return "/*chi-f0c*/xs_face_";
  if (job=='f' && enum_enum=='\0' && itm=='c' &&
      j->parse.alephKeepExpression==true)  return "/*chi-f0c*/xs_face_";
  nablaError("Could not switch in hookItem!");
  return NULL;
}


// ****************************************************************************
// * Fonction postfix à l'ENUMERATE_*
// ****************************************************************************
char* hookForAllPostfix(nablaJob *job){
  if (job->is_a_function) return "";
  if (job->item[0]=='\0') return "// job hookPostfixEnumerate\n";
  // Si un 'scope' a été trouvé, et qu'il corepond à un 'own',
  // On le traite pour l'instant en ne faisant rien: on est en  multi-thread
  if (job->scope&&!strcmp(job->scope,"own")) return "";// Should test OWN here!\n";
  if (job->region && !strcmp(job->region,"inner")){
    //if (job->item[0]=='c') return "// Should test INNER cells here!\n";
    if (job->item[0]=='f') return "// Should test INNER faces here!\n";
    //if (job->item[0]=='n') return "// Should test INNER nodes here!\n";
  }
  if (job->region && !strcmp(job->region,"outer")){
    //if (job->item[0]=='c') return "// Should test INNER cells here!\n";
    if (job->item[0]=='f') return "// Should test OUTER faces here!\n";
    //if (job->item[0]=='n') return "// Should test INNER nodes here!\n";
  }
  if (job->xyz==NULL) return filterGather(NULL,job,GATHER_SCATTER_DECL);
  nablaError("Could not switch in hookPostfixEnumerate!");
  return NULL;
}