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
// * Fonction produisant l'ENUMERATE_*
// ****************************************************************************
static char* xHookSelectEnumerate(nablaJob *job){
  const char *grp=job->scope;   // OWN||ALL
  const char *rgn=job->region;  // INNER, OUTER
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  dbg("\n\t\t[lambdaHookSelectEnumerate] function?");
  if (itm=='\0') return "\n";
  dbg("\n\t\t[lambdaHookSelectEnumerate] cell?");
  if (itm=='p' && grp==NULL && rgn==NULL)     return "FOR_EACH_PARTICLE%s(p)";
  if (itm=='c' && grp==NULL && rgn==NULL)     return "FOR_EACH_CELL%s(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER cells\n\tFOR_EACH_CELL%s(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='o')   return "FOR_EACH_OUTER_CELL%s(c)";
  if (itm=='c' && grp[0]=='o' && rgn==NULL)   return "FOR_EACH_CELL%s(c)";
  dbg("\n\t\t[lambdaHookSelectEnumerate] node?");
  if (itm=='n' && grp==NULL && rgn==NULL)     return "FOR_EACH_NODE%s(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return "#warning Should be INNER nodes\n\tFOR_EACH_NODE%s(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return "#warning Should be OUTER nodes\n\tFOR_EACH_NODE%s(n)";
  if (itm=='n' && grp[0]=='o' && rgn==NULL)   return "FOR_EACH_NODE%s(n)";
  if (itm=='n' && grp[0]=='a' && rgn==NULL)   return "FOR_EACH_NODE%s(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='i') return "#warning Should be INNER nodes\n\tFOR_EACH_NODE%s(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='o') return "#warning Should be OUTER nodes\n\tFOR_EACH_NODE%s(n)";
  dbg("\n\t\t[lambdaHookSelectEnumerate] face? (itm=%c, grp='%s', rgn='%s')", itm, grp, rgn);
  if (itm=='f' && grp==NULL && rgn==NULL)     return "FOR_EACH_FACE%s(f)";
  if (itm=='f' && grp==NULL && rgn[0]=='i')   return "FOR_EACH_INNER_FACE%s(f)";
  if (itm=='f' && grp==NULL && rgn[0]=='o')   return "FOR_EACH_OUTER_FACE%s(f)";
  // ! Tester grp==NULL avant ces prochains:
  if (itm=='f' && grp[0]=='o' && rgn==NULL)   return "FOR_EACH_FACE%s(f)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='o') return "FOR_EACH_OWN_OUTER_FACE%s(f)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='i') return "FOR_EACH_OWN_INNER_FACE%s(f)";
  dbg("\n\t\t[lambdaHookSelectEnumerate] Could not distinguish ENUMERATE!");
  nablaError("Could not distinguish ENUMERATE!");
  return NULL;
}

// ****************************************************************************
// * Fonction produisant l'ENUMERATE_*
// ****************************************************************************
char* xHookForAllDump(nablaJob *job){
  char str[NABLA_MAX_FILE_NAME];
  char format[NABLA_MAX_FILE_NAME];
  const char *forall=xHookSelectEnumerate(job);
  const char *warping=
    (job->parse.selection_statement_in_compound_statement==true &&
     job->entity->main->call!=NULL)?"":"_WARP";
  // On prépare le format grace à la partie du forall
  if (sprintf(format,"%s",forall)<=0)
    nablaError("Could not patch format!");
  dbg("\n\t[lambdaHookDumpEnumerate] format='%s'",format);
  if (sprintf(str,format,warping)<=0)
    nablaError("Could not patch warping within ENUMERATE!");
  return sdup(str);
}

// **************************************************************************** 
// * Traitement des tokens NABLA ITEMS
// ****************************************************************************
char* xHookForAllItem(nablaJob *j,
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
  if (job=='n' && enum_enum=='c'  && itm=='c') return "/*chi-ncc*/xs_node_cell";
  if (job=='n' && enum_enum=='\0' && itm=='c') return "/*chi-n0c*/xs_node_cell";
  if (job=='n' && enum_enum=='\0' && itm=='n') return "/*chi-n0n*/n";
  if (job=='f' && enum_enum=='\0' && itm=='f') return "/*chi-f0f*/f";
  if (job=='f' && enum_enum=='\0' && itm=='n') return "/*chi-f0n*/xs_face_node";
  if (job=='f' && enum_enum=='\0' && itm=='c' &&
      j->parse.alephKeepExpression==false) return "/*chi-f0c*/xs_face_cell";
  if (job=='f' && enum_enum=='\0' && itm=='c' &&
      j->parse.alephKeepExpression==true)  return "/*chi-f0c*/xs_face_cell";
  nablaError("Could not switch in hookItem!");
  return NULL;
}

// ****************************************************************************
// * Fonction postfix à l'ENUMERATE_*
// ****************************************************************************
char* xHookForAllPostfix(nablaJob *job){
  dbg("\n\t[xHookForAllPostfix]");
  return xCallFilterGather(NULL,job);
}
