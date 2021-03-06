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
char* kHookForAllDump(nablaJob *job){
  const char *grp=job->scope;   // OWN||ALL
  const char *rgn=job->region;  // INNER, OUTER
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  dbg("\n\t\t[kokkosHookSelectEnumerate] function?");
  if (itm=='\0') return "\n";
  dbg("\n\t\t[kokkosHookSelectEnumerate] cell?");
  if (itm=='p' && grp==NULL && rgn==NULL)     return "FOR_EACH_PARTICLE(p)";
  if (itm=='c' && grp==NULL && rgn==NULL)     return "FOR_EACH_CELL(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='i')   return "\n#warning Should be INNER cells\n\tFOR_EACH_CELL(c)";
  if (itm=='c' && grp==NULL && rgn[0]=='o')   return "FOR_EACH_OUTER_CELL(c)";
  if (itm=='c' && grp[0]=='o' && rgn==NULL)   return "FOR_EACH_CELL(c)";
  dbg("\n\t\t[kokkosHookSelectEnumerate] node?");
  if (itm=='n' && grp==NULL && rgn==NULL)     return "FOR_EACH_NODE(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return "\n#warning Should be INNER nodes\n\tFOR_EACH_NODE(n)";
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return "\n#warning Should be OUTER nodes\n\tFOR_EACH_NODE(n)";
  if (itm=='n' && grp[0]=='o' && rgn==NULL)   return "FOR_EACH_NODE(n)";
  if (itm=='n' && grp[0]=='a' && rgn==NULL)   return "FOR_EACH_NODE(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='i') return "\n#warning Should be INNER nodes\n\tFOR_EACH_NODE(n)";
  if (itm=='n' && grp[0]=='o' && rgn[0]=='o') return "\n#warning Should be OUTER nodes\n\tFOR_EACH_NODE(n)";
  dbg("\n\t\t[kokkosHookSelectEnumerate] face? (itm=%c, grp='%s', rgn='%s')", itm, grp, rgn);
  if (itm=='f' && grp==NULL && rgn==NULL)     return "FOR_EACH_FACE(f)";
  if (itm=='f' && grp==NULL && rgn[0]=='i')   return "FOR_EACH_INNER_FACE(of)";
  if (itm=='f' && grp==NULL && rgn[0]=='o')   return "FOR_EACH_OUTER_FACE(of)";
  // ! Tester grp==NULL avant ces prochains:
  if (itm=='f' && grp[0]=='o' && rgn==NULL)   return "FOR_EACH_FACE(f)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='o') return "FOR_EACH_OWN_OUTER_FACE(of)";
  if (itm=='f' && grp[0]=='o' && rgn[0]=='i') return "FOR_EACH_OWN_INNER_FACE(of)";
  dbg("\n\t\t[kokkosHookSelectEnumerate] Could not distinguish ENUMERATE!");
  nablaError("Could not distinguish ENUMERATE!");
  return NULL;
}

// **************************************************************************** 
// * kHookForAllPostfix
// **************************************************************************** 
char* kHookForAllPostfix(nablaJob *job){
  nablaMain *nabla=job->entity->main;
  const char itm=job->item[0];
  const char *rgn=job->region;
  
  if (itm=='f' && rgn && rgn[0]=='o')
    nprintf(nabla, NULL,
            "\n\t\tconst int f=of+NABLA_NB_FACES_INNER;");

  return xCallFilterGather(NULL,job);
}
