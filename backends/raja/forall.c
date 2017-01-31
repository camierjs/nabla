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

static char* forAllParticle(void){
  return "RAJA::forall<particle_exec_policy>(*particleList,[=] RAJA_DEVICE (int p)";
}

static char* forAllCell(void){
  return "RAJA::forall<cell_exec_policy>(*cellIdxSet,[=] RAJA_DEVICE (int c)";}
static char* forAllInnerCell(void){
  return "\n#warning Should be INNER cells\n\
\tRAJA::forall<cell_exec_policy>(*cellIdxSet,[=] RAJA_DEVICE (int c)";}
static char* forAllOuterCell(void){
  return "RAJA::forall<cell_exec_policy>(*cellIdxSet,[=] RAJA_DEVICE (int c)";}

static char* forAllNode(void){
  return "RAJA::forall<node_exec_policy>(*nodeIdxSet,[=] RAJA_DEVICE (int n)";}
static char* forAllInnerNode(void){
  return "\n#warning Should be INNER nodes\n\
\tRAJA::forall<node_exec_policy>(*nodeIdxSet,[=] RAJA_DEVICE (int n)";}
static char* forAllOuterNode(void){
  return "\n#warning Should be OUTER nodes\n\
\tRAJA::forall<node_exec_policy>(*nodeIdxSet,[=] RAJA_DEVICE (int n)";}


static char* forAllFace(void){
  return "RAJA::forall<face_exec_policy>(*faceIdxSet,[=] RAJA_DEVICE (int f)";
}
static char* forAllInnerFace(void){
  return "RAJA::forall<face_exec_policy>(*faceIdxSet,[=] RAJA_DEVICE (int f)";
}
static char* forAllOuterFace(void){
  return "RAJA::forall<face_exec_policy>(*faceIdxSet,[=] RAJA_DEVICE (int f)";
}


// ****************************************************************************
// * Fonction produisant l'ENUMERATE_*
// ****************************************************************************
char* rajaHookForAllDump(nablaJob *job){
  const char *grp=job->scope;   // OWN||ALL
  const char *rgn=job->region;  // INNER, OUTER
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  
  dbg("\n\t\t[rajaHookForAllDump] function?");
  if (itm=='\0') return "\n";
  
  dbg("\n\t\t[rajaHookForAllDump] particle?");
  if (itm=='p' && grp==NULL && rgn==NULL) return forAllParticle();
  
  dbg("\n\t\t[rajaHookForAllDump] cell?");
  if (itm=='c' && grp==NULL && rgn==NULL) return forAllCell();
  if (itm=='c' && grp==NULL && rgn[0]=='i') return forAllInnerCell();
  if (itm=='c' && grp==NULL && rgn[0]=='o') return forAllOuterCell();
  if (itm=='c' && grp[0]=='o' && rgn==NULL) return forAllCell();
  
  dbg("\n\t\t[rajaHookForAllDump] node?");
  if (itm=='n' && grp==NULL && rgn==NULL)     return forAllNode();
  if (itm=='n' && grp==NULL && rgn[0]=='i')   return forAllInnerNode();
  if (itm=='n' && grp==NULL && rgn[0]=='o')   return forAllOuterNode();
  if (itm=='n' && grp[0]=='o' && rgn==NULL)   return forAllNode();
  if (itm=='n' && grp[0]=='a' && rgn==NULL)   return forAllNode();
  if (itm=='n' && grp[0]=='o' && rgn[0]=='i') return forAllInnerNode();
  if (itm=='n' && grp[0]=='o' && rgn[0]=='o') return forAllOuterNode();
  
  dbg("\n\t\t[rajaHookForAllDump] face? (itm=%c, grp='%s', rgn='%s')", itm, grp, rgn);
  if (itm=='f' && grp==NULL && rgn==NULL)     return forAllFace();
  if (itm=='f' && grp==NULL && rgn[0]=='i')   return forAllInnerFace();
  if (itm=='f' && grp==NULL && rgn[0]=='o')   return forAllOuterFace();
  // ! Tester grp==NULL avant ces prochains:
  if (itm=='f' && grp[0]=='o' && rgn==NULL)   return forAllFace();
  if (itm=='f' && grp[0]=='o' && rgn[0]=='i') return forAllInnerFace();
  if (itm=='f' && grp[0]=='o' && rgn[0]=='o') return forAllOuterFace();
  
  dbg("\n\t\t[rajaHookForAllDump] Could not distinguish ENUMERATE!");
  nablaError("Could not distinguish ENUMERATE!");
  return NULL;
}

