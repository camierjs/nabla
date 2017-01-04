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
// * faceJobCellVar
// * ATTENTION à l'ordre!
// ****************************************************************************
char *faceJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'f') return NULL;
  if (var->item[0] != 'c') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[faceJobCellVar] scalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d",
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);

  if (scalar && forall_none && !resolve) return "[face->cell/*1*/";
  if (scalar && forall_none &&  resolve) return "[face->cell/*2*/]";
  if (scalar && !forall_none) return "[c";
  if (!scalar) return "[cell][node->cell";
  if (!scalar&&job->nb_in_item_set>0) return "[cell][node->cell";

  nablaError("Could not switch in faceJobCellVar!");
  return NULL;
}


// ****************************************************************************
// * faceJobNodeVar
// ****************************************************************************
char *faceJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'f') return NULL;
  if (var->item[0] != 'n') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[faceJobNodeVar] scalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d",
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);
  
  if (resolve && forall_cell) return "/*fn:rc*/[c]";
  if (resolve && forall_node) return "/*fn:rn*/[n]";
  if (resolve && forall_face) return "/*fn:rf*/[f]";
  //if (resolve && forall_none) return "/*fn:r0*/";
    
  //if (!resolve && forall_cell) return "/*fn:!rc*/";
  //if (!resolve && forall_node) return "/*fn:!rn*/";
  //if (!resolve && forall_face) return "/*fn:!rf*/";
  if (!resolve && forall_none) return "/*fn:!r0*/[face->node";

  nablaError("Could not switch in faceJobNodeVar!");
  return NULL;
}


// ****************************************************************************
// * faceJobFaceVar
// ****************************************************************************
char *faceJobFaceVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'f') return NULL;
  if (var->item[0] != 'f') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[faceJobFaceVar] var name: %s scalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d",var->name,
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);
  
  //if (resolve && forall_cell) return "/*ff:rc*/";
  if (resolve && forall_node) return "/*ff:rn*/[face]";
  //if (resolve && forall_face) return "/*ff:rf*/";
  if (resolve && forall_none) return "/*ff:r0*/[face]";
  
  //if (!resolve && forall_cell) return "/*ff:!rc*/";
  //if (!resolve && forall_node) return "/*ff:!rn*/";
  //if (!resolve && forall_face) return "/*ff:!rf*/";
  if (!resolve && forall_none) return "/*ff:!r0*/[face]";
  
  nablaError("[faceJobFaceVar] %s: Could not switch in faceJobFaceVar!", job->name);
  return NULL;
}


// ****************************************************************************
// * faceJobGlobalVar
// ****************************************************************************
char *faceJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'f') return NULL;
  if (var->item[0] != 'g') return NULL;
  const bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  dbg("\n\t\t[faceJobGlobalVar] scalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d",
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);
  if (left_of_assignment_operator) return "";
  return "()";
}
