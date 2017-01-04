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
// * nodeJobNodeVar
// * ATTENTION à l'ordre!
// ****************************************************************************
char *nodeJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'n') return NULL;
  if (var->item[0] != 'n') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[nodeJobNodeVar] %s %s:\t\tscalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d, isPostfixed=%d",job->name,var->name,
         scalar,resolve,forall_none,forall_node,forall_face,forall_cell,job->parse.isPostfixed);

  if (scalar && !resolve) return "";
  
  if (scalar && resolve && forall_face) return "/*srf*/[node]";
  if (scalar && resolve && forall_node) return "[n]";
  if (scalar && resolve && forall_cell) return "[node]";
  if (scalar && resolve && forall_none) return "[node]";

  // On laisse passer pour le dièse
  if (!scalar && !resolve && forall_cell) return "[node]";

  if (!scalar && resolve && forall_cell) return "[node][c.index()]";
  if (!scalar && resolve && forall_face) return "[node][f.index()]";
  if (!scalar && !resolve && forall_face) return "[node]";
  
  nablaError("Could not switch in nodeJobNodeVar!");
  return NULL;
}


// ****************************************************************************
// * nodeJobCellVar
// ****************************************************************************
char *nodeJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'n') return NULL;
  if (var->item[0] != 'c') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[nodeJobCellVar] scalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d",
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);
  
  if (!scalar) return "[c]";//cell][node->cell";
  
  if (forall_face) return "[";
  if (forall_node) return "[n]";
  if (forall_cell) return "[c]";
  if (forall_none) return "[cell->node";
  
  nablaError("Could not switch in nodeJobNodeVar!");
  return NULL;
}


// ****************************************************************************
// * nodeJobFaceVar
// ****************************************************************************
char *nodeJobFaceVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'n') return NULL;
  if (var->item[0] != 'f') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[nodeJobFaceVar] scalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d",
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);
  
  if (forall_face) return "[f]";
  if (forall_none) return "[face]";
  
  nablaError("Could not switch in nodeJobFaceVar!");
  return NULL;
}


// ****************************************************************************
// * nodeJobGlobalVar
// ****************************************************************************
char *nodeJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'n') return NULL;
  if (var->item[0] != 'g') return NULL;
  const bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  dbg("\n\t\t[nodeJobGlobalVar] scalar=%d, resolve=%d, forall_none=%d,\
 forall_node=%d, forall_face=%d, forall_cell=%d",
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);
  if (left_of_assignment_operator) return "";
  return "()";
}
