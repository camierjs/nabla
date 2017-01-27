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
// * cellJobCellVar
// ****************************************************************************
char *cellJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'c') return NULL;
  if (var->item[0] != 'c') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';

  dbg("\n\t\t[cellJobCellVar] scalar=%d, resolve=%d, \
forall_none=%d, forall_node=%d, forall_face=%d, forall_cell=%d",
      scalar,resolve,forall_none,forall_node,forall_face,forall_cell);
  
  if (scalar && !resolve) return "";
  if (scalar) return "[cell]";

  if (!scalar && !resolve) return "[cell]";
  if (!scalar && forall_node) return "[cell][n.index()]";
  if (!scalar && forall_face) return "[cell][f.index()]";
  if (!scalar) return "[cell]";

  nablaError("Could not switch in cellJobCellVar!");
  return NULL;
}


// ****************************************************************************
// * cellJobNodeVar
// * ATTENTION à l'ordre!
// ****************************************************************************
char *cellJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'c') return NULL;
  if (var->item[0] != 'n') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[cellJobNodeVar] scalar=%d, resolve=%d, \
forall_none=%d, forall_node=%d, forall_face=%d, \
forall_cell=%d isPostfixed=%d",
      scalar,resolve,forall_none,forall_node,\
      forall_face,forall_cell,job->parse.isPostfixed);

  if (resolve && forall_none) return "[cell->node";
  if (resolve && forall_face) return "[f->node";
  if (resolve && forall_node) return "[n]";

  if (!resolve && forall_none) return "[cell->node";
  if (!resolve && forall_face) return "[";
  if (!resolve && forall_node) return "[cell->node";
  
  nablaError("Could not switch in cellJobNodeVar!");
  return NULL;
}


// ****************************************************************************
// * cellJobFaceVar
// ****************************************************************************
char *cellJobFaceVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'c') return NULL;
  if (var->item[0] != 'f') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[cellJobFaceVar] scalar=%d, resolve=%d, forall_none=%d, forall_node=%d, forall_face=%d, forall_cell=%d isPostfixed=%d", scalar,resolve,forall_none,forall_node,forall_face,forall_cell,job->parse.isPostfixed);
  //nprintf(arc, "/*FaceVar*/", NULL); // FACE variable
  if (forall_face) return "[f]";
  if (forall_none) return "[cell->face";
  if (forall_node) return "[node->face"; // oups, should be checked!
 
  nablaError("Could not switch in cellJobFaceVar!");
  return NULL;
}


// ****************************************************************************
// * cellJobParticleVar
// ****************************************************************************
char *cellJobParticleVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'c') return NULL;
  if (var->item[0] != 'p') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int forall_none = job->parse.enum_enum=='\0';
  const int forall_cell = job->parse.enum_enum=='c';
  const int forall_face = job->parse.enum_enum=='f';
  const int forall_node = job->parse.enum_enum=='n';
  const int forall_particle = job->parse.enum_enum=='p';
  
  dbg("\n\t\t[cellJobParticleVar] scalar=%d, resolve=%d, forall_none=%d, forall_node=%d, forall_face=%d, forall_cell=%d isPostfixed=%d", scalar,resolve,forall_none,forall_node,forall_face,forall_cell,job->parse.isPostfixed);
  
  if (forall_particle) return "[p]";
  if (forall_none) return "[cell->particle";

  nablaError("Could not switch in cellJobParticleVar!");
  return NULL;
}


// ****************************************************************************
// * cellJobGlobalVar
// ****************************************************************************
char *cellJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'c') return NULL;
  if (var->item[0] != 'g') return NULL;
  const bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  dbg("\n\t\t[cellJobGlobalVar]");
  // "()" permet de récupérer les m_global_...()
  if (left_of_assignment_operator)
    return "";
  else
    return "()";
}
