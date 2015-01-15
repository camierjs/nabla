// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
  
  error(!0,0,"Could not switch in nodeJobNodeVar!");
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
  
  if (!scalar) return "[cell][node->cell";
  
  if (forall_face) return "[";
  if (forall_node) return "[n]";
  if (forall_cell) return "[c]";
  if (forall_none) return "[cell->node";
  
  error(!0,0,"Could not switch in nodeJobNodeVar!");
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
  
  error(!0,0,"Could not switch in nodeJobFaceVar!");
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
