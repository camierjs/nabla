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

  error(!0,0,"Could not switch in cellJobCellVar!");
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
  
  error(!0,0,"Could not switch in cellJobNodeVar!");
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
  
  error(!0,0,"Could not switch in cellJobFaceVar!");
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

  error(!0,0,"Could not switch in cellJobParticleVar!");
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
