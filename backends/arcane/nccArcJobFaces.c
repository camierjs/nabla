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
// * faceJobCellVar
// * ATTENTION à l'ordre!
// ****************************************************************************
char *faceJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'f') return NULL;
  if (var->item[0] != 'c') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[faceJobCellVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);

  if (scalar && foreach_none && !resolve) return "[face->cell";
  if (scalar && foreach_none &&  resolve) return "[face->cell]";
  if (scalar && !foreach_none) return "[c";
  if (!scalar) return "[cell][node->cell";

  error(!0,0,"Could not switch in faceJobCellVar!");
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
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[faceJobNodeVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);
  
  if (resolve && foreach_cell) return "/*fn:rc*/[c]";
  if (resolve && foreach_node) return "/*fn:rn*/[n]";
  if (resolve && foreach_face) return "/*fn:rf*/[f]";
  //if (resolve && foreach_none) return "/*fn:r0*/";
    
  //if (!resolve && foreach_cell) return "/*fn:!rc*/";
  //if (!resolve && foreach_node) return "/*fn:!rn*/";
  //if (!resolve && foreach_face) return "/*fn:!rf*/";
  if (!resolve && foreach_none) return "/*fn:!r0*/[face->node";

  nabla_error("Could not switch in faceJobNodeVar!");
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
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[faceJobFaceVar] var name: %s scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",var->name,
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);
  
  //if (resolve && foreach_cell) return "/*ff:rc*/";
  //if (resolve && foreach_node) return "/*ff:rn*/";
  //if (resolve && foreach_face) return "/*ff:rf*/";
  if (resolve && foreach_none) return "/*ff:r0*/[face]";
  
  //if (!resolve && foreach_cell) return "/*ff:!rc*/";
  //if (!resolve && foreach_node) return "/*ff:!rn*/";
  //if (!resolve && foreach_face) return "/*ff:!rf*/";
  if (!resolve && foreach_none) return "/*ff:!r0*/[face]";
  
  error(!0,0,"[faceJobFaceVar] %s: Could not switch in faceJobFaceVar!", job->name);
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
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  dbg("\n\t\t[faceJobGlobalVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);
  if (left_of_assignment_operator) return "";
  return "()";
}
