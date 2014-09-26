/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcJobFaces.c      													  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2014.09.24																	  
 * Updated  : 2014.09.24																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2014.09.24	camierjs	Creation															  *
 *****************************************************************************/
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

  error(!0,0,"Could not switch in faceJobNodeVar!");
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
  
  dbg("\n\t\t[faceJobFaceVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);
  
  //if (resolve && foreach_cell) return "/*ff:rc*/";
  //if (resolve && foreach_node) return "/*ff:rn*/";
  //if (resolve && foreach_face) return "/*ff:rf*/";
  if (resolve && foreach_none) return "/*ff:r0*/[face]";
  
  //if (!resolve && foreach_cell) return "/*ff:!rc*/";
  //if (!resolve && foreach_node) return "/*ff:!rn*/";
  //if (!resolve && foreach_face) return "/*ff:!rf*/";
  //if (!resolve && foreach_none) return "/*ff:!r0*/";
 
  error(!0,0,"Could not switch in faceJobFaceVar!");
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
