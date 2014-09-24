/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcJobNodes.c	     													  *
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
// * nodeJobNodeVar
// * ATTENTION à l'ordre!
// ****************************************************************************
char *nodeJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'n') return NULL;
  if (var->item[0] != 'n') return NULL;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[nodeJobNodeVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);

  if (scalar && !resolve) return "";
  
  if (scalar && resolve && foreach_face) return "[node]";
  if (scalar && resolve && foreach_node) return "[n]";
  if (scalar && resolve && foreach_cell) return "[node]";
  if (scalar && resolve && foreach_none) return "[node]";

  // On laisse passer pour le dièse
  if (!scalar && !resolve && foreach_cell) return "[node]";

  if (!scalar && resolve && foreach_cell) return "[node][c.index()]";
  if (!scalar && resolve && foreach_face) return "[node][f.index()]";
  if (!scalar && !resolve && foreach_face) return "[node]";
  
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
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[nodeJobCellVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);
  
  if (!scalar) return "[cell][node->cell";
  
  if (foreach_face) return "[";
  if (foreach_node) return "[n]";
  if (foreach_cell) return "[c]";
  if (foreach_none) return "[cell->node";
  
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
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  
  dbg("\n\t\t[nodeJobFaceVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);
  
  if (foreach_face) return "[f]";
  if (foreach_none) return "[face]";
  
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
  const int foreach_none = job->parse.enum_enum=='\0';
  const int foreach_cell = job->parse.enum_enum=='c';
  const int foreach_face = job->parse.enum_enum=='f';
  const int foreach_node = job->parse.enum_enum=='n';
  dbg("\n\t\t[nodeJobGlobalVar] scalar=%d, resolve=%d, foreach_none=%d,\
 foreach_node=%d, foreach_face=%d, foreach_cell=%d",
      scalar,resolve,foreach_none,foreach_node,foreach_face,foreach_cell);
  if (left_of_assignment_operator) return "";
  return "()";
}
