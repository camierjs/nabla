/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcJobParticles.c    													  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2014.10.03																	  
 * Updated  : 2014.10.03																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2014.10.03	camierjs	Creation															  *
 *****************************************************************************/
#include "nabla.h"


char *particleJobParticleVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'p') return NULL;
  if (var->item[0] != 'p') return NULL;
  return "[particle]";
}

char *particleJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'p') return NULL;
  if (var->item[0] != 'c') return NULL;
  return "[particle->cell()]";
}


char *particleJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var){
  if (job->item[0] != 'p') return NULL;
  if (var->item[0] != 'g') return NULL;
  const bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  //const int scalar = var->dim==0;
  //const int resolve = job->parse.isPostfixed!=2;
  if (left_of_assignment_operator) return "";
  return "()";
}


