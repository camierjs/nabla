/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nablaEntitys.c      									       			  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2012.11.13																	  *
 * Updated  : 2012.11.13																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2012.11.13	camierjs	Creation															  *
 *****************************************************************************/
#include "nabla.h"

// Allocation d'une nouvelle structure de entity
nablaEntity *nablaEntityNew(nablaMain *nabla){
	nablaEntity *entity;
	entity = (nablaEntity *)malloc(sizeof(nablaEntity));
 	assert(entity != NULL);
   entity->hdr=entity->src=NULL;
   entity->next=NULL;
   entity->main=nabla;
   entity->jobs=NULL;
   entity->libraries=0;// Par défaut, pas de library utilisée
  	return entity; 
}

nablaEntity *nablaEntityAddEntity(nablaMain *nabla, nablaEntity *entity) {
  nablaEntity *iterator;
  assert(entity != NULL);
  if (nabla->entity==NULL){
    nabla->entity=entity;
    return entity;
  }
  iterator = nabla->entity->next;
  if(iterator == NULL)
    iterator = entity;
  else {
    while(iterator->next != NULL)
      iterator = iterator->next;
    iterator->next = entity;
  }
  return iterator;
}

