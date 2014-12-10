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

