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

//#warning Les valeurs autorisées sont: init, compute-loop, restore, on-mesh-changed, on-mesh-refinement, build, exit


/*****************************************************************************
 * Backend ARCANE - Génération du fichier '.config'
 *****************************************************************************/
// Backend ARCANE - Header du fichier '.arc'
#define ARC_CONFIG_HEADER "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?> \
\n<arcane-config code-name=\"%s\">\
\n\t<time-loops>\
\n\t\t<time-loop name=\"%sLoop\">\
\n\t\t\t<title>%s</title>\
\n\t\t\t<description>Boucle en temps de %s</description>\
\n\t\t\t<modules>\
\n\t\t\t\t<module name=\"%s\" need=\"required\" />\
\n\t\t\t\t<module name=\"ArcanePostProcessing\" need=\"required\" />\
\n\t\t\t\t<module name=\"ArcaneCheckpoint\" need=\"required\" />\
\n\t\t\t</modules>\
\n\n\t\t\t<entry-points where=\"init\">"
NABLA_STATUS nccArcConfigHeader(nablaMain *arc){
   fprintf(arc->cfg, ARC_CONFIG_HEADER, arc->name, arc->name, arc->name, arc->name, arc->name);
	return NABLA_OK;
}


// Backend ARCANE - Footer du fichier '.config'
#define ARC_CONFIG_FOOTER "\n\t\t\t</entry-points>\
\n\t\t</time-loop>\
\n\t</time-loops>\
\n</arcane-config>"
NABLA_STATUS nccArcConfigFooter(nablaMain *arc){
   fprintf(arc->cfg, ARC_CONFIG_FOOTER);
	return NABLA_OK;
}
