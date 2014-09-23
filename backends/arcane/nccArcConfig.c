/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcaneConfig.c															  *
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
