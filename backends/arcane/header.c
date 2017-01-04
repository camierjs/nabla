///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
#include "nabla.h"
#include "backends/arcane/arcane.h"


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
static NABLA_STATUS nccArcConfigHeader(nablaMain *arc){
   fprintf(arc->cfg, ARC_CONFIG_HEADER, arc->name, arc->name, arc->name, arc->name, arc->name);
	return NABLA_OK;
}


// ****************************************************************************
// ****************************************************************************
static NABLA_STATUS nccArcaneEntityHeader(nablaMain *arc){ 
  fprintf(arc->entity->src,"#include \"%s%s%s.h\"\
\n#include <arcane/IParallelMng.h>\
\n#include <arcane/ITimeLoopMng.h>\
\n#include <arcane/anyitem/AnyItem.h>\
\n#include <arcane/ItemPairGroup.h>\
\n#include <arcane/ItemPairEnumerator.h>\n\n",
          (isAnArcaneService(arc)||
           (isAnArcaneModule(arc)&&(arc->specific_path!=NULL)))?arc->specific_path:"",
          arc->entity->name,
          nablaArcaneColor(arc));
  return NABLA_OK;
}

// ****************************************************************************
// ****************************************************************************
static NABLA_STATUS nccArcaneBeginNamespace(nablaMain *arc){
  fprintf(arc->entity->src,"\n\nusing namespace Arcane;\n\n%s\n\n",
          isAnArcaneService(arc)?"ARCANE_BEGIN_NAMESPACE":"");
  return NABLA_OK;
}



// Backend ARCANE - Footer du fichier '.config'
#define ARC_CONFIG_FOOTER "\n\t\t\t</entry-points>\
\n\t\t</time-loop>\
\n\t</time-loops>\
\n</arcane-config>"
static NABLA_STATUS nccArcConfigFooter(nablaMain *arc){
   fprintf(arc->cfg, ARC_CONFIG_FOOTER);
	return NABLA_OK;
}


// ****************************************************************************
// * hookHeaderDump
// ****************************************************************************
void aHookHeaderDump(nablaMain *nabla){
  if (isAnArcaneFamily(nabla)){
    dbg("\n[aHookHeaderDump] isAnArcaneFamily");
    aHookFamilyHeader(nabla);
    return;
  }
  dbg("\n[aHookHeaderDump] nccAxlGenerateHeader");
  nccAxlGenerateHeader(nabla);
  dbg("\n[aHookHeaderDump] nccArcaneEntityHeader");
  nccArcaneEntityHeader(nabla);
  // Dans le cas d'un service, on le fait maintenant
  if (isAnArcaneService(nabla)){
    if (nccArcaneEntityIncludes(nabla->entity)!=NABLA_OK)
      printf("error: in service HDR generation!\n");
    nccArcaneBeginNamespace(nabla);
  }
  // Première passe pour les VARIABLES du fichier AXL
  if (isAnArcaneModule(nabla))
    nccArcConfigHeader(nabla);
}


// ****************************************************************************
// * hookHeaderOpen
// ****************************************************************************
void aHookHeaderOpen(nablaMain *nabla){
  char hdrFileName[NABLA_MAX_FILE_NAME];
  dbg("\n[nccArcane] Ouverture du fichier HEADER");
  if (isAnArcaneModule(nabla) || isAnArcaneFamily(nabla)){
    sprintf(hdrFileName, "%s%s.h", nabla->name, nablaArcaneColor(nabla));
    dbg("\n[nccArcane] File %s%s.h",nabla->name, nablaArcaneColor(nabla));
    if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
  }
  dbg("\n[nccArcane] done!");
}


// ****************************************************************************
// * ENUMERATES Hooks
// ****************************************************************************
void aHookHeaderEnums(nablaMain *nabla){
  dbg("\n[aHookHeaderEnums]");
}


// ****************************************************************************
// * hookHeaderPrefix
// ****************************************************************************
void aHookHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name);
  dbg("\n[aHookHeaderPrefix]");
}


// ****************************************************************************
// * hookHeaderIncludes
// ****************************************************************************
void aHookHeaderIncludes(nablaMain *nabla){
  dbg("\n[aHookHeaderIncludes]");
}


// ****************************************************************************
// * 
// ****************************************************************************
void aHookHeaderPostfix(nablaMain *nabla){
  if (isAnArcaneModule(nabla)){ // Fermeture du CONFIG dans le cas d'un module
    nccArcConfigFooter(nabla); 
    fclose(nabla->cfg);
  }
  
  if (!isAnArcaneFamily(nabla)){
    // Fermeture de l'AXL
    fprintf(nabla->axl,"\n</%s>\n",
            (isAnArcaneModule(nabla))?"module":"service");
    fclose(nabla->axl);
  }
  
  if (isAnArcaneFamily(nabla)) aHookFamilyFooter(nabla);
    
  // Fermeture du header
  //if (isAnArcaneModule(nabla)||isAnArcaneFamily(nabla))
  fclose(nabla->entity->hdr);
}
