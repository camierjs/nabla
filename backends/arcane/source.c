///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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


// ****************************************************************************
// * hookSourceOpen
// ****************************************************************************
void aHookSourceOpen(nablaMain *nabla){
  char srcFileName[NABLA_MAX_FILE_NAME];
  char cfgFileName[NABLA_MAX_FILE_NAME];
  char axlFileName[NABLA_MAX_FILE_NAME];
 
  if (isAnArcaneModule(nabla)==true){
    dbg("\n[nccArcane] Création du fichier ARCANE main.c dans le cas d'un module");
    nccArcMain(nabla);
  }

  dbg("\n[nccArcane] Ouverture du fichier SOURCE");
  sprintf(srcFileName, "%s%s.cc", nabla->entity->name, nablaArcaneColor(nabla));
  if ((nabla->entity->src=fopen(srcFileName, "w")) == NULL) exit(NABLA_ERROR);

  if (isAnArcaneModule(nabla)==true){
    dbg("\n[nccArcane] Ouverture du fichier CONFIG pour ARCANE dans le cas d'un module");
    sprintf(cfgFileName, "%s.config", nabla->name);
    if ((nabla->cfg=fopen(cfgFileName, "w")) == NULL) exit(NABLA_ERROR);
  }
  
  // Pas d'AXL si on génère pour une Family
  if (isAnArcaneFamily(nabla)) return;  

  dbg("\n[nccArcane] Ouverture du fichier AXL pour ARCANE");
  if (isAnArcaneModule(nabla))
    sprintf(axlFileName, "%s.axl", nabla->name);
  
  if (isAnArcaneService(nabla))
    sprintf(axlFileName, "%sService.axl", nabla->name);
  
  if ((nabla->axl=fopen(axlFileName, "w")) == NULL) exit(NABLA_ERROR);
}

  
// ****************************************************************************
// * include
// ****************************************************************************
void aHookSourceInclude(nablaMain *nabla){
}

// ****************************************************************************
// * aHookSourceName
// ****************************************************************************
char* aHookSourceNamespace(nablaMain *nabla){
  return strdup(nabla->name);//nablaArcaneColor(nabla);
}
