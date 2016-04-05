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
// * Génération du fichier 'main.cc'
// ****************************************************************************
#define ARC_MAIN "#include <iostream>\n//#include <mpi.h>\n#include <arcane/impl/ArcaneMain.h>\n\
using namespace Arcane;\n\
int main(int argc,char* argv[]){\n\
  int r = 0;\n\
  ArcaneMain::arcaneInitialize();\n\
  {\n\
    ApplicationInfo app_info(&argc,&argv,\"%s\",VersionInfo(1,0,0));\n\
    r = ArcaneMain::arcaneMain(app_info);\n\
  }\n\
  ArcaneMain::arcaneFinalize();\n\
  return r;\n\
}\n"
NABLA_STATUS nccArcMain(nablaMain *arc){
  dbg("\n[nccArcMain]");
  if ((arc->main=fopen("main.cc", "w")) == NULL) exit(NABLA_ERROR); 
  fprintf(arc->main, ARC_MAIN, arc->name);
  fclose(arc->main);
  return NABLA_OK;
}


// ****************************************************************************
// * aHookMainPrefix
// ****************************************************************************
NABLA_STATUS aHookMainPrefix(nablaMain *nabla){
  dbg("\n[aHookMainPrefix]");
  // MATHEMATICA fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_mathematica))!=0){
    dbg("\n[nccArcane] MATHEMATICA initialization");
    nccArcLibMathematicaIni(nabla);
  }
  // MAIL fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_mail))!=0){
    dbg("\n[nccArcane] MAIL initialization");
    nccArcLibMailIni(nabla);
  }
  // PARTICLES fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_particles))!=0){
    dbg("\n[nccArcane] PARTICLES initialization");
    nccArcLibParticlesIni(nabla);
  }
  // ALEPH fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_aleph))!=0){
    dbg("\n[nccArcane] ALEPH initialization");
    if (isAnArcaneModule(nabla)) nccArcLibAlephIni(nabla);
    if (isAnArcaneService(nabla)) nccArcLibSchemeIni(nabla);
  }
  // CARTESIAN fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_cartesian))!=0){
    dbg("\n[nccArcane] CARTESIAN initialization");
    nccArcLibCartesianIni(nabla);
  }
  // MATERIALS fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_materials))!=0){
    dbg("\n[nccArcane] MATERIALS initialization");
    nccArcLibMaterialsIni(nabla);
  }
  // GMP fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_gmp))!=0){
    dbg("\n[nccArcane] GMP initialization");
    nccArcLibGmpIni(nabla);
  }
  // DFT fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_dft))!=0){
    dbg("\n[nccArcane] DFT initialization");
    nccArcLibDftIni(nabla);
  }
  // SLURM fonctions d'initialisation
  if ((nabla->entity->libraries&(1<<with_slurm))!=0){
    dbg("\n[nccArcane] SLURM initialization");
    nccArcLibSlurmIni(nabla);
  }
  return NABLA_OK;
}

// ****************************************************************************
// * aHookMainPreInit
// ****************************************************************************
NABLA_STATUS aHookMainPreInit(nablaMain *nabla){
  dbg("\n[aHookMainPreInit]");
  if (isAnArcaneFamily(nabla)) return NABLA_OK;
  nArcaneHLTInit(nabla);
  dbg("\n[aHookMainPreInit] AXL generation");
  if (nccAxlGenerator(nabla)!=NABLA_OK)
    printf("error: in AXL generation!\n");
  dbg("\n[aHookMainPreInit] HDR generation");
  if (nccArcaneEntityGeneratorPrivates(nabla->entity)!=NABLA_OK)
    printf("error: in HDR generation!\n");
  return NABLA_OK;
}

// ****************************************************************************
// * aHookMainVarInitKernel
// ****************************************************************************
NABLA_STATUS aHookMainVarInitKernel(nablaMain *nabla){
  dbg("\n[aHookMainVarInitKernel]");
  return NABLA_OK;
}

// ****************************************************************************
// * hookMainVarInitKernel
// ****************************************************************************
NABLA_STATUS aHookMainVarInitCall(nablaMain *nabla){
  dbg("\n[aHookMainVarInitCall]");
  return NABLA_OK;
}


// ****************************************************************************
// * hookMain
// ****************************************************************************
NABLA_STATUS aHookMainHLT(nablaMain *nabla){
  dbg("\n[aHookMainHLT]");
  nprintf(nabla, NULL,"/*aHookMainHLT*/");
  return NABLA_OK;
}


// ****************************************************************************
// * Backend POSTINIT - Génération du 'main'
// ****************************************************************************
NABLA_STATUS aHookMainPostInit(nablaMain *nabla){
  dbg("\n[aHookMainPostInit]");
  return NABLA_OK;
}


// ****************************************************************************
// * hookMainPostfix
// ****************************************************************************
NABLA_STATUS aHookMainPostfix(nablaMain *nabla){
  dbg("\n[aHookMainPostfix]");
  // Fermeture du source du entity
  if (!isAnArcaneFamily(nabla))
    fprintf(nabla->entity->src,"\n\
\n/*---------------------------------------------------------------------------*/\
\n/*---------------------------------------------------------------------------*/\
\nARCANE_REGISTER_%s_%s%s(%s%s%s%s%s);\n%s\n",
            isAnArcaneModule(nabla)?"MODULE":"SERVICE",
            toolStrUpCase(nabla->name),
            isAnArcaneService(nabla)?"SERVICE":"",
            isAnArcaneService(nabla)?nabla->service_name:"",
            isAnArcaneService(nabla)?",":"",
            nabla->name,
            isAnArcaneModule(nabla)?"Module":"Service",
            isAnArcaneService(nabla)?",SFP_None":"",
            isAnArcaneService(nabla)?"ARCANE_END_NAMESPACE\n":"");
  else dbg("\n[aHookMainPostfix] isAnArcaneFamily");
  fclose(nabla->entity->src);
  return NABLA_OK;
}
