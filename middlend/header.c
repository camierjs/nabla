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
//#include "backends/arcane/arcane.h"


// ****************************************************************************
// * Dump dans le header des 'includes'
// ****************************************************************************
NABLA_STATUS nMiddleInclude(nablaMain *nabla, const char *include){
  fprintf(nabla->entity->src, "%s\n", include);
  return NABLA_OK;
}


// ****************************************************************************
// * Dump dans le header les 'define's
// ****************************************************************************
NABLA_STATUS nMiddleDefines(nablaMain *nabla, const nWhatWith *defines){
  int i;
#ifdef ARCANE_FOUND
  FILE *target_file = isAnArcaneService(nabla)?nabla->entity->src:nabla->entity->hdr;
#else
  FILE *target_file = nabla->entity->hdr;
#endif
  fprintf(target_file,"\n\
\n// *****************************************************************************\
\n// * Defines\
\n// *****************************************************************************");
  for(i=0;defines[i].what!=NULL;i+=1)
    fprintf(target_file, "\n#define %s %s",defines[i].what,defines[i].with);
  fprintf(target_file, "\n");
  return NABLA_OK;
}


// ****************************************************************************
// * Dump dans le header les 'typedef's
// ****************************************************************************
NABLA_STATUS nMiddleTypedefs(nablaMain *nabla, const nWhatWith *typedefs){
  fprintf(nabla->entity->hdr,"\n\
\n// *****************************************************************************\
\n// * Typedefs\
\n// *****************************************************************************");
  for(int i=0;typedefs[i].what!=NULL;i+=1)
    fprintf(nabla->entity->hdr, "\ntypedef %s %s;",
            typedefs[i].what, typedefs[i].with);
  fprintf(nabla->entity->hdr, "\n");
  return NABLA_OK;
}


// ****************************************************************************
// * Dump dans le header des 'forwards's
// ****************************************************************************
NABLA_STATUS nMiddleForwards(nablaMain *nabla, const char **forwards){
  fprintf(nabla->entity->hdr,"\n\
\n// *****************************************************************************\
\n// * Forwards\
\n// *****************************************************************************");
  if (!forwards[0])
    fprintf(nabla->entity->hdr, "\n// no Forwards here!\n");
  for(int i=0;forwards[i]!=NULL;i+=1)
    fprintf(nabla->entity->hdr, "\n%s",forwards[i]);
  fprintf(nabla->entity->hdr, "\n");
  return NABLA_OK;
}

