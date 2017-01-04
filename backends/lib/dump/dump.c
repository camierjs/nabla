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

// ****************************************************************************
// * dumpExternalFile
// * NABLA_LICENSE_HEADER is tied and defined in nabla.h
// ****************************************************************************
static char *dumpExternalFile(char *file){
  return file+NABLA_LICENSE_HEADER;
}

// ****************************************************************************
// * extern definitions from dDump.S
// ****************************************************************************
extern char debug_h[];
extern char types_h[];
extern char gather_h[];
extern char scatter_h[];
extern char ostream_h[];
extern char ternary_h[];

extern char items_c[];
extern char msh1D_c[];
extern char msh2D_c[];
extern char msh3D_c[];


// ****************************************************************************
// * dumpHeader
// ****************************************************************************
void xDumpHeader(nablaMain *nabla){
  assert(nabla->entity->name);
  fprintf(nabla->entity->hdr,dumpExternalFile(types_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(ternary_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(gather_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(scatter_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(ostream_h));
  fprintf(nabla->entity->hdr,dumpExternalFile(debug_h));
}


// ****************************************************************************
// * dumpMesh
// ****************************************************************************
void xDumpMesh(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n");
  // On rajoute les tests 'is something'
  fprintf(nabla->entity->hdr,dumpExternalFile(items_c));
  // Et les constantes des connectivités
  if ((nabla->entity->libraries&(1<<with_real))!=0)
    fprintf(nabla->entity->src,dumpExternalFile(msh1D_c));
  else
    if ((nabla->entity->libraries&(1<<with_real2))!=0)
    fprintf(nabla->entity->src,dumpExternalFile(msh2D_c));
  else
    fprintf(nabla->entity->src,dumpExternalFile(msh3D_c));
}
