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
 * HEADER
 *****************************************************************************/
char* nccArcLibCartesianHeader(void){
  // On prépare le header de l'entity
  return "\n#include \"arcane/MeshArea.h\"\n\
#include \"arcane/ISubDomain.h\"\n\
#include \"arcane/MeshAreaAccessor.h\"\n\
//#include \"arcane/cea/ICartesianMesh.h\"\n\
//#include \"arcane/cea/CellDirectionMng.h\"\n\
//#include \"arcane/cea/FaceDirectionMng.h\"\n\
//#include \"arcane/cea/NodeDirectionMng.h\"\n\n\
";
}


/***************************************************************************** 
 * PRIVATES
 *****************************************************************************/
char* nccArcLibCartesianPrivates(void){
  return "\nprivate:\n\
\tvoid libCartesianInitialize(void);\n\
\tICartesianMesh* m_cartesian_mesh;";
}


/******************************************************************************
 * Initialisation
 ******************************************************************************/
void nccArcLibCartesianIni(nablaMain *arc){
  fprintf(arc->entity->src, "\
\n\nvoid %s%s::libCartesianInitialize(void){\n\
\tIMesh* mesh = defaultMesh();\n\
\tm_cartesian_mesh = arcaneCreateCartesianMesh(mesh);\n\
\tm_cartesian_mesh->computeDirections();\n}",arc->name,nablaArcaneColor(arc));
  nablaJob *libCartesianInitialize=nMiddleJobNew(arc->entity);
  libCartesianInitialize->is_an_entry_point=true;
  libCartesianInitialize->is_a_function=true;
  libCartesianInitialize->scope  = sdup("NoGroup");
  libCartesianInitialize->region = sdup("NoRegion");
  libCartesianInitialize->item   = sdup("\0");
  libCartesianInitialize->return_type  = sdup("void");
  libCartesianInitialize->name   = sdup("libCartesianInitialize");
  libCartesianInitialize->name_utf8 = sdup("libCartesianInitialize");
  libCartesianInitialize->xyz    = sdup("NoXYZ");
  libCartesianInitialize->direction  = sdup("NoDirection");
  sprintf(&libCartesianInitialize->at[0],"-huge_valf");
  libCartesianInitialize->when_index  = 1;
  libCartesianInitialize->whens[0] = ENTRY_POINT_init;
  nMiddleJobAdd(arc->entity, libCartesianInitialize);
}
