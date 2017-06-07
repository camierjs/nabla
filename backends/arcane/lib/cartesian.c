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
#include \"arcane/cartesian/ICartesianMesh.h\"\n\
#include \"arcane/cartesian/CellDirectionMng.h\"\n\
#include \"arcane/cartesian/FaceDirectionMng.h\"\n\
#include \"arcane/cartesian/NodeDirectionMng.h\"\n\n\
";
}


/***************************************************************************** 
 * PRIVATES
 *****************************************************************************/
char* nccArcLibCartesianPrivates(void){
  return "\
\n\tbool faceIsOuter(FaceEnumerator f){return f->isSubDomainBoundary();}\
\n\tbool faceIsSouth(FaceEnumerator f){return (f.index()==3)?true:false;}\
\n\tbool faceIsNorth(FaceEnumerator f){return (f.index()==1)?true:false;}\
\n\tbool faceIsWest(FaceEnumerator f){return (f.index()==0)?true:false;}\
\n\tbool faceIsEast(FaceEnumerator f){return (f.index()==2)?true:false;}\
\n\tbool faceIsOuterSouth(FaceEnumerator f){return faceIsSouth(f) and faceIsOuter(f);}\
\n\tbool faceIsOuterNorth(FaceEnumerator f){return faceIsNorth(f) and faceIsOuter(f);}\
\n\tbool faceIsOuterWest(FaceEnumerator f){return faceIsWest(f) and faceIsOuter(f);}\
\n\tbool faceIsOuterEast(FaceEnumerator f){return faceIsEast(f) and faceIsOuter(f);}\
\nprivate:\n\
\tvoid libCartesianInitialize(void);\n\
\tICartesianMesh* m_cartesian_mesh;";
}


// ****************************************************************************
// * Initialisation
// * Création des groups:
// *   - outerCells

// *   - outerWestCells
// *   - outerSouthCells
// *   - outerNorthCells
// *   - outerEastCells
// *   - outerOtherThanWestCells

// *   - innerCells
// *   - innerEastCells
// *   - innerSouthCells
// *   - innerNorthCells
// ****************************************************************************
void nccArcLibCartesianIni(nablaMain *arc){
  fprintf(arc->entity->src, "\
\n\nvoid %s%s::libCartesianInitialize(void){\n\
\tIMesh* mesh = defaultMesh();\n\
\tm_cartesian_mesh = arcaneCreateCartesianMesh(mesh);\n\
\tm_cartesian_mesh->computeDirections();\n\
\n\
\tIItemFamily* cell_family = mesh->cellFamily();\n\
\tInt32Array outer_cells_lid;\n\
\touter_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array inner_cells_lid;\n\
\tinner_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array outerWest_cells_lid;\n\
\touterWest_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array outerSouth_cells_lid;\n\
\touterSouth_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array outerNorth_cells_lid;\n\
\touterNorth_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array innerNorth_cells_lid;\n\
\tinnerNorth_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array outerOtherThanWest_cells_lid;\n\
\touterOtherThanWest_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array outerEast_cells_lid;\n\
\touterEast_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array innerEast_cells_lid;\n\
\tinnerEast_cells_lid.reserve(cell_family->allItems().size());\n\
\tInt32Array innerSouth_cells_lid;\n\
\tinnerSouth_cells_lid.reserve(cell_family->allItems().size());\n\
\tENUMERATE_FACE(iface,mesh->outerFaces()){\n\
\t\tFace face = *iface;\n\
\t\tCell front_cell = face.frontCell();\n\
\t\tCell back_cell = face.backCell();\n\
\t\tif (!front_cell.null() and !outer_cells_lid.contains(front_cell.localId()))\n\
\t\t\touter_cells_lid.add(front_cell.localId());\n\
\t\tif (!back_cell.null() and !outer_cells_lid.contains(back_cell.localId()))\n\
\t\t\touter_cells_lid.add(back_cell.localId());\n\
\t}\n\
\tCellGroup outer_cells_group = cell_family->createGroup(String(\"outerCells\"),outer_cells_lid,true);\n\
\tENUMERATE_CELL(cell,allCells()){\n\
\t\tconst Int64 c = cell->uniqueId();\n\
\t\tconst int nbx = options()->X_EDGE_ELEMS();\n\
\t\tconst int nby = options()->Y_EDGE_ELEMS();\n\
\t\tif ((opMod(c,nbx))==(nby-2) && (c>=nbx && c<nbx*(nby-1))) innerEast_cells_lid.add(cell->localId());\n\
\t\tif (c>nbx && c<(2*nbx-1)) innerSouth_cells_lid.add(cell->localId());\n\
\t\tif (c>nbx*(nby-2) && c<(nbx*(nby-1)-1)) innerNorth_cells_lid.add(cell->localId());\n\
\t\tif (c>nbx && c<nbx*(nby-1) && opMod(c,nbx)!=0 && opMod(c,nbx)!=(nbx-1)) inner_cells_lid.add(cell->localId());\n\
\t}\n\
\tCellGroup inner_cells_group = cell_family->createGroup(String(\"innerCells\"),inner_cells_lid,true);\n\
\tCellGroup innerEast_cells_group = cell_family->createGroup(String(\"innerEastCells\"),innerEast_cells_lid,true);\n\
\tCellGroup innerSouth_cells_group = cell_family->createGroup(String(\"innerSouthCells\"),innerSouth_cells_lid,true);\n\
\tCellGroup innerNorth_cells_group = cell_family->createGroup(String(\"innerNorthCells\"),innerNorth_cells_lid,true);\n\
\tENUMERATE_CELL(cell,defaultMesh()->findGroup(\"outerCells\")){\n\
\t\tconst Int64 c = cell->uniqueId();\n\
\t\tconst int nbx = options()->X_EDGE_ELEMS();\n\
\t\tconst int nby = options()->Y_EDGE_ELEMS();\n\
\t\tif (c<nbx) outerSouth_cells_lid.add(cell->localId());\n\
\t\tif (c>=nbx*(nby-1)) outerNorth_cells_lid.add(cell->localId());\n\
\t\tif (opMod(c,nbx)==0) outerWest_cells_lid.add(cell->localId());\n\
\t\tif (opMod(c,nbx)==nbx-1) outerEast_cells_lid.add(cell->localId());\n\
\t\tif (opMod(c,nbx)!=0) outerOtherThanWest_cells_lid.add(cell->localId());\n\
\t}\n\
\tCellGroup outerSouth_cells_group = cell_family->createGroup(String(\"outerSouthCells\"),outerSouth_cells_lid,true);\n\
\tCellGroup outerNorth_cells_group = cell_family->createGroup(String(\"outerNorthCells\"),outerNorth_cells_lid,true);\n\
\tCellGroup outerEast_cells_group = cell_family->createGroup(String(\"outerEastCells\"),outerEast_cells_lid,true);\n\
\tCellGroup outerWest_cells_group = cell_family->createGroup(String(\"outerWestCells\"),outerWest_cells_lid,true);\n\
\tCellGroup outerOtherThanWest_cells_group = cell_family->createGroup(String(\"outerOtherThanWestCells\"),outerOtherThanWest_cells_lid,true);\n\
}",
          arc->name,nablaArcaneColor(arc));
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
