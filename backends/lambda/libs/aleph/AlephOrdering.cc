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
#include "Aleph.h"


// *****************************************************************************
// * Minimal AlephOrdering for AlephIndexing
// *****************************************************************************
AlephOrdering::AlephOrdering(AlephKernel *kernel):TraceAccessor(kernel->parallel()->traceMng()),
                                                  m_do_swap(false),
                                                  m_kernel(kernel),
                                                  m_swap(0){}


/******************************************************************************
 *****************************************************************************/
AlephOrdering::AlephOrdering(AlephKernel *kernel,
                             int global_nb_row,
                             int local_nb_row,
                             bool do_swap):TraceAccessor(kernel->parallel()->traceMng()),
                                           m_do_swap(do_swap),
                                           m_kernel(kernel),
                                           m_swap(0)
{
  if (!do_swap){
    debug()<<"\t[AlephOrdering::AlephOrdering] No ordering!";
    return;
  }
  debug()<<"\t[AlephOrdering::AlephOrdering] Ordering!";
  
  if (m_kernel->nbRanksPerSolver()!=1)
    throw std::logic_error("[AlephOrdering] Ordering not allowed in parallel");
  
  int local_nb_cell=m_kernel->subDomain()->defaultMesh()->ownCells().size();
  int total_nb_cell=m_kernel->subDomain()->parallelMng()->reduce(Parallel::ReduceSum,local_nb_cell);

  if ((local_nb_cell==local_nb_row)&&(total_nb_cell==global_nb_row)){
    debug()<<"\t[AlephOrdering::AlephOrdering] Now cell ordering";
    this->initCellOrder();
    return;
  }
  
  if ((2*local_nb_cell==local_nb_row)&&(2*total_nb_cell==global_nb_row)){
    debug()<<"\t[AlephOrdering::AlephOrdering] Now 2*cell ordering";
    this->initTwiceCellOrder();
    return;
  }
  
  int local_nb_face=m_kernel->subDomain()->defaultMesh()->ownFaces().size();
  int total_nb_face=m_kernel->subDomain()->parallelMng()->reduce(Parallel::ReduceSum,local_nb_face);
  if (((local_nb_cell+local_nb_face)==local_nb_row)&&((total_nb_cell+total_nb_face)==global_nb_row)){
    debug()<<"\t[AlephOrdering::AlephOrdering] Now cell+face ordering";
    this->initCellFaceOrder();
    return;
  }

  if ((local_nb_face==local_nb_row)&&(total_nb_face==global_nb_row)){
    debug()<<"\t[AlephOrdering::AlephOrdering] Now face ordering";
    this->initFaceOrder();
    return;
  }

  
  int local_nb_node=m_kernel->subDomain()->defaultMesh()->ownNodes().size();
  int total_nb_node=m_kernel->subDomain()->parallelMng()->reduce(Parallel::ReduceSum,local_nb_node);
  
  if ((((local_nb_cell+local_nb_node))==local_nb_row)&&(((total_nb_cell+total_nb_node))==global_nb_row)){
    debug()<<"\t[AlephOrdering::AlephOrdering] Now (cell+node) ordering";
    this->initCellNodeOrder();
    return;
  }
  
  if (((2*(local_nb_cell+local_nb_node))==local_nb_row)&&((2*(total_nb_cell+total_nb_node))==global_nb_row)){
    debug()<<"\t[AlephOrdering::AlephOrdering] Now 2*(cell+node) ordering";
    this->initTwiceCellNodeOrder();
    return;
  }

  
  throw std::logic_error("[AlephOrdering] Could not guess cell||face||cell+face");
}

  
/******************************************************************************
 *****************************************************************************/
AlephOrdering::~AlephOrdering(){
  debug()<<"\33[5m\t[~AlephOrdering]\33[0m";
}


/******************************************************************************
 * initCellOrder
 *****************************************************************************/
void AlephOrdering::initCellOrder(void){
  debug()<<"\t[AlephOrdering::InitializeCellOrder] "<<m_kernel->topology()->gathered_nb_row(m_kernel->size());
  m_swap.resize(m_kernel->topology()->gathered_nb_row(m_kernel->size()));
  vector<int> all;
  int added=0;
  ENUMERATE_GROUP(cell,m_kernel->subDomain()->defaultMesh()->ownCells()){
    all.push_back(cell->uniqueId());
    added+=1;
  }
  debug()<<"\t[AlephOrdering::InitializeCellOrder] added="<<added;
  m_kernel->parallel()->allGatherVariable(all,m_swap);
}


/******************************************************************************
 * initTwiceCellOrder
 * Pour le mode Complexe
 *****************************************************************************/
void AlephOrdering::initTwiceCellOrder(void){
  debug()<<"\t[AlephOrdering::InitializeTwiceCellOrder] "<<m_kernel->topology()->gathered_nb_row(m_kernel->size());
  m_swap.resize(m_kernel->topology()->gathered_nb_row(m_kernel->size()));
  vector<int> all;
  int added=0;
  ENUMERATE_GROUP(cell,m_kernel->subDomain()->defaultMesh()->ownCells()){
    all.push_back(2*cell->uniqueId());
    added+=1;
    all.push_back(2*cell->uniqueId()+1);
    added+=1;
  }
  debug()<<"\t[AlephOrdering::InitializeTwiceCellOrder] added="<<added;
  m_kernel->parallel()->allGatherVariable(all,m_swap);
}


/******************************************************************************
 * initFaceOrder
 *****************************************************************************/
void AlephOrdering::initFaceOrder(void){
  debug()<<"\t[AlephOrdering::InitializeFaceOrder] "<<m_kernel->topology()->gathered_nb_row(m_kernel->size());
  m_swap.resize(m_kernel->topology()->gathered_nb_row(m_kernel->size()));
  vector<int> all;
  int added=0;
  ENUMERATE_GROUP(face,m_kernel->subDomain()->defaultMesh()->ownFaces()){
    all.push_back(face->uniqueId());
    added+=1;
  }
  debug()<<"\t[AlephOrdering::InitializeFaceOrder] added="<<added;
  m_kernel->parallel()->allGatherVariable(all,m_swap);
}


/******************************************************************************
 * initCellFaceOrder
 *****************************************************************************/
void AlephOrdering::initCellFaceOrder(void){
  debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] "<<m_kernel->topology()->gathered_nb_row(m_kernel->size());

  vector<int> all_cells;
  vector<int> all_faces;
  vector<int> gathered_nb_cells(m_kernel->size());
  vector<int> gathered_nb_faces(m_kernel->size());
  all_cells.push_back(m_kernel->subDomain()->defaultMesh()->ownCells().size());
  all_faces.push_back(m_kernel->subDomain()->defaultMesh()->ownFaces().size());
  m_kernel->parallel()->allGather(all_cells,gathered_nb_cells);
  m_kernel->parallel()->allGather(all_faces,gathered_nb_faces);
/*  for(int i=0,N=gathered_nb_cells.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] gathered_nb_cells["<<i<<"]="<<gathered_nb_cells.at(i);
  for(int i=0,N=gathered_nb_faces.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] gathered_nb_faces["<<i<<"]="<<gathered_nb_faces.at(i);
*/

  vector<int> all;
  vector<int> m_swap_cell;
  ENUMERATE_GROUP(cell,m_kernel->subDomain()->defaultMesh()->ownCells()){
    all.push_back(cell->uniqueId());
  }
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] added for cells="<<all.size();
  m_kernel->parallel()->allGatherVariable(all,m_swap_cell);
/*  for(int i=0,N=m_swap_cell.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap_cell["<<i<<"]="<<m_swap_cell.at(i);
*/
  all.clear();
  vector<int> m_swap_face;
  ENUMERATE_GROUP(face,m_kernel->subDomain()->defaultMesh()->ownFaces()){
    all.push_back(face->uniqueId());
  }
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] added for faces="<<all.size();
  m_kernel->parallel()->allGatherVariable(all,m_swap_face);
/*  for(int i=0,N=m_swap_face.size();i<N;++i)
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap_face["<<i<<"]="<<m_swap_face.at(i);
*/
  
  int cell_offset=m_swap_cell.size();
//  debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Now combining cells+faces of size="<<m_swap_cell.size()+m_swap_face.size();
/*  m_swap.resize(m_swap_cell.size()+m_swap_face.size());
  m_swap.copy(m_swap_cell.constView());
  debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Shifting faces of "<<m_swap_cell.size();
  for(int i=0,N=m_swap_face.size();i<N;++i)
    m_swap_face[i]+=cell_offset;
  m_swap.addRange(m_swap_face.constView());
*/
  
  m_swap.resize(m_swap_cell.size()+m_swap_face.size());
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Now combining cells";
  int iCell=0;
  for(int i=0;i<m_kernel->size();++i){
    int offset=m_kernel->topology()->gathered_nb_row(i);
    for(int j=0;j<gathered_nb_cells.at(i);++j){
      //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap["<<offset+j<<"]="<<m_swap_cell.at(iCell);
      m_swap[offset+j]=m_swap_cell.at(iCell);
      iCell+=1;
    }
  }
  //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Now combining faces";
  int iFace=0;
  for(int i=0;i<m_kernel->size();++i){
    int offset=0;
    if (i>0) offset=m_kernel->topology()->gathered_nb_row(i);
    offset+=gathered_nb_cells.at(i);
    for(int j=0;j<gathered_nb_faces.at(i);++j){
      //debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap["<<offset+j<<"]="<<m_swap_face.at(iFace);
      m_swap[offset+j]=cell_offset+m_swap_face.at(iFace);
      iFace+=1;
    }
  }

/*  debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] Like it?";
  for(int i=0,N=m_swap.size();i<N;++i){
    debug()<<"\t[AlephOrdering::InitializeCellFaceOrder] m_swap["<<i<<"]="<<m_swap.at(i);
    }*/
}





/******************************************************************************
 * initCellNodeOrder
 *****************************************************************************/
void AlephOrdering::initCellNodeOrder(void){
  debug()<<"\t[AlephOrdering::InitializeCellNodeOrder] "<<m_kernel->topology()->gathered_nb_row(m_kernel->size());

  vector<int> all_cells;
  vector<int> all_nodes;
  vector<int> gathered_nb_cells(m_kernel->size());
  vector<int> gathered_nb_nodes(m_kernel->size());
  all_cells.push_back(m_kernel->subDomain()->defaultMesh()->ownCells().size());
  all_nodes.push_back(m_kernel->subDomain()->defaultMesh()->ownNodes().size());
  m_kernel->parallel()->allGather(all_cells,gathered_nb_cells);
  m_kernel->parallel()->allGather(all_nodes,gathered_nb_nodes);

  vector<int> all;
  vector<int> m_swap_cell;
  ENUMERATE_GROUP(cell,m_kernel->subDomain()->defaultMesh()->ownCells()){
    all.push_back(cell->uniqueId());
  }
  m_kernel->parallel()->allGatherVariable(all,m_swap_cell);
  all.clear();
  vector<int> m_swap_node;
  ENUMERATE_GROUP(node,m_kernel->subDomain()->defaultMesh()->ownNodes()){
    all.push_back(node->uniqueId());
  }
  m_kernel->parallel()->allGatherVariable(all,m_swap_node);
  
  int cell_offset=m_swap_cell.size();

  m_swap.resize(m_swap_cell.size()+m_swap_node.size());
  int iCell=0;
  for(int i=0;i<m_kernel->size();++i){
    int offset=m_kernel->topology()->gathered_nb_row(i);
    for(int j=0;j<gathered_nb_cells.at(i);++j){
      m_swap[offset+j]=m_swap_cell.at(iCell);
      iCell+=1;
    }
  }
  int iNode=0;
  for(int i=0;i<m_kernel->size();++i){
    int offset=0;
    if (i>0) offset=m_kernel->topology()->gathered_nb_row(i);
    offset+=gathered_nb_cells.at(i);
    for(int j=0;j<gathered_nb_nodes.at(i);++j){
      m_swap[offset+j]=cell_offset+m_swap_node.at(iNode);
      iNode+=1;
    }
  }
}


void AlephOrdering::initTwiceCellNodeOrder(void){
  debug()<<"\t[AlephOrdering::initTwiceCellNodeOrder] "<<m_kernel->topology()->gathered_nb_row(m_kernel->size());

  vector<int> all_cells;
  vector<int> all_nodes;
  vector<int> gathered_nb_cells(m_kernel->size());
  vector<int> gathered_nb_nodes(m_kernel->size());
  all_cells.push_back(m_kernel->subDomain()->defaultMesh()->ownCells().size());
  all_nodes.push_back(m_kernel->subDomain()->defaultMesh()->ownNodes().size());
  m_kernel->parallel()->allGather(all_cells,gathered_nb_cells);
  m_kernel->parallel()->allGather(all_nodes,gathered_nb_nodes);

  vector<int> all;
  vector<int> m_swap_cell;
  ENUMERATE_GROUP(cell,m_kernel->subDomain()->defaultMesh()->ownCells()){
    all.push_back(2*cell->uniqueId());
    all.push_back(2*cell->uniqueId()+1);
  }
  m_kernel->parallel()->allGatherVariable(all,m_swap_cell);
  all.clear();
  vector<int> m_swap_node;
  ENUMERATE_GROUP(node,m_kernel->subDomain()->defaultMesh()->ownNodes()){
    all.push_back(2*node->uniqueId());
    all.push_back(2*node->uniqueId()+1);
  }
  m_kernel->parallel()->allGatherVariable(all,m_swap_node);
  
  int cell_offset=m_swap_cell.size();

  m_swap.resize(m_swap_cell.size()+m_swap_node.size());
  int iCell=0;
  for(int i=0;i<m_kernel->size();++i){
    int offset=m_kernel->topology()->gathered_nb_row(i);
    for(int j=0;j<gathered_nb_cells.at(i);++j){
      m_swap[offset+j]=m_swap_cell.at(iCell);
      iCell+=1;
    }
  }
  int iNode=0;
  for(int i=0;i<m_kernel->size();++i){
    int offset=0;
    if (i>0) offset=m_kernel->topology()->gathered_nb_row(i);
    offset+=gathered_nb_cells.at(i);
    for(int j=0;j<gathered_nb_nodes.at(i);++j){
      m_swap[offset+j]=cell_offset+m_swap_node.at(iNode);
      iNode+=1;
    }
  }
}
