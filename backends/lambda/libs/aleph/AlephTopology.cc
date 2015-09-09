///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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
// * Minimal AlephTopology for AlephIndexing
// *****************************************************************************
AlephTopology::AlephTopology(AlephKernel *kernel): TraceAccessor(kernel->parallel()->traceMng()),
                                                   m_kernel(kernel),
                                                   m_nb_row_size(0),
                                                   m_nb_row_rank(0),
                                                   m_gathered_nb_row_elements(0),
                                                   m_created(false),
                                                   m_has_set_row_nb_elements(false),
                                                   m_has_been_initialized(false)
{
  debug()<<"\33[1;32m\t[AlephTopology::AlephTopology] Loading MINIMALE AlephTopology"<<"\33[0m";
}


/******************************************************************************
 *****************************************************************************/
AlephTopology::AlephTopology(ITraceMng *tm,
                             AlephKernel *kernel,
                             int nb_row_size,
                             int nb_row_rank): TraceAccessor(tm),
                                                   m_kernel(kernel),
                                                   m_nb_row_size(nb_row_size),
                                                   m_nb_row_rank(nb_row_rank),
                                                   m_gathered_nb_row_elements(),
                                                   m_created(false),
                                                   m_has_set_row_nb_elements(false),
                                                   m_has_been_initialized(true){
  debug()<<"\33[1;32m\t[AlephTopology::AlephTopology] Loading AlephTopology"<<"\33[0m";
  
  m_gathered_nb_setValued.resize(m_kernel->size());
  m_gathered_nb_row.resize(m_kernel->size()+1);
  m_gathered_nb_row[0]=0;
    
  if (!m_kernel->isParallel()){
    m_gathered_nb_row[1] = m_nb_row_size;
    debug() << "\33[1;32m\t[AlephTopology::AlephTopology] SEQ done"<<"\33[0m";
    return;
  }

  if (m_kernel->isAnOther()){
    debug() << "\33[1;32m\t[AlephTopology::AlephTopology] receiving m_gathered_nb_row"<<"\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_row,0);
    return;
  }
  
  debug() << "\33[1;32m\t[AlephTopology::AlephTopology] Nous nous échangeons les indices locaux des lignes de la matrice"<<"\33[0m";
  vector<int> all_rows;
  vector<int> gathered_nb_row(m_kernel->size());
  all_rows.push_back(m_nb_row_rank);
  m_kernel->parallel()->allGather(all_rows,gathered_nb_row);
  for(int iCpu=0;iCpu<m_kernel->size();++iCpu){
    m_gathered_nb_row[iCpu+1]=m_gathered_nb_row[iCpu]+gathered_nb_row[iCpu];
    debug() << "\33[1;32m\t\t[AlephTopology::AlephTopology] "<<iCpu<<":"<<m_gathered_nb_row[iCpu]<<"\33[0m";
  }
  debug() << "\33[1;32m\t[AlephTopology::AlephTopology] m_parallel_info_partitioning done"<<"\33[0m";

  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()){
    debug() << "\33[1;32m\t[AlephTopology::AlephTopology] sending m_gathered_nb_row"<<"\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_row,0);
  }
}


/******************************************************************************
 *****************************************************************************/
AlephTopology::~AlephTopology(){
  debug() << "\33[1;5;32m\t[~AlephTopology]"<<"\33[0m";
}


/******************************************************************************
 * b1e13efe
 *****************************************************************************/
void AlephTopology::create(int setValue_idx){
  if (m_created) return;
  m_created=true;
  
  checkForInit();
  
  if (!m_kernel->isParallel()){
    debug() << "\33[1;32m\t\t\t[AlephTopology::create] SEQ m_gathered_nb_setValued[0]="<<setValue_idx<<"\33[0m";
    m_gathered_nb_setValued[0]=setValue_idx;
    return;
  }
  
  debug() << "\33[1;32m\t\t\t[AlephTopology::create]"<<"\33[0m";
  if (m_kernel->isAnOther()){
    debug() << "\33[1;32m\t[AlephTopology::create] receiving m_gathered_nb_setValued"<<"\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_setValued,0);
    return;
  }

  // Nous allons nous échanger tous les setValue_idx
  vector<int> all;
  all.push_back(setValue_idx);
  m_kernel->parallel()->allGather(all,m_gathered_nb_setValued);

  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()){
    debug() << "\33[1;32m\t[AlephTopology::create] sending m_gathered_nb_setValued"<<"\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_setValued,0);
  }
  debug() << "\33[1;32m\t\t\t[AlephTopology::create] done"<<"\33[0m";
}

  
/******************************************************************************
 * 1b264c6c
 * Ce row_nb_element est positionné afin d'aider à la construction de la matrice
 * lors des :
 *     - 'init_length' de Sloop
 *     - HYPRE_IJMatrixSetRowSizes
 *     - Trilinos Epetra_CrsMatrix
 *****************************************************************************/
void AlephTopology::setRowNbElements(vector<int> row_nb_element){
  checkForInit();
  
  
  if (m_has_set_row_nb_elements) return;
  m_has_set_row_nb_elements=true;
  debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements]"<<"\33[0m";
  
  // Nous allons nous échanger les nombre d'éléments par lignes
  debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] resize m_gathered_nb_row_elements to "
          <<m_nb_row_size<<"\33[0m";
  m_gathered_nb_row_elements.resize(m_nb_row_size);
  
  if (m_kernel->isAnOther()){
    debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] isAnOther from 0"<<"\33[0m";
    traceMng()->flush();
    m_kernel->world()->broadcast(m_gathered_nb_row_elements,0);
    debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] done"<<"\33[0m";
    traceMng()->flush();
    return;
  }
  
  if (!m_kernel->isParallel()){
    debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] SEQ, returning\33[0m";
    for( int i=0; i<m_nb_row_rank; ++i)
      m_gathered_nb_row_elements[i]=row_nb_element[i];
    return;
  }

  vector<int> local_row_nb_element(m_nb_row_rank);
  for( int i=0; i<m_nb_row_rank; ++i)
    local_row_nb_element[i]=row_nb_element[i];
  m_kernel->parallel()->allGatherVariable(local_row_nb_element, m_gathered_nb_row_elements);

  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()){
    debug()<<"\33[1;32m\t\t\t[AlephTopology::setRowNbElements] Sending m_gathered_nb_row_elements of size="<<m_gathered_nb_row_elements.size()<<"\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_row_elements,0);
  }
  debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] done"<<"\33[0m";
}

  
/******************************************************************************
 *****************************************************************************/
vector<int> AlephTopology::ptr_low_up_array(){
  debug() << "\33[1;32m\t[AlephTopology::ptr_low_up_array]"<<"\33[0m";
  return vector<int>();
}


/******************************************************************************
 *****************************************************************************/
vector<int> AlephTopology::part(){
  checkForInit();
  //debug() << "\33[1;32m\t[AlephTopology::part]"<<"\33[0m";
  return m_gathered_nb_row;
}


/******************************************************************************
 *****************************************************************************/
IParallelMng* AlephTopology::parallelMng(){
  debug() << "\33[1;32m\t[AlephTopology::parallelMng]"<<"\33[0m";
  return m_kernel->parallel();
}


/******************************************************************************
 *****************************************************************************/
void AlephTopology::rowRange(int& min_row,int& max_row){
  const int rank = m_kernel->rank();
  checkForInit();
  debug() << "\33[1;32m\t[AlephTopology::rowRange] rank="<<rank<<"\33[0m";
  min_row = m_gathered_nb_row[rank];
  max_row = m_gathered_nb_row[rank+1];
  debug() << "\33[1;32m\t[AlephTopology::rowRange] min_row=" << min_row << ", max_row=" << max_row<<"\33[0m";
}


/******************************************************************************
 *****************************************************************************/
int AlephTopology::rowLocalRange(const int index){
  int ilower=-1;
  int iupper=0;
  int range=0;
  checkForInit();
  for( int iCpu=0;iCpu<m_kernel->size();++iCpu){
    if (m_kernel->rank()!=m_kernel->solverRanks(index)[iCpu]) continue;
    if (ilower==-1) ilower=m_kernel->topology()->gathered_nb_row(iCpu);
    iupper=m_kernel->topology()->gathered_nb_row(iCpu+1)-1;
  }
  range=iupper-ilower+1;
  debug()<< "\33[1;32m\t[AlephTopology::rowLocalRange] ilower="<<ilower
         << ", iupper="<<iupper<<", range="<<range<<"\33[0m";
  return range;
}
