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


/******************************************************************************
 *****************************************************************************/
AlephVector::AlephVector(AlephKernel *kernel): TraceAccessor(kernel->parallel()->traceMng()),
                                               m_kernel(kernel),
                                               m_index(kernel->index()),
                                               m_bkp_num_values(0)
{
  if (kernel->isInitialized()==false){
    debug()<<"\33[1;36m[AlephVector::AlephVector] New Aleph vector, but kernel is not initialized!\33[0m";
    return;
  }
  m_ranks=kernel->solverRanks(m_index);
  m_participating_in_solver=kernel->subParallelMng(m_index)!=NULL;
  debug()<<"\33[1;36m[AlephVector::AlephVector] New Aleph Vector\33[0m";
  if (m_kernel->subParallelMng(m_index)==NULL){
    debug()<<"\33[1;36m[AlephVector::AlephVector] Not concerned with this one!\33[0m";
    return;
  }
  debug()<<"\33[1;36m[AlephVector::AlephVector] site size="
         <<m_kernel->subParallelMng(m_index)->commSize()
         <<" @"
         <<m_kernel->subParallelMng(m_index)->commRank()<<"\33[0m";
  m_implementation=m_kernel->factory()->GetVector(m_kernel,m_index);
}


/******************************************************************************
 *****************************************************************************/
AlephVector::~AlephVector(){
  debug()<<"\33[1;36m\t\t[~AlephVector]\33[0m";
}


/******************************************************************************
 * 6bdba30a
 *****************************************************************************/
void AlephVector::create(void){
  if (m_kernel->isInitialized()==false) return;
  if (m_kernel->configured()){
    debug()<<"\33[1;36m[AlephVector::create] kernel is configured, returning"<<"\33[0m";
    return;
  }
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()){
    m_kernel->world()->broadcast(vector<unsigned long>(1,0x6bdba30al),0);
  }
  // Si nous sommes dans le cas parallèle, mais que nous ne sommes pas le solver, on se dimensionne localement
  if (m_kernel->isParallel()){// && (!m_participating_in_solver)){
    debug()<<"\33[1;36m[AlephVector::create] // resizing idx & val to "<<m_kernel->topology()->nb_row_rank()<<"\33[0m";
	 m_aleph_vector_buffer_idx.resize(m_kernel->topology()->nb_row_rank());
	 m_aleph_vector_buffer_val.resize(m_kernel->topology()->nb_row_rank());
  }
  if (m_participating_in_solver){
    debug()<<"\33[1;36m[AlephVector::create] Participating in solver!"<<"\33[0m";
    // Initialisation des buffers du solver
    // On en profite pour initialiser en même temps les indices
    for(int iCpu=0,idx=0;iCpu<m_kernel->size();++iCpu){
      debug()<<"\33[1;36m[AlephVector::create] m_kernel->rank()="<<m_kernel->rank()<<", m_ranks["<<iCpu<<"]="<<m_ranks[iCpu]<<"\33[0m";
      if (m_kernel->rank()!=m_ranks[iCpu]) continue;
      int nbRowsInCpu=m_kernel->topology()->gathered_nb_row(iCpu+1)-m_kernel->topology()->gathered_nb_row(iCpu);
      debug()<<"\33[1;36m[AlephVector::create] Adding nb_row[iCpu="<<iCpu<<"]="<<nbRowsInCpu<<"\33[0m";
      for(int i=0;i<nbRowsInCpu;++i,idx+=1){
        m_aleph_vector_buffer_idxs.push_back(m_kernel->ordering()->swap(m_kernel->topology()->gathered_nb_row(iCpu)+i));
        //debug()<<"\33[1;36m[AlephVector::create] idxs:"<<idx<<"="<<m_aleph_vector_buffer_idxs[idx]<<"\33[0m";
      }
    }
    m_aleph_vector_buffer_vals.resize(m_aleph_vector_buffer_idxs.size());
    //m_aleph_vector_buffer_vals.fill(0.);
    debug()<<"\33[1;36m[AlephVector::create] resizing m_aleph_vector_buffer_[vals&idxs] to "<<m_aleph_vector_buffer_vals.size()<<"\33[0m"; 
  }else{
    // En parallèle, sans solveur, on a pas besoin de vecteur
    debug()<<"\33[1;36m[AlephVector::create] not participating, returning before 'create_really'"<<"\33[0m";
    return;
  }
  // Dans les autres cas, oui
  create_really();
  debug() << "\33[1;36m[AlephVector::create] done"<<"\33[0m";
}


// ****************************************************************************
// * setLocalComponents - without global indices
// ****************************************************************************
void AlephVector::setLocalComponents(vector<double> values){
  vector<int> indexs;
  if (m_bkp_num_values==0){
    int row_offset=0;
    if (m_kernel->isInitialized())
       row_offset=m_kernel->topology()->part()[m_kernel->rank()];
    debug()<<"\33[1;36m[AlephVector::setLocalComponents] m_bkp_num_values==0"<<"\33[0m";
    indexs.resize(0);
    for(int i=0,iMx=values.size();i<iMx;i+=1)
      indexs.push_back(i+row_offset);
  }
  setLocalComponents(values.size(), indexs, values);
}


// ****************************************************************************
// * reSetLocalComponents
// ****************************************************************************
void AlephVector::reSetLocalComponents(AlephVector *from){
  const int row_offset=m_kernel->topology()->part()[m_kernel->rank()];
  debug()<<"\33[1;36m[AlephVector::reSetLocalComponents] Patching with row_offset="<<row_offset<<"\33[0m";
  for(int i=0,mx=from->m_bkp_num_values;i<mx;i+=1)
    from->m_bkp_indexs[i]+=row_offset;
  setLocalComponents(from->m_bkp_num_values,from->m_bkp_indexs,from->m_bkp_values);
}


// ****************************************************************************
// * setLocalComponents - standard
// ****************************************************************************
void AlephVector::setLocalComponents(int num_values,
												 vector<int> indexs,
												 vector<double> values){
  if (!m_kernel->isInitialized()){
    debug()<<"\33[1;36m[AlephVector::setLocalComponents] Trying to setLocalComponents from an uninitialized kernel!\33[0m";
    m_bkp_num_values=num_values;
    debug()<<"\33[1;36m[AlephVector::setLocalComponents] Backuping "<<num_values<<" values and indexes!\33[0m";
    m_bkp_indexs=indexs;//.copy(indexs);
    m_bkp_values=values;//.copy(values);
    return;
  }
  if (!m_kernel->isParallel()){
    debug() << "\33[1;36m[AlephVector::setLocalComponents] implementation AlephVectorSet, num_values="<<num_values<<"\33[0m";
	 m_implementation->AlephVectorSet(&values[0],
                                     &indexs[0],
                                     num_values);
    return;
  }
  debug() << "\33[1;36m[AlephVector::setLocalComponents]\33[0m";
  for( int i=0, is=num_values; i<is; ++i ){
    // Si je suis sur le site de résolution, je place les values dans les 'vals'
    if (m_participating_in_solver && (m_kernel->rank()==m_ranks[m_kernel->rank()])){
      if (!m_kernel->isCellOrdering()){
        // Dans le cas multi-site de résolution, on shift avec la base m_aleph_vector_buffer_idxs[0]
        //debug() << "\33[1;36m[AlephVector::setLocalComponents] multi-site de résolution"<<", indexs["<<i<<"]="<<indexs[i]<<", m_aleph_vector_buffer_idxs[0]="<<m_aleph_vector_buffer_idxs[0]<<"\33[0m";
        m_aleph_vector_buffer_vals[indexs[i]-m_aleph_vector_buffer_idxs[0]]=values[i];
      }else{
        // Dans le cas de l'ordering, on est mono site, on veut pas de shift
        m_aleph_vector_buffer_vals[indexs[i]]=values[i];
     }
    }else{
      m_aleph_vector_buffer_idx[i]=m_kernel->ordering()->swap(indexs[i]);
		m_aleph_vector_buffer_val[i]=values[i];
    }
  }
}


/******************************************************************************
 *****************************************************************************/
void AlephVector::create_really(void){
  debug()<<"\33[1;36m[AlephVector::create_really] New UNconfigured vector"<<"\33[0m";
  m_implementation->AlephVectorCreate();
}

  
/******************************************************************************
 * ec7a979f
 *****************************************************************************/
void AlephVector::assemble(void){  
  if (!m_kernel->isInitialized()){
    debug()<<"\33[1;36m[AlephVector::AlephVector] Trying to assemble a vector from an uninitialized kernel!\33[0m";
    return;
  }
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther())
    m_kernel->world()->broadcast(vector<unsigned long>(1,0xec7a979fl),0);
  // Si on est en mode seq, on a rien à faire
  if (!m_kernel->isParallel()) return;
  if (m_participating_in_solver){
	 // Si je suis le solveur, je recv le reste des vecteurs provenant d'autres coeurs
    debug()<<"\33[1;36m[AlephVector::assemble] m_participating_in_solver"<<"\33[0m";
	 int nbRows;
    for(int iCpu=0,iRows=0;iCpu<m_kernel->size();++iCpu,iRows+=nbRows){
      nbRows=0;
      if (m_kernel->rank()!=m_ranks[iCpu]) continue;
      // Dans les autres cas, il faut décaler l'offset iRows
      nbRows=m_kernel->topology()->gathered_nb_row(iCpu+1)-m_kernel->topology()->gathered_nb_row(iCpu);
		if (iCpu==m_kernel->rank()) continue;
 		debug() << "\33[1;36m[AlephVector::assemble]"
              <<" recv "<<m_kernel->rank()
              <<" <= "<<iCpu
              <<", offset="<<iRows
              <<" for "<<nbRows<<"\33[0m";
		m_parallel_requests.push_back(m_kernel->world()->recv(&m_aleph_vector_buffer_vals[iRows],//,nbRows),
                                                      iCpu,
                                                      false));
	 }
  }
  if (m_kernel->rank()!=m_ranks[m_kernel->rank()] && !m_kernel->isAnOther()){ // Dans ce cas, il faut envoyer ses données
	 debug() << "\33[1;36m[AlephVector::assemble]"
            <<" send "<<m_kernel->rank()
            <<" => "<<m_ranks[m_kernel->rank()]
            <<" for "<<m_aleph_vector_buffer_val.size()<<"\33[0m";
	 m_parallel_requests.push_back(m_kernel->world()->send(m_aleph_vector_buffer_val,
                                                          m_ranks[m_kernel->rank()],
                                                          false));
  }
  debug() << "\33[1;36m[AlephVector::assemble] done"<<"\33[0m";
}


/******************************************************************************
 *****************************************************************************/
void AlephVector::assemble_waitAndFill(void){
  if (!m_kernel->isParallel()) return;
  debug() << "\33[1;36m[AlephVector::assemble_waitAndFill] wait for "
          << m_parallel_requests.size()
          <<" Requests for solver "
          <<m_index<<"\33[0m";
  m_kernel->world()->waitAllRequests(m_parallel_requests);
  m_parallel_requests.clear();
  if (!m_participating_in_solver){
    debug() << "\33[1;36m[AlephVector::assemble_waitAndFill] Not participating in solver, returning"<<"\33[0m";
    return;
  }
  debug() << "\33[1;36m[AlephVector::assemble_waitAndFill] "<<m_index <<" locfill"<<"\33[0m";
  m_implementation->AlephVectorSet(&m_aleph_vector_buffer_vals.front(),
                                   &m_aleph_vector_buffer_idxs.front(),
                                   m_aleph_vector_buffer_idxs.size());
  debug() << "\33[1;36m[AlephVector::assemble_waitAndFill] "<<m_index<<" VECTOR ASSEMBLE"<<"\33[0m";
  m_implementation->AlephVectorAssemble();
  debug() << "\33[1;36m[AlephVector::assemble_waitAndFill] done"<<"\33[0m";
}


/******************************************************************************
 *****************************************************************************/
void AlephVector::reassemble(void){
  if (!m_kernel->isParallel()) return;
  // Cas où le site de résolution n'est pas le notre, on pousse les données dans le 'val'
  if (m_kernel->rank()!=m_ranks[m_kernel->rank()] && !m_kernel->isAnOther()){
	 debug() << "\33[1;36m[AlephVector::REassemble] "
            << m_kernel->rank()
            << "<="
            << m_ranks[m_kernel->rank()]<<"\33[0m";
	 m_parallel_reassemble_requests.push_back(m_kernel->world()->recv(m_aleph_vector_buffer_val, m_ranks[m_kernel->rank()], false));
  }
  // Si je suis un site de résolution, je résupère les résultats
  if (m_participating_in_solver){
    // Je récupère déjà les valeurs depuis le vecteur
    debug() << "\33[1;36m[AlephVector::REassemble] J'ai participé, je récupère les résultats depuis l'implémentation"<<"\33[0m";
    m_implementation->AlephVectorGet(&m_aleph_vector_buffer_vals.front(),
                                     &m_aleph_vector_buffer_idxs.front(),
                                     m_aleph_vector_buffer_idxs.size());
    // Puis je send les parties des vecteurs aux autres sites
    int nbRows;
    for(int iCpu=0,iRows=0;iCpu<m_kernel->size();++iCpu,iRows+=nbRows){
      nbRows=0;
      if (m_kernel->rank()!=m_ranks[iCpu]) continue;
      nbRows=m_kernel->topology()->gathered_nb_row(iCpu+1)-m_kernel->topology()->gathered_nb_row(iCpu);
      if (iCpu==m_kernel->rank()) continue;
      debug() << "\33[1;36m[AlephVector::REassemble] send "
              << m_kernel->rank()
              << " => " << iCpu
              << ", offset=" << iRows
              << " for " << nbRows<<"\33[0m";
      m_parallel_reassemble_requests.push_back(
    m_kernel->world()->send(&m_aleph_vector_buffer_vals[iRows],//nbRows),
                            iCpu,
                            false));
    }
  }
}


/******************************************************************************
 *****************************************************************************/
void AlephVector::reassemble_waitAndFill(void){
  if (!m_kernel->isParallel()) return;
  m_kernel->world()->waitAllRequests(m_parallel_reassemble_requests);
  m_parallel_reassemble_requests.clear();
  debug() << "\33[1;36m[AlephVector::REassemble_waitAndFill]"<<"\33[0m";
}


/******************************************************************************
 *****************************************************************************/
void AlephVector::getLocalComponents(int vector_size,
												 vector<int> global_indice,
												 vector<double> vector_values){
  debug() << "\33[1;36m[AlephVector::getLocalComponents] vector_size="<<vector_size<<"\33[0m";
  if (!m_kernel->isParallel()){
    // En séquentiel, on va piocher les résultats directement
	 m_implementation->AlephVectorGet(&vector_values.front(),
                                     &global_indice.front(),
                                     vector_size);
    debug() << "\33[1;36m[AlephVector::getLocalComponents] seq done!\33[0m";
    return;
  }
  // En parallèle, on va piocher dans nos buffers qui devraient être à jour
  for( int i=0; i<vector_size; ++i ) {
    // Dans le cas où nous sommes sur notre site de résolution, les 'vals' sont nos cibles
    if (m_participating_in_solver && (m_kernel->rank()==m_ranks[m_kernel->rank()])){
      if (!m_kernel->isCellOrdering()){
        vector_values[i] = m_aleph_vector_buffer_vals[global_indice[i]-m_aleph_vector_buffer_idxs[0]];
      }else{
        vector_values[i] = m_aleph_vector_buffer_vals[global_indice[i]];
      }
    }
    // Dans le cas où le site de résolution est distant, on vient dans les 'val'
    else{
      vector_values[i] = m_aleph_vector_buffer_val[i];
    }
  }
  debug() << "\33[1;36m[AlephVector::getLocalComponents] parallel done!\33[0m";
}


// ****************************************************************************
// * getLocalComponents
// ****************************************************************************
void AlephVector::getLocalComponents(vector<double> &vector_values){
  const int vector_size=m_kernel->topology()->nb_row_rank();
  vector_values.resize(vector_size);
  m_aleph_vector_buffer_idx.resize(vector_size);
  debug() << "\33[1;36m[getLocalComponents(vector_values)] vector_size="<<vector_size<<"\33[0m";
  if (!m_kernel->isParallel()){
    for(int i=0;i<vector_size;i+=1){
      m_aleph_vector_buffer_idx[i]=i;
      //debug()<<"\33[1;36mm_aleph_vector_buffer_idx["<<i<<"]="<<m_aleph_vector_buffer_idx[i]<<"\33[0m";
    }
    // En séquentiel, on va piocher les résultats directement
	 m_implementation->AlephVectorGet(&vector_values.front(),
                                     &m_aleph_vector_buffer_idx.front(),
                                     vector_size);
    debug() << "\33[1;36m[getLocalComponents(vector_values)] seq done!\33[0m";
    return;
  }
  // En parallèle, on va piocher dans nos buffers qui devraient être à jour
  for( int i=0; i<vector_size; ++i ) {
    // Dans le cas où nous sommes sur notre site de résolution, les 'vals' sont nos cibles
    if (m_participating_in_solver && (m_kernel->rank()==m_ranks[m_kernel->rank()])){
      vector_values[i] = m_aleph_vector_buffer_vals[i];
    }else{ // Dans le cas où le site de résolution est distant, on vient dans les 'val'
      vector_values[i] = m_aleph_vector_buffer_val[i];
    }
  }
  debug() << "\33[1;36m[getLocalComponents(vector_values)] parallel done!\33[0m";
}


/******************************************************************************
 *****************************************************************************/
void AlephVector::startFilling(){
  debug()<<"\33[1;36m[AlephVector::startFilling]"<<"\33[0m";
}


/******************************************************************************
 *****************************************************************************/
void AlephVector::writeToFile(const string base_file_name){
  debug()<<"\33[1;36m[AlephVector::writeToFile]"<<"\33[0m";
  m_implementation->writeToFile(base_file_name);
}

