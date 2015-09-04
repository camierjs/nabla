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
AlephMatrix::AlephMatrix(AlephKernel *kernel): TraceAccessor(kernel->parallel()->traceMng()),
                                               m_kernel(kernel),
                                               m_index(kernel->index()),
                                               m_setValue_idx(0),
                                               m_addValue_idx(0)
{
  if (kernel->isInitialized()==false){
    debug()<<"\33[1;32m[AlephMatrix::AlephMatrix] New Aleph matrix, but kernel is not initialized!\33[0m";
    return;
  }
  // Récupération des rangs utilisés pour cette résolution
  m_ranks=kernel->solverRanks(m_index);
  // Booléen pour savoir si on participe ou pas
  m_participating_in_solver=kernel->subParallelMng(m_index)!=NULL;
  debug()<<"\33[1;32m[AlephMatrix::AlephMatrix] New Aleph matrix\33[0m";
  if (!m_participating_in_solver){
    debug()<<"\33[1;32m[AlephMatrix::AlephMatrix] Not concerned by this one!\33[0m";
    return;
  }
  debug()<<"\33[1;32m[AlephMatrix::AlephMatrix] site size="
         << m_kernel->subParallelMng(m_index)->commSize()
         << " @"
         << m_kernel->subParallelMng(m_index)->commRank()<<"\33[0m";
  // On va chercher une matrice depuis la factory qui fait l'interface aux bibliothèques externes
  m_implementation=m_kernel->factory()->GetMatrix(m_kernel,m_index);
  traceMng()->flush();
}

  
/******************************************************************************
 *****************************************************************************/
AlephMatrix::~AlephMatrix(){
  debug()<<"\33[1;32m\t\t[~AlephMatrix]\33[0m";
  rowColMap::const_iterator i = m_row_col_map.begin();
  for(;i!=m_row_col_map.end();++i)
    delete i->second;
}


/******************************************************************************
 * Matrix 'create' avec l'API 'void'
 * BaseForm[Hash["AlephMatrix::create(void)", "CRC32"], 16] = fff06e2
 *****************************************************************************/
void AlephMatrix::create(void){
  debug()<<"\33[1;32m[AlephMatrix::create(void)]\33[0m";
  // Si le kernel n'est pas initialisé, on a rien à faire
  if (!m_kernel->isInitialized()) return;
  // S'il y a des 'autres' et qu'on en fait pas partie,
  // on broadcast qu'un 'create' est à faire
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther())
    m_kernel->world()->broadcast(vector<unsigned long>(1,0xfff06e2l),0);
  // On flush en prévision du remplissage, il faut le faire même étant configuré
  // Par contre, on ne flush pas celui du addValue
  m_setValue_idx=0;
}


/******************************************************************************
 * Matrix 'create' avec l'API qui spécifie le nombre d'éléments non nuls pas lignes
 * BaseForm[Hash["AlephMatrix::create(intConstArrayView,bool)", "CRC32"], 16] = 5c3111b1
 *****************************************************************************/
void AlephMatrix::create(vector<int> row_nb_element,
                         bool has_many_elements){
  debug()<<"\33[1;32m[AlephMatrix::create(old API)] API with row_nb_element + has_many_elements\33[0m";
  this->create();
}


/*!
 * \brief reset pour flusher les tableaux des [set&add]Value
 */
void AlephMatrix::reset(void){
  debug()<<"\33[1;32m[AlephMatrix::reset]\33[0m";
  m_setValue_val.assign(m_setValue_val.size(),0.0);
  m_addValue_val.assign(m_addValue_val.size(),0.0);
}


/*!
 * \brief addValue à partir d'arguments en IVariables, items et double
 *****************************************************************************/
void AlephMatrix::addValue(const Variable &rowVar, const item* &rowItm,
                           const Variable &colVar, const item* &colItm,
                           const double val){
  addValue(rowVar,*rowItm,colVar,*colItm,val);
}
void AlephMatrix::addValue(const Variable &rowVar, const item &rowItm,
                           const Variable &colVar, const item &colItm,
                           const double val){
  int row=m_kernel->indexing()->get(rowVar,rowItm);
  int col=m_kernel->indexing()->get(colVar,colItm);
  if (m_kernel->isInitialized()){
    const int row_offset=m_kernel->topology()->part()[m_kernel->rank()];
    row+=row_offset;
    col+=row_offset;
  }
  //debug()<<"[AlephMatrix::addValue] IVariable/item add @ ["<<row<<","<<col<<"]="<<val;
  addValue(row,col,val);
}


void AlephMatrix::updateKnownRowCol(int row,
                                    int col,
                                    double val){
  //debug()<<"\33[1;32m[AlephMatrix::updateKnownRowCol]\33[0m";
  m_addValue_row.push_back(row);
  m_addValue_col.push_back(col);
  m_addValue_val.push_back(val);
  m_addValue_idx+=1;
  // On fait de même coté 'set' pour avoir la bonne taille
  m_setValue_row.push_back(row);
  m_setValue_col.push_back(col);
  m_setValue_val.push_back(0.);
}

void AlephMatrix::rowMapMapCol(int row,
                               int col,
                               double val){
  rowColMap::const_iterator iRowMap = m_row_col_map.find(row);
  // Si la row n'est même pas encore connue
  // On rajoute une entrée map(map(m_addValue_idx))
  if (iRowMap==m_row_col_map.end()){
    colMap *jMap=new colMap();
    /*debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] row "
          <<row<<" inconue, m_addValue_idx="
          <<m_addValue_idx<<"\33[0m";*/
    m_row_col_map.insert(std::make_pair(row,jMap));
    jMap->insert(std::make_pair(col,m_addValue_idx));
    updateKnownRowCol(row,col,val);
    return;
  }
  // On focus sur la seconde dimension
  colMap *jMap = iRowMap->second;
  colMap::const_iterator iColMap = jMap->find(col);
  // Si cet col n'est pas connue de cette row
  // On rajoute une entrée
  if (iColMap==jMap->end()){
    /*debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] col "
          <<col<<" inconue, m_addValue_idx="
          <<m_addValue_idx<<"\33[0m";*/
    jMap->insert(std::make_pair(col,m_addValue_idx));
    updateKnownRowCol(row,col,val);
    return;
  }
  // Sinon on ajoute
  //debug()<<"\33[1;32m[AlephMatrix::rowMapMapCol] hit\33[0m";
  //debug()<<"[AlephMatrix::rowMapMapCol] += for ["<<row<<","<<col<<"]="<<val; traceMng()->flush();
  m_addValue_val[iColMap->second]+=val;
}

/*!
 * \brief addValue standard en (i,j,val)
 */
void AlephMatrix::addValue(int row, int col, double val){
  //debug()<<"\33[32m[AlephMatrix::addValue] addValue("<<row<<","<<col<<")="<<val<<"\33[0m";
  row=m_kernel->ordering()->swap(row);
  col=m_kernel->ordering()->swap(col);
  // Recherche de la case (row,j) si elle existe déjà
  rowMapMapCol(row,col,val);
}


/*!
 * \brief setValue à partir d'arguments en IVariables, item* et double
 */
void AlephMatrix::setValue(const Variable &rowVar, const item* &rowItm,
                           const Variable &colVar, const item* &colItm,
                           const double val){
  setValue(rowVar,*rowItm,colVar,*colItm,val);
}


/*!
 * \brief setValue à partir d'arguments en IVariables, items et double
 */
void AlephMatrix::setValue(const Variable &rowVar, const item &rowItm,
                           const Variable &colVar, const item &colItm,
                           const double val){
  int row=m_kernel->indexing()->get(rowVar,rowItm);
  int col=m_kernel->indexing()->get(colVar,colItm);
  //debug()<<"[AlephMatrix::setValue] dof #"<<m_setValue_idx<<" ["<<row<<","<<col<<"]="<<val;
  if (m_kernel->isInitialized()){
    const int row_offset=m_kernel->topology()->part()[m_kernel->rank()];
    row+=row_offset;
    col+=row_offset;
  }
  setValue(row,col,val);
}
 

/*!
 * \brief setValue standard à partir d'arguments (row,col,val)
 */
void AlephMatrix::setValue(int row, int col, double val){
  // Re-ordering si besoin
  row=m_kernel->ordering()->swap(row);
  col=m_kernel->ordering()->swap(col);
  // Si le kernel a déjà été configuré,
  // on s'assure que la 'géométrie/support' n'a pas changée entre les résolutions
  if (m_kernel->configured()){
	 if ((m_setValue_row[m_setValue_idx] != row) ||
        (m_setValue_col[m_setValue_idx] != col))
      throw std::logic_error("[Aleph::setValue] Row|Col have changed!");
    m_setValue_row[m_setValue_idx]=row;
    m_setValue_col[m_setValue_idx]=col;
    m_setValue_val[m_setValue_idx]=val;
  }else{
    m_setValue_row.push_back(row);
    m_setValue_col.push_back(col);
    m_setValue_val.push_back(val);
  }
  m_setValue_idx+=1;
}


/*!
 * \brief reIdx recherche la correspondance de l'AlephIndexing
 */
int AlephMatrix::reIdx(int ij,     
                         vector<int*>&known_items_own_address){
  return *known_items_own_address[ij];
}


/*!
 * \brief reSetValuesIn rejoue les setValue avec les indexes calculés via l'AlephIndexing
 */
void AlephMatrix::reSetValuesIn(AlephMatrix *thisMatrix,
                                vector<int*> &known_items_own_address){
  for(int k=0,kMx=m_setValue_idx;k<kMx;k+=1){
    int i=reIdx(m_setValue_row[k], known_items_own_address);
    int j=reIdx(m_setValue_col[k], known_items_own_address);
    thisMatrix->setValue(i,j,m_setValue_val[k]);
  }
}


/*!
 * \brief reAddValuesIn rejoue les addValue avec les indexes calculés via l'AlephIndexing
 */
void AlephMatrix::reAddValuesIn(AlephMatrix *thisMatrix,
                                vector<int*> &known_items_own_address){
  for(int k=0,kMx=m_addValue_row.size();k<kMx;k+=1){
    const int row=reIdx(m_addValue_row[k], known_items_own_address);
    const int col=reIdx(m_addValue_col[k], known_items_own_address);
    const double val=m_addValue_val[k];
    thisMatrix->addValue(row,col,val);
  }
}


/*!
 * \brief assemble les matrices avant résolution
 */
void AlephMatrix::assemble(void){
  // Si le kernel n'est pas initialisé, on ne fait toujours rien
  if (!m_kernel->isInitialized()){
    debug()<<"\33[1;32m[AlephMatrix::assemble] Trying to assemble a matrix"
           <<"from an uninitialized kernel!\33[0m";
    return;
  }
  // Si aucun [set|add]Value n'a été perçu, ce n'est pas normal
  if (m_addValue_idx!=0 && m_setValue_idx!=0)
    throw std::logic_error("[AlephMatrix::assemble] Still exclusives [add||set]Value required!");
  // Si des addValue ont été captés, il faut les 'rejouer'
  // Attention: pour l'instant les add et les set sont disjoints!
  if (m_addValue_idx!=0){
    debug()<<"\33[1;32m[AlephMatrix::assemble] m_addValue_idx!=0\33[0m";
    // On flush notre index des setValues
    m_setValue_idx=0;
    debug()<<"\t\33[32m[AlephMatrix::assemble] Flatenning addValues size="<<m_addValue_row.size()<<"\33[0m";
    for(int k=0,kMx=m_addValue_row.size();k<kMx;++k){
      m_setValue_row[k]=m_addValue_row[k];
      m_setValue_col[k]=m_addValue_col[k];
      m_setValue_val[k]=m_addValue_val[k];
      /*debug()<<"\t\33[32m[AlephMatrix::assemble] setValue ("<<m_setValue_row[k]
        <<","<<m_setValue_col[k]<<")="<<m_setValue_val[k]<<"\33[0m";*/
      m_setValue_idx+=1;
    }
  }
  // S'il y a des 'autres' et qu'on en fait pas parti, on les informe de l'assemblage
  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()){
    debug()<<"\33[1;32m[AlephMatrix::assemble] On informe les autres kappa que l'on assemble"<<"\33[0m";
    m_kernel->world()->broadcast(vector<unsigned long>(1,0x74f253cal),0);
    // Et on leur donne l'info du m_setValue_idx
    m_kernel->world()->broadcast(vector<int>(1,m_setValue_idx),0);
  }
  // On initialise la topologie si cela n'a pas été déjà fait
  if (!m_kernel->isAnOther()){
    debug()<<"\33[1;32m[AlephMatrix::assemble] Initializing topology"<<"\33[0m";
    m_kernel->topology()->create(m_setValue_idx);
  }
  // Si on a pas déjà calculé le nombre d'éléments non nuls par lignes
  // c'est le moment de le déclencher
  debug()<<"\33[1;32m[AlephMatrix::assemble] Updating row_nb_element"<<"\33[0m";
  if (!m_kernel->topology()->hasSetRowNbElements()){
    vector<int> row_nb_element;
    row_nb_element.resize(m_kernel->topology()->nb_row_rank());
    row_nb_element.assign(m_kernel->topology()->nb_row_rank(),0);
    // Quand on est pas un Autre, il faut mettre à jour le row_nb_element si cela n'a pas été spécifié lors du matrice->create
    if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()){
      debug()<<"\33[1;32m[AlephMatrix::assemble] Kernel's topology has not set its nb_row_elements, now doing it!"<<"\33[0m";
      const int row_offset=m_kernel->topology()->part()[m_kernel->rank()];
      debug()<<"\33[1;32m[AlephMatrix::assemble] row_offset="<<row_offset<<"\33[0m";
      debug()<<"\33[1;32m[AlephMatrix::assemble] filled, row_nb_element.size="<<row_nb_element.size()<<"\33[0m";
      // On le fait pour l'instant en une passe pour avoir une borne max
      for(int i=0,iMx=m_setValue_row.size();i<iMx;++i)
        row_nb_element[m_setValue_row.at(i)-row_offset]+=1;
    }
    m_kernel->topology()->setRowNbElements(row_nb_element);
    debug()<<"\33[1;32m[AlephMatrix::assemble] done hasSetRowNbElements"<<"\33[0m";    
  }
  // Dans le cas //, le solveur se prépare à récupérer les parties de matrices venant des autres
  debug()<<"\33[1;32m[AlephMatrix::assemble] Récupération des parties de matrices"<<"\33[0m";
  if (m_participating_in_solver && (!m_kernel->configured())){
    vector<int> nbValues(m_kernel->size());
    {
      nbValues.assign(nbValues.size(),0);
      for(int iCpu=0;iCpu<m_kernel->size();++iCpu){
        if (m_kernel->rank()!=m_ranks[iCpu]) continue;
        //debug()<<"\33[1;32m[AlephMatrix::assemble] Adding nb_values from iCpu "<<iCpu<<"\33[0m";
        nbValues[iCpu]=m_kernel->topology()->gathered_nb_setValued(iCpu);
      }
    }
    {
      // Pour l'instant, on ne sait faire que du séquentiel
      assert(!m_kernel->isParallel());
      m_aleph_matrix_buffer_rows.resize(nbValues[0]);
      m_aleph_matrix_buffer_cols.resize(nbValues[0]);
      m_aleph_matrix_buffer_vals.resize(nbValues[0]);
    }
  }
  // Si on est pas en //, on a rien d'autre à faire
  if (!m_kernel->isParallel()) return;
  // Si je participe à la résolution, je reçois les contributions des autres participants
  if (m_participating_in_solver){
    debug()<<"\33[1;32m[AlephMatrix::assemble] I am part of the solver, let's iRecv"<<"\33[0m";
	 // Si je suis le solveur, je recv le reste des matrices provenant soit des autres coeurs, soit de moi-même
	 for(int iCpu=0;iCpu<m_kernel->size();++iCpu){
      // Sauf de moi-même
		if (iCpu==m_kernel->rank()) continue;
      // Sauf de ceux qui ne participent pas
      if (m_kernel->rank()!=m_ranks[iCpu]) continue;
		debug() << "\33[1;32m[AlephMatrix::assemble] "
              <<" recv "<<m_kernel->rank()
              <<" <= "<<iCpu
              <<" size="<<m_aleph_matrix_buffer_cols[iCpu].size()<<"\33[0m";
		m_aleph_matrix_mpi_data_requests.push_back(m_kernel->world()->recv(m_aleph_matrix_buffer_vals[iCpu], iCpu, false));
      // Une fois configuré, nous connaissons tous les (i,j): pas besoin de les renvoyer
		if (!m_kernel->configured()){
		  m_aleph_matrix_mpi_data_requests.push_back(m_kernel->world()->recv(m_aleph_matrix_buffer_rows[iCpu], iCpu, false));
		  m_aleph_matrix_mpi_data_requests.push_back(m_kernel->world()->recv(m_aleph_matrix_buffer_cols[iCpu], iCpu, false));
		}
	 }
  }
  // Si je suis un rang qui a des données à envoyer, je le fais
  if ((m_kernel->rank()!=m_ranks[m_kernel->rank()])&&(!m_kernel->isAnOther())){
	 debug() << "\33[1;32m[AlephMatrix::assemble]"
            <<" send "<<m_kernel->rank()
            <<" => "<<m_ranks[m_kernel->rank()]
            <<" for "<<m_setValue_val.size()<<"\33[0m";
    m_aleph_matrix_mpi_data_requests.push_back(m_kernel->world()->send(m_setValue_val, m_ranks[m_kernel->rank()], false));
    if (!m_kernel->configured()){
      debug()<<"\33[1;32m[AlephMatrix::assemble] iSend my row to "<< m_ranks[m_kernel->rank()]<<"\33[0m";
      m_aleph_matrix_mpi_data_requests.push_back(m_kernel->world()->send(m_setValue_row, m_ranks[m_kernel->rank()], false));
      debug()<<"\33[1;32m[AlephMatrix::assemble] iSend my col to "<< m_ranks[m_kernel->rank()]<<"\33[0m";
      m_aleph_matrix_mpi_data_requests.push_back(m_kernel->world()->send(m_setValue_col, m_ranks[m_kernel->rank()], false));
    }
  }
}


/*!
 * \brief create_really transmet l'ordre de création à la bibliothèque externe
 */
void AlephMatrix::create_really(void){
  //Timer::Action ta(m_kernel->subDomain(),"AlephMatrix::create_really");
  debug()<<"\33[1;32m[AlephMatrix::create_really]"<<"\33[0m";
  // Il nous faut alors dans tous les cas une matrice de travail  
  debug() << "\33[1;32m[AlephMatrix::create_really] new MATRIX"<<"\33[0m";
  // et on déclenche la création au sein de l'implémentation
  m_implementation->AlephMatrixCreate();
  debug()<<"\33[1;32m[AlephMatrix::create_really] done"<<"\33[0m";
}

 
/*!
 * \brief assemble_waitAndFill attend que les requètes précédemment postées aient été traitées
 */
void AlephMatrix::assemble_waitAndFill(void){
  //Timer::Action ta(m_kernel->subDomain(),"AlephMatrix::assemble_waitAndFill");
  debug()<<"\33[1;32m[AlephMatrix::assemble_waitAndFill]"<<"\33[0m";
  if (m_kernel->isParallel()){
    debug()<<"\33[1;32m[AlephMatrix::assemble_waitAndFill] wait for "
           <<m_aleph_matrix_mpi_data_requests.size()<<" Requests"<<"\33[0m";
    m_kernel->world()->waitAllRequests(m_aleph_matrix_mpi_data_requests);
	 m_aleph_matrix_mpi_data_requests.clear();
    debug()<<"\33[1;32m[AlephMatrix::assemble_waitAndFill] clear"<<"\33[0m";
	 if (m_participating_in_solver==false) {
      debug()<<"\33[1;32m[AlephMatrix::assemble_waitAndFill] nothing more to do"<<"\33[0m";
    }
  }
  // Si je ne participe pas, je ne participe pas
  if (!m_participating_in_solver) return;
  // Sinon, on prend le temps de construire la matrice, les autres devraient le faire aussi
  if (!m_kernel->configured()){
    debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] solver "<<m_index<<" create_really"<<"\33[0m";
    create_really();
  }
  // Et on enchaîne alors avec le remplissage de la matrice
  { 
	 if (m_kernel->configured())
		m_implementation->AlephMatrixSetFilled(false);// Activation de la protection de remplissage
	 debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] "<<m_index<<" fill"<<"\33[0m";
    int *bfr_row_implem;
    int *bfr_col_implem;
    double *bfr_val_implem;
 	 for( int iCpu=0;iCpu<m_kernel->size();++iCpu){
      if (m_kernel->rank()!=m_ranks[iCpu]) continue;  
      if (iCpu==m_kernel->rank()) {
        bfr_row_implem  = reinterpret_cast<int*>(&m_setValue_row.front());
        bfr_col_implem  = reinterpret_cast<int*>(&m_setValue_col.front());
        bfr_val_implem  = reinterpret_cast<double*>(&m_setValue_val.front());
        m_implementation->AlephMatrixFill(m_setValue_val.size(),
                                          bfr_row_implem,
                                          bfr_col_implem,
                                          bfr_val_implem);
      }else{
        bfr_row_implem  = reinterpret_cast<int*>(&m_aleph_matrix_buffer_rows[iCpu].front());
        bfr_col_implem  = reinterpret_cast<int*>(&m_aleph_matrix_buffer_cols[iCpu].front());
        bfr_val_implem  = reinterpret_cast<double*>(&m_aleph_matrix_buffer_vals[iCpu].front());
        m_implementation->AlephMatrixFill(m_aleph_matrix_buffer_vals[iCpu].size(),
                                          bfr_row_implem,
                                          bfr_col_implem,
                                          bfr_val_implem);
      }
	 }
  }
  { // On déclare alors la matrice comme remplie, et on lance la configuration
	 m_implementation->AlephMatrixSetFilled(true);// Désactivation de la protection de remplissage
	 if (!m_kernel->configured()){
		debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] "<<m_index<<" MATRIX ASSEMBLE"<<"\33[0m";
      int assrtnd=0;
      assrtnd=m_implementation->AlephMatrixAssemble();
		debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] AlephMatrixAssemble="<<assrtnd<<"\33[0m";
      // throw FatalErrorException("AlephMatrix::assemble_waitAndFill", "configuration failed");
	 }
  }
  debug() << "\33[1;32m[AlephMatrix::assemble_waitAndFill] done"<<"\33[0m";
}


/*!
  \brief 'Poste' le solver au scheduler de façon asynchrone ou pas
*/
void AlephMatrix::solve(AlephVector* x,
                        AlephVector* b,
                        int& nb_iteration,
                        double* residual_norm,
                        AlephParams* solver_param,
                        bool async){
  debug() << "\33[1;32m[AlephMatrix::solve] Queuing solver "<<m_index<<"\33[0m";
  m_kernel->postSolver(solver_param,this,x,b);
  // Si on nous a spécifié le post, on ne déclenche pas le mode synchrone
  if (async) return; 
  debug() << "\33[1;32m[AlephMatrix::solve] SYNCHRONOUS MODE has been requested, syncing!"<<"\33[0m";
  m_kernel->syncSolver(0, nb_iteration, residual_norm);
  return;
}


/*!
 * \brief Résout le système linéraire
 * \param x solution du système Ax=b (en sortie)
 * \param b second membre du système (en entrée)
 * \param nb_iteration nombre d'itérations du système (en sortie)
 * \param residual_norm résidu de convergence du système (en sortie)
 * \param info parametres de l'application parallele (en entrée)
 * \param solver_param Parametres du Solveur du solveur Ax=b (en entrée)
 */
void AlephMatrix::solveNow(AlephVector* x,
                           AlephVector* b,
                           AlephVector* tmp,
                           int& nb_iteration,
                           double* residual_norm,
                           AlephParams* params){
  const bool dump_to_compare=
    (m_index==0) &&                                 // Si on est à la première résolution
    (m_kernel->rank()==0) &&                        // et qu'on est le 'master'
    (params->writeMatrixToFileErrorStrategy()) &&   // et qu'on a demandé un write_matrix !
    //(m_kernel->subDomain()->commonVariables().globalIteration()==1) && // et la première itération ou la deuxième
    (m_kernel->nbRanksPerSolver()==1);
  if (!m_participating_in_solver){
   debug()<<"\33[1;32m[AlephMatrix::solveNow] Nothing to do here!"<<"\33[0m";
   return;
  }
  debug()<<"\33[1;32m[AlephMatrix::solveNow]"<<"\33[0m";
  if (dump_to_compare){
    int globalIteration = 0;//m_kernel->subDomain()->commonVariables().globalIteration();
    string mtxFilename("m_aleph_matrix_A_");// + globalIteration;
    string rhsFilename("m_aleph_vector_b_");// + globalIteration;
    //std::string iteration_str = std::to_string(0);
    mtxFilename+=std::to_string(globalIteration);
    //rhsFilename+=globalIteration;
    warning()<<"[AlephMatrix::solveNow] mtxFileName rhsFileName write_to_file";
    writeToFile(mtxFilename.c_str());
    b->writeToFile(rhsFilename.c_str());
  }
  // Déclenche la résolution au sein de la bibliothèque externe
  m_implementation->AlephMatrixSolve(x,b,tmp,
                                     nb_iteration,
                                     residual_norm,
                                     params);
  if (dump_to_compare){
    const int globalIteration = 0;//m_kernel->subDomain()->commonVariables().globalIteration();
    string lhsFilename("m_aleph_vector_x_");
    //lhsFilename+=std::string(globalIteration);
    x->writeToFile(lhsFilename.c_str());
  }
  if (m_kernel->isCellOrdering())
    debug() << "\33[1;32m[AlephMatrix::solveSync_waitAndFill] // nb_iteration="
            <<nb_iteration<<", residual_norm="<<*residual_norm<<"\33[0m";
  if (m_kernel->isParallel()) return;
  debug() << "\33[1;32m[AlephMatrix::solveSync_waitAndFill] // nb_iteration="
          <<nb_iteration<<", residual_norm="<<*residual_norm<<"\33[0m";
  return;
}


/*!
 *\brief Déclenche l'ordre de récupération des résultats
 */
void AlephMatrix::reassemble(int& nb_iteration,
                             double* residual_norm){
   // Si on est pas en mode parallèle, on en a finit pour le solve
  if (!m_kernel->isParallel()) return;
  m_aleph_matrix_buffer_n_iteration.resize(1);
  m_aleph_matrix_buffer_n_iteration[0]=nb_iteration;
  m_aleph_matrix_buffer_residual_norm.resize(4);
  m_aleph_matrix_buffer_residual_norm[0]=residual_norm[0];
  m_aleph_matrix_buffer_residual_norm[1]=residual_norm[1];
  m_aleph_matrix_buffer_residual_norm[2]=residual_norm[2];
  m_aleph_matrix_buffer_residual_norm[3]=residual_norm[3];
  // Il faut recevoir des données
  if (m_kernel->rank()!=m_ranks[m_kernel->rank()] && !m_kernel->isAnOther()){
 	 debug() << "\33[1;32m[AlephMatrix::REassemble] "<<m_kernel->rank()
            <<"<="<<m_ranks[m_kernel->rank()]<<"\33[0m";
	 m_aleph_matrix_mpi_results_requests.push_back(m_kernel->world()->recv(m_aleph_matrix_buffer_n_iteration,
                                                                    m_ranks[m_kernel->rank()], false));
 	 m_aleph_matrix_mpi_results_requests.push_back(m_kernel->world()->recv(m_aleph_matrix_buffer_residual_norm,
                                                                    m_ranks[m_kernel->rank()], false));
  }
  if (m_participating_in_solver){
    debug() << "\33[1;32m[AlephMatrix::REassemble] have participated, should send:"<<"\33[0m";
    for(int iCpu=0;iCpu<m_kernel->size();++iCpu){
      if (iCpu==m_kernel->rank()) continue;
      if (m_kernel->rank()!=m_ranks[iCpu]) continue;
      debug() << "\33[1;32m[AlephMatrix::REassemble] "<<m_kernel->rank()<<" => "<<iCpu<<"\33[0m";
      m_aleph_matrix_mpi_results_requests.push_back(m_kernel->world()->send(m_aleph_matrix_buffer_n_iteration,
                                                                      iCpu,false));
      m_aleph_matrix_mpi_results_requests.push_back(m_kernel->world()->send(m_aleph_matrix_buffer_residual_norm,
                                                                      iCpu,false));
    }
  }
}


/*!
 *\brief Synchronise les réceptions des résultats
 */
void AlephMatrix::reassemble_waitAndFill(int& nb_iteration, double* residual_norm){
  if (!m_kernel->isParallel()) return;
  debug() << "\33[1;32m[AlephMatrix::REassemble_waitAndFill]"<<"\33[0m";
  //if (m_kernel->isAnOther()) return;
  m_kernel->world()->waitAllRequests(m_aleph_matrix_mpi_results_requests);
  m_aleph_matrix_mpi_results_requests.clear();
  if (!m_participating_in_solver){
	 nb_iteration=m_aleph_matrix_buffer_n_iteration[0];
	 residual_norm[0]=m_aleph_matrix_buffer_residual_norm[0];
	 residual_norm[1]=m_aleph_matrix_buffer_residual_norm[1];
	 residual_norm[2]=m_aleph_matrix_buffer_residual_norm[2];
	 residual_norm[3]=m_aleph_matrix_buffer_residual_norm[3];
  }
  debug() << "\33[1;32m[AlephMatrix::REassemble_waitAndFill] // nb_iteration="
          <<nb_iteration<<", residual_norm="<<*residual_norm<<"\33[0m";
}


/*!
 *\brief Permet de spécifier le début d'une phase de remplissage
 */
void AlephMatrix::startFilling(){
  /* Nothing here to do with this m_implementation */
  debug()<<"[AlephMatrix::startFilling] void"<<"\33[0m";
}


/*!
 *\brief Déclenche l'écriture de la matrice dans un fichier
 */
void AlephMatrix::writeToFile(const string file_name){
  debug()<<"\33[1;32m[AlephMatrix::writeToFile] Dumping matrix to "<<file_name<<"\33[0m";
  m_implementation->writeToFile(file_name);
}
