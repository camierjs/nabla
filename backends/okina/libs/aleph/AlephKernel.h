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
#ifndef _ALEPH_KERNEL_H_
#define _ALEPH_KERNEL_H_

#include "Aleph.h"

class IAlephFactory;
class IAlephTopology;
class AlephTopology;
class AlephMatrix;
class AlephOrdering;
class AlephIndexing;
class AlephVector;
class IParallelMng;
class ISubDomain;
class AlephParams;


/******************************************************************************
 *****************************************************************************/
class AlephKernelResults{
 public:
  int m_nb_iteration;
  double m_residual_norm[4];
};


/******************************************************************************
 *****************************************************************************/
class AlephKernelArguments: public TraceAccessor{
public:
  AlephKernelArguments(ITraceMng* tm,
                       AlephVector *x_vector,
                       AlephVector *b_vector,
                       AlephVector *tmp_vector,
                       IAlephTopology *topology):
    TraceAccessor(tm),
    m_x_vector(x_vector),
    m_b_vector(b_vector),
    m_tmp_vector(tmp_vector),
    m_topology_implementation(topology),
    m_params(NULL){} // m_params sera initialisé via le postSolver
  ~AlephKernelArguments(){
    debug()<<"\33[1;5;31m[~AlephKernelArguments]"<<"\33[0m";
  };
public:
  AlephVector *m_x_vector;
  AlephVector *m_b_vector;
  AlephVector *m_tmp_vector;
  IAlephTopology *m_topology_implementation;
  AlephParams *m_params;
};


/******************************************************************************
 *****************************************************************************/
class AlephKernel: public TraceAccessor{
 public:
  AlephKernel(IParallelMng*, int, IAlephFactory*, int=0, int=0, bool=false);
  AlephKernel(ITraceMng*, ISubDomain*, IAlephFactory*, int=0, int=0, bool=false);
  AlephKernel(ISubDomain*,int alephUnderlyingSolver=0, int alephNumberOfCores=0);
  ~AlephKernel(void);
  void setup(void);
  void initialize(int, int);
  void break_and_return(void);
  AlephVector* createSolverVector(void);
  AlephMatrix* createSolverMatrix(void);
  void postSolver(AlephParams*,AlephMatrix*,AlephVector*,AlephVector*);
  void workSolver(void);  
  AlephVector* syncSolver(int, int&, double*);

public:
  IAlephFactory* factory(void){return m_factory;}
  AlephTopology *topology(void){return m_topology;}
  AlephOrdering *ordering(void){return m_ordering;}
  AlephIndexing *indexing(void){return m_indexing;}
  int rank(void){return m_rank;}
  int size(void){return m_size;}
  ISubDomain* subDomain(void){
    if (!m_sub_domain && !m_i_am_an_other)
      throw std::logic_error("No sub-domain to work on!");
    return m_sub_domain;
  }
  bool isParallel(void){return m_isParallel;}
  bool isInitialized(void){return m_has_been_initialized;}
  bool thereIsOthers(void){return m_there_are_idles;}
  bool isAnOther(void){return m_i_am_an_other;}
  IParallelMng* parallel(void){ return m_parallel;}
  IParallelMng* world(void){ return m_world_parallel;}
  int underlyingSolver(void){return m_underlying_solver;}
  bool isCellOrdering(void){return m_reorder;}
  int index(void){return m_solver_index;}
  bool configured(void){return m_configured;}
  void mapranks(vector<int>&);
  bool hitranks(int, vector<int>);
  int nbRanksPerSolver(void){return m_solver_size;}
  vector<int> solverRanks(int i){return m_solver_ranks.at(i);}//.view();}
  IParallelMng* subParallelMng(int i){return m_sub_parallel_mng_queue.at(i);}
  IAlephTopology* getTopologyImplementation(int i){
    return m_arguments_queue.at(i)->m_topology_implementation;
  }
private:
  IParallelMng *createUnderlyingParallelMng(int);
private: 
  ISubDomain* m_sub_domain;
  bool m_isParallel;
  int m_rank;
  int m_size;
  int m_world_size;
  bool m_there_are_idles;
  bool m_i_am_an_other;
  IParallelMng* m_parallel;
  IParallelMng* m_world_parallel;
private:
  bool m_configured;
  IAlephFactory *m_factory;
  AlephTopology *m_topology;
  AlephOrdering *m_ordering;
  AlephIndexing *m_indexing;
  int m_aleph_vector_idx;
  const int m_underlying_solver;
  const bool m_reorder;
  int m_solver_index;
  int m_solver_size;
  bool m_solved;
  bool m_has_been_initialized;
  
private:
  vector<vector<int> > m_solver_ranks;
  vector<IParallelMng*> m_sub_parallel_mng_queue;
  vector<AlephMatrix*> m_matrix_queue;
  vector<AlephKernelArguments*> m_arguments_queue;
  vector<AlephKernelResults*> m_results_queue;
};

#endif  

