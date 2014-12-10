// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
  Integer m_nb_iteration;
  Real m_residual_norm[4];
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
  AlephKernel(IParallelMng*, Integer, IAlephFactory*, Integer=0, Integer=0, bool=false);
  AlephKernel(ITraceMng*, ISubDomain*, IAlephFactory*, Integer=0, Integer=0, bool=false);
  AlephKernel(ISubDomain*,Integer alephUnderlyingSolver=0, Integer alephNumberOfCores=0);
  ~AlephKernel(void);
  void setup(void);
  void initialize(Integer, Integer);
  void break_and_return(void);
  AlephVector* createSolverVector(void);
  AlephMatrix* createSolverMatrix(void);
  void postSolver(AlephParams*,AlephMatrix*,AlephVector*,AlephVector*);
  void workSolver(void);  
  AlephVector* syncSolver(Integer, Integer&, Real*);

public:
  IAlephFactory* factory(void){return m_factory;}
  AlephTopology *topology(void){return m_topology;}
  AlephOrdering *ordering(void){return m_ordering;}
  AlephIndexing *indexing(void){return m_indexing;}
  Integer rank(void){return m_rank;}
  Integer size(void){return m_size;}
  ISubDomain* subDomain(void){
    if (!m_sub_domain && !m_i_am_an_other)
      throw FatalErrorException("[AlephKernel::subDomain]", "No sub-domain to work on!");
    return m_sub_domain;
  }
  bool isParallel(void){return m_isParallel;}
  bool isInitialized(void){return m_has_been_initialized;}
  bool thereIsOthers(void){return m_there_are_idles;}
  bool isAnOther(void){return m_i_am_an_other;}
  IParallelMng* parallel(void){ return m_parallel;}
  IParallelMng* world(void){ return m_world_parallel;}
  Integer underlyingSolver(void){return m_underlying_solver;}
  bool isCellOrdering(void){return m_reorder;}
  Integer index(void){return m_solver_index;}
  bool configured(void){return m_configured;}
  void mapranks(Array<Integer>&);
  bool hitranks(Integer, ArrayView<Integer>);
  Integer nbRanksPerSolver(void){return m_solver_size;}
  ArrayView<Integer> solverRanks(Integer i){return m_solver_ranks.at(i);}//.view();}
  IParallelMng* subParallelMng(Integer i){return m_sub_parallel_mng_queue.at(i);}
  IAlephTopology* getTopologyImplementation(Integer i){
    return m_arguments_queue.at(i)->m_topology_implementation;
  }
private:
  IParallelMng *createUnderlyingParallelMng(Integer);
private: 
  ISubDomain* m_sub_domain;
  bool m_isParallel;
  Integer m_rank;
  Integer m_size;
  Integer m_world_size;
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
  Integer m_aleph_vector_idx;
  const Integer m_underlying_solver;
  const bool m_reorder;
  Integer m_solver_index;
  Integer m_solver_size;
  bool m_solved;
  bool m_has_been_initialized;
  
private:
  Array<Array<Integer> > m_solver_ranks;
  Array<IParallelMng*> m_sub_parallel_mng_queue;
  Array<AlephMatrix*> m_matrix_queue;
  Array<AlephKernelArguments*> m_arguments_queue;
  Array<AlephKernelResults*> m_results_queue;
};

#endif  

