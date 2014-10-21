#ifndef ALEPH_VECTOR_H
#define ALEPH_VECTOR_H

#include "AlephGlobal.h"

class IAlephVector;
class AlephTopology;
class AlephKernel;

class AlephVector: public TraceAccessor{
 public:
  AlephVector(AlephKernel*);
  ~AlephVector();
 
 public:
  void create(void);
  void create_really(void);
  void update(void){throw NotImplementedException(A_FUNCINFO);}
  void reSetLocalComponents(AlephVector*);
  void setLocalComponents(Integer num_values,
                          ConstArrayView<int> glob_indices,
                          ConstArrayView<double> values);
  void setLocalComponents(ConstArrayView<double> values);
  void getLocalComponents(Integer vector_size,
                          ConstArrayView<int> global_indice,
                          ArrayView<double>   vector_values);  
  void getLocalComponents(Array<Real> &values);  
  void startFilling(void);
  void assemble(void);
  void assemble_waitAndFill(void);
  void reassemble(void);
  void reassemble_waitAndFill(void);
  void copy(AlephVector*){throw NotImplementedException(A_FUNCINFO);}
  void writeToFile(const String);
  IAlephVector* implementation(void){return m_implementation;}
 private:
  AlephKernel* m_kernel;
  Integer m_index;
  ArrayView<Integer> m_ranks;
  bool m_participating_in_solver;
  IAlephVector* m_implementation;
 private:
  // Buffers utilisés dans le cas où nous sommes le solveur
  Array<Int32> m_aleph_vector_buffer_idxs;
  Array<Real> m_aleph_vector_buffer_vals;
 private:
  Array<Int32> m_aleph_vector_buffer_idx;
  Array<Real> m_aleph_vector_buffer_val;
 private:
  Array<Parallel::Request> m_parallel_requests;
  Array<Parallel::Request> m_parallel_reassemble_requests;
 public:
  Integer m_bkp_num_values;
  Array<int> m_bkp_indexs;
  Array<double> m_bkp_values;
};

#endif  

