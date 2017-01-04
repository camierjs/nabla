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
#ifndef ALEPH_VECTOR_H
#define ALEPH_VECTOR_H

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
  void update(void){throw std::domain_error(A_FUNCINFO);}
  void reSetLocalComponents(AlephVector*);
  void setLocalComponents(int num_values,
                          vector<int> glob_indices,
                          vector<double> values);
  void setLocalComponents(vector<double> values);
  void getLocalComponents(int vector_size,
                          vector<int> global_indice,
                          vector<double>   vector_values);  
  void getLocalComponents(vector<double> &values);  
  void startFilling(void);
  void assemble(void);
  void assemble_waitAndFill(void);
  void reassemble(void);
  void reassemble_waitAndFill(void);
  void copy(AlephVector*){throw std::domain_error(A_FUNCINFO);}
  void writeToFile(const string);
  IAlephVector* implementation(void){return m_implementation;}
 private:
  AlephKernel* m_kernel;
  int m_index;
  vector<int> m_ranks;
  bool m_participating_in_solver;
  IAlephVector* m_implementation;
 private:
  // Buffers utilisés dans le cas où nous sommes le solveur
  vector<int> m_aleph_vector_buffer_idxs;
  vector<double> m_aleph_vector_buffer_vals;
 private:
  vector<int> m_aleph_vector_buffer_idx;
  vector<double> m_aleph_vector_buffer_val;
 private:
  vector<Parallel::Request> m_parallel_requests;
  vector<Parallel::Request> m_parallel_reassemble_requests;
 public:
  int m_bkp_num_values;
  vector<int> m_bkp_indexs;
  vector<double> m_bkp_values;
};

#endif  

