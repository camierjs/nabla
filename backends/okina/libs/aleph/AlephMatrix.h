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
#ifndef ALEPH_MATRIX_H
#define ALEPH_MATRIX_H

#include <map>
#include "Aleph.h"

class IAlephMatrix;

class AlephMatrix: public TraceAccessor{
public:
  AlephMatrix(AlephKernel*);
  ~AlephMatrix();
public:
  void create(void);
  void create(IntegerConstArrayView, bool=false);
  void create_really(void);
  void reset(void);
  Integer reIdx(Integer,Array<Int32*>&);
  void reSetValuesIn(AlephMatrix*,Array<Int32*>&);
  void reAddValuesIn(AlephMatrix*,Array<Int32*>&);
  void updateKnownRowCol(Integer,Integer,Real);
  void rowMapMapCol(Integer,Integer,Real);
  void addValue(const VariableRef&, const Item&,
                const VariableRef&, const Item&, const Real);
  void addValue(const VariableRef&, const ItemEnumerator&,
                const VariableRef&, const ItemEnumerator&, const Real);
  void setValue(const VariableRef&, const Item&,
                const VariableRef&, const Item&, const Real);
  void setValue(const VariableRef&, const ItemEnumerator&,
                const VariableRef&, const ItemEnumerator&, const Real);
  void addValue(Integer,Integer,Real);
  void setValue(Integer,Integer,Real);
  void writeToFile(const String);
  void startFilling();
  void assemble();
  void assemble_waitAndFill();
  void reassemble(Integer&, Real*);
  void reassemble_waitAndFill(Integer&, Real*);
  void solve(AlephVector*, AlephVector*, Integer&, Real*, AlephParams*,bool=false);
  void solveNow(AlephVector*, AlephVector*, AlephVector*, Integer&, Real*, AlephParams* );
private:
  AlephKernel* m_kernel;
  Integer m_index;
  ArrayView<Integer> m_ranks;
  bool m_participating_in_solver;
  IAlephMatrix* m_implementation;
private:
  // Matrice utilisée dans le cas où nous sommes le solveur
  MultiArray2Int32 m_aleph_matrix_buffer_rows;
  MultiArray2Int32 m_aleph_matrix_buffer_cols;
  MultiArray2Real m_aleph_matrix_buffer_vals;
  // Tableaux tampons des setValues
  Integer m_setValue_idx;
  Array<Int32> m_setValue_row;
  Array<Int32> m_setValue_col;
  Array<Real> m_setValue_val;
 private:  // Tableaux tampons des addValues
  typedef std::map<Integer,Integer> colMap;
  typedef std::map<Integer,colMap*> rowColMap;
  rowColMap m_row_col_map;
  Integer m_addValue_idx;
  Array<Integer> m_addValue_row;
  Array<Integer> m_addValue_col;
  Array<Real> m_addValue_val;
 private:  // Tableaux des requètes
  Array<Parallel::Request> m_aleph_matrix_mpi_data_requests;
  Array<Parallel::Request> m_aleph_matrix_mpi_results_requests;
 private: // Résultats. Placés ici afin de les conserver hors du scope de la fonction les utilisant
  Array<Int32> m_aleph_matrix_buffer_n_iteration;
  Array<Real> m_aleph_matrix_buffer_residual_norm;
};


#endif  

