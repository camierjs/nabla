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
  void create(vector<int>, bool=false);
  void create_really(void);
  void reset(void);
  int reIdx(int,vector<int*>&);
  void reSetValuesIn(AlephMatrix*,vector<int*>&);
  void reAddValuesIn(AlephMatrix*,vector<int*>&);
  void updateKnownRowCol(int,int,double);
  void rowMapMapCol(int,int,double);
  void addValue(const Variable&, const item&,
                const Variable&, const item&, const double);
  void addValue(const Variable&, const item*&,
                const Variable&, const item*&, const double);
  void setValue(const Variable&, const item&,
                const Variable&, const item&, const double);
  void setValue(const Variable&, const item*&,
                const Variable&, const item*&, const double);
  void addValue(int,int,double);
  void setValue(int,int,double);
  void writeToFile(const string);
  void startFilling();
  void assemble();
  void assemble_waitAndFill();
  void reassemble(int&, double*);
  void reassemble_waitAndFill(int&, double*);
  void solve(AlephVector*, AlephVector*, int&, double*, AlephParams*,bool=false);
  void solveNow(AlephVector*, AlephVector*, AlephVector*, int&, double*, AlephParams* );
private:
  AlephKernel* m_kernel;
  int m_index;
  vector<int> m_ranks;
  bool m_participating_in_solver;
  IAlephMatrix* m_implementation;
private:
  // Matrice utilisée dans le cas où nous sommes le solveur
  vector<vector<int> > m_aleph_matrix_buffer_rows;
  vector<vector<int> > m_aleph_matrix_buffer_cols;
  vector<vector<double> > m_aleph_matrix_buffer_vals;
  // Tableaux tampons des setValues
  int m_setValue_idx;
  vector<int> m_setValue_row;
  vector<int> m_setValue_col;
  vector<double> m_setValue_val;
 private:  // Tableaux tampons des addValues
  typedef std::map<int,int> colMap;
  typedef std::map<int,colMap*> rowColMap;
  rowColMap m_row_col_map;
  int m_addValue_idx;
  vector<int> m_addValue_row;
  vector<int> m_addValue_col;
  vector<double> m_addValue_val;
 private:  // Tableaux des requètes
  vector<Parallel::Request> m_aleph_matrix_mpi_data_requests;
  vector<Parallel::Request> m_aleph_matrix_mpi_results_requests;
 private: // Résultats. Placés ici afin de les conserver hors du scope de la fonction les utilisant
  vector<int> m_aleph_matrix_buffer_n_iteration;
  vector<double> m_aleph_matrix_buffer_residual_norm;
};


#endif  

