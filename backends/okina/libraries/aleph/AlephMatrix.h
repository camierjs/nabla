#ifndef ALEPH_MATRIX_H
#define ALEPH_MATRIX_H

#include <map>
#include "AlephGlobal.h"

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

