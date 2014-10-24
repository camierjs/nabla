/*****************************************************************************
 * IAlephTrilinos.h                                            (C) 2010-2012 *
 * constants for output types
 #define AZ_all             -4  Print out everything including matrix
 #define AZ_none             0  Print out no results (not even warnings)
 #define AZ_last            -1  Print out final residual and warnings
 #define AZ_summary         -2  Print out summary, final residual and warnings
 #define AZ_warnings        -3  Print out only warning messages
 *****************************************************************************/
#ifndef _ALEPH_INTERFACE_TRILINOS_H_
#define _ALEPH_INTERFACE_TRILINOS_H_


#include "Epetra_config.h"
#include "Epetra_Vector.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "AztecOO.h"
#include "ml_MultiLevelPreconditioner.h"
#include "Ifpack_IC.h"

#include "AlephInterface.h"


/******************************************************************************
 AlephVectorTrilinos
*****************************************************************************/
class AlephVectorTrilinos: public IAlephVector{
public:
/******************************************************************************
 * AlephVectorTrilinos
 *****************************************************************************/
  AlephVectorTrilinos(ITraceMng *tm,
                      AlephKernel *kernel,
                      Integer index);
  void AlephVectorCreate(void);
  void AlephVectorSet(const double *bfr_val, const int *bfr_idx, Integer size);
  int AlephVectorAssemble(void);
  void AlephVectorGet(double *bfr_val, const int *bfr_idx, Integer size);
  void writeToFile(const String filename);
  Real LinftyNorm(void);
  void fill(Real value);
public:
  Epetra_Vector *m_trilinos_vector;
  Epetra_Comm *m_trilinos_Comm;
};
 

/******************************************************************************
 AlephMatrixTrilinos
*****************************************************************************/
class AlephMatrixTrilinos: public IAlephMatrix{
public:
  AlephMatrixTrilinos(ITraceMng *tm,
                      AlephKernel *kernel,
                      Integer index);
  void AlephMatrixCreate(void);
  void AlephMatrixSetFilled(bool);
  int AlephMatrixAssemble(void);
  void AlephMatrixFill(int size, int *rows, int *cols, double *values);
  Real LinftyNormVectorProductAndSub(AlephVector* x,
                                     AlephVector* b);
  bool isAlreadySolved(AlephVectorTrilinos* x,
                       AlephVectorTrilinos* b,
                       AlephVectorTrilinos* tmp,
                       Real* residual_norm,
                       AlephParams* params);
  int AlephMatrixSolve(AlephVector* x,
                       AlephVector* b,
                       AlephVector* t,
                       Integer& nb_iteration,
                       Real* residual_norm,
                       AlephParams* solver_param);
  void writeToFile(const String filename);
private:
  Epetra_CrsMatrix *m_trilinos_matrix;
  Epetra_Comm *m_trilinos_Comm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TrilinosAlephFactoryImpl: public TraceAccessor,
                             public IAlephFactoryImpl{
public:
  TrilinosAlephFactoryImpl(ITraceMng*);
  ~TrilinosAlephFactoryImpl();
  virtual void initialize();
  virtual IAlephTopology* createTopology(ITraceMng *tm,
                                         AlephKernel *kernel,
                                         Integer index,
                                         Integer nb_row_size);
  virtual IAlephVector* createVector(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index);  
  virtual IAlephMatrix* createMatrix(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index);
private:
  Array<IAlephVector*> m_IAlephVectors;
  Array<IAlephMatrix*> m_IAlephMatrixs;
};


#endif // _ALEPH_INTERFACE_TRILINOS_H_
