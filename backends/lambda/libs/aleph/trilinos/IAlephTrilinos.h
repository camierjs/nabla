///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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
/*****************************************************************************
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
//#include "Epetra_MpiComm.h"
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
