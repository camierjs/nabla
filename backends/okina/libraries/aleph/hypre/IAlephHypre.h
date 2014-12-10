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
#ifndef ALEPH_INTERFACE_HYPRE_H
#define ALEPH_INTERFACE_HYPRE_H

#define HAVE_MPI
#define MPI_COMM_SUB (*(MPI_Comm*)(m_kernel->subParallelMng(m_index)->getMPICommunicator()))
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include "HYPRE.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "krylov.h"

#include "AlephInterface.h"


/******************************************************************************
 AlephVectorHypre
 *****************************************************************************/
class AlephVectorHypre: public IAlephVector{
public:
  AlephVectorHypre(ITraceMng *tm,AlephKernel *kernel, Integer index);
public:
/******************************************************************************
 * The Create() routine creates an empty vector object that lives on the comm communicator. This is
 * a collective call, with each process passing its own index extents, jLower and jupper. The names
 * of these extent parameters begin with a j because we typically think of matrix-vector multiplies
 * as the fundamental operation involving both matrices and vectors. For matrix-vector multiplies,
 * the vector partitioning should match the column partitioning of the matrix (which also uses the j
 * notation). For linear system solves, these extents will typically match the row partitioning of the
 * matrix as well.
*****************************************************************************/
  void AlephVectorCreate(void);
  void AlephVectorSet(const double *bfr_val, const int *bfr_idx, Integer size);
  int AlephVectorAssemble(void);
  void AlephVectorGet(double *bfr_val, const int *bfr_idx, Integer size);
  Real norm_max();
  void writeToFile(const String filename);
public:
  HYPRE_IJVector m_hypre_ijvector;
  HYPRE_ParVector m_hypre_parvector;
  HYPRE_Int jSize;
  HYPRE_Int jUpper;
  HYPRE_Int jLower;
};


/******************************************************************************
 AlephMatrixHypre
 *****************************************************************************/
class AlephMatrixHypre: public IAlephMatrix{
 public:
/******************************************************************************
 AlephMatrixHypre
 *****************************************************************************/
  AlephMatrixHypre(ITraceMng *tm,
                   AlephKernel *kernel,
                   Integer index);
  
  ~AlephMatrixHypre();
  
 public:


/******************************************************************************
 * Each submatrix Ap is "owned" by a single process and its first and last row numbers are
 * given by the global indices ilower and iupper in the Create() call below.
 *******************************************************************************
 * The Create() routine creates an empty matrix object that lives on the comm communicator. This
 * is a collective call (i.e., must be called on all processes from a common synchronization point),
 * with each process passing its own row extents, ilower and iupper. The row partitioning must be
 * contiguous, i.e., iupper for process i must equal ilower-1 for process i+1. Note that this allows
 * matrices to have 0- or 1-based indexing. The parameters jlower and jupper define a column
 * partitioning, and should match ilower and iupper when solving square linear systems.
 *****************************************************************************/
  void AlephMatrixCreate(void);
  void AlephMatrixSetFilled(bool);
  int AlephMatrixAssemble(void);
  void AlephMatrixFill(int size, int *rows, int *cols, double *values);
  bool isAlreadySolved(AlephVectorHypre *x,
                       AlephVectorHypre *b,
                       AlephVectorHypre *tmp,
                       Real* residual_norm,
                       AlephParams* params);
  int AlephMatrixSolve(AlephVector* x,
                       AlephVector* b,
                       AlephVector* t,
                       Integer& nb_iteration,
                       Real* residual_norm,
                       AlephParams* solver_param);
  void writeToFile(const String filename);
  void initSolverPCG(const AlephParams* solver_param,HYPRE_Solver& solver);
  void initSolverBiCGStab(const AlephParams* solver_param,HYPRE_Solver& solver);
  void initSolverGMRES(const AlephParams* solver_param,HYPRE_Solver& solver);
  void setDiagonalPreconditioner( const TypesSolver::eSolverMethod solver_method ,
                                  const HYPRE_Solver&              solver        ,
                                  HYPRE_Solver&                    precond       );
  void setILUPreconditioner(const TypesSolver::eSolverMethod solver_method ,
                            const HYPRE_Solver& solver,
                            HYPRE_Solver& precond);
  void setSpaiStatPreconditioner(const TypesSolver::eSolverMethod solver_method ,
                                 const HYPRE_Solver& solver,
                                 const AlephParams* solver_param,
                                 HYPRE_Solver& precond);
  void setAMGPreconditioner(const TypesSolver::eSolverMethod solver_method ,
                            const HYPRE_Solver& solver        ,
                            const AlephParams* solver_param  ,
                            HYPRE_Solver& precond);
  bool solvePCG
  ( const AlephParams* solver_param ,
    HYPRE_Solver&       solver       ,
    HYPRE_ParCSRMatrix& M            ,
    HYPRE_ParVector&    B            ,
    HYPRE_ParVector&    X            ,
    int&                iteration    ,
    double&             residue      );
  bool solveBiCGStab
  ( HYPRE_Solver&       solver       ,
    HYPRE_ParCSRMatrix& M            ,
    HYPRE_ParVector&    B            ,
    HYPRE_ParVector&    X            ,
    int&                iteration    ,
    double&             residue      );
  bool solveGMRES
  ( HYPRE_Solver&       solver       ,
    HYPRE_ParCSRMatrix& M            ,
    HYPRE_ParVector&    B            ,
    HYPRE_ParVector&    X            ,
    int&                iteration    ,
    double&             residue      );
 private:
  HYPRE_IJMatrix m_hypre_ijmatrix; 
  HYPRE_ParCSRMatrix m_hypre_parmatrix; 
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class HypreAlephFactoryImpl: public TraceAccessor, public IAlephFactoryImpl{
public:
  HypreAlephFactoryImpl(ITraceMng*);
  ~HypreAlephFactoryImpl();
public:
  virtual void initialize();
  
  virtual IAlephTopology* createTopology(ITraceMng* tm,
                                         AlephKernel* kernel,
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

#endif // _ALEPH_INTERFACE_HYPRE_H_
