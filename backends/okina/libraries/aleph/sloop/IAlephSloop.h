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
#ifndef _ALEPH_SLOOP_H
#define _ALEPH_SLOOP_H

#define HASMPI
#define SLOOP_MATH 
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#define PARALLEL_SLOOP

#include <mpi.h>
#include "SLOOP.h"
#include "AlephInterface.h"

// ****************************************************************************
// * Class AlephTopologySloop
// ****************************************************************************
class AlephTopologySloop: public IAlephTopology{
public:
  AlephTopologySloop(ITraceMng* tm,
                     AlephKernel *kernel,
                     Integer index,
                     Integer nb_row_size);
  ~AlephTopologySloop();
public:
  void backupAndInitialize();
  void restore();
public:
  SLOOP::SLOOPCommInfo *m_sloop_comminfo;
  SLOOP::SLOOPCommInfo *m_world_comminfo;
  SLOOP::SLOOPMsg *m_sloop_msg;
  SLOOP::SLOOPTopology *m_sloop_topology; 
};


// ****************************************************************************
// * AlephVectorSloop
// ****************************************************************************
class AlephVectorSloop: public IAlephVector{
public:
  AlephVectorSloop(ITraceMng* tm,
                   AlephKernel *kernel,
                   Integer index);
  ~AlephVectorSloop();
  void AlephVectorCreate(void);
  void AlephVectorSet(const double *bfr_val, const int *bfr_idx, Integer size);
  int AlephVectorAssemble(void);
  void AlephVectorGet(double *bfr_val, const int *bfr_idx, Integer size);
  void writeToFile(const String file_name);
public:
  SLOOP::SLOOPDistVector* m_sloop_vector;
};


/******************************************************************************
 AlephMatrixSloop
*****************************************************************************/
class AlephMatrixSloop: public IAlephMatrix{
public:
  AlephMatrixSloop(ITraceMng* tm,
                   AlephKernel *kernel,
                   Integer index);
  ~AlephMatrixSloop();
  void AlephMatrixCreate(void);
  void AlephMatrixSetFilled(bool toggle);
  int AlephMatrixAssemble(void);
  void AlephMatrixFill(int size, int *rows, int *cols, double *values);
  bool isAlreadySolved(SLOOP::SLOOPDistVector* x,
                       SLOOP::SLOOPDistVector* b,
                       SLOOP::SLOOPDistVector* tmp,
                       Real* residual_norm,
                       AlephParams* params);
  int AlephMatrixSolve(AlephVector* x,
                       AlephVector* b,
                       AlephVector* tmp,
                       Integer& nb_iteration,
                       Real* residual_norm,
                       AlephParams* solver_param);
  void writeToFile(const String file_name);
  SLOOP::SLOOPSolver* createSloopSolver(AlephParams* solver_param, SLOOP::SLOOPMsg& info_sloop_msg);
  SLOOP::SLOOPPreconditioner* createSloopPreconditionner(AlephParams* solver_param,
                                                         SLOOP::SLOOPMsg& info_sloop_msg);
  SLOOP::SLOOPStopCriteria * createSloopStopCriteria(AlephParams* solver_param,
                                                     SLOOP::SLOOPMsg& info_sloop_msg) ;
  void setSloopSolverParameters(AlephParams* solver_param,
                                SLOOP::SLOOPSolver* sloop_solver);
  void setSloopPreconditionnerParameters(AlephParams* solver_param,
                                         SLOOP::SLOOPPreconditioner* preconditionner);
  bool normalizeSolverMatrix(AlephParams* solver_param);
public:
  SLOOP::SLOOPMatrix* m_sloop_matrix;
};



class SloopAlephFactoryImpl: public IAlephFactoryImpl{
public:
  SloopAlephFactoryImpl();
  ~SloopAlephFactoryImpl();
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
  Array<IAlephTopology*> m_IAlephTopologys;
};


#endif // _ALEPH_SLOOP_H_
