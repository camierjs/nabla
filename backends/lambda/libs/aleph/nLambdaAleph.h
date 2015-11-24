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
#ifdef LAMBDA_HAS_PACKAGE_HYPRE
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include "HYPRE.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "krylov.h"
#endif // LAMBDA_HAS_PACKAGE_HYPRE

//#ifdef LAMBDA_HAS_PACKAGE_TRILINOS
//#include "Epetra_config.h"
//#include "Epetra_Vector.h"
//#include "Epetra_MpiComm.h"
//#include "Epetra_Map.h"
//#include "Epetra_CrsMatrix.h"
//#include "Epetra_LinearProblem.h"
//#include "AztecOO.h"
//#include "Ifpack_IC.h"
//#include "ml_MultiLevelPreconditioner.h"
//#endif // LAMBDA_HAS_PACKAGE_TRILINOS

//#ifdef LAMBDA_HAS_PACKAGE_CUDA
//#include <cuda.h>
//#include <cublas.h>
//#include <cuda_runtime.h>
//#endif // LAMBDA_HAS_PACKAGE_CUDA

//#include <IAleph.h>
//#include <IAlephFactory.h>


// ****************************************************************************
// * AlephRealMatrix
// ****************************************************************************
class AlephRealMatrix{
public:
  void error(){throw std::logic_error("AlephRealArray");}
/*   void addValue(const double* iVar, const item* iItmEnum,
     const double* jVar, const item jItm, double value){
     m_aleph_mat->addValue(iVar,*iItmEnum,jVar,jItm,value);t}
     void addValue(const double* iVar, const item* iItmEnum,
                 const double* jVar, const item* jItmEnum, double value){
      m_aleph_mat->addValue(iVar,*iItmEnum,jVar,*jItmEnum,value);t}
   void addValue(const double* iVar, const item iItm,
                 const double* jVar, const item jItm, double value){
      m_aleph_mat->addValue(iVar,iItm,jVar,jItm,value);t}
   void addValue(const double* iVar, const item iItm,
   const double* jVar, const item* jItmEnum, double value){
   m_aleph_mat->addValue(iVar,iItm,jVar,*jItmEnum,value);t}*/
  
  void addValue(double *iVar, int iItm,
                double *jVar, int jItm,
                double value){
    /*debug()<<"\33[1;32m[matrix::setValue("
           <<"iVar @ 0x"<<iVar<<" ["<<iItm<<"], "
           <<"jVar @ 0x"<<jVar<<" ["<<jItm<<"], "
           <<"value="<<value<<")]\33[0m";*/
    m_aleph_mat->addValue(iVar,iItm,jVar,jItm,value);
  }
  void setValue(double *iVar, int iItm,
                double *jVar, int jItm,
                double value){
    /*debug()<<"\33[1;32m[matrix::setValue("
           <<"iVar @ 0x"<<iVar<<" ["<<iItm<<"], "
           <<"jVar @ 0x"<<jVar<<" ["<<jItm<<"], "
           <<"value="<<value<<")]\33[0m";*/
    m_aleph_mat->setValue(iVar,iItm,jVar,jItm,value);
  }
  /*
  void setValue(const double* iVar, const item* iItmEnum,
                const double* jVar, const item jItm,
                double value){
    m_aleph_mat->setValue(iVar,*iItmEnum,jVar,jItm,value);
  }
  
  void setValue(const double* iVar, const item* iItmEnum,
                const double* jVar, const item* jItmEnum,
                double value){
    m_aleph_mat->setValue(iVar,*iItmEnum,jVar,*jItmEnum,value);
    }*/
  public:
  AlephKernel *m_aleph_kernel;
  AlephMatrix *m_aleph_mat;
};


// ****************************************************************************
// * AlephRealArray
// ****************************************************************************
class AlephRealArray:public vector<double>{
public:
  void reset(){
    debug()<<"\33[1;33m[vector::reset]\33[0m";
    resize(0);
  }
  void error(){throw std::logic_error("[AlephRealArray] Error");}
  void newValue(double value){
    push_back(value);
  }
  //void addValue(const double* var, const item* itmEnum, double value){
  //   return addValue(var,*itmEnum,value);
  //}
  void addValue(double *var, int itm, double value){
    unsigned int idx=m_aleph_kernel->indexing()->get(var,itm);
    debug()<<"\33[1;33m[vector::addValue["<<idx<<"]\33[0m";
    if (idx==size()){
      resize(idx+1);
      index.push_back(idx);
      this->at(idx)=value;
    }else{
      this->at(idx)=at(idx)+value;
    }
  }
  //void setValue(const Variable &var, const item* itmEnum, double value){
  //  return setValue(var,*itmEnum,value);
  //}
  void setValue(double *var, int itm, double value){
    //debug()<<"\33[1;33m[vector::setValue(...)]\33[0m";
    int topology_row_offset=0;
    unsigned int idx=m_aleph_kernel->indexing()->get(var,itm)-topology_row_offset;
    if(idx==size()){
      resize(idx+1);
      index.push_back(idx);
      this->at(idx)=value;
    }else{
      this->at(idx)=value;
    }
  }
  double getValue(double *var, int itmEnum){
    //debug()<<"\33[1;33m[vector::getValue]\33[0m";
    return at(m_aleph_kernel->indexing()->get(var,itmEnum));
  }
public:
  vector<int> index;
  AlephKernel *m_aleph_kernel;
};


// ****************************************************************************
// * Globals for Aleph
// ****************************************************************************
IAlephFactory *m_aleph_factory;
AlephKernel *m_aleph_kernel;
AlephParams *m_aleph_params;
AlephMatrix *m_aleph_mat;
AlephVector *m_aleph_rhs;
AlephVector *m_aleph_sol;

vector<int> vector_indexs;
vector<double> vector_zeroes;

AlephRealArray lhs;
AlephRealArray rhs;
AlephRealMatrix mtx;


// ****************************************************************************
// * Globals for Simulation
// ****************************************************************************
IMesh *thisMesh=new IMesh(NABLA_NB_CELLS_X_AXIS,NABLA_NB_CELLS_Y_AXIS,NABLA_NB_CELLS_Z_AXIS);

ITraceMng *thisTraceMng=new ITraceMng();

SequentialMng *thisParallelMng=new SequentialMng(thisTraceMng);

ISubDomain *thisSubDomain=new ISubDomain(thisMesh,thisParallelMng);


IMesh* mesh(void){return thisMesh;}
ITraceMng* traceMng(void){return thisTraceMng;}
ISubDomain* subDomain(void){return thisSubDomain;}


// ****************************************************************************
// * Forward Declarations
// ****************************************************************************
void alephInitialize(void){
  debug()<<"\33[1;37m[alephInitialize] createSolverMatrix\33[0m";
  m_aleph_mat=m_aleph_kernel->createSolverMatrix();
  debug()<<"\33[1;37m[alephInitialize] RHS createSolverVector\33[0m";
  m_aleph_rhs=m_aleph_kernel->createSolverVector();
  debug()<<"\33[1;37m[alephInitialize] SOL createSolverVector\33[0m";
  m_aleph_sol=m_aleph_kernel->createSolverVector();
  m_aleph_mat->create();
  m_aleph_rhs->create();
  m_aleph_sol->create();
  m_aleph_mat->reset();
  mtx.m_aleph_mat=m_aleph_mat;
  debug()<<"\33[1;37m[alephInitialize] done\33[0m";
}


// ****************************************************************************
// * alephIni
// ****************************************************************************
void alephIni(%s%s){ // we have to match args & params
#ifndef ALEPH_INDEX
  debug()<<"\33[1;37;45m[alephIni] NO ALEPH_INDEX\33[0m";
  //#warning NO ALEPH_INDEX
  // Pourrait être enlevé, mais est encore utilisé dans le test DDVF sans auto-index
  vector_indexs.resize(0);
  vector_zeroes.resize(0);
  rhs.resize(0);
  mesh()->checkValidMeshFull();
  m_aleph_factory=new AlephFactory(traceMng());
  m_aleph_kernel=new AlephKernel(traceMng(), subDomain(),
                                 m_aleph_factory,
                                 alephUnderlyingSolver,
                                 alephNumberOfCores,
                                 false);
#else
  debug()<<"\33[1;37;45m[alephIni] with ALEPH_INDEX\33[0m";
  //#warning ALEPH_INDEX
  m_aleph_kernel=new AlephKernel(subDomain(),
                                 alephUnderlyingSolver,
                                 alephNumberOfCores);
  debug()<<"\33[1;37;45m[alephIni] Kernel set, setting rhs,lhs&mtx!\33[0m";
  rhs.m_aleph_kernel=m_aleph_kernel;
  lhs.m_aleph_kernel=m_aleph_kernel;
  mtx.m_aleph_kernel=m_aleph_kernel;
#endif
  m_aleph_params=
    new AlephParams(traceMng(),
                    alephEpsilon,      // epsilon epsilon de convergence
                    alephMaxIterations,// max_iteration nb max iterations
                    (TypesSolver::ePreconditionerMethod)alephPreconditionerMethod,// préconditionnement: ILU, DIAGONAL, AMG, IC
                    (TypesSolver::eSolverMethod)alephSolverMethod, // méthode de résolution
                    -1,                             // gamma
                    -1.0,                           // alpha
                    true,                           // xo_user par défaut Xo n'est pas égal à 0
                    false,                          // check_real_residue
                    false,                          // print_real_residue
                    false,                          // debug_info
                    1.e-20,                         // min_rhs_norm
                    false,                          // convergence_analyse
                    true,                           // stop_error_strategy
                    option_aleph_dump_matrix,       // write_matrix_to_file_error_strategy
                    "SolveErrorAlephMatrix.dbg",    // write_matrix_name_error_strategy
                    true,                           // listing_output 
                    0.,                             // threshold
                    false,                          // print_cpu_time_resolution
                    -1,                             // amg_coarsening_method
                    -3,                             // output_level
                    1,                              // 1: V, 2: W, 3: Full Multigrid V
                    1,                              // amg_solver_iterations
                    1,                              // amg_smoother_iterations
                    TypesSolver::SymHybGSJ_smoother,// amg_smootherOption
                    TypesSolver::ParallelRugeStuben,// amg_coarseningOption
                    TypesSolver::CG_coarse_solver,  // amg_coarseSolverOption
                    false,                          // keep_solver_structure
                    false,                          // sequential_solver
                    TypesSolver::RB);               // criteria_stop");
  debug()<<"\33[1;37;45m[alephIni] done!\33[0m";
}


// ****************************************************************************
// * alephAddValue
// ****************************************************************************
void alephAddValue(double *rowVar, int rowItm,
                   double *colVar, int colItm, double val){
  debug()<<"\33[1;31m[alephAddValue(...)]\33[0m";
  m_aleph_mat->addValue(rowVar,rowItm,colVar,colItm,val);
}

void alephAddValue(int i, int j, double val){
  debug()<<"\33[1;31m[alephAddValue(i,j,val)]\33[0m";
  m_aleph_mat->addValue(i,j,val);
}


// ****************************************************************************
// * alephRhsSet
// ****************************************************************************
void alephRhsSet(int row, double value){
  const int kernel_rank = m_aleph_kernel->rank();
  const int rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];
  debug()<<"\33[1;31m[alephRhsSet(row,val)]\33[0m";
  rhs[row-rank_offset]=value;
}

void alephRhsSet(double *var, int itm, double value){
  debug()<<"\33[1;31m[alephRhsSet(...)]\33[0m";
  rhs[m_aleph_kernel->indexing()->get(var,itm)]=value;
}


// ****************************************************************************
// * alephRhsGet
// ****************************************************************************
double alephRhsGet(int row){
  const int kernel_rank = m_aleph_kernel->rank();
  const register int rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];
  debug()<<"\33[1;31m[alephRhsGet]\33[0m";
  return rhs[row-rank_offset];
}


// ****************************************************************************
// * alephRhsAdd
// ****************************************************************************
void alephRhsAdd(int row, double value){
  const int kernel_rank = m_aleph_kernel->rank();
  const int rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];
  debug()<<"\33[1;31m[alephRhsAdd]\33[0m";
  rhs[row-rank_offset]+=value;
}


// ****************************************************************************
// * alephSolveWithoutIndex
// ****************************************************************************
void alephSolveWithoutIndex(void){
  int nb_iteration;
  double residual_norm[4];
  debug()<<"\33[1;31m[alephSolveWithoutIndex]\33[0m";
  m_aleph_mat->assemble();
  m_aleph_rhs->setLocalComponents(rhs);
  m_aleph_rhs->assemble();
  vector_zeroes.resize(rhs.size());
  vector_zeroes.assign(rhs.size(),0.0);
  m_aleph_sol->setLocalComponents(vector_zeroes);
  m_aleph_sol->assemble();
  m_aleph_mat->solve(m_aleph_sol, m_aleph_rhs, nb_iteration, &residual_norm[0], m_aleph_params, true);
  AlephVector *solution=m_aleph_kernel->syncSolver(0,nb_iteration,&residual_norm[0]);
  info() << "Solved in \33[7m" << nb_iteration << "\33[m iterations,"
         << "residuals=[\33[1m" << residual_norm[0] <<"\33[m,"<< residual_norm[3]<<"]";
  lhs.reset(); lhs.resize(rhs.size()); //lhs.fill(0.0);
  solution->getLocalComponents(lhs);
}


// ****************************************************************************
// * alephSolve
// ****************************************************************************
void alephSolve(void){
  debug()<<"\33[1;31m[alephSolve]\33[0m";
#ifndef ALEPH_INDEX
  int nb_iteration;
  double residual_norm[4];
  m_aleph_mat->assemble();
  m_aleph_rhs->setLocalComponents(vector_indexs.size(),vector_indexs,rhs);
  m_aleph_rhs->assemble();
  m_aleph_sol->setLocalComponents(vector_indexs.size(),vector_indexs,vector_zeroes);
  m_aleph_sol->assemble();
  m_aleph_mat->solve(m_aleph_sol, m_aleph_rhs, nb_iteration, &residual_norm[0], m_aleph_params, true);
  AlephVector *solution=m_aleph_kernel->syncSolver(0,nb_iteration,&residual_norm[0]);
  info() << "Solved in \33[7m" << nb_iteration << "\33[m iterations,"
         << "residuals=[\33[1m" << residual_norm[0] <<"\33[m,"<< residual_norm[3]<<"]";
  solution->getLocalComponents(vector_indexs.size(),vector_indexs,rhs);
//#warning weird getLocalComponents with rhs
#else
  alephSolveWithoutIndex();
#endif
}
