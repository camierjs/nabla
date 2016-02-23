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
#ifndef LAMBDA_ALEPH_INTERFACE_HYPRE_H
#define LAMBDA_ALEPH_INTERFACE_HYPRE_H

#define HAVE_MPI
#define MPI_COMM_SUB MPI_COMM_WORLD // (*(MPI_Comm*)(m_kernel->subParallelMng(m_index)->getMPICommunicator()))
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include "HYPRE.h"
#include "HYPRE_utilities.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "krylov.h"

#ifndef ItacRegion
#define ItacRegion(a,x)
#endif

using std::ostream;
//using std::string;

static void check(const char* hypre_func,
                  int error_code)  {
  if (error_code==0) return;
  char buf[8192];
  HYPRE_DescribeError(error_code,buf);
  std::cout << "\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            << "\nHYPRE ERROR in function "
            << hypre_func
            << "\nError_code=" << error_code
            << "\nMessage=" << buf
            << "\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            << "\n"
            << std::flush;
  throw  std::logic_error("HYPRE Check");
}


static void hypreCheck(const char* hypre_func,
                       int error_code){
  check(hypre_func,error_code);
  int r = HYPRE_GetError();
  if (r!=0)
    std::cout << "HYPRE GET ERROR r=" << r
         << " error_code=" << error_code << " func=" << hypre_func << '\n';
}


/******************************************************************************
 AlephVectorHypre
 *****************************************************************************/
class AlephVectorHypre: public IAlephVector{
public:
  AlephVectorHypre(ITraceMng *tm,AlephKernel *kernel, int index):
  IAlephVector(tm,kernel,index),
  m_hypre_ijvector(0),
  m_hypre_parvector(0),
  jSize(0),
  jUpper(0),
  jLower(-1){
  debug()<<"[AlephVectorHypre::AlephVectorHypre] new SolverVectorHypre";
}

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
  void AlephVectorCreate(void){
  debug()<<"[AlephVectorHypre::AlephVectorCreate] HYPRE VectorCreate";
  void* object;

  for( int iCpu=0;iCpu<m_kernel->size();++iCpu){
    if (m_kernel->rank()!=m_kernel->solverRanks(m_index)[iCpu]) continue;
    debug() << "[AlephVectorHypre::AlephVectorCreate] adding contibution of core #"<<iCpu;
    if (jLower==-1) jLower=m_kernel->topology()->gathered_nb_row(iCpu);
    jUpper=m_kernel->topology()->gathered_nb_row(iCpu+1)-1;
  }

  debug()<<"[AlephVectorHypre::AlephVectorCreate] jLower="<<jLower<<", jupper="<<jUpper;
  // Mise à jour de la taille locale du buffer pour le calcul plus tard de la norme max, par exemple
  jSize=jUpper-jLower+1;
  
  hypreCheck("IJVectorCreate",             
             HYPRE_IJVectorCreate(MPI_COMM_SUB,
                                  jLower,
                                  jUpper,
                                  &m_hypre_ijvector));
  
  debug()<<"[AlephVectorHypre::AlephVectorCreate] HYPRE IJVectorSetObjectType"; 
  hypreCheck("IJVectorSetObjectType", HYPRE_IJVectorSetObjectType(m_hypre_ijvector,HYPRE_PARCSR));
  
  debug()<<"[AlephVectorHypre::AlephVectorCreate] HYPRE IJVectorInitialize";
  hypreCheck("HYPRE_IJVectorInitialize", HYPRE_IJVectorInitialize(m_hypre_ijvector));
 
  HYPRE_IJVectorGetObject(m_hypre_ijvector,&object);
  m_hypre_parvector = (HYPRE_ParVector)object;
  debug()<<"[AlephVectorHypre::AlephVectorCreate] done";
}

  
/******************************************************************************
 *****************************************************************************/
  void AlephVectorSet(const double *bfr_val, const int *bfr_idx, int size){
    debug()<<"[AlephVectorHypre::AlephVectorSet] size="<<size;
    hypreCheck("IJVectorSetValues", HYPRE_IJVectorSetValues(m_hypre_ijvector, size, bfr_idx, bfr_val));
  }
  
/******************************************************************************
 *****************************************************************************/
  int AlephVectorAssemble(void){
    debug()<<"[AlephVectorHypre::AlephVectorAssemble]";
    hypreCheck("IJVectorAssemble", HYPRE_IJVectorAssemble(m_hypre_ijvector));
    return 0;
  }
  
/******************************************************************************
 *****************************************************************************/
  void AlephVectorGet(double *bfr_val, const int *bfr_idx, int size){
    debug()<<"[AlephVectorHypre::AlephVectorGet] size="<<size;
    hypreCheck("HYPRE_IJVectorGetValues", HYPRE_IJVectorGetValues(m_hypre_ijvector, size, bfr_idx, bfr_val));
  }

  
 /******************************************************************************
  * norm_max
  *****************************************************************************/
  double norm_max(){
    double normInf=0.0;
    vector<HYPRE_Int> bfr_idx(jSize);
    vector<double> bfr_val(jSize);
 
    for(HYPRE_Int i=0;i<jSize;++i)
      bfr_idx[i]=jLower+i;
    
    hypreCheck("HYPRE_IJVectorGetValues", HYPRE_IJVectorGetValues(m_hypre_ijvector,
                                                                  jSize,
                                                                  &bfr_idx[0],
                                                                  &bfr_val[0]));
    for(HYPRE_Int i=0;i<jSize;++i){
      const double abs_val = ::fabs(bfr_val[i]);
      debug()<<"\t[AlephVectorHypre::norm_max] abs_val="<<abs_val;
      if (abs_val > normInf) normInf = abs_val;
    }
    debug()<<"\t[AlephVectorHypre::norm_max] normInf="<<normInf;
    normInf=m_kernel->subParallelMng(m_index)->reduce(Parallel::ReduceMax,normInf);
    debug()<<"\t[AlephVectorHypre::norm_max] reduced normInf="<<normInf;
    return normInf;
  }

  
/******************************************************************************
 *****************************************************************************/
  void writeToFile(const string filename){
    string filename_idx(filename);// + "_" + (int)m_kernel->subDomain()->commonVariables().globalIteration();
    debug()<<"[AlephVectorHypre::writeToFile]";
    hypreCheck("HYPRE_IJVectorPrint",
               HYPRE_IJVectorPrint(m_hypre_ijvector, filename_idx.c_str()));
  }
  
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
                   int index):
  IAlephMatrix(tm,kernel,index),
  m_hypre_ijmatrix(0){
  debug()<<"[AlephMatrixHypre] new AlephMatrixHypre";
}
  
  ~AlephMatrixHypre(){
    debug()<<"[~AlephMatrixHypre]";
  }
  
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
  void AlephMatrixCreate(void){
    debug()<<"[AlephMatrixHypre::AlephMatrixCreate] HYPRE MatrixCreate idx:"<<m_index;
    void *object;
    int ilower=-1;
    int iupper=0;
    for( int iCpu=0;iCpu<m_kernel->size();++iCpu){
      if (m_kernel->rank()!=m_kernel->solverRanks(m_index)[iCpu]) continue;
      if (ilower==-1) ilower=m_kernel->topology()->gathered_nb_row(iCpu);
      iupper=m_kernel->topology()->gathered_nb_row(iCpu+1)-1;
      debug() << "[AlephMatrixHypre::AlephMatrixCreate] ilower="<<ilower;
      debug() << "[AlephMatrixHypre::AlephMatrixCreate] iupper="<<iupper;
    }
    debug()<<"[AlephMatrixHypre::AlephMatrixCreate] ilower="<<ilower<<", iupper="<<iupper;

    int jlower=ilower;//0;
    int jupper=iupper;//m_kernel->topology()->gathered_nb_row(m_kernel->size())-1;
    debug()<<"[AlephMatrixHypre::AlephMatrixCreate] jlower="<<jlower<<", jupper="<<jupper;
    
    hypreCheck("HYPRE_IJMatrixCreate",
               HYPRE_IJMatrixCreate(MPI_COMM_SUB,
                                    ilower, iupper,
                                    jlower, jupper,
                                    &m_hypre_ijmatrix));

    debug()<<"[AlephMatrixHypre::AlephMatrixCreate] HYPRE IJMatrixSetObjectType";
    HYPRE_IJMatrixSetObjectType(m_hypre_ijmatrix,HYPRE_PARCSR);
    
    debug()<<"[AlephMatrixHypre::AlephMatrixCreate] HYPRE IJMatrixSetRowSizes";
    HYPRE_IJMatrixSetRowSizes(m_hypre_ijmatrix,
                              (HYPRE_Int*)&m_kernel->topology()->gathered_nb_row_elements()[0]);
    debug()<<"[AlephMatrixHypre::AlephMatrixCreate] HYPRE IJMatrixInitialize";
    HYPRE_IJMatrixInitialize(m_hypre_ijmatrix);
    HYPRE_IJMatrixGetObject(m_hypre_ijmatrix,&object);
    m_hypre_parmatrix = (HYPRE_ParCSRMatrix)object;
  }
 

/******************************************************************************
 *****************************************************************************/
  void AlephMatrixSetFilled(bool){}

/******************************************************************************
 *****************************************************************************/
  int AlephMatrixAssemble(void){
    debug()<<"[AlephMatrixHypre::AlephMatrixAssemble]";
    hypreCheck("HYPRE_IJMatrixAssemble",
               HYPRE_IJMatrixAssemble(m_hypre_ijmatrix));
    return 0;
  }

/******************************************************************************
 *****************************************************************************/
  void AlephMatrixFill(int size, int *rows, int *cols, double *values){
    debug()<<"[AlephMatrixHypre::AlephMatrixFill] size="<<size;
    int rtn=0;
    int col[1]={1};
    for(int i=0;i<size;i++){
      rtn+=HYPRE_IJMatrixSetValues(m_hypre_ijmatrix, 1, col, &rows[i], &cols[i], &values[i]);
    }
    hypreCheck("HYPRE_IJMatrixSetValues",rtn);
    //HYPRE_IJMatrixSetValues(m_hypre_ijmatrix, nrows, ncols, rows, cols, values);
    debug()<<"[AlephMatrixHypre::AlephMatrixFill] done";
  }


  
/******************************************************************************
 * isAlreadySolved
 *****************************************************************************/
  bool isAlreadySolved(AlephVectorHypre *x,
                       AlephVectorHypre *b,
                       AlephVectorHypre *tmp,
                       double* residual_norm,
                       AlephParams* params) {
    const bool convergence_analyse = params->convergenceAnalyse();
    
    // test le second membre du système linéaire
    const double res0 = b->norm_max();
	
    if (convergence_analyse)
      info() << "analyse convergence : norme max du second membre res0 : " << res0;
   
    const double considered_as_null = params->minRHSNorm();
    if (res0 < considered_as_null) {
		HYPRE_ParVectorSetConstantValues(x->m_hypre_parvector, 0.0);
		residual_norm[0]= res0;
		if (convergence_analyse)
        info() << "analyse convergence : le second membre du système linéaire est inférieur à : " << considered_as_null;
		return true;
    }

    if (params->xoUser()) {
      // on test si b est déjà solution du système à epsilon près
      //matrix->vectorProduct(b, tmp_vector); tmp_vector->sub(x);
      //M->vector_multiply(*tmp,*x);  // tmp=A*x
      //tmp->substract(*tmp,*b);      // tmp=A*x-b

      // X= alpha* M.B + beta * X (lu dans les sources de HYPRE)
      HYPRE_ParCSRMatrixMatvec(1.0,m_hypre_parmatrix,x->m_hypre_parvector,0.,tmp->m_hypre_parvector);
      HYPRE_ParVectorAxpy(-1.0,b->m_hypre_parvector,tmp->m_hypre_parvector);
      
      const double residu= tmp->norm_max();
      //info() << "[IAlephHypre::isAlreadySolved] residu="<<residu;

      if (residu < considered_as_null) {
        if (convergence_analyse) {
          info() << "analyse convergence : |Ax0-b| est inférieur à " << considered_as_null;
          info() << "analyse convergence : x0 est déjà solution du système.";
        }
        residual_norm[0] = residu;
        return true;
      }

      const double relative_error = residu / res0;
      if (convergence_analyse)
        info() << "analyse convergence : résidu initial : " << residu
               << " --- residu relatif initial (residu/res0) : " << residu / res0;
     
      if (relative_error < (params->epsilon())) {
        if (convergence_analyse)
          info() << "analyse convergence : X est déjà solution du système";
        residual_norm[0] = residu;
        return true;
      }
    }
    return false;
  }

  
/******************************************************************************
 *****************************************************************************/
  int AlephMatrixSolve(AlephVector* x,
                       AlephVector* b,
                       AlephVector* t,
                       int& nb_iteration,
                       double* residual_norm,
                       AlephParams* solver_param){
    solver_param->setAmgCoarseningMethod(TypesSolver::AMG_COARSENING_AUTO);
    const string func_name("SolverMatrixHypre::solve");
    void* object;
    int ierr = 0;
  
    HYPRE_IJVector solution = (dynamic_cast<AlephVectorHypre*> (x->implementation()))->m_hypre_ijvector;
    HYPRE_IJVector RHS      = (dynamic_cast<AlephVectorHypre*> (b->implementation()))->m_hypre_ijvector;
    //HYPRE_IJVector tmp      = (dynamic_cast<AlephVectorHypre*> (t->implementation()))->m_hypre_ijvector;
  
    HYPRE_IJMatrixGetObject(m_hypre_ijmatrix,&object);
    HYPRE_ParCSRMatrix M = (HYPRE_ParCSRMatrix)object;
    HYPRE_IJVectorGetObject(solution,&object);
    HYPRE_ParVector X = (HYPRE_ParVector)object;
    HYPRE_IJVectorGetObject(RHS,&object);
    HYPRE_ParVector B = (HYPRE_ParVector)object;
    //HYPRE_IJVectorGetObject(tmp,&object);
    //HYPRE_ParVector T = (HYPRE_ParVector)object;

    if (isAlreadySolved((dynamic_cast<AlephVectorHypre*> (x->implementation())),
                        (dynamic_cast<AlephVectorHypre*> (b->implementation())),
                        (dynamic_cast<AlephVectorHypre*> (t->implementation())),
                        residual_norm,solver_param)){
      ItacRegion(isAlreadySolved,AlephMatrixHypre);
      debug() << "[AlephMatrixHypre::AlephMatrixSolve] isAlreadySolved !";
      nb_iteration = 0;
      return 0;
    }

    TypesSolver::ePreconditionerMethod preconditioner_method = solver_param->precond();
    //  TypesSolver::ePreconditionerMethod preconditioner_method = TypesSolver::NONE;
    TypesSolver::eSolverMethod solver_method = solver_param->method();
  
    // déclaration et initialisation du solveur
    HYPRE_Solver solver = 0;
  
    switch(solver_method){
    case TypesSolver::PCG      : initSolverPCG     (solver_param,solver); break;
    case TypesSolver::BiCGStab : initSolverBiCGStab(solver_param,solver); break;
    case TypesSolver::GMRES    : initSolverGMRES   (solver_param,solver); break;
    default : throw std::logic_error("solveur inconnu");
    }
  
    // déclaration et initialisation du preconditionneur
    HYPRE_Solver precond = 0;
  
    switch(preconditioner_method){
    case TypesSolver::NONE     : break;
    case TypesSolver::DIAGONAL : setDiagonalPreconditioner(solver_method,solver,precond); break;
    case TypesSolver::ILU      : setILUPreconditioner     (solver_method,solver,precond); break;
    case TypesSolver::SPAIstat : setSpaiStatPreconditioner(solver_method,solver,solver_param,precond); break;
    case TypesSolver::AMG      : setAMGPreconditioner     (solver_method,solver,solver_param,precond); break;
    case TypesSolver::AINV     : throw std::logic_error("preconditionnement AINV indisponible");
    case TypesSolver::SPAIdyn  : throw std::logic_error("preconditionnement SPAIdyn indisponible");
    case TypesSolver::ILUp     : throw std::logic_error("preconditionnement ILUp indisponible");
    case TypesSolver::IC       : throw std::logic_error("preconditionnement IC indisponible");
    case TypesSolver::POLY     : throw std::logic_error("preconditionnement POLY indisponible");
    default : throw std::logic_error("preconditionnement inconnu");
    }
  
    // résolution du système algébrique
    int    iteration  = 0;
    double residue    = 0.0;
  
    switch(solver_method){
    case TypesSolver::PCG      : ierr = solvePCG(solver_param,solver,M,B,X,iteration,residue); break;
    case TypesSolver::BiCGStab : ierr = solveBiCGStab(solver,M,B,X,iteration,residue); break;
    case TypesSolver::GMRES    : ierr = solveGMRES   (solver,M,B,X,iteration,residue); break;
    default : ierr=-3; return ierr;
    }
    nb_iteration  = static_cast<int>(iteration);
    residual_norm[0] = static_cast<double>(residue);

/* for(int i=0;i<8;++i){
   int idx[1];
   double valx[1]={-1.};
   //double valb[1]={-1.};
   idx[0]=i;
   HYPRE_IJVectorGetValues(solution, 1, idx, valx);
   debug()<<"[AlephMatrixHypre::AlephMatrixSolve] X["<<i<<"]="<<valx[0];
   //(static_cast<AlephVectorHypre*> (x->implementation()))->AlephVectorGet(valx,idx,1);
   //debug()<<"[AlephMatrixHypre::AlephMatrixSolve] x["<<i<<"]="<<valx[0];
   //HYPRE_IJVectorGetValues(RHS, 1, idx, valb);
   //debug()<<"[AlephMatrixHypre::AlephMatrixSolve] B["<<i<<"]="<<valb[0];
   }
*/
    switch(preconditioner_method){
    case TypesSolver::NONE     : break;
    case TypesSolver::DIAGONAL : break;
    case TypesSolver::ILU      : HYPRE_ParCSRPilutDestroy    (precond); break;
    case TypesSolver::SPAIstat : HYPRE_ParCSRParaSailsDestroy(precond); break;
    case TypesSolver::AMG      : HYPRE_BoomerAMGDestroy      (precond); break;
    default : throw std::logic_error("preconditionnement inconnu");
    }
  
    if ( iteration == solver_param->maxIter() && solver_param->stopErrorStrategy()){
      info() << "\n============================================================";
      info() << "\nCette erreur est retournée après " << iteration << "\n";
      info() << "\nOn a atteind le nombre max d'itérations du solveur.";
      info() << "\nIl est possible de demander au code de ne pas tenir compte de cette erreur.";
      info() << "\nVoir la documentation du jeu de données concernant le service solveur.";
      info() << "\n======================================================";
      throw  std::logic_error("AlephMatrixHypre::Solve On a atteind le nombre max d'itérations du solveur");
    }
    return ierr;
  }

  
/******************************************************************************
 *****************************************************************************/
  void writeToFile(const string filename){
    HYPRE_IJMatrixPrint(m_hypre_ijmatrix, filename.c_str());
  }


/******************************************************************************
 *****************************************************************************/
  void initSolverPCG(const AlephParams* solver_param,HYPRE_Solver& solver){
    //const string func_name=string("SolverMatrixHypre::initSolverPCG");
    double epsilon = solver_param->epsilon();
    int    max_it  = solver_param->maxIter();
    int output_level=solver_param->getOutputLevel();

    HYPRE_ParCSRPCGCreate(MPI_COMM_SUB, &solver);
    HYPRE_ParCSRPCGSetMaxIter(solver,max_it);
    HYPRE_ParCSRPCGSetTol(solver,epsilon);
    HYPRE_ParCSRPCGSetTwoNorm(solver,1);
    HYPRE_ParCSRPCGSetPrintLevel(solver,output_level);
  }


/******************************************************************************
 *****************************************************************************/
  void initSolverBiCGStab(const AlephParams* solver_param,HYPRE_Solver& solver){
    //const String func_name="SolverMatrixHypre::initSolverBiCGStab";
    double epsilon = solver_param->epsilon();
    int    max_it  = solver_param->maxIter();
    int output_level=solver_param->getOutputLevel();

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_SUB, &solver);
    HYPRE_ParCSRBiCGSTABSetMaxIter(solver, max_it);
    HYPRE_ParCSRBiCGSTABSetTol(solver,epsilon);
    HYPRE_ParCSRBiCGSTABSetPrintLevel(solver,output_level);
  }


/******************************************************************************
 *****************************************************************************/
  void initSolverGMRES(const AlephParams* solver_param,HYPRE_Solver& solver){
    //const String func_name="SolverMatrixHypre::initSolverGMRES";
    double epsilon = solver_param->epsilon();
    int    max_it  = solver_param->maxIter();
    int output_level=solver_param->getOutputLevel();

    HYPRE_ParCSRGMRESCreate(MPI_COMM_SUB, &solver);
    const int krylov_dim = 20; // dimension Krylov space for GMRES
    HYPRE_ParCSRGMRESSetKDim(solver, krylov_dim);
    HYPRE_ParCSRGMRESSetMaxIter(solver, max_it);
    HYPRE_ParCSRGMRESSetTol(solver, epsilon);
    HYPRE_ParCSRGMRESSetPrintLevel(solver, output_level);
  }

  
/******************************************************************************
 *****************************************************************************/
  void setDiagonalPreconditioner( const TypesSolver::eSolverMethod solver_method ,
                                  const HYPRE_Solver&              solver        ,
                                  HYPRE_Solver&                    precond       ){
    //const String func_name="SolverMatrixHypre::setDiagonalPreconditioner";
    switch(solver_method){
    case TypesSolver::PCG :
      HYPRE_ParCSRPCGSetPrecond(solver,
                                HYPRE_ParCSRDiagScale,
                                HYPRE_ParCSRDiagScaleSetup,
                                precond);
      break;
    case TypesSolver::BiCGStab :
      HYPRE_ParCSRBiCGSTABSetPrecond(solver,
                                     HYPRE_ParCSRDiagScale,
                                     HYPRE_ParCSRDiagScaleSetup,
                                     precond);
      break;
    case TypesSolver::GMRES :
      HYPRE_ParCSRGMRESSetPrecond(solver,
                                  HYPRE_ParCSRDiagScale,
                                  HYPRE_ParCSRDiagScaleSetup,
                                  precond);
      break;
    default:
      std::logic_error("solveur inconnu pour preconditionnement 'Diagonal'");
    }
  }


/******************************************************************************
 *****************************************************************************/
  void setILUPreconditioner(const TypesSolver::eSolverMethod solver_method ,
                            const HYPRE_Solver& solver,
                            HYPRE_Solver& precond){
    //const String func_name="SolverMatrixHypre::setILUPreconditioner";
    switch(solver_method){
    case TypesSolver::PCG :
      throw std::logic_error("solveur PCG indisponible avec le preconditionnement 'ILU'");
      break;
    case TypesSolver::BiCGStab :
      HYPRE_ParCSRPilutCreate(MPI_COMM_SUB, &precond );
      HYPRE_ParCSRBiCGSTABSetPrecond(solver,
                                     HYPRE_ParCSRPilutSolve ,
                                     HYPRE_ParCSRPilutSetup ,
                                     precond);
      break;
    case TypesSolver::GMRES :
      HYPRE_ParCSRPilutCreate(MPI_COMM_SUB,
                              &precond );
      HYPRE_ParCSRGMRESSetPrecond(solver,
                                  HYPRE_ParCSRPilutSolve,
                                  HYPRE_ParCSRPilutSetup,
                                  precond);
      break;
    default:
      throw std::logic_error("solveur inconnu pour preconditionnement ILU\n");
    }
  }


/******************************************************************************
 *****************************************************************************/
  void setSpaiStatPreconditioner(const TypesSolver::eSolverMethod solver_method ,
                                 const HYPRE_Solver& solver,
                                 const AlephParams* solver_param,
                                 HYPRE_Solver& precond){
    HYPRE_ParCSRParaSailsCreate(MPI_COMM_SUB,&precond);
    double alpha = solver_param->alpha();
    int gamma = solver_param->gamma();
    if (alpha < 0.0) alpha = 0.1; // valeur par defaut pour le parametre de tolerance
    if (gamma == -1) gamma = 1;   // valeur par defaut pour le parametre de remplissage
    HYPRE_ParCSRParaSailsSetParams(precond,alpha,gamma);
    switch(solver_method){
    case TypesSolver::PCG:
      HYPRE_ParCSRPCGSetPrecond( solver,HYPRE_ParCSRParaSailsSolve ,HYPRE_ParCSRParaSailsSetup ,precond);
      break;
    case TypesSolver::BiCGStab:
      throw std::logic_error("AlephMatrixHypre::setSpaiStatPreconditionersolveur 'BiCGStab' invalide pour preconditionnement 'SPAIstat'");
      break;
    case TypesSolver::GMRES:
      // matrice non symétrique
      HYPRE_ParCSRParaSailsSetSym(precond,0);
      HYPRE_ParCSRGMRESSetPrecond( solver,HYPRE_ParaSailsSolve ,HYPRE_ParaSailsSetup ,precond);
      break;
    default:
      throw std::logic_error("AlephMatrixHypre::setSpaiStatPreconditionersolveur inconnu pour preconditionnement 'SPAIstat'\n");
      break;
    }
  }


/******************************************************************************
 *****************************************************************************/
  void setAMGPreconditioner(const TypesSolver::eSolverMethod solver_method ,
                            const HYPRE_Solver& solver        ,
                            const AlephParams* solver_param  ,
                            HYPRE_Solver& precond){
    // defaults for BoomerAMG from hypre example -- lc
    // TODO : options and defaults values must be completed
    double trunc_factor = 0.1;      // set AMG interpolation truncation factor = val
    int    cycle_type   =  solver_param->getAmgCycle();          // set AMG cycles (1=V, 2=W, etc.)
    int    coarsen_type = solver_param->amgCoarseningMethod();
    // Ruge coarsening (local) if <val> == 1
    int    relax_default = 3;       // relaxation type <val> :
    //        0=Weighted Jacobi
    //        1=Gauss-Seidel (very slow!)
    //        3=Hybrid Jacobi/Gauss-Seidel
    int    num_sweep = 1;           // Use <val> sweeps on each level (here 1)
    int    hybrid = 1;              // no switch in coarsening if -1
    int    measure_type = 1;        // use globale measures
    double max_row_sum = 1.0;       // set AMG maximum row sum threshold for dependency weakening

    int max_levels = 50; // 25;  // maximum number of AMG levels
    const int gamma = solver_param->gamma();
    if (gamma != -1) max_levels = gamma ; // utilisation de la valeur du jeu de donnees

    double strong_threshold = 0.1; // 0.25; // set AMG threshold Theta = val
    const double alpha = solver_param->alpha();
    if (alpha>0.0) strong_threshold = alpha; // utilisation de la valeur du jeu de donnees
    // news
    int output_level=solver_param->getOutputLevel();

    int*    num_grid_sweeps   = hypre_TAlloc(int,4);
    int*    grid_relax_type   = hypre_TAlloc(int,4);
    int**   grid_relax_points = hypre_TAlloc(int*,4);
    double* relax_weight      = hypre_TAlloc(double,max_levels);

    for (int i=0; i<max_levels; i++) relax_weight[i] = 1.0;

    if (coarsen_type == 5){
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = relax_default;
      grid_relax_points[0] = hypre_TAlloc(int, 3);
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;

      /* down cycle */
      num_grid_sweeps[1] = 4; 
      grid_relax_type[1] = relax_default;
      grid_relax_points[1] = hypre_CTAlloc(int, 4);
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;

      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = relax_default;
      grid_relax_points[2] = hypre_TAlloc(int,4);
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
    }
    else{
      /* fine grid */
      num_grid_sweeps[0] = 2*num_sweep;
      grid_relax_type[0] = relax_default;
      grid_relax_points[0] = hypre_TAlloc(int,(2*num_sweep));
      for (int i=0; i<2*num_sweep; i+=2){
        grid_relax_points[0][i] = -1;
        grid_relax_points[0][i+1] = 1;
      }

      /* down cycle */
      num_grid_sweeps[1] = 2*num_sweep;
      grid_relax_type[1] = relax_default;
      grid_relax_points[1] = hypre_TAlloc(int,(2*num_sweep));
      for (int i=0; i<2*num_sweep; i+=2){
        grid_relax_points[1][i] = -1;
        grid_relax_points[1][i+1] = 1;
      }

      /* up cycle */
      num_grid_sweeps[2] = 2*num_sweep;
      grid_relax_type[2] = relax_default;
      grid_relax_points[2] = hypre_TAlloc(int,(2*num_sweep));
      for (int i=0; i<2*num_sweep; i+=2){
        grid_relax_points[2][i] = -1;
        grid_relax_points[2][i+1] = 1;
      }
    }

    /* coarsest grid */
    num_grid_sweeps[3] = 1;
    grid_relax_type[3] = 9;
    grid_relax_points[3] = hypre_TAlloc(int,1);
    grid_relax_points[3][0] = 0;

    // end of default seting

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond,output_level);
    HYPRE_BoomerAMGSetCoarsenType(precond,(hybrid*coarsen_type));
    HYPRE_BoomerAMGSetMeasureType(precond,measure_type);
    HYPRE_BoomerAMGSetStrongThreshold(precond, strong_threshold);
    HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);
    HYPRE_BoomerAMGSetCycleType(precond, cycle_type);
    HYPRE_BoomerAMGSetNumGridSweeps(precond, num_grid_sweeps);
    HYPRE_BoomerAMGSetGridRelaxType(precond, grid_relax_type);
    HYPRE_BoomerAMGSetRelaxWeight(precond, relax_weight);
    HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
    HYPRE_BoomerAMGSetTol(precond,0.0);
    HYPRE_BoomerAMGSetMaxLevels(precond,max_levels);
    HYPRE_BoomerAMGSetMaxRowSum(precond,max_row_sum);

    switch(solver_method){
    case TypesSolver::PCG :
      HYPRE_ParCSRPCGSetPrecond
        ( solver               ,
          HYPRE_BoomerAMGSolve ,
          HYPRE_BoomerAMGSetup ,
          precond              )
        ;
      break;
    case TypesSolver::BiCGStab :
      HYPRE_ParCSRBiCGSTABSetPrecond
        ( solver               ,
          HYPRE_BoomerAMGSolve ,
          HYPRE_BoomerAMGSetup ,
          precond              )
        ;
      break;
    case TypesSolver::GMRES :
      HYPRE_ParCSRGMRESSetPrecond
        ( solver               ,
          HYPRE_BoomerAMGSolve ,
          HYPRE_BoomerAMGSetup ,
          precond              )
        ;
      break;
    default :
      throw std::logic_error("AlephMatrixHypre::setAMGPreconditionersolveur inconnu pour preconditionnement 'AMG'\n");
    }
  }


/******************************************************************************
 *****************************************************************************/
  bool solvePCG
  ( const AlephParams* solver_param ,
    HYPRE_Solver&       solver       ,
    HYPRE_ParCSRMatrix& M            ,
    HYPRE_ParVector&    B            ,
    HYPRE_ParVector&    X            ,
    int&                iteration    ,
    double&             residue      )
  {
    //const String func_name="SolverMatrixHypre::solvePCG";
    const bool xo = solver_param->xoUser();
    bool error = false;

    if (!xo) HYPRE_ParVectorSetConstantValues(X, 0.0);
    HYPRE_ParCSRPCGSetup(solver, M, B, X);
    HYPRE_ParCSRPCGSolve(solver, M, B, X);
    HYPRE_ParCSRPCGGetNumIterations(solver,&iteration);
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver,&residue);

    int converged = 0;
    HYPRE_PCGGetConverged(solver, &converged);
    error |= (!converged);

    HYPRE_ParCSRPCGDestroy(solver);

    return !error;
  }


/******************************************************************************
 *****************************************************************************/
  bool solveBiCGStab
  ( HYPRE_Solver&       solver       ,
    HYPRE_ParCSRMatrix& M            ,
    HYPRE_ParVector&    B            ,
    HYPRE_ParVector&    X            ,
    int&                iteration    ,
    double&             residue      )
  {
    //const String func_name="SolverMatrixHypre::solveBiCGStab";
    bool error = false;
    HYPRE_ParVectorSetRandomValues(X, 775);
    HYPRE_ParCSRBiCGSTABSetup(solver, M, B, X);
    HYPRE_ParCSRBiCGSTABSolve(solver, M, B, X);
    HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &iteration);
    HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver,&residue);

    int converged = 0;
    hypre_BiCGSTABGetConverged(solver, &converged);
    error |= (!converged);

    HYPRE_ParCSRBiCGSTABDestroy(solver);

    return !error;
  }


/******************************************************************************
 *****************************************************************************/
  bool solveGMRES
  ( HYPRE_Solver&       solver       ,
    HYPRE_ParCSRMatrix& M            ,
    HYPRE_ParVector&    B            ,
    HYPRE_ParVector&    X            ,
    int&                iteration    ,
    double&             residue      )
  {
    //const String func_name="SolverMatrixHypre::solveGMRES";
    bool error = false;
    HYPRE_ParCSRGMRESSetup(solver, M, B, X);
    HYPRE_ParCSRGMRESSolve(solver, M, B, X);
    HYPRE_ParCSRGMRESGetNumIterations(solver, &iteration);
    HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver,&residue);

    int converged = 0;
    HYPRE_GMRESGetConverged(solver, &converged);
    error |= (!converged);

    HYPRE_ParCSRGMRESDestroy(solver);
    return !error;
  }

 private:
  HYPRE_IJMatrix m_hypre_ijmatrix; 
  HYPRE_ParCSRMatrix m_hypre_parmatrix; 
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HypreAlephFactoryImpl: public IAlephFactoryImpl{
public:
  HypreAlephFactoryImpl();
  ~HypreAlephFactoryImpl();
public:
  virtual void initialize();
  virtual IAlephTopology* createTopology(ITraceMng* tm,
                                         AlephKernel* kernel,
                                         int index,
                                         int nb_row_size);
  virtual IAlephVector* createVector(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     int index);
  virtual IAlephMatrix* createMatrix(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     int index);
private:
  vector<IAlephVector*> m_IAlephVectors;
  vector<IAlephMatrix*> m_IAlephMatrixs;
};

#endif // _ALEPH_INTERFACE_HYPRE_H_
