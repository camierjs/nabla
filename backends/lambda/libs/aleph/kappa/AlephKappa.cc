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
#include "AlephStd.h"
#include "Aleph.h"
#include "AlephInterface.h"
#include "IAlephFactory.h"

#include "AlephKappa.h"

AlephKappaService::
AlephKappaService(ITraceMng *tm):TraceAccessor(tm),
                                 m_kernel(NULL),
                                 m_world_parallel(NULL),
                                 m_world_rank(-1),
                                 m_size(-1),
                                 m_world_size(-1),
                                 m_factory(NULL),
                                 m_underlying_solver(-1),
  m_solver_size(-1),
                                 m_reorder(false)
{
   debug() << "[AlephKappaService] NEW"; traceMng()->flush();
   //m_application = sbi.application();
}

AlephKappaService::~AlephKappaService(){
  if (m_kernel) delete m_kernel;
  //if (m_factory) delete m_factory;
}


void AlephKappaService::execute(void){
  debug() << "[AlephKappaService] Retrieving world size...";
  m_world_size=m_world_parallel->commSize();
  m_world_rank=m_world_parallel->commRank();
  
  debug()<<"[AlephKappaService] I should be an additional site #"
           << m_world_rank <<" among "<<m_world_size;

  debug() << "[AlephKappaService] Retrieving configuration...";
  // cfg(0): m_underlying_solver
  // cfg(1): m_solver_size
  // cfg(2): m_reorder
  // cfg(3): m_size
  Array<Integer> cfg(4);
  m_world_parallel->broadcast(cfg,0);
  for(Integer rnk=0, max=cfg.size(); rnk<max;rnk+=1){
    debug() << "[AlephKappaService] cfg["<<rnk<<"]="<<cfg[rnk];
  }

  debug() << "[AlephKappaService] factory";
  m_factory=new AlephFactory(m_world_parallel->traceMng());
  debug() << "[AlephKappaService] kernel";
  m_kernel=new AlephKernel(m_world_parallel,
                           m_size=cfg.at(3),
                           m_factory,
                           m_underlying_solver=cfg.at(0),
                           m_solver_size=cfg.at(1),
                           m_reorder=(cfg.at(2)==1)?true:false);
  
  AlephParams *params=
    new AlephParams(traceMng(),
                    1.0e-10,                         // m_param_epsilon epsilon de convergence
                    2000,                            // m_param_max_iteration nb max iterations
                    TypesSolver::DIAGONAL,           // m_param_preconditioner_method: DIAGONAL, AMG, IC
                    TypesSolver::PCG,                // m_param_solver_method m�thode de r�solution
                    -1,                              // m_param_gamma 
                    -1.0,                            // m_param_alpha
                    false,                           // m_param_xo_user par d�faut Xo n'est pas �gal � 0
                    false,                           // m_param_check_real_residue
                    false,                           // m_param_print_real_residue
                    false,                           // m_param_debug_info
                    1.e-40,                          // m_param_min_rhs_norm
                    false,                           // m_param_convergence_analyse
                    true,                            // m_param_stop_error_strategy
                    false,                           // m_param_write_matrix_to_file_error_strategy
                    "SolveErrorAlephMatrix.dbg",     // m_param_write_matrix_name_error_strategy
                    false,                           // m_param_listing_output
                    0.,                              // m_param_threshold
                    false,                           // m_param_print_cpu_time_resolution
                    0,                               // m_param_amg_coarsening_method
                    0,                               // m_param_output_level
                    1,                               // m_param_amg_cycle
                    1,                               // m_param_amg_solver_iterations
                    1,                               // m_param_amg_smoother_iterations
                    TypesSolver::SymHybGSJ_smoother, // m_param_amg_smootherOption
                    TypesSolver::ParallelRugeStuben, // m_param_amg_coarseningOption
                    TypesSolver::CG_coarse_solver,   // m_param_amg_coarseSolverOption
                    false,                           // m_param_keep_solver_structure
                    false,                           // m_param_sequential_solver
                    TypesSolver::RB);                // m_param_criteria_stop
  
  
  Array<AlephMatrix*> A_matrix_queue;
  Integer aleph_vector_idx=0;
  Array<AlephVector*>b;
  Array<AlephVector*>x;
  // Ce flag permet d'�viter de prendre en compte le create() du vecteur temporaire des arguments du solveur
  bool firstVectorCreateForTmp=true;

  traceMng()->flush();
  
  while(true){
    Array<unsigned long> token(1);
    debug() << "[AlephKappaService] listening for a token...";
    traceMng()->flush();
    m_world_parallel->broadcast(token.view(),0);
    traceMng()->flush();
    debug() << "[AlephKappaService] found token "<<token.at(0);
    traceMng()->flush();
 
    switch (token.at(0)){
      
      /************************************************************************
       * AlephKernel::initialize
       ************************************************************************/
    case (0xd80dee82l):{
      debug() << "[AlephKappaService] AlephKernel::initialize!";
      Array<Integer> args(2);
      // R�cup�ration des global_nb_row et local_nb_row
      m_world_parallel->broadcast(args.view(),0);
      m_kernel->initialize((Integer)args.at(0),(Integer)args.at(1));
      break;
    }

      /************************************************************************
       * AlephKernel::createSolverMatrix
       ************************************************************************/
    case (0xef162166l):
      debug() << "[AlephKappaService] AlephKernel::createSolverMatrix (new A["<<A_matrix_queue.size()<<"])!";
      firstVectorCreateForTmp=true; // On indique que le prochain create() est relatif au vecteur tmp
      A_matrix_queue.add(m_kernel->createSolverMatrix());
      break;

      /************************************************************************
       * AlephKernel::postSolver
       ************************************************************************/
    case (0xba9488bel):{
      debug() << "[AlephKappaService] AlephKernel::postSolver!";
      Array<Real> real_args(4);
      m_world_parallel->broadcast(real_args.view(),0);
      params->setEpsilon(real_args.at(0));
      params->setAlpha(real_args.at(1));
      params->setMinRHSNorm(real_args.at(2));
      params->setDDMCParameterAmgDiagonalThreshold(real_args.at(3));
      
      Array<int> bool_args(11);
      m_world_parallel->broadcast(bool_args.view(),0);
      params->setXoUser((bool)bool_args.at(0));
      params->setCheckRealResidue((bool)bool_args.at(1));
      params->setPrintRealResidue((bool)bool_args.at(2));
      params->setDebugInfo((bool)bool_args.at(3));
      params->setConvergenceAnalyse((bool)bool_args.at(4));
      params->setStopErrorStrategy((bool)bool_args.at(5));
      params->setWriteMatrixToFileErrorStrategy((bool)bool_args.at(6));
      params->setDDMCParameterListingOutput((bool)bool_args.at(7));
      params->setPrintCpuTimeResolution((bool)bool_args.at(8));
      params->setKeepSolverStructure((bool)bool_args.at(9));
      params->setSequentialSolver((bool)bool_args.at(10));
    
      Array<Integer> int_args(13);
      m_world_parallel->broadcast(int_args.view(),0);
      params->setMaxIter(int_args.at(0));
      params->setGamma(int_args.at(1));
      params->setPrecond((TypesSolver::ePreconditionerMethod)int_args.at(2));
      params->setMethod((TypesSolver::eSolverMethod)int_args.at(3));
      params->setAmgCoarseningMethod((TypesSolver::eAmgCoarseningMethod)int_args.at(4));
      params->setOutputLevel(int_args.at(5));
      params->setAmgCycle(int_args.at(6));
      params->setAmgSolverIter(int_args.at(7));
      params->setAmgSmootherIter(int_args.at(8));
      params->setAmgSmootherOption((TypesSolver::eAmgSmootherOption)int_args.at(9));
      params->setAmgCoarseningOption((TypesSolver::eAmgCoarseningOption)int_args.at(10));
      params->setAmgCoarseSolverOption((TypesSolver::eAmgCoarseSolverOption)int_args.at(11));
      params->setCriteriaStop((TypesSolver::eCriteriaStop)int_args.at(12));
      
      m_kernel->postSolver(params,NULL,NULL,NULL);
      break;
    }
     
      /************************************************************************
       * AlephKernel::createSolverVector
       ************************************************************************/
    case (0xc4b28f2l):{
      if ((aleph_vector_idx%2)==0){
        debug() << "[AlephKappaService] AlephKernel::createSolverVector (new b["<<b.size()<<"])";
        b.add(m_kernel->createSolverVector());
      }else{
        debug() << "[AlephKappaService] AlephKernel::createSolverVector (new x["<<x.size()<<"])";
        x.add(m_kernel->createSolverVector());
      }
      aleph_vector_idx+=1;
      break;
    }
      
      /************************************************************************
       * AlephMatrix::create(void)
       ************************************************************************/
    case (0xfff06e2l):{
      debug() << "[AlephKappaService] AlephMatrix::create(void)!";
      A_matrix_queue.at(m_kernel->index())->create();
      break;
    }
      
      /************************************************************************
       * AlephVector::create
       ************************************************************************/
    case (0x6bdba30al):{
      if (firstVectorCreateForTmp){     // Si c'est pour le vecteur tmp, on skip
        debug() << "[AlephKappaService] firstVectorCreateForTmp";
        firstVectorCreateForTmp=false;  // Et on annonce que c'est fait pour le tmp
        break;
      }
      
      if ((aleph_vector_idx%2)==0){
        debug() << "[AlephKappaService] AlephVector::create (b["<<m_kernel->index()<<"])";
        b.at(m_kernel->index())->create();
      }else{
        debug() << "[AlephKappaService] AlephVector::create (x["<<m_kernel->index()<<"])";
        x.at(m_kernel->index())->create();
      }
      aleph_vector_idx+=1;
      break;
    }
      
      /************************************************************************
       * AlephMatrix::assemble
       ************************************************************************/
    case (0x74f253cal):{
      debug() << "[AlephKappaService] AlephMatrix::assemble! (kernel->index="<<m_kernel->index()<<")";
      Array<Integer> setValue_idx(1);
      m_world_parallel->broadcast(setValue_idx.view(),0);
      // On le fait avant pour seter le flag � true
      m_kernel->topology()->create(setValue_idx.at(0));
      A_matrix_queue.at(m_kernel->index())->assemble();
      break;
    }
      
      /************************************************************************
       * AlephVector::assemble
       ************************************************************************/
    case (0xec7a979fl):{
      if ((aleph_vector_idx%2)==0){
        debug() << "[AlephKappaService] AlephVector::assemble! (b"<<m_kernel->index()<<")";
        b.at(m_kernel->index())->assemble();
      }else{
        debug() << "[AlephKappaService] AlephVector::assemble! (x"<<m_kernel->index()<<")";
        x.at(m_kernel->index())->assemble();
      }
      aleph_vector_idx+=1;
      break;
    }
      
      /************************************************************************
       * AlephKernel::syncSolver
       ************************************************************************/
    case (0xbf8d3adfl):{
      debug() << "[AlephKappaService] AlephKernel::syncSolver";
      traceMng()->flush();
      Array<Integer> gid(1);
      m_world_parallel->broadcast(gid.view(),0);
      
      Integer nb_iteration;
      Real residual_norm[4];
      debug() << "[AlephKappaService] AlephKernel::syncSolver group id="<<gid.at(0);
      traceMng()->flush();
      m_kernel->syncSolver(gid.at(0), nb_iteration, &residual_norm[0]);
      break;
    }
      
      /************************************************************************
       * SessionExec::executeRank
       ************************************************************************/
    case (0xdfeb699fl):{
      debug() << "[AlephKappaService] AlephKernel::finalize!";
      traceMng()->flush();
      delete params;
      return;
    }
      
      /************************************************************************
       * Should never happen
       ************************************************************************/
    default:
      debug() << "[AlephKappaService] default";
      traceMng()->flush();
      throw FatalErrorException("execute", "Unknown token for handshake");
    }
    traceMng()->flush();
  }
  throw FatalErrorException("execute", "Should never be there!");
}
