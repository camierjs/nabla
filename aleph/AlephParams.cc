///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
#include "Aleph.h"

AlephParams::AlephParams():
  m_param_epsilon(1.0e-10),
  m_param_max_iteration(1024),
  m_param_preconditioner_method(TypesSolver::DIAGONAL),
  m_param_solver_method(TypesSolver::PCG),
  m_param_gamma(-1),
  m_param_alpha(-1.0),
  m_param_xo_user(false),
  m_param_check_real_residue(false),
  m_param_print_real_residue(false),
  m_param_debug_info(false),
  m_param_min_rhs_norm(1.e-20),
  m_param_convergence_analyse(false),
  m_param_stop_error_strategy(true),
  m_param_write_matrix_to_file_error_strategy(false),
  m_param_write_matrix_name_error_strategy("SolveErrorAlephMatrix.dbg"),
  m_param_listing_output(false),
  m_param_threshold(0.0),
  m_param_print_cpu_time_resolution(false),
  m_param_amg_coarsening_method(0),
  m_param_output_level(0),
  m_param_amg_cycle(1),
  m_param_amg_solver_iterations(1),
  m_param_amg_smoother_iterations(1),
  m_param_amg_smootherOption(TypesSolver::SymHybGSJ_smoother),
  m_param_amg_coarseningOption(TypesSolver::ParallelRugeStuben),
  m_param_amg_coarseSolverOption(TypesSolver::CG_coarse_solver),
  m_param_keep_solver_structure(false),
  m_param_sequential_solver(false),
  m_param_criteria_stop(TypesSolver::RB)
{
  //debug() << "\33[1;4;33m\t[AlephParams] New"<<"\33[0m";
}

AlephParams::AlephParams(ITraceMng *tm,
                         double epsilon, // epsilon de convergence
								 int max_iteration, // nb max iterations
								 TypesSolver::ePreconditionerMethod preconditioner_method, //  préconditionnement utilisé (defaut DIAG)
								 TypesSolver::eSolverMethod solver_method, // méthode de résolution par defaut PCG
								 int gamma, // destine au parametrage des préconditionnements
								 double alpha, // destine au parametrage des préconditionnements
								 bool xo_user, // permet a l'utilisateur d'initialiser le PGC avec un Xo different de zero
								 bool check_real_residue,
								 bool print_real_residue,
								 bool debug_info,
								 double min_rhs_norm,
								 bool convergence_analyse,
								 bool stop_error_strategy,
								 bool write_matrix_to_file_error_strategy,
								 string write_matrix_name_error_strategy,
								 bool listing_output,
								 double threshold,
								 bool print_cpu_time_resolution,
								 int amg_coarsening_method,
								 int output_level,
								 int amg_cycle,
								 int amg_solver_iterations,
								 int amg_smoother_iterations,
								 TypesSolver::eAmgSmootherOption amg_smootherOption,
								 TypesSolver::eAmgCoarseningOption amg_coarseningOption,
								 TypesSolver::eAmgCoarseSolverOption amg_coarseSolverOption,
								 bool keep_solver_structure,
								 bool sequential_solver,
                         TypesSolver::eCriteriaStop param_criteria_stop):
  m_param_epsilon(epsilon),
  m_param_max_iteration(max_iteration),
  m_param_preconditioner_method(preconditioner_method),
  m_param_solver_method(solver_method),
  m_param_gamma(gamma),
  m_param_alpha(alpha),
  m_param_xo_user(xo_user),
  m_param_check_real_residue(check_real_residue),
  m_param_print_real_residue(print_real_residue),
  m_param_debug_info(debug_info),
  m_param_min_rhs_norm(min_rhs_norm),
  m_param_convergence_analyse(convergence_analyse),
  m_param_stop_error_strategy(stop_error_strategy),
  m_param_write_matrix_to_file_error_strategy(write_matrix_to_file_error_strategy),
  m_param_write_matrix_name_error_strategy(write_matrix_name_error_strategy),
  m_param_listing_output(listing_output),
  m_param_threshold(threshold),
  m_param_print_cpu_time_resolution(print_cpu_time_resolution),
  m_param_amg_coarsening_method(amg_coarsening_method),
  m_param_output_level(output_level),
  m_param_amg_cycle(amg_cycle),
  m_param_amg_solver_iterations(amg_solver_iterations),
  m_param_amg_smoother_iterations(amg_smoother_iterations),
  m_param_amg_smootherOption(amg_smootherOption),
  m_param_amg_coarseningOption(amg_coarseningOption),
  m_param_amg_coarseSolverOption(amg_coarseSolverOption),
  m_param_keep_solver_structure(keep_solver_structure),
  m_param_sequential_solver(sequential_solver),
  m_param_criteria_stop(param_criteria_stop){
  //debug() << "\33[1;4;33m\t[AlephParams] New"<<"\33[0m";
  }

AlephParams::~AlephParams() {
  //debug() << "\33[1;4;33m\t[~AlephParams]"<<"\33[0m";
}

// set
void AlephParams::setEpsilon(const double epsilon){m_param_epsilon = epsilon;}
void AlephParams::setMaxIter(const int max_iteration){m_param_max_iteration = max_iteration;}
void AlephParams::setPrecond(const TypesSolver::ePreconditionerMethod preconditioner_method){m_param_preconditioner_method=preconditioner_method;}
void AlephParams::setMethod(const TypesSolver::eSolverMethod solver_method){m_param_solver_method=solver_method;}
void AlephParams::setAlpha(const double alpha){m_param_alpha=alpha;}
void AlephParams::setGamma(const int gamma){m_param_gamma=gamma;}
void AlephParams::setXoUser(const bool xo_user){m_param_xo_user=xo_user;}
void AlephParams::setCheckRealResidue(const bool check_real_residue){m_param_check_real_residue=check_real_residue;}
void AlephParams::setPrintRealResidue(const bool print_real_residue){m_param_print_real_residue=print_real_residue;}
void AlephParams::setDebugInfo(const bool debug_info){m_param_debug_info=debug_info;}
void AlephParams::setMinRHSNorm(const double min_rhs_norm){m_param_min_rhs_norm=min_rhs_norm;}
void AlephParams::setConvergenceAnalyse(const bool convergence_analyse){m_param_convergence_analyse=convergence_analyse;}
void AlephParams::setStopErrorStrategy(const bool stop_error_strategy){m_param_stop_error_strategy=stop_error_strategy;}
void AlephParams::setWriteMatrixToFileErrorStrategy(const bool write_matrix_to_file_error_strategy){m_param_write_matrix_to_file_error_strategy=write_matrix_to_file_error_strategy;}
void AlephParams::setWriteMatrixNameErrorStrategy(const string& write_matrix_name_error_strategy){m_param_write_matrix_name_error_strategy=write_matrix_name_error_strategy;}
void AlephParams::setDDMCParameterListingOutput(const bool listing_output){m_param_listing_output=listing_output;}
void AlephParams::setDDMCParameterAmgDiagonalThreshold(const double threshold){m_param_threshold=threshold;}
void AlephParams::setPrintCpuTimeResolution(const bool print_cpu_time_resolution){m_param_print_cpu_time_resolution=print_cpu_time_resolution;}

void AlephParams::setAmgCoarseningMethod(const TypesSolver::eAmgCoarseningMethod method){
  switch (method) {
  case TypesSolver::AMG_COARSENING_AUTO     : m_param_amg_coarsening_method =  6; break;
  case TypesSolver::AMG_COARSENING_HYPRE_0  : m_param_amg_coarsening_method =  0; break;
  case TypesSolver::AMG_COARSENING_HYPRE_1  : m_param_amg_coarsening_method =  1; break;
  case TypesSolver::AMG_COARSENING_HYPRE_3  : m_param_amg_coarsening_method =  3; break;
  case TypesSolver::AMG_COARSENING_HYPRE_6  : m_param_amg_coarsening_method =  6; break;
  case TypesSolver::AMG_COARSENING_HYPRE_7  : m_param_amg_coarsening_method =  7; break;
  case TypesSolver::AMG_COARSENING_HYPRE_8  : m_param_amg_coarsening_method =  8; break;
  case TypesSolver::AMG_COARSENING_HYPRE_9  : m_param_amg_coarsening_method =  9; break;
  case TypesSolver::AMG_COARSENING_HYPRE_10 : m_param_amg_coarsening_method = 10; break;
  case TypesSolver::AMG_COARSENING_HYPRE_11 : m_param_amg_coarsening_method = 11; break;
  case TypesSolver::AMG_COARSENING_HYPRE_21 : m_param_amg_coarsening_method = 21; break;
  case TypesSolver::AMG_COARSENING_HYPRE_22 : m_param_amg_coarsening_method = 22; break;
  default: std::logic_error(A_FUNCINFO);
  }
}
void AlephParams::setOutputLevel(const int output_level){m_param_output_level=output_level;}
void AlephParams::setAmgCycle(const int amg_cycle){m_param_amg_cycle=amg_cycle;}
void AlephParams::setAmgSolverIter(const int amg_solver_iterations){m_param_amg_solver_iterations=amg_solver_iterations;}
void AlephParams::setAmgSmootherIter(const int amg_smoother_iterations){m_param_amg_smoother_iterations=amg_smoother_iterations;}
void AlephParams::setAmgSmootherOption(const TypesSolver::eAmgSmootherOption amg_smootherOption){m_param_amg_smootherOption=amg_smootherOption;}
void AlephParams::setAmgCoarseningOption(const TypesSolver::eAmgCoarseningOption amg_coarseningOption){m_param_amg_coarseningOption=amg_coarseningOption;}
void AlephParams::setAmgCoarseSolverOption(const TypesSolver::eAmgCoarseSolverOption amg_coarseSolverOption){m_param_amg_coarseSolverOption=amg_coarseSolverOption;}
void AlephParams::setKeepSolverStructure(const bool keep_solver_structure){m_param_keep_solver_structure=keep_solver_structure;}
void AlephParams::setSequentialSolver(const bool sequential_solver){m_param_sequential_solver=sequential_solver;}
void AlephParams::setCriteriaStop(const TypesSolver::eCriteriaStop criteria_stop){m_param_criteria_stop=criteria_stop;}


// get
double AlephParams::epsilon()const{return m_param_epsilon;}
int AlephParams::maxIter()const{return m_param_max_iteration;}
double AlephParams::alpha()const{return m_param_alpha;}
int AlephParams::gamma()const{return m_param_gamma;}
TypesSolver::ePreconditionerMethod AlephParams::precond(){return m_param_preconditioner_method;}
TypesSolver::eSolverMethod AlephParams::method(){return m_param_solver_method;}
bool AlephParams::xoUser()const{return m_param_xo_user;}
bool AlephParams::checkRealResidue()const{return m_param_check_real_residue;}
bool AlephParams::printRealResidue()const{return m_param_print_real_residue;}
bool AlephParams::debugInfo(){return m_param_debug_info;}
double AlephParams::minRHSNorm(){return m_param_min_rhs_norm;}
bool AlephParams::convergenceAnalyse(){return m_param_convergence_analyse;}
bool AlephParams::stopErrorStrategy(){return m_param_stop_error_strategy;}
bool AlephParams::writeMatrixToFileErrorStrategy(){return m_param_write_matrix_to_file_error_strategy;}
string AlephParams::writeMatrixNameErrorStrategy(){return m_param_write_matrix_name_error_strategy;}
bool AlephParams::DDMCParameterListingOutput()const{return m_param_listing_output;}
double AlephParams::DDMCParameterAmgDiagonalThreshold()const{return m_param_threshold;}
bool AlephParams::printCpuTimeResolution()const{return m_param_print_cpu_time_resolution;}
int AlephParams::amgCoarseningMethod()const{return m_param_amg_coarsening_method;}// -1 pour Sloop
int AlephParams::getOutputLevel()const{return m_param_output_level;}
int AlephParams::getAmgCycle()const{return m_param_amg_cycle;}
int AlephParams::getAmgSolverIter()const{return m_param_amg_solver_iterations;}
int AlephParams::getAmgSmootherIter()const{return m_param_amg_smoother_iterations;}
TypesSolver::eAmgSmootherOption AlephParams::getAmgSmootherOption()const{return m_param_amg_smootherOption;}
TypesSolver::eAmgCoarseningOption AlephParams::getAmgCoarseningOption()const{return m_param_amg_coarseningOption;}
TypesSolver::eAmgCoarseSolverOption AlephParams::getAmgCoarseSolverOption()const{return m_param_amg_coarseSolverOption;}
bool AlephParams::getKeepSolverStructure()const{return m_param_keep_solver_structure;}
bool AlephParams::getSequentialSolver()const{return m_param_sequential_solver;}
TypesSolver::eCriteriaStop AlephParams::getCriteriaStop()const{return m_param_criteria_stop;}

