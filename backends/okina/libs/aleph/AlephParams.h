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
#ifndef ALEPH_PARAMS_H
#define ALEPH_PARAMS_H

#include "Aleph.h"

class ITraceMng;

class AlephParams{
 public:
  AlephParams();
  AlephParams(double, int,
				  int,
				  int,
				  int, double, bool, bool, bool, bool, double, bool, bool, bool,
				  string, bool, double, bool, int, int, int, int, int,
				  TypesSolver::eAmgSmootherOption,
				  TypesSolver::eAmgCoarseningOption,
				  TypesSolver::eAmgCoarseSolverOption,
				  bool, bool,
              TypesSolver::eCriteriaStop);
  AlephParams(ITraceMng*, double, int,
				  TypesSolver::ePreconditionerMethod,
				  TypesSolver::eSolverMethod,
				  int, double, bool, bool, bool, bool, double, bool, bool, bool,
				  string, bool, double, bool, int, int, int, int, int,
				  TypesSolver::eAmgSmootherOption,
				  TypesSolver::eAmgCoarseningOption,
				  TypesSolver::eAmgCoarseSolverOption,
				  bool, bool,
              TypesSolver::eCriteriaStop);
  ~AlephParams();
  
  // set
  void setEpsilon(const double epsilon);
  void setMaxIter(const int max_iteration);
  void setPrecond(const TypesSolver::ePreconditionerMethod preconditioner_method);
  void setMethod(const TypesSolver::eSolverMethod solver_method);
  void setAlpha(const double alpha);
  void setGamma(const int gamma);
  void setXoUser(const bool xo_user);
  void setCheckRealResidue(const bool check_real_residue);
  void setPrintRealResidue(const bool print_real_residue);
  void setDebugInfo(const bool debug_info);
  void setMinRHSNorm(const double min_rhs_norm);
  void setConvergenceAnalyse(const bool convergence_analyse);
  void setStopErrorStrategy(const bool stop_error_strategy);
  void setWriteMatrixToFileErrorStrategy(const bool write_matrix_to_file_error_strategy);
  void setWriteMatrixNameErrorStrategy(const string& write_matrix_name_error_strategy);
  void setDDMCParameterListingOutput(const bool listing_output);
  void setDDMCParameterAmgDiagonalThreshold(const double threshold);
  void setPrintCpuTimeResolution(const bool print_cpu_time_resolution);
  void setAmgCoarseningMethod(const TypesSolver::eAmgCoarseningMethod method);
  void setOutputLevel(const int output_level);
  void setAmgCycle(const int amg_cycle);
  void setAmgSolverIter(const int amg_solver_iterations);
  void setAmgSmootherIter(const int amg_smoother_iterations);
  void setAmgSmootherOption(const TypesSolver::eAmgSmootherOption amg_smootherOption);
  void setAmgCoarseningOption(const TypesSolver::eAmgCoarseningOption amg_coarseningOption);
  void setAmgCoarseSolverOption(const TypesSolver::eAmgCoarseSolverOption amg_coarseSolverOption);
  void setKeepSolverStructure(const bool keep_solver_structure);
  void setSequentialSolver(const bool sequential_solver);
  void setCriteriaStop(const TypesSolver::eCriteriaStop criteria_stop);

  // get
  double epsilon()const;
  int maxIter()const;
  double alpha()const;
  int gamma()const;
  TypesSolver::ePreconditionerMethod precond();
  TypesSolver::eSolverMethod method();
  bool xoUser()const;
  bool checkRealResidue()const;
  bool printRealResidue()const;
  bool debugInfo();
  double minRHSNorm();
  bool convergenceAnalyse();
  bool stopErrorStrategy();
  bool writeMatrixToFileErrorStrategy();
  string writeMatrixNameErrorStrategy();
  bool DDMCParameterListingOutput()const;
  double DDMCParameterAmgDiagonalThreshold()const;
  bool printCpuTimeResolution()const;
  int amgCoarseningMethod()const;
  int getOutputLevel()const;
  int getAmgCycle()const;
  int getAmgSolverIter()const;
  int getAmgSmootherIter()const;
  TypesSolver::eAmgSmootherOption getAmgSmootherOption()const;
  TypesSolver::eAmgCoarseningOption getAmgCoarseningOption()const;
  TypesSolver::eAmgCoarseSolverOption getAmgCoarseSolverOption()const;
  bool getKeepSolverStructure()const;
  bool getSequentialSolver()const;
  TypesSolver::eCriteriaStop getCriteriaStop()const;

 private:
  double m_param_epsilon; // epsilon de convergence
  int m_param_max_iteration; // nb max iterations
  TypesSolver::ePreconditionerMethod m_param_preconditioner_method; //  préconditionnement utilisé (defaut DIAG)
  TypesSolver::eSolverMethod m_param_solver_method; // méthode de résolution par defaut PCG
  int m_param_gamma; // destine au parametrage des préconditionnements
  double m_param_alpha; // destine au parametrage des préconditionnements
  bool m_param_xo_user; // permet a l'utilisateur d'initialiser le PGC avec un Xo different de zero
  bool m_param_check_real_residue;
  bool m_param_print_real_residue;
  bool m_param_debug_info;
  double m_param_min_rhs_norm;
  bool m_param_convergence_analyse;
  bool m_param_stop_error_strategy; 
  bool m_param_write_matrix_to_file_error_strategy;
  string m_param_write_matrix_name_error_strategy;
  bool m_param_listing_output;
  double m_param_threshold;
  bool m_param_print_cpu_time_resolution;
  int m_param_amg_coarsening_method;
  int m_param_output_level;
  int m_param_amg_cycle;
  int m_param_amg_solver_iterations;
  int m_param_amg_smoother_iterations;
  TypesSolver::eAmgSmootherOption m_param_amg_smootherOption;
  TypesSolver::eAmgCoarseningOption m_param_amg_coarseningOption;
  TypesSolver::eAmgCoarseSolverOption m_param_amg_coarseSolverOption;
  bool m_param_keep_solver_structure;
  bool m_param_sequential_solver;
  TypesSolver::eCriteriaStop m_param_criteria_stop;
};

#endif  

