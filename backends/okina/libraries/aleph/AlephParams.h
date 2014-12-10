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
#ifndef ALEPH_PARAMS_H
#define ALEPH_PARAMS_H

#include "Aleph.h"

class ITraceMng;

class AlephParams{
 public:
  AlephParams();
  AlephParams(Real, Integer,
				  Integer,
				  Integer,
				  Integer, Real, bool, bool, bool, bool, Real, bool, bool, bool,
				  String, bool, Real, bool, Integer, Integer, Integer, Integer, Integer,
				  TypesSolver::eAmgSmootherOption,
				  TypesSolver::eAmgCoarseningOption,
				  TypesSolver::eAmgCoarseSolverOption,
				  bool, bool,
              TypesSolver::eCriteriaStop);
  AlephParams(ITraceMng*, Real, Integer,
				  TypesSolver::ePreconditionerMethod,
				  TypesSolver::eSolverMethod,
				  Integer, Real, bool, bool, bool, bool, Real, bool, bool, bool,
				  String, bool, Real, bool, Integer, Integer, Integer, Integer, Integer,
				  TypesSolver::eAmgSmootherOption,
				  TypesSolver::eAmgCoarseningOption,
				  TypesSolver::eAmgCoarseSolverOption,
				  bool, bool,
              TypesSolver::eCriteriaStop);
  ~AlephParams();
  
  // set
  void setEpsilon(const Real epsilon);
  void setMaxIter(const Integer max_iteration);
  void setPrecond(const TypesSolver::ePreconditionerMethod preconditioner_method);
  void setMethod(const TypesSolver::eSolverMethod solver_method);
  void setAlpha(const Real alpha);
  void setGamma(const Integer gamma);
  void setXoUser(const bool xo_user);
  void setCheckRealResidue(const bool check_real_residue);
  void setPrintRealResidue(const bool print_real_residue);
  void setDebugInfo(const bool debug_info);
  void setMinRHSNorm(const Real min_rhs_norm);
  void setConvergenceAnalyse(const bool convergence_analyse);
  void setStopErrorStrategy(const bool stop_error_strategy);
  void setWriteMatrixToFileErrorStrategy(const bool write_matrix_to_file_error_strategy);
  void setWriteMatrixNameErrorStrategy(const String& write_matrix_name_error_strategy);
  void setDDMCParameterListingOutput(const bool listing_output);
  void setDDMCParameterAmgDiagonalThreshold(const Real threshold);
  void setPrintCpuTimeResolution(const bool print_cpu_time_resolution);
  void setAmgCoarseningMethod(const TypesSolver::eAmgCoarseningMethod method);
  void setOutputLevel(const Integer output_level);
  void setAmgCycle(const Integer amg_cycle);
  void setAmgSolverIter(const Integer amg_solver_iterations);
  void setAmgSmootherIter(const Integer amg_smoother_iterations);
  void setAmgSmootherOption(const TypesSolver::eAmgSmootherOption amg_smootherOption);
  void setAmgCoarseningOption(const TypesSolver::eAmgCoarseningOption amg_coarseningOption);
  void setAmgCoarseSolverOption(const TypesSolver::eAmgCoarseSolverOption amg_coarseSolverOption);
  void setKeepSolverStructure(const bool keep_solver_structure);
  void setSequentialSolver(const bool sequential_solver);
  void setCriteriaStop(const TypesSolver::eCriteriaStop criteria_stop);

  // get
  Real epsilon()const;
  int maxIter()const;
  Real alpha()const;
  int gamma()const;
  TypesSolver::ePreconditionerMethod precond();
  TypesSolver::eSolverMethod method();
  bool xoUser()const;
  bool checkRealResidue()const;
  bool printRealResidue()const;
  bool debugInfo();
  Real minRHSNorm();
  bool convergenceAnalyse();
  bool stopErrorStrategy();
  bool writeMatrixToFileErrorStrategy();
  String writeMatrixNameErrorStrategy();
  bool DDMCParameterListingOutput()const;
  Real DDMCParameterAmgDiagonalThreshold()const;
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
  Real m_param_epsilon; // epsilon de convergence
  Integer m_param_max_iteration; // nb max iterations
  TypesSolver::ePreconditionerMethod m_param_preconditioner_method; //  pr�conditionnement utilis� (defaut DIAG)
  TypesSolver::eSolverMethod m_param_solver_method; // m�thode de r�solution par defaut PCG
  Integer m_param_gamma; // destine au parametrage des pr�conditionnements
  Real m_param_alpha; // destine au parametrage des pr�conditionnements
  bool m_param_xo_user; // permet a l'utilisateur d'initialiser le PGC avec un Xo different de zero
  bool m_param_check_real_residue;
  bool m_param_print_real_residue;
  bool m_param_debug_info;
  Real m_param_min_rhs_norm;
  bool m_param_convergence_analyse;
  bool m_param_stop_error_strategy; 
  bool m_param_write_matrix_to_file_error_strategy;
  String m_param_write_matrix_name_error_strategy;
  bool m_param_listing_output;
  Real m_param_threshold;
  bool m_param_print_cpu_time_resolution;
  Integer m_param_amg_coarsening_method;
  Integer m_param_output_level;
  Integer m_param_amg_cycle;
  Integer m_param_amg_solver_iterations;
  Integer m_param_amg_smoother_iterations;
  TypesSolver::eAmgSmootherOption m_param_amg_smootherOption;
  TypesSolver::eAmgCoarseningOption m_param_amg_coarseningOption;
  TypesSolver::eAmgCoarseSolverOption m_param_amg_coarseSolverOption;
  bool m_param_keep_solver_structure;
  bool m_param_sequential_solver;
  TypesSolver::eCriteriaStop m_param_criteria_stop;
};

#endif  

