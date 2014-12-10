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
#ifndef ALEPH_SOLVER_TYPESSOLVER_H
#define ALEPH_SOLVER_TYPESSOLVER_H


class TypesSolver{
public:

  enum ePreconditionerMethod{
    DIAGONAL=0 ,
    AINV=1     ,
    AMG=2      ,
    IC=3       ,
    POLY=4     ,
    ILU=5      ,
    ILUp=6     ,
    SPAIstat=7 ,
    SPAIdyn=8  ,
    DDMCriteriaAdaptedSelector=9 ,
    NONE=10
  };

  /*!
   * \brief Strat�gie � adopter en cas d'erreur du solveur.
   */
  enum eErrorStrategy{
    ES_Continue = 0,
    ES_Stop = 1,
    ES_GoBackward = 2
  };

  enum eSolverMethod{
    PCG=0 ,
    BiCGStab=1 ,
    BiCGStab2=2,
    GMRES=3,
    SAMG=4,
    QMR=5,
    SuperLU=6
  };

  enum eAmgCoarseningMethod{
    AMG_COARSENING_AUTO     =  0 ,
    AMG_COARSENING_HYPRE_0  =  1 ,
    AMG_COARSENING_HYPRE_1  =  2 ,
    AMG_COARSENING_HYPRE_3  =  3 ,
    AMG_COARSENING_HYPRE_6  =  4 ,
    AMG_COARSENING_HYPRE_7  =  5 ,
    AMG_COARSENING_HYPRE_8  =  6 ,
    AMG_COARSENING_HYPRE_9  =  7 ,
    AMG_COARSENING_HYPRE_10 =  8 ,
    AMG_COARSENING_HYPRE_11 =  9 ,
    AMG_COARSENING_HYPRE_21 = 10 ,
    AMG_COARSENING_HYPRE_22 = 11
  };

  // ordre de eAmgSmootherOption est le meme que celui de Sloop
  // attention de ne pas le modifier voir Mireille ou bien AMG.h dans Sloop
  enum eAmgSmootherOption{
    Rich_IC_smoother         =  0 ,
    Rich_ILU_smoother        =  1 ,
    Rich_AINV_smoother       =  2 ,
    CG_smoother              =  3 ,
    HybGSJ_smoother          =  4 ,
    SymHybGSJ_smoother       =  5 ,
    HybGSJ_block_smoother    =  6 ,
    SymHybGSJ_block_smoother =  7 ,
    Rich_IC_block_smoother   =  8 ,
    Rich_ILU_block_smoother  =  9
  };


    // ordre de eAmgCoarseningOption est le meme que celui de Sloop
    // attention de ne pas le modifier voir Mireille ou bien AMG.h dans Sloop
  enum eAmgCoarseningOption{
    ParallelRugeStuben              =  0 ,
    CLJP                            =  1 ,
    Falgout                         =  2 ,
    ParallelRugeStubenBoundaryFirst =  3 ,
    FalgoutBoundaryFirst            =  4 ,
    PRS                             =  5 ,
    PRSBF                           =  6 ,
    FBF                             =  7 ,
    GMBF                            =  8
  };

  // ordre de eAmgCoarseSolverOption est le meme que celui de Sloop
  // attention de ne pas le modifier voir Mireille ou bien AMG.h dans Sloop
  enum  eAmgCoarseSolverOption {
	 CG_coarse_solver			= 0 ,
	 BiCGStab_coarse_solver	= 1 ,
	 LU_coarse_solver			= 2 ,
  	 Cholesky_coarse_solver	= 3 ,
  	 Smoother_coarse_solver	= 4 ,
  	 SuperLU_coarse_solver	= 5
  } ;



   // critere d'arret du solveur Sloop
  enum eCriteriaStop  {
    RR0		=0 ,
    RB		=1 ,
    R		   =2 ,
    RCB		=3 ,
    RBinf	=4 ,
    EpsA	   =5 ,
    NIter	=6 ,
    RR0inf	=7 ,
    STAG    =8
  };
};


#endif
