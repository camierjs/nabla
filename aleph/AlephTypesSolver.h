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
   * \brief Stratégie à adopter en cas d'erreur du solveur.
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
