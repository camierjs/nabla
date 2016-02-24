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
#include "nabla.h"
#include "backends/arcane/arcane.h"


/***************************************************************************** 
 *
 *****************************************************************************/
char* nccArcLibAlephHeader(void){
  // On prépare le header de l'entity
  return "\n\
#include <arcane/aleph/AlephTypesSolver.h>\n\
#include <arcane/aleph/Aleph.h>\n\n\
//#include <arcane/hyoda/Hyoda.h>\n\n\
#ifdef ARCANE_HAS_PACKAGE_SLOOP\n\
#define HASMPI\n\
#define SLOOP_MATH\n\
#define OMPI_SKIP_MPICXX\n\
#define MPICH_SKIP_MPICXX\n\
#define PARALLEL_SLOOP\n\
#include \"mpi.h\"\n\
#include \"SLOOP.h\"\n\
#endif // ARCANE_HAS_PACKAGE_SLOOP\n\
\n\
#ifdef ARCANE_HAS_PACKAGE_HYPRE\n\
#define OMPI_SKIP_MPICXX\n\
#define MPICH_SKIP_MPICXX\n\
#include \"HYPRE.h\"\n\
#include \"HYPRE_utilities.h\"\n\
#include \"HYPRE_IJ_mv.h\"\n\
#include \"HYPRE_parcsr_mv.h\"\n\
#include \"HYPRE_parcsr_ls.h\"\n\
#include \"_hypre_parcsr_mv.h\"\n\
#include \"krylov.h\"\n\
#endif // ARCANE_HAS_PACKAGE_HYPRE\n\
\n\
//#ifdef ARCANE_HAS_PACKAGE_TRILINOS\n\
//#include \"Epetra_config.h\"\n\
//#include \"Epetra_Vector.h\"\n\
//#include \"Epetra_MpiComm.h\"\n\
//#include \"Epetra_Map.h\"\n\
//#include \"Epetra_CrsMatrix.h\"\n\
//#include \"Epetra_LinearProblem.h\"\n\
//#include \"AztecOO.h\"\n\
//#include \"Ifpack_IC.h\"\n\
//#include \"ml_MultiLevelPreconditioner.h\"\n\
//#endif // ARCANE_HAS_PACKAGE_TRILINOS\n\
\n\
//#ifdef ARCANE_HAS_PACKAGE_CUDA\n\
//#include <cuda.h>\n\
//#include <cublas.h>\n\
//#include <cuda_runtime.h>\n\
//#endif // ARCANE_HAS_PACKAGE_CUDA\n\n\
\n\
#include <arcane/aleph/IAleph.h>\n\
#include <arcane/aleph/IAlephFactory.h>\n\n\
\n\
using namespace Arcane;\n\
///////////////////////////////////////////\n\
class AlephRealArray:public RealArray{\n\
public:\n\
\tvoid reset(){resize(0);}\n\
\tvoid error(){throw FatalErrorException(\"AlephRealArray\",\"Error\");}\n\
\tvoid newValue(double value){\n\
\t\tadd(value);\n\
\t}\n\
\tvoid addValue(const VariableRef &var, const ItemEnumerator &itmEnum, double value){\n\
\t\treturn addValue(var,*itmEnum,value);\n\
\t}\n\
\tvoid addValue(const VariableRef &var, const Item &itm, double value){\n\
\t\tInteger idx=m_aleph_kernel->indexing()->get(var,itm);\n\
\t\tif(idx==size()){\n\
\t\t\tresize(idx+1);\n\
\t\t\tindex.add(idx);\n\
\t\t\tsetAt(idx,value);\n\
\t\t}else{\n\
\t\t\tsetAt(idx,at(idx)+value);\n\
\t\t}\n\
\t}\n\
\tvoid setValue(const VariableRef &var, const ItemEnumerator &itmEnum, double value){\n\
\t\treturn setValue(var,*itmEnum,value);\n\
\t}\n\
\tvoid setValue(const VariableRef &var, const Item &itm, double value){\n\
\t\tInteger topology_row_offset=0;\n\
\t\tInteger idx=m_aleph_kernel->indexing()->get(var,itm)-topology_row_offset;\n\
\t\tif(idx==size()){\n\
\t\t\tresize(idx+1);\n\
\t\t\tindex.add(idx);\n\
\t\t\tsetAt(idx,value);\n\
\t\t}else{\n\
\t\t\tsetAt(idx,value);\n\
\t\t}\n\
\t}\n\
\tdouble getValue(const VariableRef &var, ItemEnumerator &itmEnum){\n\
\t\treturn at(m_aleph_kernel->indexing()->get(var,*itmEnum));\n\
}\n\
\tdouble getValue(const VariableRef &var, Item itm){\n\
\t\treturn at(m_aleph_kernel->indexing()->get(var,itm));\n\
}\n\
public:\n\
\tIntegerArray index;\n\
\tAlephKernel *m_aleph_kernel;\n\
};\n\
///////////////////////////////////////////\n\
class AlephRealMatrix{\n\
public:\n\
\tvoid error(){throw FatalErrorException(\"AlephRealArray\",\"Error\");}\n\
\tvoid addValue(const VariableRef &iVar, const ItemEnumerator &iItmEnum,\n\
                 const VariableRef &jVar, const Item &jItm, double value){\n\
\t\tm_aleph_mat->addValue(iVar,*iItmEnum,jVar,jItm,value);\n\t}\n\
\tvoid addValue(const VariableRef &iVar, const ItemEnumerator &iItmEnum,\n\
                 const VariableRef &jVar, const ItemEnumerator &jItmEnum, double value){\n\
\t\tm_aleph_mat->addValue(iVar,*iItmEnum,jVar,*jItmEnum,value);\n\t}\n\
\tvoid addValue(const VariableRef &iVar, const Item &iItm,\n\
                 const VariableRef &jVar, const Item &jItm, double value){\n\
\t\tm_aleph_mat->addValue(iVar,iItm,jVar,jItm,value);\n\t}\n\
\tvoid addValue(const VariableRef &iVar, const Item &iItm,\n\
                 const VariableRef &jVar, const ItemEnumerator &jItmEnum, double value){\n\
\t\tm_aleph_mat->addValue(iVar,iItm,jVar,*jItmEnum,value);\n\t}\n\
\tvoid setValue(const VariableRef &iVar, const Item &iItm,\n\
                 const VariableRef &jVar, const Item &jItm, double value){\n\
\t\tm_aleph_mat->setValue(iVar,iItm,jVar,jItm,value);\n\t}\n\
\tvoid setValue(const VariableRef &iVar, const ItemEnumerator &iItmEnum,\n\
                 const VariableRef &jVar, const Item &jItm, double value){\n\
\t\tm_aleph_mat->setValue(iVar,*iItmEnum,jVar,jItm,value);\n\t}\n\
\tvoid setValue(const VariableRef &iVar, const ItemEnumerator &iItmEnum,\n\
                 const VariableRef &jVar, const ItemEnumerator &jItmEnum, double value){\n\
\t\tm_aleph_mat->setValue(iVar,*iItmEnum,jVar,*jItmEnum,value);\n\t}\n\
public:\n\
\tAlephKernel *m_aleph_kernel;\n\
\tAlephMatrix *m_aleph_mat;\n\
};";  
}


// ****************************************************************************
// * nccArcLibAlephPrivates
// ****************************************************************************
char* nccArcLibAlephPrivates(void){
  return   "\nprivate:\n\
\tvoid alephInitialize(void);\n\
\tvoid alephIni(void);\n\
\tvoid alephAddValue(const VariableRef&, const Item&, const VariableRef&, const Item&,double);\n\
\tvoid alephAddValue(int,int,double);\n\
\tvoid alephRhsSet(const VariableRef&, const Item&, Real);\n\
\tvoid alephRhsSet(Integer, Real);\n\
\tvoid alephRhsAdd(Integer, Real);\n\
\tdouble alephRhsGet(Integer row);\n\
\tvoid alephSolve();\n\
\tvoid alephSolveWithoutIndex();\n\
private:\n\
\tIAlephFactory *m_aleph_factory;\n\
\tAlephKernel *m_aleph_kernel;\n\
\tAlephParams *m_aleph_params;\n\
\tAlephMatrix *m_aleph_mat;\n\
\tAlephVector *m_aleph_rhs;\n\
\tAlephVector *m_aleph_sol;\n\
\tArray<Integer> vector_indexs;\n\
\tArray<Real> vector_zeroes;\n\
\tAlephRealArray lhs;\n\
\tAlephRealArray rhs;\n\
\tAlephRealMatrix mtx;";
}



/******************************************************************************
 * nccArcLibAlephInitialize
 ******************************************************************************/
static void nccArcLibAlephInitialize(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid %s%s::alephInitialize(void){\n\
\tm_aleph_mat=m_aleph_kernel->createSolverMatrix();\n\
\tm_aleph_rhs=m_aleph_kernel->createSolverVector();\n\
\tm_aleph_sol=m_aleph_kernel->createSolverVector();\n\
\tm_aleph_mat->create();\n\
\tm_aleph_rhs->create();\n\
\tm_aleph_sol->create();\n\
\tm_aleph_mat->reset();\n\
\tmtx.m_aleph_mat=m_aleph_mat;\n\
}",arc->name,nablaArcaneColor(arc));
}


/******************************************************************************
 * nccArcLibAlephMatAddValue with IVariable and Item
 ******************************************************************************/
static void nccArcLibAlephMatAddValueWithIVariableAndItem(nablaMain *arc){
  fprintf(arc->entity->src, "\
\nvoid %s%s::alephAddValue(const VariableRef &rowVar, const Item &rowItm,\n\
\t\t\t\t\t\t\t\t\t\tconst VariableRef &colVar, const Item &colItm,double val){\n\
\tm_aleph_mat->addValue(rowVar,rowItm,colVar,colItm,val);\n}",
          arc->name,
          nablaArcaneColor(arc));
}

/******************************************************************************
 * nccArcLibAlephMatAddValue
 ******************************************************************************/
static void nccArcLibAlephMatAddValue(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid %s%s::alephAddValue(int i, int j, double val){\n\
\tm_aleph_mat->addValue(i,j,val);\n}",
          arc->name,
          nablaArcaneColor(arc));
}
  

/******************************************************************************
 * nccArcLibAlephRhsSet
 ******************************************************************************/
static void nccArcLibAlephRhsSet(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid %s%s::alephRhsSet(Integer row, Real value){\n\
\tconst Integer kernel_rank = m_aleph_kernel->rank();\n\
\tconst Integer rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];\n\
\trhs[row-rank_offset]=value;\n\
}",arc->name,nablaArcaneColor(arc));
  fprintf(arc->entity->src, "\nvoid %s%s::alephRhsSet(const VariableRef &var, const Item &itm, Real value){\n\
\trhs[m_aleph_kernel->indexing()->get(var,itm)]=value;\n\
}",arc->name,nablaArcaneColor(arc));
}

/******************************************************************************
 * nccArcLibAlephRhsAdd
 ******************************************************************************/
static void nccArcLibAlephRhsAdd(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid %s%s::alephRhsAdd(Integer row, Real value){\n\
\tconst Integer kernel_rank = m_aleph_kernel->rank();\n\
\tconst Integer rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];\n\
\trhs[row-rank_offset]+=value;\n}",
          arc->name,
          nablaArcaneColor(arc));
}


/******************************************************************************
 * nccArcLibAlephRhsSet
 ******************************************************************************/
static void nccArcLibAlephRhsGet(nablaMain *arc){
  fprintf(arc->entity->src, "\ndouble %s%s::alephRhsGet(Integer row){\n\
\tconst Integer kernel_rank = m_aleph_kernel->rank();\n\
\tconst register Integer rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];\n\
\treturn rhs[row-rank_offset];\n}",
          arc->name,
          nablaArcaneColor(arc));
}


/******************************************************************************
 * nccArcLibAlephSolve
 ******************************************************************************/
static void nccArcLibAlephSolve(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid %s%s::alephSolve(){\n\
#ifndef ALEPH_INDEX\n\
\tInteger nb_iteration;\n\
\tReal residual_norm[4];\n\
\tm_aleph_mat->assemble();\n\
\tm_aleph_rhs->setLocalComponents(vector_indexs.size(),vector_indexs.view(),rhs.view());\n\
\tm_aleph_rhs->assemble();\n\
\tm_aleph_sol->setLocalComponents(vector_indexs.size(),vector_indexs.view(),vector_zeroes.view());\n\
\tm_aleph_sol->assemble();\n\
\tm_aleph_mat->solve(m_aleph_sol, m_aleph_rhs, nb_iteration, &residual_norm[0], m_aleph_params, true);\n\
\tAlephVector *solution=m_aleph_kernel->syncSolver(0,nb_iteration,&residual_norm[0]);\n\
\tinfo() << \"Solved in \33[7m\" << nb_iteration << \"\33[m iterations,\"\n\
\t\t<< \"residuals=[\33[1m\" << residual_norm[0] <<\"\33[m,\"<< residual_norm[3]<<\"]\";\n\
\tsolution->getLocalComponents(vector_indexs.size(),vector_indexs,rhs);\n\
//#warning weird getLocalComponents with rhs\n\
#else\n\
\talephSolveWithoutIndex();\n\
#endif\n}",arc->name,
          nablaArcaneColor(arc));
}


/******************************************************************************
 * nccArcLibAlephSolveWithoutIndex
 ******************************************************************************/
static void nccArcLibAlephSolveWithoutIndex(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid %s%s::alephSolveWithoutIndex(){\n\
\tInteger nb_iteration;\n\
\tReal residual_norm[4];\n\
\tm_aleph_mat->assemble();\n\
\tm_aleph_rhs->setLocalComponents(rhs.view());\n\
\tm_aleph_rhs->assemble();\n\
\tvector_zeroes.resize(rhs.size());\n\
\tvector_zeroes.fill(0.0);\n\
\tm_aleph_sol->setLocalComponents(vector_zeroes.view());\n\
\tm_aleph_sol->assemble();\n\
\tm_aleph_mat->solve(m_aleph_sol, m_aleph_rhs, nb_iteration, &residual_norm[0], m_aleph_params, true);\n\
\tAlephVector *solution=m_aleph_kernel->syncSolver(0,nb_iteration,&residual_norm[0]);\n\
\tinfo() << \"Solved in \33[7m\" << nb_iteration << \"\33[m iterations,\"\n\
\t\t<< \"residuals=[\33[1m\" << residual_norm[0] <<\"\33[m,\"<< residual_norm[3]<<\"]\";\n\
\tlhs.reset(); lhs.resize(rhs.size()); //lhs.fill(0.0);\n\
\tsolution->getLocalComponents(lhs);\n\
}",
          arc->name,
          nablaArcaneColor(arc));
}


/******************************************************************************
 * nccArcLibAlephIni
 ******************************************************************************/
void nccArcLibAlephIni(nablaMain *arc){
  fprintf(arc->entity->src, "\n\n\
// ****************************************************************************\n\
// * alephIni\n\
// ****************************************************************************\
\nvoid %s%s::alephIni(void){\
\n#ifndef ALEPH_INDEX\
\n\t// Pourrait être enlevé, mais est encore utilisé dans le test DDVF sans auto-index\
\n\tvector_indexs.resize(0);\
\n\tvector_zeroes.resize(0);\
\n\trhs.resize(0);\
\n\tmesh()->checkValidMeshFull();",
          arc->name,
          nablaArcaneColor(arc));
  fprintf(arc->entity->src, "\n\tm_aleph_factory=new AlephFactory(subDomain()->application(),traceMng());");
  fprintf(arc->entity->src, "\
\n\tm_aleph_kernel=new AlephKernel(traceMng(), subDomain(), m_aleph_factory,\n\
                                   options()->alephUnderlyingSolver(),\n\
                                   options()->alephNumberOfCores(),\n\
                                   false);//options()->alephMatchSequential());");
  fprintf(arc->entity->src, "\n#else");
  fprintf(arc->entity->src, "\
\n\tm_aleph_kernel=new AlephKernel(subDomain(),\n\
                                  options()->alephUnderlyingSolver(),\n\
                                  options()->alephNumberOfCores());");
  fprintf(arc->entity->src, "\n\trhs.m_aleph_kernel=m_aleph_kernel;");
  fprintf(arc->entity->src, "\n\tlhs.m_aleph_kernel=m_aleph_kernel;");
  fprintf(arc->entity->src, "\n\tmtx.m_aleph_kernel=m_aleph_kernel;");
  fprintf(arc->entity->src, "\n#endif");
  fprintf(arc->entity->src, "\n\tm_aleph_params=\n\
    new AlephParams(traceMng(),\n\
                    options()->alephEpsilon(),      // epsilon epsilon de convergence\n\
                    options()->alephMaxIterations(),// max_iteration nb max iterations\n\
                    (TypesSolver::ePreconditionerMethod)options()->alephPreconditionerMethod(),// préconditionnement: ILU, DIAGONAL, AMG, IC\n\
                    (TypesSolver::eSolverMethod)options()->alephSolverMethod(), // méthode de résolution\n\
                    -1,                             // gamma\n\
                    -1.0,                           // alpha\n\
                    true,                           // xo_user par défaut Xo n'est pas égal à 0\n\
                    false,                          // check_real_residue\n\
                    false,                          // print_real_residue\n\
                    false,                          // debug_info\n\
                    1.e-20,                         // min_rhs_norm\n\
                    false,                          // convergence_analyse\n\
                    true,                           // stop_error_strategy\n\
                    options()->option_aleph_dump_matrix,// write_matrix_to_file_error_strategy\n\
                    \"SolveErrorAlephMatrix.dbg\",  // write_matrix_name_error_strategy\n\
                    true,                           // listing_output\n \
                    0.,                             // threshold\n\
                    false,                          // print_cpu_time_resolution\n\
                    -1,                             // amg_coarsening_method\n\
                    -3,                             // output_level\n\
                    1,                              // 1: V, 2: W, 3: Full Multigrid V\n\
                    1,                              // amg_solver_iterations\n\
                    1,                              // amg_smoother_iterations\n\
                    TypesSolver::SymHybGSJ_smoother,// amg_smootherOption\n\
                    TypesSolver::ParallelRugeStuben,// amg_coarseningOption\n\
                    TypesSolver::CG_coarse_solver,  // amg_coarseSolverOption\n\
                    false,                          // keep_solver_structure\n\
                    false,                          // sequential_solver\n\
                    TypesSolver::RB);               // criteria_stop\n");
  fprintf(arc->entity->src, "\n}");
  nccArcLibAlephInitialize(arc);
  nccArcLibAlephMatAddValueWithIVariableAndItem(arc);
  nccArcLibAlephMatAddValue(arc);
  nccArcLibAlephRhsSet(arc);
  nccArcLibAlephRhsAdd(arc);
  nccArcLibAlephRhsGet(arc);
  nccArcLibAlephSolve(arc);
  nccArcLibAlephSolveWithoutIndex(arc);

  nablaJob *alephInitFunction=nMiddleJobNew(arc->entity);
  alephInitFunction->is_an_entry_point=true;
  alephInitFunction->is_a_function=true;
  alephInitFunction->scope  = strdup("NoScope");
  alephInitFunction->region = strdup("NoRegion");
  alephInitFunction->item   = strdup("\0");
  alephInitFunction->return_type  = strdup("void");
  alephInitFunction->name   = strdup("alephIni");
  alephInitFunction->name_utf8 = strdup("ℵIni");
  alephInitFunction->xyz    = strdup("NoXYZ");
  alephInitFunction->direction  = strdup("NoDirection");
  sprintf(&alephInitFunction->at[0],"-huge_valf");
  alephInitFunction->when_index  = 1;
  alephInitFunction->whens[0] = ENTRY_POINT_init;
  nMiddleJobAdd(arc->entity, alephInitFunction);  
}
