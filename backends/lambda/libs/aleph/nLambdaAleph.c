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
#include "nabla.h"

// ****************************************************************************
// * dumpExternalFile
// * NABLA_LICENSE_HEADER is tied and defined in nabla.h
// ****************************************************************************
static char *dumpExternalFile(char *file){
  return file+NABLA_LICENSE_HEADER;
}

extern char AlephStd_h[];
extern char Aleph_h[];
extern char AlephTypesSolver_h[];
extern char AlephParams_h[];
extern char AlephVector_h[];
extern char AlephMatrix_h[];
extern char AlephKernel_h[];
extern char AlephOrdering_h[];
extern char AlephIndexing_h[];
extern char AlephTopology_h[];
extern char AlephInterface_h[];
extern char IAlephFactory_h[];

// ****************************************************************************
// *
// ****************************************************************************
char* lambdaAlephHeader(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n");
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephStd_h'*/\n%s",dumpExternalFile(AlephStd_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephTypesSolver_h'*/\n%s",dumpExternalFile(AlephTypesSolver_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephParams_h'*/\n%s",dumpExternalFile(AlephParams_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephVector_h'*/\n%s",dumpExternalFile(AlephVector_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephMatrix_h'*/\n%s",dumpExternalFile(AlephMatrix_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephKernel_h'*/\n%s",dumpExternalFile(AlephKernel_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephOrdering_h'*/\n%s",dumpExternalFile(AlephOrdering_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephIndexing_h'*/\n%s",dumpExternalFile(AlephIndexing_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephTopology_h'*/\n%s",dumpExternalFile(AlephTopology_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'AlephInterface_h'*/\n%s",dumpExternalFile(AlephInterface_h));
  fprintf(nabla->entity->hdr,"/*lambdaAlephHeader: dumping 'IAlephFactory_h'*/\n%s",dumpExternalFile(IAlephFactory_h));
  // On prépare le header de l'entity
  return "\n\
//#include <AlephTypesSolver.h>\n\
//#include <Aleph.h>\n\n       \
\n\
#ifdef LAMBDA_HAS_PACKAGE_HYPRE\n\
#define OMPI_SKIP_MPICXX\n\
#define MPICH_SKIP_MPICXX\n\
#include \"HYPRE.h\"\n\
#include \"HYPRE_utilities.h\"\n\
#include \"HYPRE_IJ_mv.h\"\n\
#include \"HYPRE_parcsr_mv.h\"\n\
#include \"HYPRE_parcsr_ls.h\"\n\
#include \"_hypre_parcsr_mv.h\"\n\
#include \"krylov.h\"\n\
#endif // LAMBDA_HAS_PACKAGE_HYPRE\n\
\n\
//#ifdef LAMBDA_HAS_PACKAGE_TRILINOS\n\
//#include \"Epetra_config.h\"\n\
//#include \"Epetra_Vector.h\"\n\
//#include \"Epetra_MpiComm.h\"\n\
//#include \"Epetra_Map.h\"\n\
//#include \"Epetra_CrsMatrix.h\"\n\
//#include \"Epetra_LinearProblem.h\"\n\
//#include \"AztecOO.h\"\n\
//#include \"Ifpack_IC.h\"\n\
//#include \"ml_MultiLevelPreconditioner.h\"\n\
//#endif // LAMBDA_HAS_PACKAGE_TRILINOS\n\
\n\
//#ifdef LAMBDA_HAS_PACKAGE_CUDA\n\
//#include <cuda.h>\n\
//#include <cublas.h>\n\
//#include <cuda_runtime.h>\n\
//#endif // LAMBDA_HAS_PACKAGE_CUDA\n\n\
\n\
//#include <IAleph.h>\n\
//#include <IAlephFactory.h>\n\n\
\n\
///////////////////////////////////////////\n\
class AlephRealArray:public vector<double>{\n\
public:\n\
\tvoid reset(){resize(0);}\n\
\tvoid error(){throw std::logic_error(\"[AlephRealArray] Error\");}\n\
\tvoid newValue(double value){\n\
\t\tpush_back(value);\n\
\t}\n\
\t//void addValue(const double* var, const item* itmEnum, double value){\n\
\t//\treturn addValue(var,*itmEnum,value);\n\
\t//}\n\
\tvoid addValue(const double* var, const item* itm, double value){\n\
\t\tunsigned int idx=0;//m_aleph_kernel->indexing()->get(var,itm);\n\
\t\tif(idx==size()){\n\
\t\t\tresize(idx+1);\n\
\t\t\tindex.push_back(idx);\n\
\t\t\tthis->at(idx)=value;\n\
\t\t}else{\n\
\t\t\tthis->at(idx)=at(idx)+value;\n\
\t\t}\n\
\t}\n\
\tvoid setValue(const Variable &var, const item* &itmEnum, double value){\n\
\t\treturn setValue(var,*itmEnum,value);\n\
\t}\n\
\tvoid setValue(const Variable &var, const item &itm, double value){\n\
\t\tint topology_row_offset=0;\n\
\t\tunsigned int idx=0;//m_aleph_kernel->indexing()->get(var,itm)-topology_row_offset;\n\
\t\tif(idx==size()){\n\
\t\t\tresize(idx+1);\n\
\t\t\tindex.push_back(idx);\n\
\t\t\tthis->at(idx)=value;\n\
\t\t}else{\n\
\t\t\tthis->at(idx)=value;\n\
\t\t}\n\
\t}\n\
\tdouble getValue(const double* var, item* itmEnum){\n\
\t\treturn 0.0;//at(m_aleph_kernel->indexing()->get(var,*itmEnum));\n\
}\n\
public:\n\
\tvector<int> index;\n\
\tAlephKernel *m_aleph_kernel;\n\
};\n\
///////////////////////////////////////////\n\
class AlephRealMatrix{\n\
public:\n\
\tvoid error(){throw std::logic_error(\"AlephRealArray\");}\n\
/*\tvoid addValue(const double* iVar, const item* iItmEnum,\n\
                 const double* jVar, const item jItm, double value){\n\
\t\tm_aleph_mat->addValue(iVar,*iItmEnum,jVar,jItm,value);\n\t}\n\
\tvoid addValue(const double* iVar, const item* iItmEnum,\n\
                 const double* jVar, const item* jItmEnum, double value){\n\
\t\tm_aleph_mat->addValue(iVar,*iItmEnum,jVar,*jItmEnum,value);\n\t}\n\
\tvoid addValue(const double* iVar, const item iItm,\n\
                 const double* jVar, const item jItm, double value){\n\
\t\tm_aleph_mat->addValue(iVar,iItm,jVar,jItm,value);\n\t}\n\
\tvoid addValue(const double* iVar, const item iItm,\n\
                 const double* jVar, const item* jItmEnum, double value){\n\
\t\tm_aleph_mat->addValue(iVar,iItm,jVar,*jItmEnum,value);\n\t}\n\
\tvoid setValue(const double* iVar, const item iItm,\n\
                 const double* jVar, const item jItm, double value){\n\
\t\tm_aleph_mat->setValue(iVar,iItm,jVar,jItm,value);\n\t}\n\
\tvoid setValue(const double* iVar, const item* iItmEnum,\n\
                 const double* jVar, const item jItm, double value){\n\
\t\tm_aleph_mat->setValue(iVar,*iItmEnum,jVar,jItm,value);\n\t}\n\
\tvoid setValue(const double* iVar, const item* iItmEnum,\n\
                 const double* jVar, const item* jItmEnum, double value){\n\
\t\tm_aleph_mat->setValue(iVar,*iItmEnum,jVar,*jItmEnum,value);\n\t}\n\
*/\n\
public:\n\
\tAlephKernel *m_aleph_kernel;\n\
\tAlephMatrix *m_aleph_mat;\n\
};";  
}


// ****************************************************************************
// * lambdaAlephPrivates
// ****************************************************************************
char* lambdaAlephPrivates(void){
  return   "\n\n\
void alephInitialize(void);\n\
void alephIni(void);\n\
void alephAddValue(const double*&, const item&, const double*&, const item&,double);\n\
void alephAddValue(int,int,double);\n\
void alephRhsSet(const double*&, const item&, double);\n\
void alephRhsSet(int, double);\n\
void alephRhsAdd(int, double);\n\
double alephRhsGet(int row);\n\
void alephSolve();\n\
void alephSolveWithoutIndex();\n\
\n\
IAlephFactory *m_aleph_factory;\n\
AlephKernel *m_aleph_kernel;\n\
AlephParams *m_aleph_params;\n\
AlephMatrix *m_aleph_mat;\n\
AlephVector *m_aleph_rhs;\n\
AlephVector *m_aleph_sol;\n\
vector<int> vector_indexs;\n\
vector<double> vector_zeroes;\n\
AlephRealArray lhs;\n\
AlephRealArray rhs;\n\
AlephRealMatrix mtx;\n\
\n\
ISubDomain sub_domain;\n\
ISubDomain* subDomain(){return &sub_domain;}\n\
\n\
IMesh thisMesh;\n\
IMesh* mesh(){return &thisMesh;}\n\
\n\
ITraceMng trcMng;\n\
ITraceMng* traceMng(){return &trcMng;}\n\
";
}



/******************************************************************************
 * lambdaAlephInitialize
 ******************************************************************************/
static void lambdaAlephInitialize(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid alephInitialize(void){\n\
\tm_aleph_mat=m_aleph_kernel->createSolverMatrix();\n\
\tm_aleph_rhs=m_aleph_kernel->createSolverVector();\n\
\tm_aleph_sol=m_aleph_kernel->createSolverVector();\n\
\tm_aleph_mat->create();\n\
\tm_aleph_rhs->create();\n\
\tm_aleph_sol->create();\n\
\tm_aleph_mat->reset();\n\
\tmtx.m_aleph_mat=m_aleph_mat;\n}");
}


/******************************************************************************
 * lambdaAlephMatAddValue with IVariable and item
 ******************************************************************************/
static void lambdaAlephMatAddValueWithIVariableAndItem(nablaMain *arc){
  fprintf(arc->entity->src, "\
\nvoid alephAddValue(const double* &rowVar, const item &rowItm,\n\
\t\t\t\t\t\t\t\t\t\tconst double* &colVar, const item &colItm,double val){\n\
\tm_aleph_mat->addValue(rowVar,rowItm,colVar,colItm,val);\n}");
}

/******************************************************************************
 * lambdaAlephMatAddValue
 ******************************************************************************/
static void lambdaAlephMatAddValue(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid alephAddValue(int i, int j, double val){\n\
\tm_aleph_mat->addValue(i,j,val);\n}");
}
  

/******************************************************************************
 * lambdaAlephRhsSet
 ******************************************************************************/
static void lambdaAlephRhsSet(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid alephRhsSet(int row, double value){\n\
\tconst int kernel_rank = m_aleph_kernel->rank();\n\
\tconst int rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];\n\
\trhs[row-rank_offset]=value;\n}");
  fprintf(arc->entity->src, "\nvoid alephRhsSet(const double* &var, const item &itm, double value){\n\
\trhs[m_aleph_kernel->indexing()->get(var,itm)]=value;\n}");
}

/******************************************************************************
 * lambdaAlephRhsAdd
 ******************************************************************************/
static void lambdaAlephRhsAdd(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid alephRhsAdd(int row, double value){\n\
\tconst int kernel_rank = m_aleph_kernel->rank();\n\
\tconst int rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];\n\
\trhs[row-rank_offset]+=value;\n}");
}


/******************************************************************************
 * lambdaAlephRhsSet
 ******************************************************************************/
static void lambdaAlephRhsGet(nablaMain *arc){
  fprintf(arc->entity->src, "\ndouble alephRhsGet(int row){\n\
\tconst int kernel_rank = m_aleph_kernel->rank();\n\
\tconst register int rank_offset=m_aleph_kernel->topology()->part()[kernel_rank];\n\
\treturn rhs[row-rank_offset];\n}");
}


/******************************************************************************
 * lambdaAlephSolve
 ******************************************************************************/
static void lambdaAlephSolve(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid alephSolve(){\n\
#ifndef ALEPH_INDEX\n\
\tint nb_iteration;\n\
\tdouble residual_norm[4];\n\
\tm_aleph_mat->assemble();\n\
\tm_aleph_rhs->setLocalComponents(vector_indexs.size(),vector_indexs,rhs);\n\
\tm_aleph_rhs->assemble();\n\
\tm_aleph_sol->setLocalComponents(vector_indexs.size(),vector_indexs,vector_zeroes);\n\
\tm_aleph_sol->assemble();\n\
\tm_aleph_mat->solve(m_aleph_sol, m_aleph_rhs, nb_iteration, &residual_norm[0], m_aleph_params, true);\n\
\tAlephVector *solution=m_aleph_kernel->syncSolver(0,nb_iteration,&residual_norm[0]);\n\
\tinfo() << \"Solved in \33[7m\" << nb_iteration << \"\33[m iterations,\"\n\
\t\t<< \"residuals=[\33[1m\" << residual_norm[0] <<\"\33[m,\"<< residual_norm[3]<<\"]\";\n\
\tsolution->getLocalComponents(vector_indexs.size(),vector_indexs,rhs);\n\
//#warning weird getLocalComponents with rhs\n\
#else\n\
\talephSolveWithoutIndex();\n\
#endif\n}");
}


/******************************************************************************
 * lambdaAlephSolveWithoutIndex
 ******************************************************************************/
static void lambdaAlephSolveWithoutIndex(nablaMain *arc){
  fprintf(arc->entity->src, "\nvoid alephSolveWithoutIndex(){\n\
\tint nb_iteration;\n\
\tdouble residual_norm[4];\n\
\tm_aleph_mat->assemble();\n\
\tm_aleph_rhs->setLocalComponents(rhs);\n\
\tm_aleph_rhs->assemble();\n\
\tvector_zeroes.resize(rhs.size());\n\
\tvector_zeroes.assign(rhs.size(),0.0);\n\
\tm_aleph_sol->setLocalComponents(vector_zeroes);\n\
\tm_aleph_sol->assemble();\n\
\tm_aleph_mat->solve(m_aleph_sol, m_aleph_rhs, nb_iteration, &residual_norm[0], m_aleph_params, true);\n\
\tAlephVector *solution=m_aleph_kernel->syncSolver(0,nb_iteration,&residual_norm[0]);\n\
\tinfo() << \"Solved in \33[7m\" << nb_iteration << \"\33[m iterations,\"\n\
\t\t<< \"residuals=[\33[1m\" << residual_norm[0] <<\"\33[m,\"<< residual_norm[3]<<\"]\";\n\
\tlhs.reset(); lhs.resize(rhs.size()); //lhs.fill(0.0);\n\
\tsolution->getLocalComponents(lhs);\n}");
}


/******************************************************************************
 * lambdaAlephIni
 ******************************************************************************/
void lambdaAlephIni(nablaMain *arc){
  fprintf(arc->entity->src, "\n\n\
// ****************************************************************************\n\
// * alephIni\n\
// ****************************************************************************\
\nvoid alephIni(void){\
\n#ifndef ALEPH_INDEX\
\n\t// Pourrait être enlevé, mais est encore utilisé dans le test DDVF sans auto-index\
\n\tvector_indexs.resize(0);\
\n\tvector_zeroes.resize(0);\
\n\trhs.resize(0);\
\n\tmesh()->checkValidMeshFull();");
fprintf(arc->entity->src, "\n\tm_aleph_factory=NULL;//new AlephFactory(subDomain()->application(),traceMng());");
  fprintf(arc->entity->src, "\
\n\tm_aleph_kernel=new AlephKernel(traceMng(), subDomain(), m_aleph_factory,\n\
                                   alephUnderlyingSolver,\n\
                                   alephNumberOfCores,\n\
                                   false);//options()->alephMatchSequential());");
  fprintf(arc->entity->src, "\n#else");
  fprintf(arc->entity->src, "\
\n\tm_aleph_kernel=new AlephKernel(subDomain(),\n\
                                  alephUnderlyingSolver,\n\
                                  alephNumberOfCores);");
  fprintf(arc->entity->src, "\n\trhs.m_aleph_kernel=m_aleph_kernel;");
  fprintf(arc->entity->src, "\n\tlhs.m_aleph_kernel=m_aleph_kernel;");
  fprintf(arc->entity->src, "\n\tmtx.m_aleph_kernel=m_aleph_kernel;");
  fprintf(arc->entity->src, "\n#endif");
  fprintf(arc->entity->src, "\n\tm_aleph_params=\n\
    new AlephParams(traceMng(),\n\
                    alephEpsilon,      // epsilon epsilon de convergence\n\
                    alephMaxIterations,// max_iteration nb max iterations\n\
                    (TypesSolver::ePreconditionerMethod)alephPreconditionerMethod,// préconditionnement: ILU, DIAGONAL, AMG, IC\n\
                    (TypesSolver::eSolverMethod)alephSolverMethod, // méthode de résolution\n\
                    -1,                             // gamma\n\
                    -1.0,                           // alpha\n\
                    true,                           // xo_user par défaut Xo n'est pas égal à 0\n\
                    false,                          // check_real_residue\n\
                    false,                          // print_real_residue\n\
                    false,                          // debug_info\n\
                    1.e-20,                         // min_rhs_norm\n\
                    false,                          // convergence_analyse\n\
                    true,                           // stop_error_strategy\n\
                    option_aleph_dump_matrix,       // write_matrix_to_file_error_strategy\n\
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
  lambdaAlephInitialize(arc);
  lambdaAlephMatAddValueWithIVariableAndItem(arc);
  lambdaAlephMatAddValue(arc);
  lambdaAlephRhsSet(arc);
  lambdaAlephRhsAdd(arc);
  lambdaAlephRhsGet(arc);
  lambdaAlephSolve(arc);
  lambdaAlephSolveWithoutIndex(arc);

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
