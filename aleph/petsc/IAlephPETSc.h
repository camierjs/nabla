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
#ifndef _ALEPH_INTERFACE_PETSC_H_
#define _ALEPH_INTERFACE_PETSC_H_

class AlephTopologyPETSc: public IAlephTopology{
public:
  AlephTopologyPETSc(ITraceMng* tm,
                     AlephKernel *kernel,
                     Integer index,
                     Integer nb_row_size):
    IAlephTopology(tm, kernel, index, nb_row_size){

    if (!m_participating_in_solver){
      debug() << "\33[1;32m\t[AlephTopologyPETSc] Not concerned with this solver, returning\33[0m";
      return;    
    }
    debug() << "\33[1;32m\t\t[AlephTopologyPETSc] @"<<this<<"\33[0m";
    if (!m_kernel->isParallel()){
      PETSC_COMM_WORLD = PETSC_COMM_SELF;
    }else{
      PETSC_COMM_WORLD = *(MPI_Comm*)(kernel->subParallelMng(index)->getMPICommunicator());
    }
    PetscInitializeNoArguments();
  }
  ~AlephTopologyPETSc(){
    debug() << "\33[1;5;32m\t\t\t[~AlephTopologyPETSc]\33[0m";
  }
public:
  void backupAndInitialize(){}
  void restore(){}
};


class AlephVectorPETSc: public IAlephVector{
public:
  AlephVectorPETSc(ITraceMng*, AlephKernel*, Integer);
  void AlephVectorCreate(void);
  void AlephVectorSet(const double*, const int*, Integer);
  int AlephVectorAssemble(void);
  void AlephVectorGet(double*, const int*, Integer);
  void writeToFile(const String);
  Real LinftyNorm(void);
public:
  Vec m_petsc_vector;
  PetscInt jSize,jUpper,jLower;
};
  

class AlephMatrixPETSc: public IAlephMatrix{
public:
  AlephMatrixPETSc(ITraceMng*,AlephKernel*,Integer);
  void AlephMatrixCreate(void);
  void AlephMatrixSetFilled(bool);
  int AlephMatrixAssemble(void);
  void AlephMatrixFill(int, int*, int*, double*);
  Real LinftyNormVectorProductAndSub(AlephVector*, AlephVector*);
  bool isAlreadySolved(AlephVectorPETSc*, AlephVectorPETSc*,
                       AlephVectorPETSc*, Real*, AlephParams*) ;
  int AlephMatrixSolve(AlephVector*, AlephVector*,
                       AlephVector*, Integer&, Real*, AlephParams*);
  void writeToFile(const String);
private:
  Mat m_petsc_matrix;
  KSP m_ksp_solver;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class PETScAlephFactoryImpl: public TraceAccessor,
                             public IAlephFactoryImpl{
public:
  PETScAlephFactoryImpl(ITraceMng *tm):
    TraceAccessor(tm),
    m_IAlephVectors(0),
    m_IAlephMatrixs(0),
    m_IAlephTopologys(0){}
  ~PETScAlephFactoryImpl(){
    debug() << "\33[1;32m[~PETScAlephFactoryImpl]\33[0m";
    for(Integer i=0,iMax=m_IAlephVectors.size(); i<iMax; ++i)
      delete m_IAlephVectors.at(i);
    for(Integer i=0,iMax=m_IAlephMatrixs.size(); i<iMax; ++i)
      delete m_IAlephMatrixs.at(i);
    for(Integer i=0,iMax=m_IAlephTopologys.size(); i<iMax; ++i)
      delete m_IAlephTopologys.at(i);
  }
public:
  virtual void initialize() {}
  virtual IAlephTopology* createTopology(ITraceMng *tm,
                                         AlephKernel *kernel,
                                         Integer index,
                                         Integer nb_row_size){
    IAlephTopology *new_topology=new AlephTopologyPETSc(tm, kernel, index, nb_row_size);
    m_IAlephTopologys.add(new_topology);
    return new_topology;
  }
  virtual IAlephVector* createVector(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index){
    IAlephVector *new_vector=new AlephVectorPETSc(tm,kernel,index);
    m_IAlephVectors.add(new_vector);
    return new_vector;
  }
  
  virtual IAlephMatrix* createMatrix(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index){
    IAlephMatrix *new_matrix=new AlephMatrixPETSc(tm,kernel,index);
    m_IAlephMatrixs.add(new_matrix);
    return new_matrix;
  }
private:
  Array<IAlephVector*> m_IAlephVectors;
  Array<IAlephMatrix*> m_IAlephMatrixs;
  Array<IAlephTopology*> m_IAlephTopologys;
};



#endif // _ALEPH_INTERFACE_PETSC_H_
