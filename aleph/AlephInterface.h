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
#ifndef _ALEPH_INTERFACE_H_
#define _ALEPH_INTERFACE_H_

class AlephKernel;
class IAlephVector;
class AlephVector;

/******************************************************************************
 * IAlephTopology
 *****************************************************************************/
class IAlephTopology: public TraceAccessor{
public:
  IAlephTopology(ITraceMng* tm,
                 AlephKernel *kernel,
                 int index,
                 int nb_row_size):TraceAccessor(tm),
                                      m_index(index),
                                      m_kernel(kernel),
                                      m_participating_in_solver(kernel->subParallelMng(index)!=NULL){
    debug() << "\33[1;34m\t\t[IAlephTopology] NEW IAlephTopology"<<"\33[0m";
    debug() << "\33[1;34m\t\t[IAlephTopology] m_participating_in_solver="
            << m_participating_in_solver<<"\33[0m";
  }
  ~IAlephTopology(){
    debug()<<"\33[1;5;34m\t\t[~IAlephTopology]"<<"\33[0m";
  }
public:
  virtual void backupAndInitialize()=0;
  virtual void restore()=0;
protected:
  int m_index;
  AlephKernel *m_kernel;
  bool m_participating_in_solver;
};


/******************************************************************************
 * IAlephVector
 *****************************************************************************/
class IAlephVector: public TraceAccessor{
public:
  IAlephVector(ITraceMng* tm,
               AlephKernel *kernel,
               int index):TraceAccessor(tm),
                              m_index(index),
                              m_kernel(kernel)
  {
    debug()<<"\33[1;34m\t\t[IAlephVector] NEW IAlephVector"<<"\33[0m";
  }
  ~IAlephVector(){
    debug()<<"\33[1;5;34m\t\t[~IAlephVector]"<<"\33[0m";
  }
public:
  virtual void AlephVectorCreate(void)=0;
  virtual void AlephVectorSet(const double*, const long long int*,int)=0;
  virtual int AlephVectorAssemble(void)=0;
  virtual void AlephVectorGet(double*, const long long int*,int)=0;
  virtual void writeToFile(const string)=0;
protected:
  int m_index;
  AlephKernel *m_kernel;
};


/******************************************************************************
 * IAlephMatrix
 *****************************************************************************/
class IAlephMatrix: public TraceAccessor{
public:
  IAlephMatrix(ITraceMng* tm,
               AlephKernel *kernel,
               int index):TraceAccessor(tm),
                              m_index(index),
                              m_kernel(kernel)
  {
    debug()<<"\33[1;34m\t\t[IAlephMatrix] NEW IAlephMatrix"<<"\33[0m";
  }
  ~IAlephMatrix(){
    debug()<<"\33[1;5;34m\t\t[~IAlephMatrix]"<<"\33[0m";
  }
public:
  virtual void AlephMatrixCreate(void)=0;
  virtual void AlephMatrixSetFilled(bool)=0;
  virtual int AlephMatrixAssemble(void)=0;
  virtual void AlephMatrixFill(int, long long int*, long long int*, double*)=0;
  virtual int AlephMatrixSolve(AlephVector*,
										 AlephVector*,
                               AlephVector*,
										 int&,
										 double*,
										 AlephParams*)=0;
  virtual void writeToFile(const string)=0;
protected:
  int m_index;
  AlephKernel *m_kernel;
};


class IAlephFactory: public TraceAccessor{
public:
  IAlephFactory(ITraceMng *tm):TraceAccessor(tm){
    debug()<<"\33[1;34m[IAlephFactory] NEW IAlephFactory"<<"\33[0m";
  }
  ~IAlephFactory(){
    debug()<<"\33[1;5;34m[~IAlephFactory]"<<"\33[0m";
  }
  virtual IAlephTopology* GetTopology(AlephKernel *, int, int)=0;
  virtual IAlephVector* GetVector(AlephKernel*, int)=0;
  virtual IAlephMatrix* GetMatrix(AlephKernel*, int)=0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une fabrique d'implémentation pour Aleph.
 *
 * Cette interface est utilisée par AlephFactory pour choisir la
 * bibliothèque d'algèbre linéaire sous-jacente (par exemple sloop, hypre,...)
 */
class IAlephFactoryImpl{
 public:
  virtual ~IAlephFactoryImpl(){}
  virtual void initialize()=0;
  virtual IAlephTopology* createTopology(ITraceMng*,AlephKernel*, int, int)=0;
  virtual IAlephVector* createVector(ITraceMng*,AlephKernel*, int)=0;
  virtual IAlephMatrix* createMatrix(ITraceMng*,AlephKernel*, int)=0;
};

#endif // _ALEPH_INTERFACE_H_
