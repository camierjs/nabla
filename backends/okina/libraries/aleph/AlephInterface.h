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
#ifndef _ALEPH_INTERFACE_H_
#define _ALEPH_INTERFACE_H_

#include "AlephKernel.h"


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
                 Integer index,
                 Integer nb_row_size):TraceAccessor(tm),
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
  Integer m_index;
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
               Integer index):TraceAccessor(tm),
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
  virtual void AlephVectorSet(const double*,const int*,Integer)=0;
  virtual int AlephVectorAssemble(void)=0;
  virtual void AlephVectorGet(double*,const int*,Integer)=0;
  virtual void writeToFile(const String)=0;
protected:
  Integer m_index;
  AlephKernel *m_kernel;
};


/******************************************************************************
 * IAlephMatrix
 *****************************************************************************/
class IAlephMatrix: public TraceAccessor{
public:
  IAlephMatrix(ITraceMng* tm,
               AlephKernel *kernel,
               Integer index):TraceAccessor(tm),
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
  virtual void AlephMatrixFill(int, int*, int*, double*)=0;
  virtual int AlephMatrixSolve(AlephVector*,
										 AlephVector*,
                               AlephVector*,
										 Integer&,
										 Real*,
										 AlephParams*)=0;
  virtual void writeToFile(const String)=0;
protected:
  Integer m_index;
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
  virtual IAlephTopology* GetTopology(AlephKernel *, Integer, Integer)=0;
  virtual IAlephVector* GetVector(AlephKernel*, Integer)=0;
  virtual IAlephMatrix* GetMatrix(AlephKernel*, Integer)=0;
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
  virtual IAlephTopology* createTopology(ITraceMng*,AlephKernel*, Integer, Integer)=0;
  virtual IAlephVector* createVector(ITraceMng*,AlephKernel*, Integer)=0;
  virtual IAlephMatrix* createMatrix(ITraceMng*,AlephKernel*, Integer)=0;
};

#endif // _ALEPH_INTERFACE_H_
