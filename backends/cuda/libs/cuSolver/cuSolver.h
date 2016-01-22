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

#include <cuda_runtime.h>
#include "cusparse.h"


// ****************************************************************************
// * AlephRealMatrix
// ****************************************************************************
class AlephRealMatrix{
public:
/*  void error(){throw std::logic_error("AlephRealArray");}  
  void addValue(double *iVar, int iItm,
                double *jVar, int jItm,
                double value){
    m_aleph_mat->addValue(iVar,iItm,jVar,jItm,value);
  }
  void setValue(double *iVar, int iItm,
                double *jVar, int jItm,
                double value){
    m_aleph_mat->setValue(iVar,iItm,jVar,jItm,value);
  }
  public:
  AlephKernel *m_aleph_kernel;
  AlephMatrix *m_aleph_mat;*/
};


// ****************************************************************************
// * AlephRealArray
// ****************************************************************************
class AlephRealArray{//:public vector<double>{
public:
/*  void reset(){
    debug()<<"\33[1;33m[vector::reset]\33[0m";
    resize(0);
  }
  void error(){throw std::logic_error("[AlephRealArray] Error");}
  void newValue(double value){
    push_back(value);
  }
  void addValue(double *var, int itm, double value){
    unsigned int idx=m_aleph_kernel->indexing()->get(var,itm);
    debug()<<"\33[1;33m[vector::addValue["<<idx<<"]\33[0m";
    if (idx==size()){
      resize(idx+1);
      index.push_back(idx);
      this->at(idx)=value;
    }else{
      this->at(idx)=at(idx)+value;
    }
  }
  void setValue(double *var, int itm, double value){
    //debug()<<"\33[1;33m[vector::setValue(...)]\33[0m";
    int topology_row_offset=0;
    unsigned int idx=m_aleph_kernel->indexing()->get(var,itm)-topology_row_offset;
    if(idx==size()){
      resize(idx+1);
      index.push_back(idx);
      this->at(idx)=value;
    }else{
      this->at(idx)=value;
    }
  }
  double getValue(double *var, int itmEnum){
    //debug()<<"\33[1;33m[vector::getValue]\33[0m";
    return at(m_aleph_kernel->indexing()->get(var,itmEnum));
  }
public:
  vector<int> index;
  AlephKernel *m_aleph_kernel;*/
};


// ****************************************************************************
// * Globals for Aleph
// ****************************************************************************
/*
IAlephFactory *m_aleph_factory;
AlephKernel *m_aleph_kernel;
AlephParams *m_aleph_params;
AlephMatrix *m_aleph_mat;
AlephVector *m_aleph_rhs;
AlephVector *m_aleph_sol;

vector<int> vector_indexs;
vector<double> vector_zeroes;

AlephRealArray lhs;
AlephRealArray rhs;
AlephRealMatrix mtx;
*/

// ****************************************************************************
// * Forward Declarations
// ****************************************************************************
void alephInitialize(void){
  //m_aleph_mat=m_aleph_kernel->createSolverMatrix();
  //m_aleph_rhs=m_aleph_kernel->createSolverVector();
  //m_aleph_sol=m_aleph_kernel->createSolverVector();
  //m_aleph_mat->create();
  //m_aleph_rhs->create();
  //m_aleph_sol->create();
  //m_aleph_mat->reset();
  //mtx.m_aleph_mat=m_aleph_mat;
}


// ****************************************************************************
// * alephIni
// ****************************************************************************
void alephIni(%s%s){ // we have to match args & params
}


// ****************************************************************************
// * alephAddValue
// ****************************************************************************
void alephAddValue(double *rowVar, int rowItm,
                   double *colVar, int colItm, double val){
  //m_aleph_mat->addValue(rowVar,rowItm,colVar,colItm,val);
}

void alephAddValue(int i, int j, double val){
  //m_aleph_mat->addValue(i,j,val);
}


// ****************************************************************************
// * alephRhsSet
// ****************************************************************************
void alephRhsSet(int row, double value){
  //rhs[row-rank_offset]=value;
}

void alephRhsSet(double *var, int itm, double value){
  //rhs[m_aleph_kernel->indexing()->get(var,itm)]=value;
}


// ****************************************************************************
// * alephRhsGet
// ****************************************************************************
double alephRhsGet(int row){
  return 0.0;
}


// ****************************************************************************
// * alephRhsAdd
// ****************************************************************************
void alephRhsAdd(int row, double value){
}


// ****************************************************************************
// * alephSolveWithoutIndex
// ****************************************************************************
void alephSolveWithoutIndex(void){
}


// ****************************************************************************
// * alephSolve
// ****************************************************************************
void alephSolve(void){
}
