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
#include "../Aleph.h"

#include "IAlephHypre.h"


HypreAlephFactoryImpl::HypreAlephFactoryImpl():
  m_IAlephVectors(0),
  m_IAlephMatrixs(0){}

HypreAlephFactoryImpl::~HypreAlephFactoryImpl(){
  std::cout << "\33[1;5;31m[~HypreAlephFactoryImpl]\33[0m";
  for(int i=0,iMax=m_IAlephVectors.size(); i<iMax; ++i)
    delete m_IAlephVectors.at(i);
  for(int i=0,iMax=m_IAlephMatrixs.size(); i<iMax; ++i)
    delete m_IAlephMatrixs.at(i);
}

void HypreAlephFactoryImpl::initialize() {}

IAlephTopology* HypreAlephFactoryImpl::createTopology(ITraceMng* tm,
                                                      AlephKernel* kernel,
                                                      int index,
                                                      int nb_row_size){
  return NULL;
}

IAlephVector* HypreAlephFactoryImpl::createVector(ITraceMng* tm,
                                                  AlephKernel* kernel,
                                                  int index){
  IAlephVector *new_vector=new AlephVectorHypre(tm,kernel,index);
  m_IAlephVectors.push_back(new_vector);
  return new_vector;
}

IAlephMatrix* HypreAlephFactoryImpl::createMatrix(ITraceMng* tm,
                                                  AlephKernel* kernel,
                                                  int index){
  IAlephMatrix *new_matrix=new AlephMatrixHypre(tm,kernel,index);
  m_IAlephMatrixs.push_back(new_matrix);
  return new_matrix;
}


IAlephFactoryImpl* fetchHypre(void){
  return dynamic_cast<IAlephFactoryImpl*>(new HypreAlephFactoryImpl());
}
