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
#ifndef _KN_MIC_OSTREAM_H_
#define _KN_MIC_OSTREAM_H_


// ****************************************************************************
// * INTEGERS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const integer &a){
  int *ip = (int*)&(a);
  return os << "["<<*(ip+0)<<","<<*(ip+1)<<","<<*(ip+2)<<","<<*(ip+3)<<","<<*(ip+4)<<","<<*(ip+5)<<","<<*(ip+6)<<","<<*(ip+7)<<"]";
}


// ****************************************************************************
// * REALS
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const __m512d v){
  const double *fp = (double*)&v;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3)
            <<","<<*(fp+4)<<","<<*(fp+5)<<","<<*(fp+6)<<","<<*(fp+7)<<"]";
}

std::ostream& operator<<(std::ostream &os, const real &a){
  const double *fp = (double*)&a;
  return os << "["<<*(fp+0)<<","<<*(fp+1)<<","<<*(fp+2)<<","<<*(fp+3)
            <<","<<*(fp+4)<<","<<*(fp+5)<<","<<*(fp+6)<<","<<*(fp+7)<<"]";
}


// ****************************************************************************
// * REALS_3
// ****************************************************************************
std::ostream& operator<<(std::ostream &os, const real3 &a){
  double *x = (double*)&(a.x);
  double *y = (double*)&(a.y);
  double *z = (double*)&(a.z);
  return os << "[("<<*(x+0)<<","<<*(y+0)<<","<<*(z+0)<< "), "
            <<  "("<<*(x+1)<<","<<*(y+1)<<","<<*(z+1)<< "), "
            <<  "("<<*(x+2)<<","<<*(y+2)<<","<<*(z+2)<< "), "
            <<  "("<<*(x+3)<<","<<*(y+3)<<","<<*(z+3)<< "), "
            <<  "("<<*(x+4)<<","<<*(y+4)<<","<<*(z+4)<< "), "
            <<  "("<<*(x+5)<<","<<*(y+5)<<","<<*(z+5)<< "), "
            <<  "("<<*(x+6)<<","<<*(y+6)<<","<<*(z+6)<< "), "
            <<  "("<<*(x+7)<<","<<*(y+7)<<","<<*(z+7)<< ")]";
}


#endif // _KN_MIC_OSTREAM_H_
