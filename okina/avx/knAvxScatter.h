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
#ifndef _KN_SCATTER_H_
#define _KN_SCATTER_H_


// *****************************************************************************
// * Scatter: (X is the data @ offset x)
// * scatter: |ABCD| and offsets:    a                 b       c   d
// * data:    |....|....|....|....|..A.|....|....|....|B...|...C|..D.|....|....|
// * ! à la séquence car quand c et d sont sur le même warp, ça percute
// ******************************************************************************
inline void scatterk(const int a, const int b,
                     const int c, const int d,
                     real *scatter, real *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  p[a]=s[0];
  p[b]=s[1];
  p[c]=s[2];
  p[d]=s[3];
}


// *****************************************************************************
// * Scatter for real3
// *****************************************************************************
inline void scatter3k(const int a, const int b,
                      const int c, const int d,
                      real3 *scatter, real3 *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  
  p[4*(3*WARP_BASE(a)+0)+WARP_OFFSET(a)]=s[0];
  p[4*(3*WARP_BASE(b)+0)+WARP_OFFSET(b)]=s[1];
  p[4*(3*WARP_BASE(c)+0)+WARP_OFFSET(c)]=s[2];
  p[4*(3*WARP_BASE(d)+0)+WARP_OFFSET(d)]=s[3];

  p[4*(3*WARP_BASE(a)+1)+WARP_OFFSET(a)]=s[4];
  p[4*(3*WARP_BASE(b)+1)+WARP_OFFSET(b)]=s[5];
  p[4*(3*WARP_BASE(c)+1)+WARP_OFFSET(c)]=s[6];
  p[4*(3*WARP_BASE(d)+1)+WARP_OFFSET(d)]=s[7];

  p[4*(3*WARP_BASE(a)+2)+WARP_OFFSET(a)]=s[8];
  p[4*(3*WARP_BASE(b)+2)+WARP_OFFSET(b)]=s[9];
  p[4*(3*WARP_BASE(c)+2)+WARP_OFFSET(c)]=s[10];
  p[4*(3*WARP_BASE(d)+2)+WARP_OFFSET(d)]=s[11];
}

#endif //  _KN_SCATTER_H_
