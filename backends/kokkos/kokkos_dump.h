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
#ifndef _NABLA_KOKKOS_GATHER_H_
#define _NABLA_KOKKOS_GATHER_H_


struct minFunctor {
  double value;
  KOKKOS_INLINE_FUNCTION minFunctor():value(1.0e20){}
  KOKKOS_INLINE_FUNCTION minFunctor(double v):value(v){}
  KOKKOS_INLINE_FUNCTION void operator+=(const volatile minFunctor& f) volatile {
    value = fmin(value,f.value);
  }
};


inline real rgatherk(const int a, const Kokkos::View<real*>& data){
  return data[a];
}
inline real3 rgather3k(const int a, const Kokkos::View<real3*>& data){
  return data[a];
}
inline real3x3 rgather3x3k(const int a, const Kokkos::View<real3x3*>& data){
  return data[a];
}

inline real rGatherAndZeroNegOnes(const int a, const Kokkos::View<real*>& data){
  if (a>=0) return data[a];
  return 0.0;
}
inline real3 rGatherAndZeroNegOnes(const int a, const Kokkos::View<real3*>& data){
  if (a>=0) return data[a];
  return 0.0;
}
inline real3x3 rGatherAndZeroNegOnes(const int a, const Kokkos::View<real3x3*>& data){
  if (a>=0) return data[a];
  return 0.0;
}
inline real3 rGatherAndZeroNegOnes(const int a, const int corner, const Kokkos::View<real3*>& data){
  const int i=3*8*a+3*corner;
  const double *p=(double*) data.ptr_on_device();
  if (a>=0) return real3(p[i+0],p[i+1],p[i+2]);
  return 0.0;
}


inline void scatterk(const int a, Kokkos::View<real*>& gathered, Kokkos::View<real*>& data){
  if (a<0) return; // Skipping to fake write
  data[a]=gathered[0];
}


inline void scatter3k(const int a, real3* gathered, Kokkos::View<real3*> data){
  if (a<0) return; // Skipping to fake write
  double *p=(double *)data.ptr_on_device();
  p[3*a+0]=gathered[0].x;
  p[3*a+1]=gathered[0].y;
  p[3*a+2]=gathered[0].z;
}
inline void scatter3k(const int a, Kokkos::View<real3*>& gathered, Kokkos::View<real3*>& data){
  if (a<0) return; // Skipping to fake write
  double *p=(double *)data.ptr_on_device();
  p[3*a+0]=gathered[0].x;
  p[3*a+1]=gathered[0].y;
  p[3*a+2]=gathered[0].z;
}


inline void scatter3x3k(const int a, real3x3* gathered, Kokkos::View<real3x3*> data){
  if (a<0) return; // Skipping to fake write
  data[a]=gathered[0];
}
inline void scatter3x3k(const int a, Kokkos::View<real3x3*> gathered, Kokkos::View<real3x3*> data){
  if (a<0) return; // Skipping to fake write
  data[a]=gathered[0];
}

#endif // _NABLA_KOKKOS_GATHER_H_

