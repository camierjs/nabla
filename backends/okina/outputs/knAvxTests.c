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


#ifdef __AVX__
  // Test AVX des gathers
__attribute__((unused)) static void avxTest(void){
  real data[4];
  real gthr;
  real scttr=set(0.00001,0.00002,0.00003,0.00004);
  data[0]=set(1.+1./9.,1.+2./9.,1.+3./9.,1.+4./9.);dbgReal(DBG_ALL,data[0]);dbg(DBG_INI,"\n");
  data[1]=set(2.+1./9.,2.+2./9.,2.+3./9.,2.+4./9.);dbgReal(DBG_ALL,data[1]);dbg(DBG_INI,"\n");
  data[2]=set(3.+1./9.,3.+2./9.,3.+3./9.,3.+4./9.);dbgReal(DBG_ALL,data[2]);dbg(DBG_INI,"\n");
  data[3]=set(4.+1./9.,4.+2./9.,4.+3./9.,4.+4./9.);dbgReal(DBG_ALL,data[3]);dbg(DBG_INI,"\n");
  dbg(DBG_INI,"On veut la diagonale:\n\t\t\t");
  gatherk(0+0,4+1,8+2,12+3,data,&gthr);dbgReal(DBG_ALL,gthr);dbg(DBG_INI,"\n");
  dbg(DBG_INI,"Now scattering:");
  scatterk(0+0,4+1,8+2,12+3,&scttr,data);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[0]);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[1]);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[2]);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[3]);
  //Flush
  data[0]=set(1.+1./9.,1.+2./9.,1.+3./9.,1.+4./9.);
  data[1]=set(2.+1./9.,2.+2./9.,2.+3./9.,2.+4./9.);
  data[2]=set(3.+1./9.,3.+2./9.,3.+3./9.,3.+4./9.);
  data[3]=set(4.+1./9.,4.+2./9.,4.+3./9.,4.+4./9.);
  dbg(DBG_INI,"\nNow K-scattering:");
  scatterk(0+0,4+1,8+2,12+3,&scttr,data);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[0]);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[1]);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[2]);
  dbg(DBG_INI,"\n");dbgReal(DBG_ALL,data[3]); 
  Real3 data3[4];
  Real3 gthr3;
  Real3 scttr3=Real3(set(0.00001,0.00002,0.00003,0.00004),
                     set(1.00001,1.00002,1.00003,1.00004),
                     set(2.00001,2.00002,2.00003,2.00004));
  dbg(DBG_INI,"\nX:\n");
  data3[0].x=set(10.+1./9., 10.+2./9., 10.+3./9., 10.+4./9.);dbgReal(DBG_ALL,data3[0].x);dbg(DBG_INI,"\n");
  data3[1].x=set(20.+1./9., 20.+2./9., 20.+3./9., 20.+4./9.);dbgReal(DBG_ALL,data3[1].x);dbg(DBG_INI,"\n");
  data3[2].x=set(30.+1./9., 30.+2./9., 30.+3./9., 30.+4./9.);dbgReal(DBG_ALL,data3[2].x);dbg(DBG_INI,"\n");
  data3[3].x=set(40.+1./9., 40.+2./9., 40.+3./9., 40.+4./9.);dbgReal(DBG_ALL,data3[3].x);dbg(DBG_INI,"\n");
  dbg(DBG_INI,"Y:\n");
  data3[0].y=set(11.+1./9., 11.+2./9., 11.+3./9., 11.+4./9.);dbgReal(DBG_ALL,data3[0].y);dbg(DBG_INI,"\n");
  data3[1].y=set(21.+1./9., 21.+2./9., 21.+3./9., 21.+4./9.);dbgReal(DBG_ALL,data3[1].y);dbg(DBG_INI,"\n");
  data3[2].y=set(31.+1./9., 31.+2./9., 31.+3./9., 31.+4./9.);dbgReal(DBG_ALL,data3[2].y);dbg(DBG_INI,"\n");
  data3[3].y=set(41.+1./9., 41.+2./9., 41.+3./9., 41.+4./9.);dbgReal(DBG_ALL,data3[3].y);dbg(DBG_INI,"\n");
  dbg(DBG_INI,"Z:\n");
  data3[0].z=set(12.+1./9., 12.+2./9., 12.+3./9., 12.+4./9.);dbgReal(DBG_ALL,data3[0].z);dbg(DBG_INI,"\n");
  data3[1].z=set(22.+1./9., 22.+2./9., 22.+3./9., 22.+4./9.);dbgReal(DBG_ALL,data3[1].z);dbg(DBG_INI,"\n");
  data3[2].z=set(32.+1./9., 32.+2./9., 32.+3./9., 32.+4./9.);dbgReal(DBG_ALL,data3[2].z);dbg(DBG_INI,"\n");
  data3[3].z=set(42.+1./9., 42.+2./9., 42.+3./9., 42.+4./9.);dbgReal(DBG_ALL,data3[3].z);dbg(DBG_INI,"\n");
  dbg(DBG_INI,"En memoire:\n");
  dbgReal(DBG_ALL,data3[0].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[0].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[0].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].z);dbg(DBG_INI,"\n");
  dbg(DBG_INI,"\nOn veut la diagonale:\n");
  gather3k(0+0,4+1,8+2,12+3,data3,&gthr3);dbgReal3(DBG_ALL,gthr3);dbg(DBG_INI,"\n");
  dbg(DBG_INI,"\n");
  dbg(DBG_INI,"Now scattering:\n");
  /*scatter3(0+0,4+1,8+2,12+3,scttr3,data3);
  dbgReal(DBG_ALL,data3[0].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[0].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[0].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].z);dbg(DBG_INI,"\n");
  //Flush
  data3[0].x=set(10.+1./9., 10.+2./9., 10.+3./9., 10.+4./9.);
  data3[1].x=set(20.+1./9., 20.+2./9., 20.+3./9., 20.+4./9.);
  data3[2].x=set(30.+1./9., 30.+2./9., 30.+3./9., 30.+4./9.);
  data3[3].x=set(40.+1./9., 40.+2./9., 40.+3./9., 40.+4./9.);
  data3[0].y=set(11.+1./9., 11.+2./9., 11.+3./9., 11.+4./9.);
  data3[1].y=set(21.+1./9., 21.+2./9., 21.+3./9., 21.+4./9.);
  data3[2].y=set(31.+1./9., 31.+2./9., 31.+3./9., 31.+4./9.);
  data3[3].y=set(41.+1./9., 41.+2./9., 41.+3./9., 41.+4./9.);
  data3[0].z=set(12.+1./9., 12.+2./9., 12.+3./9., 12.+4./9.);
  data3[1].z=set(22.+1./9., 22.+2./9., 22.+3./9., 22.+4./9.);
  data3[2].z=set(32.+1./9., 32.+2./9., 32.+3./9., 32.+4./9.);
  data3[3].z=set(42.+1./9., 42.+2./9., 42.+3./9., 42.+4./9.);*/
  dbg(DBG_INI,"\nNow K-scattering:\n");
  scatter3k(0+0,4+1,8+2,12+3,&scttr3,data3);
  dbgReal(DBG_ALL,data3[0].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[0].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[0].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[1].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[2].z);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].x);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].y);dbg(DBG_INI,"\n");
  dbgReal(DBG_ALL,data3[3].z);dbg(DBG_INI,"\n");
  //exit(0);
}

#endif
