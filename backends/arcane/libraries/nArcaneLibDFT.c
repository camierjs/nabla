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
#include <nabla.h>


/***************************************************************************** 
 * Préparation du header de l'entity
 *****************************************************************************/
char* nccArcLibDftHeader(void){
  return "\
\n#include <arcane/utils/TraceClassConfig.h>\
\n#include <arcane/utils/ITraceMng.h>\
\n#include <arcane/ObserverPool.h>\
\n#include <arcane/IVariableMng.h>\
\nextern \"C\" void lltIni(void);\
\nextern \"C\" int lltWork(int,int*,int,int*,long*,long*,long*,int**,int*,int);\
\nextern \"C\" unsigned int get_default_fft_length(unsigned int);\
";
}


// ****************************************************************************
// * nccArcLibDftDelete
// ****************************************************************************
char *nccArcLibDftDelete(void){
  return "\n\t\t// nccArcLibDftDelete";
}


// ****************************************************************************
// * nccArcLibDftPrivates
// ****************************************************************************
char* nccArcLibDftPrivates(void){
  return "\nprivate:\t//DFT stuff\
\n\tvoid dftIni(void);\
\n\tvoid dftContinueInit(void);\
\n\tvoid dftReadBackup(void);\
\n\tvoid dftWriteBackup(void);\
\n\tObserverPool m_dft_observers;\
\n\tVariableCellInt32 m_dft_i;\
\n\tVariableCellInt64 m_dft_sh0;// mod-(2^64) residue\
\n\tVariableCellInt64 m_dft_sh1;// Selfridge/Hurwitz Residue mod 2^35-1\
\n\tVariableCellInt64 m_dft_sh2;// Selfridge/Hurwitz Residue mod 2^36-1\
\n\tVariableCellArrayByte m_dft_data;\
\n\t// ****************************************************************************\
\n\t// * dftLlt\
\n\t// ****************************************************************************\
\n\tint dftLlt(CellEnumerator cell, int exponent, int i){\
\n\t\tbool continuing=i!=0;\
\n\t\t//if (continuing) info()<<\"\\33[1;36m[dftLlt] CONTINUING\\33[m\";\
\n\t\t// On pré-calcule la taille dont on va avoir besoin\
\n\t\tunsigned int lenDFT=get_default_fft_length(exponent);\
\n\t\tint returned=0;\
\n\t\tint bytes_array_length;\
\n\t\tint *bytes_array;\
\n\t\t// On met à jour la taille des backups\
\n\t\t// Attention: il faut que toutes les dft soient de la même taille!\
\n\t\t// Alloc this to the old-style nalloc*int size, so it'll be large\
\n\t\t// enough to hold either an old-style integer residue or a (more compact)\
\n\t\t// bytewise residue\
\n\t\tif (!continuing){\
\n\t\t\tinfo()<<\"\\33[1;36m[dftLlt] DFT lenght set to \"<<lenDFT<<\"\\33[m\";\
\n\t\t\tm_dft_data.resize(lenDFT*1024*sizeof(double)+512);\
\n\t\t\tdebug()<<\"m_dft_data resized to \"<<lenDFT*1024*sizeof(double)+512;\
\n\t\t}\
\n\t\tdebug()<<\"\\33[1;36m[dftLlt] unguardedBasePointer()=\"<<(void*)m_dft_data[cell].unguardedBasePointer()<<\"\\33[m\";\
\n\t\treturned=lltWork(exponent,\
\n\t\t\t\t\t\t\t\t\t\t(int*)m_dft_data[cell].unguardedBasePointer(),\
\n\t\t\t\t\t\t\t\t\t\tlenDFT,\
\n\t\t\t\t\t\t\t\t\t\t&m_dft_i[cell],\
\n\t\t\t\t\t\t\t\t\t\t&m_dft_sh0[cell],\
\n\t\t\t\t\t\t\t\t\t\t&m_dft_sh1[cell],\
\n\t\t\t\t\t\t\t\t\t\t&m_dft_sh2[cell],\
\n\t\t\t\t\t\t\t\t\t\t&bytes_array, &bytes_array_length,\
\n\t\t\t\t\t\t\t\t\t\tcontinuing);\
\n\t\tdebug()<<\"\\33[1;36m[dftLlt] bytes_array_length=\"<<bytes_array_length<<\"\\33[m\";\
\n\t\tdebug()<<\"\\33[1;36m[dftLlt] bytes_array=\"<<bytes_array<<\"\\33[m\";\
\n\t\tdebug()<<\"\\33[1;36m[dftLlt] return code: \"<<returned<<\"\\33[m\";\
\n\t\tif (returned<0)\
\n\t\t\tfatal()<<\"\\33[1;36m[dftLlt] returned error=\"<<returned<<\"\\33[m\";\
\n\t\treturn returned;\
\n\t}";
}


// ****************************************************************************
// * nccArcLibDftWriteBackup
// ****************************************************************************
void nccArcLibDftWriteBackup(nablaMain *arc){
  fprintf(arc->entity->src, "\n\
\n// ****************************************************************************\
\n// * dftWriteBackup\
\n// ****************************************************************************\
\nvoid %s%s::dftWriteBackup(void){                                      \
\n\tinfo()<<\"\\33[7m[dftWriteBackup]\"<<\"\\33[m\";\
}",arc->name,
          nablaArcaneColor(arc));
}

// ****************************************************************************
// * nccArcLibDftReadBackup
// ****************************************************************************
void nccArcLibDftReadBackup(nablaMain *arc){
  fprintf(arc->entity->src, "\n\
\n// ****************************************************************************\
\n// * dftReadBackup\
\n// ****************************************************************************\
\nvoid %s%s::dftReadBackup(void){\
\n\tinfo()<<\"\\33[7m[dftReadBackup]\"<<\"\\33[m\";\
\n}",arc->name,
          nablaArcaneColor(arc));
}



  
// ****************************************************************************
// * nccArcLibDftIni
// ****************************************************************************
void nccArcLibDftIni(nablaMain *arc){
  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * dftInit\
\n// ****************************************************************************\
\nvoid %s%s::dftIni(void){\
\n\tinfo()<<\"\\33[7m[dftIni]\"<<\"\\33[m\";\
\n\tIVariableMng* vm = subDomain()->variableMng();\
\n\tm_dft_observers.addObserver(this, &%s%s::dftWriteBackup,vm->writeObservable());\
\n\tlltIni();\n}",
          arc->name, nablaArcaneColor(arc),
          arc->name, nablaArcaneColor(arc));
  nablaJob *dftInitFunction=nMiddleJobNew(arc->entity);
  dftInitFunction->is_an_entry_point=true;
  dftInitFunction->is_a_function=true;
  dftInitFunction->scope  = strdup("NoGroup");
  dftInitFunction->region = strdup("NoRegion");
  dftInitFunction->item   = strdup("\0");
  dftInitFunction->return_type  = strdup("void");
  dftInitFunction->name   = strdup("dftIni");
  dftInitFunction->name_utf8 = strdup("dftIni");
  dftInitFunction->xyz    = strdup("NoXYZ");
  dftInitFunction->direction  = strdup("NoDirection");
  sprintf(&dftInitFunction->at[0],"-huge_valf");
  dftInitFunction->when_index  = 1;
  dftInitFunction->whens[0] = ENTRY_POINT_init;
  nMiddleJobAdd(arc->entity, dftInitFunction);

  nccArcLibDftReadBackup(arc);
  nccArcLibDftWriteBackup(arc);

  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * dftContinueInit\
\n// ****************************************************************************\
\nvoid %s%s::dftContinueInit(void){\
\n\tinfo()<<\"\\33[7m[dftContinueInit]\"<<\"\\33[m\";\
\n\tdftReadBackup();\
\n}\n\n",arc->name,
          nablaArcaneColor(arc));
  nablaJob *dftContinueInit=nMiddleJobNew(arc->entity);
  dftContinueInit->is_an_entry_point=true;
  dftContinueInit->is_a_function=true;
  dftContinueInit->scope  = strdup("NoGroup");
  dftContinueInit->region = strdup("NoRegion");
  dftContinueInit->item   = strdup("\0");
  dftContinueInit->return_type  = strdup("void");
  dftContinueInit->name   = strdup("dftContinueInit");
  dftContinueInit->name_utf8   = strdup("dftContinueInit");
  dftContinueInit->xyz    = strdup("NoXYZ");
  dftContinueInit->direction  = strdup("NoDirection");
  sprintf(&dftContinueInit->at[0],"-0.0");
  dftContinueInit->when_index  = 1;
  dftContinueInit->whens[0] = ENTRY_POINT_continue_init;
  nMiddleJobAdd(arc->entity, dftContinueInit);
  
}
