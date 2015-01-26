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
#include <nabla.h>


/***************************************************************************** 
 * Préparation du header du entity
 *****************************************************************************/
char* nccArcLibGmpHeader(void){
  return "\
\n#include <math.h>\
\n#include <time.h>\
\n#include <stdio.h>\
\n#include <gmp.h>\
\n#include <gmpxx.h>\
\n#include <arcane/utils/TraceClassConfig.h>\
\n#include <arcane/utils/ITraceMng.h>\
\n#include <arcane/ObserverPool.h>\
\n#include <arcane/IVariableMng.h>\
\n#include <arcane/MultiArray2VariableRef.h>\
\n";
}


// ****************************************************************************
// * nccArcLibGmpDelete
// ****************************************************************************
char *nccArcLibGmpDelete(void){
  return "\n\t\t// Should mpz_clear each mpz_t in m_gmp;";
}


// ****************************************************************************
// * nccArcLibGmpPrivates
// ****************************************************************************
char* nccArcLibGmpPrivates(void){
  return   "\nprivate:\t//Gmp stuff\
\n\tvoid gmpInit(void);\
\n\tvoid gmpContinueInit(void);\
\n\tvoid gmpWritBfrDump(void);\
\n\tvoid gmpReadFrmDump(void);\
\n\tArray<Array<mpz_class> > m_gmp;\
\n\tObserverPool m_gmp_observers;\
\n\tVariableMultiArray2Byte m_gmp_backups;\
\n\tInteger m_gmp_iteration;";
}


// ****************************************************************************
// * nccArcLibGmpWritBfrDump
// ****************************************************************************
void nccArcLibGmpWritBfrDump(nablaMain *arc){
  int i;
  int gmpNumberOfPrecise=nablaVariableGmpRank(arc->variables);
  fprintf(arc->entity->src, "\n\
\n// ****************************************************************************\
\n// * gmpWritBfrDump\
\n// ****************************************************************************\
\n// void * mpz_export (void *rop, size t *countp, int order, size t size, int endian, size t nails, mpz t op ) \
\n// Fill rop with word data from op.\
\n// The parameters specify the format of the data produced.\
\n// Each word will be size bytes and order can be 1 for MSW first or -1 for LSW first.\
\n// Within each word endian can be 1 for MSB first, -1 for LSB first, or 0 for the native endianness.\
\n// The most significant nails bits of each word are unused and set to zero, this can be 0 to produce full words.\
\n// The number of words produced is written to *countp , or countp can be NULL to discard the count.\
\n// rop must have enough space for the data,\
\n// if rop is NULL then a result array of the necessary size is allocated using the current GMP allocation function.\
\n// In either case the return value is the destination used, either rop or the allocated block.");  
  fprintf(arc->entity->src, "\nvoid %s%s::gmpWritBfrDump(void){\
\n\tconst int gmpNumberOfPrecise=%d;\
\n\tif (m_gmp_iteration==subDomain()->commonVariables().globalIteration()){\
\n\t\tinfo()<<\"\\33[7;33m[gmpWritBfrDump] double call, returning!\"<<\"\\33[m\";\
\n\t\treturn;\
\n\t}\
\n\tm_gmp_iteration=subDomain()->commonVariables().globalIteration();\
\n\tinfo()<<\"\\33[7;33m[gmpWritBfrDump]\"<<\"\\33[m\";\
\n\tIntegerArray gmp_backups_sizes;\
\n\tENUMERATE_CELL(cell,ownCells()){\
\n\t\tint count;\
\n\t\t//size_t countp=0;\
\n\t\tconst size_t size=1; // Each word will be 1 byte\
\n\t\t//const int order=1;   // Most significant word first\
\n\t\t//const int endian=0;  // Native endianness of the host CPU\
\n\t\tconst int nails=0;   // Full words production\
\n\t\tconst int numb = 8*size - nails;\
\n\t\tmpz_t imported;\
\n\t\tmpz_init(imported);\
\n\t\t//info()<<\"cell #\"<<cell->uniqueId();\
\n\t\tfor(int i=0;i<gmpNumberOfPrecise;i+=1){\
\n\t\t\tcount = (mpz_sizeinbase(m_gmp[i][cell->localId()].get_mpz_t(), 2) + numb-1) / numb;\
\n\t\t\t//info()<<\"Should resize to \"<<count;\
\n\t\t\tgmp_backups_sizes.add(count);\
\n\t\t}\n\t}",
          arc->name,
          nablaArcaneColor(arc),
          gmpNumberOfPrecise);
   if (nablaVariableGmpDumpNumber(arc->variables)!=0){
     fprintf(arc->entity->src, "\n\
\n\t// On a fait cela pour resizer les tableaux de backups\
\n\tm_gmp_backups.resize(gmp_backups_sizes);");
     fprintf(arc->entity->src, "\
\n\t// Et on revient sérialiser\
\n\tENUMERATE_CELL(cell,ownCells()){\
\n\t\t//int count;\
\n\t\tsize_t countp=0;\
\n\t\tconst size_t size=1; // Each word will be 1 byte\
\n\t\tconst int order=1;   // Most significant word first\
\n\t\tconst int endian=0;  // Native endianness of the host CPU\
\n\t\tconst int nails=0;   // Full words production\
\n\t\t//const int numb = 8*size - nails;\
\n\t\tmpz_t imported;\
\n\t\tmpz_init(imported);\
\n\t\t//info()<<\"cell #\"<<cell->uniqueId();");
     for(i=0;i<gmpNumberOfPrecise;i+=1){
       char *name=nablaVariableGmpNameRank(arc->variables,i);
       if (nablaVariableGmpDumpRank(arc->variables,i)==false){
         fprintf(arc->entity->src,"\n\t\t// Not dumping cell variable m_cell_%s",name);
         continue;
       }
       fprintf(arc->entity->src,"\
\n\t\t// Dumping cell variable m_cell_%s\
\n\t\tmpz_export((void*)m_gmp_backups[gmpNumberOfPrecise*cell->localId()+%d].unguardedBasePointer(),\
\n\t\t\t\t\t\t\t\t&countp, order, size, endian, nails,\
\n\t\t\t\t\t\t\t\tm_gmp[%d][cell->localId()].get_mpz_t());\
\n\t\tmpz_import(imported, countp, order, size, endian, nails, (void*)m_gmp_backups[gmpNumberOfPrecise*cell->localId()+%d].unguardedBasePointer());\
\n\t\tif (mpz_cmp(m_gmp[%d][cell->localId()].get_mpz_t(),imported)!=0)\
\n\t\t\tfatal()<<\"gmpWritBfrDump error while comparing its backup\";",
               name,i,i,i,i);
     }
     fprintf(arc->entity->src, "\
\n\t\tmpz_clear(imported);\
\n\t}");
   }
   fprintf(arc->entity->src,"\n}");
}


// ****************************************************************************
// * nccArcLibGmpReadFrmDump
// ****************************************************************************
void nccArcLibGmpReadFrmDump(nablaMain *arc){
  int i;
  int gmpNumberOfPrecise=nablaVariableGmpRank(arc->variables);
fprintf(arc->entity->src, "\n\
\n// ****************************************************************************\
\n// * gmpReadFrmDump\
\n// ****************************************************************************");  
 fprintf(arc->entity->src, "\nvoid %s%s::gmpReadFrmDump(void){",
         arc->name,
         nablaArcaneColor(arc));
 if (nablaVariableGmpDumpNumber(arc->variables)!=0){
   fprintf(arc->entity->src, "\
\n\tconst int gmpNumberOfPrecise=%d;\
\n\tinfo()<<\"\\33[7m[gmpReadFrmDump]\"<<\"\\33[m\";\
\n\tENUMERATE_CELL(cell,ownCells()){\
\n\t\tsize_t countp;\
\n\t\tconst size_t size=1; // Each word will be 1 byte\
\n\t\tconst int order=1;   // Most significant word first\
\n\t\tconst int endian=0;  // Native endianness of the host CPU\
\n\t\tconst int nails=0;   // Full words production\
\n\t\t//const int numb = 8*size - nails;\
\n\t\t//info()<<\"cell #\"<<cell->uniqueId();",gmpNumberOfPrecise);
   for(i=0;i<gmpNumberOfPrecise;i+=1){
     char *name=nablaVariableGmpNameRank(arc->variables,i);
     if (nablaVariableGmpDumpRank(arc->variables,i)==false){
       fprintf(arc->entity->src,"\n\t\t// Not restoring cell variable m_cell_%s",name);
       continue;
     }
     fprintf(arc->entity->src,"\n\t\t// Restoring cell variable m_cell_%s\
\n\t\tcountp=m_gmp_backups[gmpNumberOfPrecise*cell->localId()+%d].size();\
\n\t\tmpz_import(m_gmp[%d][cell->localId()].get_mpz_t(), countp, order, size, endian, nails, (void*)m_gmp_backups[gmpNumberOfPrecise*cell->localId()+%d].unguardedBasePointer());",name,i,i,i);
   }
   fprintf(arc->entity->src, "\n\t}");
 } 
 fprintf(arc->entity->src, "\n}");
}

  
// ****************************************************************************
// * nccArcLibGmpIni
// ****************************************************************************
void nccArcLibGmpIni(nablaMain *arc){
  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * gmpInit\
\n// ****************************************************************************\
\nvoid %s%s::gmpInit(void){\
\n\tconst int gmpNumberOfPrecise=%d;\
\n\tinfo()<<\"\\33[7m[gmpInit]\"<<\"\\33[m\";\
\n\t// None Lowest Low Medium High Highest\
\n\tTraceClassConfig tcCfg=TraceClassConfig(true,true,Trace::None);\
\n\t//TraceClassConfig tcCfg=TraceClassConfig(!true,true,Trace::None);\
\n\tsubDomain()->parallelMng()->traceMng()->setClassConfig(\"Master\",tcCfg);\
\n\tsubDomain()->parallelMng()->traceMng()->setClassConfig(\"ArcaneCheckpoint\",tcCfg);\
\n\tsubDomain()->parallelMng()->traceMng()->setClassConfig(\"Variable\",tcCfg);\
\n\tIVariableMng* vm = subDomain()->variableMng();\
\n\tm_gmp_observers.addObserver(this, &%s%s::gmpWritBfrDump,vm->writeObservable());\
\n\t// Resize en fonction du nombre de cells\
\n\tm_gmp.resize(gmpNumberOfPrecise);\
\n\tfor(int i=0;i<gmpNumberOfPrecise;i+=1){\
\n\t\tm_gmp[i].resize(ownCells().size());\
\n\t};\
\n\tENUMERATE_CELL(cell,ownCells()){\
\n\t\tfor(int i=0;i<gmpNumberOfPrecise;i+=1){\
\n\t\t\tmpz_init(m_gmp[i][cell->localId()].get_mpz_t());\
\n\t\t};\n\t}\n}",
          arc->name,nablaArcaneColor(arc),
          nablaVariableGmpRank(arc->variables),
          arc->name, nablaArcaneColor(arc));
  nablaJob *gmpInitFunction=nablaJobNew(arc->entity);
  gmpInitFunction->is_an_entry_point=true;
  gmpInitFunction->is_a_function=true;
  gmpInitFunction->scope  = strdup("NoGroup");
  gmpInitFunction->region = strdup("NoRegion");
  gmpInitFunction->item   = strdup("\0");
  gmpInitFunction->rtntp  = strdup("void");
  gmpInitFunction->name   = strdup("gmpInit");
  gmpInitFunction->name_utf8   = strdup("gmpInit");
  gmpInitFunction->xyz    = strdup("NoXYZ");
  gmpInitFunction->drctn  = strdup("NoDirection");
  sprintf(&gmpInitFunction->at[0],"-huge_valf");
  gmpInitFunction->whenx  = 1;
  gmpInitFunction->whens[0] = ENTRY_POINT_init;
  nablaJobAdd(arc->entity, gmpInitFunction);
  
  nccArcLibGmpWritBfrDump(arc);
  
  nccArcLibGmpReadFrmDump(arc);
  
  fprintf(arc->entity->src, "\
\nvoid %s%s::gmpContinueInit(void){\
\n\tinfo()<<\"\\33[7m[gmpContinueInit]\"<<\"\\33[m\";\
\n\tgmpReadFrmDump();\n}",
          arc->name,
          nablaArcaneColor(arc));
  nablaJob *gmpContinueInit=nablaJobNew(arc->entity);
  gmpContinueInit->is_an_entry_point=true;
  gmpContinueInit->is_a_function=true;
  gmpContinueInit->scope  = strdup("NoGroup");
  gmpContinueInit->region = strdup("NoRegion");
  gmpContinueInit->item   = strdup("\0");
  gmpContinueInit->rtntp  = strdup("void");
  gmpContinueInit->name   = strdup("gmpContinueInit");
  gmpContinueInit->name_utf8   = strdup("gmpContinueInit");
  gmpContinueInit->xyz    = strdup("NoXYZ");
  gmpContinueInit->drctn  = strdup("NoDirection");
  sprintf(&gmpContinueInit->at[0],"-0.0");
  gmpContinueInit->whenx  = 1;
  gmpContinueInit->whens[0] = ENTRY_POINT_continue_init;
  nablaJobAdd(arc->entity, gmpContinueInit);
}
