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
#include "nabla.h"


/***************************************************************************** 
 * Préparation du header du module
 *****************************************************************************/
char* nccArcLibSlurmHeader(void){
  return "\n#include </usr/local/sr/include/s8job.h>\n";
}


// ****************************************************************************
// * nccArcLibSlurmPrivates
// ****************************************************************************
char* nccArcLibSlurmPrivates(void){
  return   "\nprivate:\t//Slurm stuff\n\
\tInteger slurmTremain(void);\n\
\tint slurmTlimit(void);\n\
\tint m_tlimit;\n\
\tInteger m_tremain;";
}


// ****************************************************************************
// * nccArcLibSlurmTremain
// ****************************************************************************
void nccArcLibSlurmTremain(nablaMain *arc){
  fprintf(arc->entity->src, "\
\nInteger %s%s::slurmTremain(void){\
\n\tInteger values_to_sync[1];\
\n\tIntegerArrayView broadcasted(1,values_to_sync);\
\n\tif (parallelMng()->commRank()==0){\
\n\t\tdouble trmn;\
\n\t\t::tremain(&trmn);\
\n\t\tvalues_to_sync[0]=(Integer)trmn;\
\n\t}\
\n\tparallelMng()->broadcast(broadcasted,0);\
\n\tif (((100*(m_tremain-broadcasted[0]))/(1+m_tremain)) > 50)\
\n\t\treturn m_tremain;\
\n\tm_tremain=broadcasted[0];\
\n\t//info()<<\"globalCPUTime=\"<<globalCPUTime;\
\n\tif (globalCPUTime<4)\
\n\t\tm_tremain=slurmTlimit()+1;\
\n\treturn m_tremain;\
\n}",arc->name,nablaArcaneColor(arc));
}


// ****************************************************************************
// * nccArcLibSlurmLimit
// ****************************************************************************
void nccArcLibSlurmTlimit(nablaMain *arc){
  fprintf(arc->entity->src, "\
\nint %s%s::slurmTlimit(void){\
\n\treturn ::s8_tlimi(&m_tlimit);\
\n}",arc->name,nablaArcaneColor(arc));
}


// ****************************************************************************
// * nccArcLibSlurmIni
// ****************************************************************************
void nccArcLibSlurmIni(nablaMain *arc){
  nccArcLibSlurmTremain(arc);
  nccArcLibSlurmTlimit(arc);
}
