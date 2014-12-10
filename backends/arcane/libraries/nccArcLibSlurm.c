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
