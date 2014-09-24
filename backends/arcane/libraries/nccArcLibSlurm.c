/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArcLibSlurm.c        													  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2013.04.04																	  *
 * Updated  : 2013.04.04																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2013.04.04	camierjs	Creation															  *
 *****************************************************************************/
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
