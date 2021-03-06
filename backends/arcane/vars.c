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
#include "nabla.h"
#include "nabla.tab.h"
#include "backends/arcane/arcane.h"


// ****************************************************************************
// * aHookVariablesODecl
// ****************************************************************************
char* aHookVariablesODecl(nablaMain *nabla){
  return "&";
}


// ****************************************************************************
// * Initialisation des besoins vis-à-vis des variables (globales)
// ****************************************************************************
void aHookVariablesInit(nablaMain *nabla){}


/***************************************************************************** 
 * DEFINES LIST ==> ARCANE
 *****************************************************************************/
//static nWhatWith noDefines[] ={ {NULL,NULL}};
static nWhatWith arcaneDefines[] ={
  {"m_global_greek_deltat","m_global_deltat"},
  {"reducemin(a)","0.0"},
  {"dot(a,b)","math::scaMul(a,b)"},
  {"opScaMul(a,b)","math::scaMul(a,b)"},
  {"opVecMul(a,b)","math::vecMul(a,b)"},
  {"cross(a,b)","math::vecMul(a,b)"},
  {"cross2D(a,b)","math::vecMul2D(a,b)"},
  {"opProdTens(a,b)","math::prodTens(a,b)"},
  {"opProdTensVec(a,b)","math::prodTensVec(a,b)"},
  {"opMatrixProduct(a,b)","math::matrixProduct(a,b)"},
  {"mixteMul(a,b,c)","math::mixteMul(a,b,c)"},
  {"matrix3x3Id","math::matrix3x3Id"},
  {"norm(v)","(v).abs()"},
  {"absolute(v)","math::abs(v)"},
  {"nbNodes","defaultMesh()->nbNode()"},
  {"File", "ofstream&"},
  {"file(name,ext)", "ofstream name(#name \".\" #ext)"},
  {"rabs(v)","math::abs(v)"},
  {"cube_root(v)","::cbrt(v)"},
  {"knAt(a)",""},
  {"mat","int"},
  {"matrixDeterminant(A)","math::matrixDeterminant(A)"},
  {"inverseMatrix(A,d)","math::inverseMatrix(A,d)"},
  {"xyz","eMeshDirection"},
  {"isZero(a)","math::isZero(a)"},
  {"square_root(a)","math::sqrt(a)"},
  {"ElapsedTime", "subDomain()->commonVariables().globalElapsedTime()"},
  {"globalCPUTime", "subDomain()->commonVariables().globalCPUTime()"},
  {"GlobalIteration", "subDomain()->commonVariables().globalIteration()"},
  {"synchronize(a)","a.synchronize()"},
  {"mpi_reduce(how,what)","subDomain()->parallelMng()->reduce(how,what)"},
  {"mpInteger","mpz_class"},
  {"restrict",""},
  {NULL,NULL}
};
nWhatWith arcaneOpCodesDefines[] ={
  {"opMul(a,b)","(a*b)"},
  {"opDiv(a,b)","(a/b)"},
  {"opMod(a,b)","(a%b)"},
  {"opAdd(a,b)","(a+b)"},
  {"opSub(a,b)","(a-b)"},
  {"opTernary(c,i,e)","((c)?(i):(e))"},
  {"log(a)","::log(a)"},
  {"exp(a)","::exp(a)"},
  {"min(a,b)","math::min(a,b)"},
  {"max(a,b)","math::max(a,b)"},
  {"adrs(a)","(a.unguardedBasePointer())"},
  {"ReduceMinToDouble(r)", "r"},
  {"ternaryEqOp(a,b,c,d)","(a==b)?c:d"},
  {NULL,NULL}
};


/***************************************************************************** 
 *
 *****************************************************************************/
NABLA_STATUS nccArcaneEntityIncludes(const nablaEntity *entity){
  FILE *target_file = isAnArcaneService(entity->main)?entity->src:entity->hdr;
  // Pas de ça pour un service
  if (isAnArcaneModule(entity->main))
    fprintf(target_file, "#ifndef %s_ENTITY_H\n#define %s_ENTITY_H\n\n",
            entity->name_upcase,
            entity->name_upcase);
  // Ni ça
  if (isAnArcaneAlone(entity->main))
    fprintf(target_file, "\
#include \"math.h\"\n\
#include \"assert.h\"\n\
#include \"stdint.h\"\n\
#include \"fenv.h\"\n\
#include \"fpu_control.h\"\n\
#include \"values.h\"\n");
  // Par contre, on veut bien cela dans le source d'un service
  fprintf(target_file, "\
#include <arcane/MathUtils.h>\n\
#include <arcane/utils/Real3x3.h>\n\
#include <arcane/utils/Real2x2.h>\n\
#include <arcane/utils/Real3.h>\n\
#include <arcane/hyoda/Hyoda.h>\n\
//#define ARCANE_HYODA_SOFTBREAK(dummy)\n\
#include <arcane/ArcaneTypes.h>\n\
#include <arcane/utils/ArcanePrecomp.h>\n\
#include <arcane/utils/ArcaneGlobal.h>\n\
#include <arcane/IItemFamily.h>\n\
#include <arcane/utils/String.h>\n\
#include <arcane/utils/StringBuilder.h>\n\
#include <arcane/utils/OStringStream.h>\n\
#include <arcane/utils/TraceAccessor.h>\n\
#include <arcane/utils/TraceClassConfig.h>\n\
#include <arcane/IMesh.h>\n\
#include <arcane/cartesian/ICartesianMesh.h>\n\
#include <arcane/ArcaneTypes.h>\n\
#include <arcane/IEntryPoint.h>\n\
#include <arcane/IEntryPointMng.h>\n\
#include <arcane/IParallelMng.h>\n\
");
  // Mais ça encore, ne nous intéresse pas dans le cas d'un service
  if (isAnArcaneModule(entity->main))
    fprintf(target_file, "\n\
%s\n\
%s\n\
%s\n\
%s\n\
%s\n\
%s\n\
%s\n\
%s\n\
%s\n\
#include \"%s%s%s_axl.h\"\n\n\
using namespace Arcane;\n\
//using namespace math;\n\
using namespace Parallel;\n\
\n\
/*inline bool opTernary(bool cond, bool t, bool f){if (cond) return t; return f;}\n\
inline Cell opTernary(bool cond, Cell ifStatement, Cell elseStatement){\n \
if (cond==true) return ifStatement;\n\
return elseStatement;\n\
}\n\
inline Node opTernary(bool cond, Node ifStatement, Node elseStatement){\n \
if (cond==true) return ifStatement;\n\
return elseStatement;\n\
}\n\
inline ItemUniqueId opTernary(bool cond, ItemUniqueId ifStatement, ItemUniqueId elseStatement){\n \
if (cond==true) return ifStatement;\n\
return elseStatement;\n\
}\n\
\n\
inline Real3 opTernary(bool cond, double ifStatement, const Real3 elseStatement){\n \
if (cond==true) return Real3(ifStatement,ifStatement,ifStatement);\n\
return elseStatement;\n\
}\n\
inline Real opTernary(bool cond, Real ifStatement, const Real elseStatement){\n\
if (cond) return (ifStatement);\n\
return elseStatement;\n\
}\n\
inline int opTernary(bool cond, int ifStatement, const int elseStatement){\n\
if (cond) return (ifStatement);\n\
return elseStatement;\n\
}\n\
inline double unglitch(double a){\n\
   const union{\n\
      unsigned long long i;\n\
      double d;\n\
   } mask = { 0xffffffffffff0000ull};\n\
   union{\n\
      unsigned long long i;\n\
      double d;\n\
   } data;\n\
  data.d= a;\n\
  data.i&= mask.i;\n\
  return data.d;\n \
}*/\n",
            ((entity->libraries&(1<<with_mathematica))!=0)?nccArcLibMathematicaHeader():"",
            ((entity->libraries&(1<<with_gmp))!=0)?nccArcLibDftHeader():"",
            ((entity->libraries&(1<<with_gmp))!=0)?nccArcLibGmpHeader():"",
            ((entity->libraries&(1<<with_mail))!=0)?nccArcLibMailHeader():"",
            ((entity->libraries&(1<<with_aleph))!=0)?
            isAnArcaneModule(entity->main)?nccArcLibAlephHeader():nccArcLibSchemeHeader():"",
            ((entity->libraries&(1<<with_cartesian))!=0)?nccArcLibCartesianHeader():"",
            ((entity->libraries&(1<<with_materials))!=0)?nccArcLibMaterialsHeader():"",
            ((entity->libraries&(1<<with_slurm))!=0)?nccArcLibSlurmHeader():"",
            ((entity->libraries&(1<<with_particles))!=0)?nccArcLibParticlesHeader():"",
            (isAnArcaneModule(entity->main)
             &&(entity->main->specific_path!=NULL))?entity->main->specific_path:"",
            entity->name,isAnArcaneModule(entity->main)?"":"Service");
  // Dans le cas d'un service, on ne souhaite que les opCodes en DEFINES
  if (isAnArcaneService(entity->main))
    return nMiddleDefines(entity->main, arcaneOpCodesDefines);
  // Sinon pour le module, on fait les opCodes et les autres defines
  nMiddleDefines(entity->main, arcaneOpCodesDefines);
  return nMiddleDefines(entity->main, arcaneDefines);
}


/***************************************************************************** 
 *
 *****************************************************************************/
static NABLA_STATUS nccArcaneEntityConstructor(const nablaEntity *entity){
  nablaVariable *var;
  // Dans le cas d'un service, pour l'instant on ne fait rien dans le header
  if (isAnArcaneService(entity->main)) return NABLA_OK;
  fprintf(entity->hdr, "\n\nclass %s%s : public Arcane%sObject{\npublic:\
\n\t%s%s(const %sBuildInfo& mbi):Arcane%sObject(mbi)%s%s%s%s%s",
          entity->name, nablaArcaneColor(entity->main), entity->name,
          entity->name, nablaArcaneColor(entity->main), nablaArcaneColor(entity->main),
          entity->name,
          isAnArcaneModule(entity->main)?"\n\t\t,m_physic_type_code(VariableBuildInfo(this,\"PhysicTypeCode\"))":"",
          ((entity->libraries&(1<<with_particles))!=0)?",m_particle_family(0)":"",
          ((entity->libraries&(1<<with_particles))!=0)?nccArcLibParticlesConstructor(entity):"",
          ((entity->libraries&(1<<with_dft))!=0)?"\
\n\t\t, m_dft_i(VariableBuildInfo(this,\"DFTi\"))\
\n\t\t, m_dft_sh0(VariableBuildInfo(this,\"DFTsh0\"))\
\n\t\t, m_dft_sh1(VariableBuildInfo(this,\"DFTsh1\"))\
\n\t\t, m_dft_sh2(VariableBuildInfo(this,\"DFTsh2\"))\
\n\t\t, m_dft_data(VariableBuildInfo(this,\"DFTdata\",IVariable::PPersistant))":"",
          ((entity->libraries&(1<<with_gmp))!=0)?"\
\n\t\t, m_gmp_backups(VariableBuildInfo(mbi.subDomain(),\"GmpMPZBackups\"))\
\n\t\t, m_gmp_iteration(0){\
\n\t\t//addEntryPoint(this,\"ContinueInit\",&naheaModule::gmpContinueInit,IEntryPoint::WContinueInit);":"{");
  for(var=entity->main->variables;var!=NULL;var=var->next){
    if (var->dim==0) continue;
    if (var->gmpRank!=-1) continue;
    fprintf(entity->hdr, "\n\t\tm_%s_%s.resize(%ld);", var->item, var->name, var->size);
  } 
  
  if ((entity->libraries&(1<<with_particles))!=0){
    fprintf(entity->hdr, "\n\t\tm_particle_family = mbi.mesh()->createItemFamily(IK_Particle,\"particles\");");
  }
//#warning WTF m_physic_type_code
  fprintf(entity->hdr,
          "\n\t}\n\t~%s%s(){%s}\npublic:%s",
          entity->name,
          nablaArcaneColor(entity->main),
          ((entity->libraries&(1<<with_mathematica))!=0)?nccArcLibMathematicaDelete():"",
          isAnArcaneModule(entity->main)?"\n\tVariableScalarInteger m_physic_type_code;":""
          );  
  return NABLA_OK;
}


// ****************************************************************************
// *
// ****************************************************************************
NABLA_STATUS nccArcaneEntityVirtuals(const nablaEntity *entity){
  nablaJob *job=entity->jobs;
  nablaMain *nabla=entity->main;
  // Dans le cas d'un service, pour l'instant on ne fait rien dans le header
  if (isAnArcaneService(nabla)) return NABLA_OK;
  
  for(;job!=NULL;job=job->next){
    if (job->is_a_function){
      // Si on est une Family et que la fonction a un '@', on ne fait rien!
      if (isAnArcaneFamily(nabla) && job->is_an_entry_point) continue;
      fprintf(entity->hdr, "\n\tvirtual ");
      nMiddleFunctionDumpHeader(entity->hdr, job->jobNode);
      fprintf(entity->hdr, ";");
      continue;
    }
    // Si on est une Family et que le job a un '@', on ne fait rien!
    if (isAnArcaneFamily(nabla) && job->is_an_entry_point) continue;
    // Cela se fait aussi lors de la génération de l'axl afin de pouvoir traiter les @ -4,4
    dbg("\n\t[nccArcaneEntityVirtuals] virtuals for job %s", job->name);
    // On remplit la ligne du fichier HDR
    fprintf(entity->hdr, "\n\tvirtual void %s(", job->name);
    if (isAnArcaneFamily(nabla)){
      int i=0;
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next,i+=1)
        fprintf(entity->hdr, "%s%s%s",i==0?"":",",
                arcaneHookDfsArgType(nabla,var),
                aHookVariablesODecl(nabla));
    }else
      nMiddleDumpParameterTypeList(entity->main,entity->hdr, job->stdParamsNode);
    fprintf(entity->hdr, ");");
  }
  return NABLA_OK;
}


// ****************************************************************************
// * Dump des variables dans le header
// * Utile pour les variables static, par exemple
// ****************************************************************************
void aHookVariablesPrefix(nablaMain *nabla){
  if (isAnArcaneFamily(nabla)){
    aHookFamilyVariablesPrefix(nabla);
    nccArcaneEntityVirtuals(nabla->entity);
    return;
  }

  // Dans le cas d'un module, on le fait maintenant
  if (isAnArcaneModule(nabla)){
    dbg("\n[nccArcane] nccArcaneEntityIncludes initialization");
    if (nccArcaneEntityIncludes(nabla->entity)!=NABLA_OK)
      printf("error: in Includes generation!\n");
  }
  
  dbg("\n[nccArcane] nccArcaneEntityConstructor initialization");
  if (nccArcaneEntityConstructor(nabla->entity)!=NABLA_OK)
    printf("error: in EntityConstructor generation!\n");

  dbg("\n[nccArcane] nccArcaneEntityVirtuals initialization");
  if (nccArcaneEntityVirtuals(nabla->entity)!=NABLA_OK)
    printf("error: in EntityVirtuals generation!\n");
}


// ****************************************************************************
// * Malloc des variables
// ****************************************************************************
void aHookVariablesMalloc(nablaMain *nabla){
}


// ****************************************************************************
// * Variables Postfix
// ****************************************************************************
void aHookVariablesFree(nablaMain *nabla){
}
