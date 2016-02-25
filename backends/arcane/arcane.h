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
#ifndef _NABLA_ARCANE_H_
#define _NABLA_ARCANE_H_

NABLA_STATUS aHookMainPrefix(nablaMain*);
NABLA_STATUS aHookMainPreInit(nablaMain*);
NABLA_STATUS aHookMainVarInitKernel(nablaMain*);
NABLA_STATUS aHookMainVarInitCall(nablaMain*);
NABLA_STATUS aHookMainHLT(nablaMain*);
NABLA_STATUS aHookMainPostInit(nablaMain*);
NABLA_STATUS aHookMainPostfix(nablaMain*);

void aHookVariablesInit(nablaMain*);
void aHookVariablesPrefix(nablaMain*);
void aHookVariablesMalloc(nablaMain*);
void aHookVariablesFree(nablaMain*);

void aHookMeshPrefix(nablaMain*);
void aHookMeshCore(nablaMain*);
void aHookMeshPostfix(nablaMain*);

void aHookSourceOpen(nablaMain*);
void aHookSourceInclude(nablaMain*);
char* aHookSourceNamespace(nablaMain*);

bool aHookPrimaryExpressionToReturn(nablaMain*,nablaJob*,astNode*);

void aHookAddExtraParametersDFS(nablaMain*,nablaJob*,int*);
void aHookDumpNablaParameterListDFS(nablaMain*,nablaJob*,astNode*,int*);
void aHookDfsForCalls(nablaMain*,nablaJob*,astNode*,const char*,astNode*);

void aHookHeaderDump(nablaMain*);
void aHookHeaderOpen(nablaMain*);
void aHookHeaderEnums(nablaMain*);
void aHookHeaderPrefix(nablaMain*);
void aHookHeaderIncludes(nablaMain*);
void aHookHeaderPostfix(nablaMain*);

char *nArcanePragmaGccIvdep(void);
char *nArcanePragmaGccAlign(void);
char* arcaneEntryPointPrefix(nablaMain*,nablaJob*);
void arcaneAddArguments(nablaMain*,nablaJob*);
void arcaneAddCallNames(nablaMain*,nablaJob*,astNode*);
void arcaneFatal(nablaMain*);
void arcaneTime(nablaMain*);
void arcaneExit(nablaMain*,nablaJob*);
void arcaneIteration(nablaMain*);
void arcaneTurnTokenToOption(nablaMain*,nablaOption*);
void arcaneHookReduction(nablaMain*,astNode*);

bool arcaneHookDfsVariable(void);

//char *nccArcBits(void);
//char* nccArcGather(nablaJob*,nablaVariable* var, GATHER_SCATTER_PHASE);
//char* nccArcScatter(nablaVariable* var);

char* nccArcSystemPrefix(void);
//char* nccArcSystemPostfix(void);
//char* nccArcPrevCell(int);
//char* nccArcNextCell(int);
//char* nccArcIncludes(void);

char *nablaArcaneColor(nablaMain*);
bool isAnArcaneAlone(nablaMain*);
bool isAnArcaneModule(nablaMain*);
bool isAnArcaneService(nablaMain*);

void actFunctionDumpHdr(FILE*, astNode*);

char* nccArcLibMailHeader(void);
char* nccArcLibMailPrivates(void);
void nccArcLibMailIni(nablaMain*);
char *nccArcLibMailDelete(void);

void nccArcLibSchemeIni(nablaMain*);
char* nccArcLibSchemeHeader(void);
char* nccArcLibSchemePrivates(void);

void nccArcLibAlephIni(nablaMain*);
char* nccArcLibAlephHeader(void);
char* nccArcLibAlephPrivates(void);

void nccArcLibCartesianIni(nablaMain*);
char* nccArcLibCartesianHeader(void);
char* nccArcLibCartesianPrivates(void);

void nccArcLibMaterialsIni(nablaMain*);
char* nccArcLibMaterialsHeader(void);
char* nccArcLibMaterialsPrivates(void);

char* nccArcLibMathematicaHeader(void);
char* nccArcLibMathematicaPrivates(void);
void nccArcLibMathematicaIni(nablaMain*);
char *nccArcLibMathematicaDelete(void);

char* nccArcLibDftHeader(void);
char* nccArcLibDftPrivates(void);
void nccArcLibDftIni(nablaMain*);

char* nccArcLibGmpHeader(void);
char* nccArcLibGmpPrivates(void);
void nccArcLibGmpIni(nablaMain*);

char* nccArcLibSlurmHeader(void);
char* nccArcLibSlurmPrivates(void);
void nccArcLibSlurmIni(nablaMain*);

char* nccArcLibParticlesHeader(void);
char* nccArcLibParticlesPrivates(const nablaEntity*);
void nccArcLibParticlesIni(nablaMain*);
char *nccArcLibParticlesDelete(void);
char* nccArcLibParticlesConstructor(const nablaEntity*);


NABLA_STATUS nccArcMain(nablaMain*);
//NABLA_STATUS nccArcConfigHeader(nablaMain*);
//NABLA_STATUS nccArcConfigFooter(nablaMain*);

void nArcaneHLTInit(nablaMain*);
char* nccAxlGeneratorEntryPointWhenName(double when);
NABLA_STATUS nccAxlGenerateHeader(nablaMain*);

NABLA_STATUS nccAxlGenerator(nablaMain*);
NABLA_STATUS nccHdrEntityGeneratorInclude(const nablaEntity*);
NABLA_STATUS nccHdrEntityGeneratorConstructor(const nablaEntity*);
NABLA_STATUS nccHdrEntityGeneratorPrivates(const nablaEntity*);

//NABLA_STATUS nccArcaneEntityHeader(nablaMain*);
//NABLA_STATUS nccArcaneBeginNamespace(nablaMain *);
NABLA_STATUS nccArcaneEntityIncludes(const nablaEntity*);
//NABLA_STATUS nccArcaneEntityConstructor(const nablaEntity*);
NABLA_STATUS nccArcaneEntityVirtuals(const nablaEntity*);
NABLA_STATUS nccArcaneEntityGeneratorPrivates(const nablaEntity*);

char* nccArcLibAlephHeader(void);
void nccArcLibAlephIni(nablaMain*);

// Main Entry Backend
NABLA_STATUS arcaneOld(nablaMain*,astNode*, const char*);
hooks* arcane(nablaMain*);

// Hooks 
void arcaneJob(nablaMain*, astNode*);
void arcaneHookFunctionName(nablaMain*);
void arcaneHookFunction(nablaMain*, astNode*);
void arcaneItemDeclaration(astNode*,int,nablaMain*);
void arcaneOptionsDeclaration(astNode*, int, nablaMain*);
char* arcaneHookPrefixEnumerate(nablaJob*);
char* arcaneHookDumpEnumerateXYZ(nablaJob*);
char* arcaneHookDumpEnumerate(nablaJob*);
char* arcaneHookPostfixEnumerate(nablaJob*);
char* arcaneHookItem(nablaJob*,const char, const char, char);
void arcaneHookSwitchToken(astNode*, nablaJob*);
nablaVariable *arcaneHookTurnTokenToVariable(astNode*,nablaMain*,nablaJob*);
void arcaneHookTurnBracketsToParentheses(nablaMain*, nablaJob*, nablaVariable*, char);
void arcaneHookSystem(astNode*,nablaMain*, const char, char);
char* arcaneHookTokenPrefix(nablaMain*);
char* arcaneHookTokenPostfix(nablaMain*);
void arcaneHookIsTest(nablaMain*,nablaJob*,astNode*,int);

// Transformations
char *cellJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *cellJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *cellJobFaceVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *cellJobParticleVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *cellJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);

char *nodeJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *nodeJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *nodeJobFaceVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *nodeJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);

char *faceJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *faceJobNodeVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *faceJobFaceVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *faceJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);

char *particleJobParticleVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *particleJobCellVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);
char *particleJobGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);

char *functionGlobalVar(const nablaMain *arc, const nablaJob *job,  const nablaVariable *var);

void nArcaneHLTEntryPoint(nablaMain *arc,
                          nablaJob *entry_point,
                          int number_of_entry_points,double*);
#endif // _NABLA_ARCANE_H_
