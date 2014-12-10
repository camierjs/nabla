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
#ifndef _NABLA_ARCANE_H_
#define _NABLA_ARCANE_H_

char *nccArcBits(void);
char* nccArcGather(nablaJob*,nablaVariable* var, enum_phase phase);
char* nccArcScatter(nablaVariable* var);
char* nccArcPrevCell(void);
char* nccArcNextCell(void);
char* nccArcIncludes(void);

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
char* nccArcLibParticlesPrivates(nablaEntity*);
void nccArcLibParticlesIni(nablaMain*);
char *nccArcLibParticlesDelete(void);
char* nccArcLibParticlesConstructor(nablaEntity*);


NABLA_STATUS nccArcMain(nablaMain*);
NABLA_STATUS nccArcConfigHeader(nablaMain*);
NABLA_STATUS nccArcConfigFooter(nablaMain*);

NABLA_STATUS nccAxlGenerateHeader(nablaMain*);

NABLA_STATUS nccAxlGenerator(nablaMain*);
NABLA_STATUS nccHdrEntityGeneratorInclude(nablaEntity*);
NABLA_STATUS nccHdrEntityGeneratorConstructor(nablaEntity*);
NABLA_STATUS nccHdrEntityGeneratorPrivates(nablaEntity*);

NABLA_STATUS nccArcaneEntityHeader(nablaMain*);
NABLA_STATUS nccArcaneBeginNamespace(nablaMain *);
NABLA_STATUS nccArcaneEntityIncludes(nablaEntity*);
NABLA_STATUS nccArcaneEntityConstructor(nablaEntity*);
NABLA_STATUS nccArcaneEntityVirtuals(nablaEntity*);
NABLA_STATUS nccArcaneEntityGeneratorPrivates(nablaEntity*);

char* nccArcLibAlephHeader(void);
void nccArcLibAlephIni(nablaMain*);

// Main Entry Backend
NABLA_STATUS nccArcane(nablaMain*,astNode*, const char*);

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

#endif // _NABLA_ARCANE_H_
