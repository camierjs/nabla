/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccArc.h      																  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 18.01.2010																	  *
 * Updated  : 18.01.2010																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 18.01.2010	jscamier	Creation															  *
 *****************************************************************************/
#ifndef _NCC_ARC_H_
#define _NCC_ARC_H_


char *nccArcBits(void);
char* nccArcGather(nablaVariable* var, enum_phase phase);
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
NABLA_STATUS nccArcane(nablaMain*,astNode*, const char*, const char*);

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
#endif // _NABLA_ARCANE_H_
