/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccTools.c         														  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2012.11.13																	  *
 * Updated  : 2012.11.13																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2012.11.13	camierjs	Creation															  *
 *****************************************************************************/
#ifndef _NCC_TOOLS_H_
#define _NCC_TOOLS_H_

int nprintf(const struct nablaMainStruct *nabla, const char *debug, const char *format, ...);

int hprintf(const struct nablaMainStruct *nabla, const char *debug, const char *format, ...);

char *toolStrDownCase(const char *);

char *toolStrUpCase(const char *);

char *op2name(char *op);

void nUtf8(char**);

int nablaMakeTempFile(const char *, char *);

#endif // _NCC_TOOLS_H_
