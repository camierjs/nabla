/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM                                                  *
 *****************************************************************************
 * File     : nccAst.c        										    				  *
 * Author   : Camier Jean-Sylvain														  *
 * Created  : 2014.10.22																	  *
 *****************************************************************************
 * Description: 																				  *
 *****************************************************************************
 * Date			Author	Description														  *
 * 2014.10.22	camierjs	Creation															  *
 *****************************************************************************/
#include "nabla.h"
#include "nabla.tab.h"

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

extern const char *yytname[];
extern const yytype_uint16 yyr1[];
//extern const YYTYPE_UINT16 yyr1[];

/*void Z1(int yyn, int token, astNode *lhs, astNode *n1){
  printf("\nyyn=%d, YYTRANSLATE(yyn)=%d",yyn,yyTranslate(yyn));
  //lhs=astNewNodeRule(yytname[yyr1[yyn]],yyr1[yyn]);
  //astAddChild((lhs),(n1));
}
*/
