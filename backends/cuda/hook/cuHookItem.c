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
#include "nabla.tab.h"


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* cuHookSysPrefix(void){ return ""; }
char* cuHookSysPostfix(void){ return ""; }

char* cuHookPrevCell(int direction){
  return "gatherk_and_zero_neg_ones(cell_prev[direction*NABLA_NB_CELLS+tcid],";
}


// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* cuHookNextCell(int direction){
  return "gatherk_and_zero_neg_ones(cell_next[direction*NABLA_NB_CELLS+tcid],";
}


/***************************************************************************** 
 * Traitement des tokens NABLA ITEMS
 *****************************************************************************/
char* cuHookItem(nablaJob* job, const char j, const char itm, char enum_enum){
  if (j=='c' && enum_enum=='\0' && itm=='c') return "/*chi-c0c*/c";
  if (j=='c' && enum_enum=='\0' && itm=='n') return "/*chi-c0n*/c->";
  if (j=='c' && enum_enum=='f'  && itm=='n') return "/*chi-cfn*/f->";
  if (j=='c' && enum_enum=='f'  && itm=='c') return "/*chi-cfc*/f->";
  if (j=='n' && enum_enum=='f'  && itm=='n') return "/*chi-nfn*/f->";
  if (j=='n' && enum_enum=='f'  && itm=='c') return "/*chi-nfc*/f->";
  if (j=='n' && enum_enum=='\0' && itm=='n') return "/*chi-n0n*/n";
  if (j=='f' && enum_enum=='\0' && itm=='f') return "/*chi-f0f*/f";
  if (j=='f' && enum_enum=='\0' && itm=='n') return "/*chi-f0n*/f->";
  if (j=='f' && enum_enum=='\0' && itm=='c') return "/*chi-f0c*/xs_face_";
  nablaError("Could not switch in cuHookItem!");
  return NULL;
}


/***************************************************************************** 
 * Traitement des transformations '[', '(' & ''
 *****************************************************************************/
void cuHookTurnBracketsToParentheses(nablaMain* nabla, nablaJob *job, nablaVariable *var, char cnfg){
  dbg("\n\t[actJobItemParse] primaryExpression hits Cuda variable");
  if (  (cnfg=='c' && var->item[0]=='n')
      ||(cnfg=='c' && var->item[0]=='f')
      ||(cnfg=='n' && var->item[0]!='n')            
      ||(cnfg=='f' && var->item[0]!='f')
      ||(cnfg=='e' && var->item[0]!='e')
      ||(cnfg=='m' && var->item[0]!='m')
      ){
    //nprintf(nabla, "/*turnBracketsToParentheses@true*/", "/*%c %c*/", cnfg, var->item[0]);
    job->parse.turnBracketsToParentheses=true;
  }else{
    if (job->parse.postfix_constant==true
        && job->parse.variableIsArray==true) return;
    if (job->parse.isDotXYZ==1) nprintf(nabla, "/*cuHookTurnBracketsToParentheses_X*/", NULL);
    if (job->parse.isDotXYZ==2) nprintf(nabla, "/*cuHookTurnBracketsToParentheses_Y*/", NULL);
    if (job->parse.isDotXYZ==3) nprintf(nabla, "/*cuHookTurnBracketsToParentheses_Z*/", NULL);
    job->parse.isDotXYZ=0;
    job->parse.turnBracketsToParentheses=false;
  }
}


/***************************************************************************** 
 * Traitement des tokens SYSTEM
 *****************************************************************************/
void cuHookSystem(astNode * n,nablaMain *arc, const char cnf, char enum_enum){
  char *itm=(cnf=='c')?"cell":(cnf=='n')?"node":"face";
  //char *etm=(enum_enum=='c')?"c":(enum_enum=='n')?"n":"f";
  if (n->tokenid == LID)           nprintf(arc, "/*chs*/", "[%s->localId()]",itm);//asInteger
  if (n->tokenid == SID)           nprintf(arc, "/*chs*/", "[subDomain()->subDomainId()]");
  if (n->tokenid == THIS)          nprintf(arc, "/*chs THIS*/", NULL);
  if (n->tokenid == NBNODE)        nprintf(arc, "/*chs NBNODE*/", NULL);
  if (n->tokenid == NBCELL)        nprintf(arc, "/*chs NBCELL*/", NULL);
  //if (n->tokenid == INODE)         nprintf(arc, "/*chs INODE*/", NULL);
  if (n->tokenid == BOUNDARY_CELL) nprintf(arc, "/*chs BOUNDARY_CELL*/", NULL);
  if (n->tokenid == FATAL)         nprintf(arc, "/*chs*/", "throw FatalErrorException]");
  
  if (n->tokenid == BACKCELL)      nprintf(arc, "/*chs*/", "[face_cell[tfid+NABLA_NB_FACES*0]]");
  if (n->tokenid == BACKCELLUID)   nprintf(arc, "/*chs*/", "[face_cell[tfid+NABLA_NB_FACES*0]]");
  if (n->tokenid == FRONTCELL)     nprintf(arc, "/*chs*/", "[face_cell[tfid+NABLA_NB_FACES*1]]");
  if (n->tokenid == FRONTCELLUID)  nprintf(arc, "/*chs*/", "[face_cell[tfid+NABLA_NB_FACES*1]]");
  
  if (n->tokenid == NEXTCELL)      nprintf(arc, "/*chs NEXTCELL*/", ")");
  if (n->tokenid == PREVCELL)      nprintf(arc, "/*chs PREVCELL*/", ")");
  if (n->tokenid == NEXTNODE)      nprintf(arc, "/*chs NEXTNODE*/", "[nextNode]");
  if (n->tokenid == PREVNODE)      nprintf(arc, "/*chs PREVNODE*/", "[prevNode]");
  if (n->tokenid == PREVLEFT)      nprintf(arc, "/*chs PREVLEFT*/", "[cn.previousLeft()]");
  if (n->tokenid == PREVRIGHT)     nprintf(arc, "/*chs PREVRIGHT*/", "[cn.previousRight()]");
  if (n->tokenid == NEXTLEFT)      nprintf(arc, "/*chs NEXTLEFT*/", "[cn.nextLeft()]");
  if (n->tokenid == NEXTRIGHT)     nprintf(arc, "/*chs NEXTRIGHT*/", "[cn.nextRight()]");
  //error(!0,0,"Could not switch Cuda Hook System!");
}
