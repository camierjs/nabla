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



/******************************************************************************
 * tr \" to spaces
 ******************************************************************************/
static char *strKillQuote(const char * str){
  char *p=strdup(str);
  char *bkp=p;
  //printf("\n[strKillQuote] str=%s", str);
  for(;*p!=0;p++)
    if (*p==34) *p=32;
  return bkp;
} 


/*****************************************************************************
 * astTreeSaveNodes
 *****************************************************************************/
static unsigned int astTreeSaveNodes(FILE *fTreeOutput, astNode *l, unsigned int id){
  for(;l not_eq NULL;l=l->next){
    if (l->rule not_eq NULL)
      fprintf(fTreeOutput,
              "\n\tnode_%d [label=\"%s\" color=\"#CCDDCC\"]",
              l->id=id++,
              strKillQuote(l->rule));
    else if (l->token not_eq NULL)
      fprintf(fTreeOutput,
              "\n\tnode_%d [label=\"%s\" color=\"#CCCCDD\"]",
              l->id=id++,
              strKillQuote(l->token));
    else /* */ ;
    id=astTreeSaveNodes(fTreeOutput, l->children, id);
  }
  return id;
}


/*****************************************************************************
 * astTreeSaveEdges
 *****************************************************************************/
static NABLA_STATUS astTreeSaveEdges(FILE *fTreeOutput, astNode *l, astNode *father){
  for(;l not_eq NULL;l=l->next){
    fprintf(fTreeOutput, "\n\tnode_%d -> node_%d;", father->id, l->id);
    astTreeSaveEdges(fTreeOutput, l->children, l);
  }
  return NABLA_OK;
}


/*****************************************************************************
 * astTreeSave
 *****************************************************************************/
NABLA_STATUS astTreeSave(const char *nabla_entity_name, astNode *root){
  FILE *fDot;
  char fName[NABLA_MAX_FILE_NAME];
  sprintf(fName, "%s.dot", nabla_entity_name);
  // Saving tree file
  if ((fDot=fopen(fName, "w")) == 0) return NABLA_ERROR|dbg("[nccTreeSave] fopen ERROR");
  fprintf(fDot, "digraph {\nordering=out;\n\tnode [style = filled, shape = circle];");
  astTreeSaveNodes(fDot, root, 0);
  if (astTreeSaveEdges(fDot, root->children, root) not_eq NABLA_OK)
    return NABLA_ERROR|dbg("[nccTreeSave] ERROR");
  fprintf(fDot, "\n}\n");
  fclose(fDot);
  return NABLA_OK;
}		




/*****************************************************************************
 * getInOutPutsNodes
 *****************************************************************************/
void getInOutPutsNodes(FILE *fOut, astNode *n, char *color){
  if (fOut==NULL) return;
  if (n->token not_eq NULL){
    if (n->tokenid == CELL){
      fprintf(fOut, "\n\tcell_");color="CCDDCC";
    }else if (n->tokenid == NODE){
      fprintf(fOut, "\n\tnode_");color="DDCCCC";
    }else if (n->tokenid == FACE){
      fprintf(fOut, "\n\tface_");color="CCCCDD";
    }else{
      fprintf(fOut, "%s[shape = circle, color=\"#%s\"];", n->token, color);
      dbg("\n[getInOutPutsNodes] %s", n->token);
    }
  }
  if(n->children != NULL) getInOutPutsNodes(fOut, n->children, color);
  if(n->next != NULL) getInOutPutsNodes(fOut, n->next, color);
}


/*****************************************************************************
 * getInOutPutsEdges
 *****************************************************************************/
void getInOutPutsEdges(FILE *fOut, astNode *n, int inout, char *nName1, char* nName2){
  if (fOut==NULL) return;
  if (n->token != NULL){
    if (n->tokenid == CELL){
      if (inout == OUT) fprintf(fOut, "\n\t%sJob_%s -> c", nName1, nName2);
      else fprintf(fOut, "\n\tc");
    }else if (n->tokenid == NODE){
      if (inout == OUT) fprintf(fOut, "\n\t%sJob_%s -> n", nName1, nName2);
      else fprintf(fOut, "\n\tn");
    }else if (n->tokenid == FACE){
      if (inout == OUT) fprintf(fOut, "\n\t%sJob_%s -> f", nName1, nName2);
      else fprintf(fOut, "\n\tn");
    }else{
      if (inout == IN) fprintf(fOut, "%s -> %sJob_%s;", n->token, nName1, nName2);
      else fprintf(fOut, "%s", n->token);
    }
  }
  if(n->children != NULL) getInOutPutsEdges(fOut, n->children, inout, nName1, nName2);
  if(n->next != NULL) getInOutPutsEdges(fOut, n->next, inout, nName1, nName2);
}
