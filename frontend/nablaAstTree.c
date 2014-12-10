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
  if (astTreeSaveEdges(fDot, root->children, root) not_eq NABLA_OK) return NABLA_ERROR|dbg("[nccTreeSave] ERROR");
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
