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
#include "nabla.h"
#include "nabla.tab.h"

// ****************************************************************************
// * Upcase de la chaîne donnée en argument
// ****************************************************************************
static inline char *itemUPCASE(const char *itm){
  if (itm[0]=='c') return "CELLS";
  if (itm[0]=='n') return "NODES";
  if (itm[0]=='g') return "GLOBAL";
  dbg("\n\t[itemUPCASE] itm=%s", itm);
  exit(NABLA_ERROR|fprintf(stderr, "\n[itemUPCASE] Error with given item\n"));
  return NULL;
}


// ****************************************************************************
// * enums pour les différents dumps à faire: déclaration, malloc et free
// ****************************************************************************
typedef enum {
  OKINA_VARIABLES_DECLARATION=0,
  OKINA_VARIABLES_MALLOC,
  OKINA_VARIABLES_FREE
} OKINA_VARIABLES_SWITCH;


// ****************************************************************************
// * Pointeur de fonction vers une qui dump ce que l'on souhaite
// ****************************************************************************
typedef NABLA_STATUS (*pFunDump)(nablaMain *nabla, nablaVariable *var, char *postfix, char *depth);


// ****************************************************************************
// * Dump d'un MALLOC d'une variables dans le fichier source
// ****************************************************************************
static NABLA_STATUS okinaGenerateSingleVariableMalloc(nablaMain *nabla,
                                                    nablaVariable *var,
                                                    char *postfix,
                                                    char *depth){
  nprintf(nabla,"\n\t// okinaGenerateSingleVariableMalloc",NULL);
  return NABLA_OK;
}


// ****************************************************************************
// * Dump d'un FREE d'une variables dans le fichier source
// ****************************************************************************
static NABLA_STATUS okinaGenerateSingleVariableFree(nablaMain *nabla,
                                                  nablaVariable *var,
                                                  char *postfix,
                                                  char *depth){  
  nprintf(nabla,"\n\t// okinaGenerateSingleVariableFree",NULL);
  return NABLA_OK;
}


// ****************************************************************************
// * Dump d'une variables dans le fichier
// ****************************************************************************
static NABLA_STATUS okinaGenerateSingleVariable(nablaMain *nabla,
                                              nablaVariable *var,
                                              char *postfix,
                                              char *depth){  
  nprintf(nabla,"\n\t// okinaGenerateSingleVariable",NULL);
  if (strncmp(var->name,"coord",5)==0){
    if ((nabla->entity->libraries&(1<<with_real))!=0){
      fprintf(nabla->entity->hdr,
              "\nreal/*3*/ node_coord[NABLA_NB_NODES_WARP] __attribute__ ((aligned(WARP_ALIGN)));");
      return NABLA_OK;
    }
  }
  if (var->dim==0)
    fprintf(nabla->entity->hdr,
            "\n%s %s_%s%s%s[NABLA_NB_%s_WARP] __attribute__ ((aligned(WARP_ALIGN)));",
            postfix?"real":var->type,
            var->item,
            var->name,
            postfix?postfix:"",
            depth?depth:"",
            itemUPCASE(var->item));
  if (var->dim==1)
    fprintf(nabla->entity->hdr,"\n%s %s_%s%s[%ld*NABLA_NB_%s_WARP] __attribute__ ((aligned(WARP_ALIGN)));;",
            postfix?"real":var->type,
            var->item,
            var->name,
            postfix?postfix:"",
            var->size,
            itemUPCASE(var->item));
  return NABLA_OK;
}


// ****************************************************************************
// * Retourne quelle fonction selon l'enum donné
// ****************************************************************************
static pFunDump witch2func(OKINA_VARIABLES_SWITCH witch){
  switch (witch){
  case (OKINA_VARIABLES_DECLARATION): return okinaGenerateSingleVariable;
  case (OKINA_VARIABLES_MALLOC): return okinaGenerateSingleVariableMalloc;
  case (OKINA_VARIABLES_FREE): return okinaGenerateSingleVariableFree;
  default: exit(NABLA_ERROR|fprintf(stderr, "\n[witch2switch] Error with witch\n"));
  }
}


// ****************************************************************************
// * Dump d'une variables de dimension 1
// ****************************************************************************
static NABLA_STATUS okinaGenericVariableDim1(nablaMain *nabla, nablaVariable *var, pFunDump fDump){
  //int i;
  //char depth[]="[0]";
  dbg("\n[okinaGenerateVariableDim1] variable %s", var->name);
  //for(i=0;i<NABLA_HARDCODED_VARIABLE_DIM_1_DEPTH;++i,depth[1]+=1) fDump(nabla, var, NULL, depth);
  fDump(nabla, var, NULL, "/*8*/");
  return NABLA_OK;
}


// ****************************************************************************
// * Dump d'une variables de dimension 0
// ****************************************************************************
static NABLA_STATUS okinaGenericVariableDim0(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  dbg("\n[okinaGenerateVariableDim0] variable %s", var->name);
  if (strcmp(var->type,"real3")!=0)
    return fDump(nabla, var, NULL, NULL);
  else
    return fDump(nabla, var, NULL, NULL);
  /*return fDump(nabla, var, "_x", NULL)|
           fDump(nabla, var, "_y", NULL)|
           fDump(nabla, var, "_z", NULL);*/
  return NABLA_ERROR;
}


// ****************************************************************************
// * Dump d'une variables
// ****************************************************************************
static NABLA_STATUS okinaGenericVariable(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  if (!var->axl_it) return NABLA_OK;
  if (var->item==NULL) return NABLA_ERROR;
  if (var->name==NULL) return NABLA_ERROR;
  if (var->type==NULL) return NABLA_ERROR;
  if (var->dim==0) return okinaGenericVariableDim0(nabla,var,fDump);
  if (var->dim==1) return okinaGenericVariableDim1(nabla,var,fDump);
  dbg("\n[okinaGenericVariable] variable dim error: %d", var->dim);
  exit(NABLA_ERROR|fprintf(stderr, "\n[okinaGenericVariable] Error with given variable\n"));
}


// ****************************************************************************
// * Dump des options
// ****************************************************************************
static void okinaOptions(nablaMain *nabla){
  nablaOption *opt;
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * Options\n\
// ********************************************************");
  for(opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->hdr,
            "\n#define %s %s",
            opt->name, opt->dflt);
}


// ****************************************************************************
// * Dump des globals
// ****************************************************************************
static void okinaGlobals(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\n\
// ********************************************************\n\
// * Temps de la simulation\n\
// ********************************************************\n\
Real global_deltat[1];\n\
int global_iteration;\n\
double global_time;\n");
}


// ****************************************************************************
// * Dump des variables
// ****************************************************************************
void okinaVariablesPrefix(nablaMain *nabla){
  nablaVariable *var;

  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Variables\n\
// ********************************************************");
  for(var=nabla->variables;var!=NULL;var=var->next){
    if (okinaGenericVariable(nabla, var, witch2func(OKINA_VARIABLES_DECLARATION))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[okinaVariables] Error with variable %s\n", var->name));
    if (okinaGenericVariable(nabla, var, witch2func(OKINA_VARIABLES_MALLOC))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[okinaVariables] Error with variable %s\n", var->name));
  }
  okinaOptions(nabla);
  okinaGlobals(nabla);
}


// ****************************************************************************
// * okinaVariablesPostfix
// ****************************************************************************
void okinaVariablesPostfix(nablaMain *nabla){
  nablaVariable *var;
  for(var=nabla->variables;var!=NULL;var=var->next)
    if (okinaGenericVariable(nabla, var, witch2func(OKINA_VARIABLES_FREE))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[okinaVariables] Error with variable %s\n", var->name));
}



