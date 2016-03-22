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

// ****************************************************************************
// * enums pour les différents dumps à faire: déclaration, malloc et free
// ****************************************************************************
typedef enum {
  VARIABLES_DECLARATION=0,
  VARIABLES_MALLOC,
  VARIABLES_FREE
} VARIABLES_SWITCH;


// ****************************************************************************
// * Pointeur de fonction vers une qui dump ce que l'on souhaite
// ****************************************************************************
typedef NABLA_STATUS (*pFunDump)(nablaMain*,nablaVariable*,char*,char*);


// ****************************************************************************
// * Upcase de la chaîne donnée en argument
// * Utilisé lors des déclarations des Variables
// ****************************************************************************
static inline char *itemUPCASE(const char *itm){
  if (itm[0]=='c') return "NABLA_NB_CELLS";
  if (itm[0]=='n') return "NABLA_NB_NODES";
  if (itm[0]=='f') return "NABLA_NB_FACES";
  if (itm[0]=='g') return "NABLA_NB_GLOBAL";
  if (itm[0]=='p') return "NABLA_NB_PARTICLES";
  dbg("\n\t[itemUPCASE] itm=%s", itm);
  exit(NABLA_ERROR|fprintf(stderr, "\n[itemUPCASE] Error with given item\n"));
  return NULL;
}

// ****************************************************************************
// * 
// ****************************************************************************
static char *dimType(nablaMain *nabla,
                     char *type){
  const bool dim1D = (nabla->entity->libraries&(1<<with_real))!=0;
  const bool dim2D = (nabla->entity->libraries&(1<<with_real2))!=0;
  if (strncmp(type,"real3x3",7)==0) return type;
  if (strncmp(type,"real3",5)==0 and dim1D) return "real";
  if (strncmp(type,"real3",5)==0 and dim2D) return "real2";
  if (strncmp(type,"real2",5)==0 and dim1D) return "real";
  return type;
}

// ****************************************************************************
// * Dump d'un MALLOC d'une variables dans le fichier source
// * void *malloc(size_t size);
// ****************************************************************************
static NABLA_STATUS generateSingleVariableMalloc(nablaMain *nabla,
                                                 nablaVariable *var,
                                                 char *postfix,
                                                 char *depth){
  const char *type=dimType(nabla,var->type);
  nprintf(nabla,NULL,"\n\t// generateSingleVariableMalloc %s",var->name);
  if (var->dim==0)
    fprintf(nabla->entity->src,"\n\t%s* %s_%s=(%s*)malloc(sizeof(%s)*%s);",
            type,
            var->item,var->name,
            type,
            type,
            itemUPCASE(var->item));
  if (var->dim==1)
    fprintf(nabla->entity->src,"\n\t%s* %s_%s=(%s*)malloc(sizeof(%s)*%ld*%s);// __attribute__ ((aligned(WARP_ALIGN)));",
            type,
            var->item,var->name,
            type,
            type,
            var->size,
            itemUPCASE(var->item));
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'un FREE d'une variables dans le fichier source
 *****************************************************************************/
static NABLA_STATUS generateSingleVariableFree(nablaMain *nabla,
                                               nablaVariable *var,
                                               char *postfix,
                                               char *depth){  
  //if (var->item[0]!='p') return NABLA_OK;
  //nprintf(nabla,NULL,"\n\t// generateSingleVariableFree %s",var->name);
  if (var->dim==0)
    fprintf(nabla->entity->src,"\n\tfree(%s_%s);",
            var->item, var->name);
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'une variables dans le fichier
 *****************************************************************************/
static NABLA_STATUS generateSingleVariable(nablaMain *nabla,
                                           nablaVariable *var,
                                           char *postfix,
                                           char *depth){  
  const bool dim1D = (nabla->entity->libraries&(1<<with_real))!=0;
  const bool dim2D = (nabla->entity->libraries&(1<<with_real2))!=0;
  // Cas de la variable 'coord'
  if (strncmp(var->name,"coord",5)==0){
    if (dim2D){
      fprintf(nabla->entity->hdr,
              "\n//real3 node_coord[NABLA_NB_NODES] __attribute__ ((aligned(WARP_ALIGN)));");
     return NABLA_OK;
    }
    if (dim1D){
      fprintf(nabla->entity->hdr,
              "\n//real/*3*/ node_coord[NABLA_NB_NODES] __attribute__ ((aligned(WARP_ALIGN)));");
      return NABLA_OK;
    }
  }
  
  // Dans le cas d'une particule, on test le malloc
  if (var->item[0]=='p'){
    fprintf(nabla->entity->hdr,
            "\n%s *%s_%s%s%s __attribute__ ((aligned(WARP_ALIGN)));",
            postfix?"real":var->type,
            var->item,
            var->name,
            postfix?postfix:"",
            depth?depth:"");
    return NABLA_OK;
  }
  //nprintf(nabla,NULL,"\n\t// generateSingleVariable");
  const char *type =
    (var->item[0]!='g' && dim2D && strncmp(var->type,"real2",5)==0)?
    "real3/*2D->3D*/":
    var->type;
  if (var->dim==0)
    fprintf(nabla->entity->hdr,"\n//%s %s_%s%s%s[%s] __attribute__ ((aligned(WARP_ALIGN)));",
            postfix?"real":type,
            var->item, var->name,
            postfix?postfix:"",
            depth?depth:"",
            itemUPCASE(var->item));
  if (var->dim==1)
    fprintf(nabla->entity->hdr,"\n//%s %s_%s%s[%ld*%s] __attribute__ ((aligned(WARP_ALIGN)));",
            postfix?"real":var->type,
            var->item,var->name,
            postfix?postfix:"",
            var->size,
            itemUPCASE(var->item));
  return NABLA_OK;
}


/***************************************************************************** 
 * Retourne quelle fonction selon l'enum donné
 *****************************************************************************/
static pFunDump witch2func(VARIABLES_SWITCH witch){
  switch (witch){
  case (VARIABLES_DECLARATION): return generateSingleVariable;
  case (VARIABLES_MALLOC): return generateSingleVariableMalloc;
  case (VARIABLES_FREE): return generateSingleVariableFree;
  default: exit(NABLA_ERROR|fprintf(stderr, "\n[witch2switch] Error with witch\n"));
  }
}


/***************************************************************************** 
 * Dump d'une variables de dimension 1
 *****************************************************************************/
static NABLA_STATUS genericVariableDim1(nablaMain *nabla, nablaVariable *var, pFunDump fDump){
  fDump(nabla, var, NULL, "");
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump d'une variables de dimension 0
 *****************************************************************************/
static NABLA_STATUS genericVariableDim0(nablaMain *nabla, nablaVariable *var, pFunDump fDump){  
  dbg("\n[generateVariableDim0] variable %s", var->name);
  return fDump(nabla, var, NULL, NULL);
}


/***************************************************************************** 
 * Dump d'une variables
 *****************************************************************************/
static NABLA_STATUS genericVariable(nablaMain *nabla,
                                    nablaVariable *var,
                                    pFunDump fDump){  
  if (!var->axl_it) return NABLA_OK;
  if (var->item==NULL) return NABLA_ERROR;
  if (var->name==NULL) return NABLA_ERROR;
  if (var->type==NULL) return NABLA_ERROR;
  if (var->dim==0) return genericVariableDim0(nabla,var,fDump);
  if (var->dim==1) return genericVariableDim1(nabla,var,fDump);
  dbg("\n[genericVariable] variable dim error: %d", var->dim);
  exit(NABLA_ERROR|
       fprintf(stderr,
               "\n[genericVariable] Error with given variable\n"));
}


// ****************************************************************************
// * Initialisation des besoins vis-à-vis des variables (globales)
// ****************************************************************************
void xHookVariablesInit(nablaMain *nabla){
  // Rajout de la variable globale 'iteration'
  nablaVariable *iteration = nMiddleVariableNew(nabla);
  nMiddleVariableAdd(nabla, iteration);
  iteration->axl_it=false;
  iteration->item=strdup("global");
  iteration->type=strdup("int");
  iteration->name=strdup("iteration");
}


/***************************************************************************** 
 * Dump des options
 *****************************************************************************/
static void options(nablaMain *nabla){
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
// * Dump des variables dans le header
// * Utile pour les variables static, par exemple
// ****************************************************************************
void xHookVariablesPrefix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * Variables\n\
// ********************************************************");
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (genericVariable(nabla, var, witch2func(VARIABLES_DECLARATION))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[variables] Error with variable %s\n", var->name));
  }
  options(nabla);
}


// ****************************************************************************
// * Malloc des variables
// ****************************************************************************
void xHookVariablesMalloc(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * Déclaration & Malloc des Variables\n\
\t// ********************************************************");
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (genericVariable(nabla,
                        var,
                        witch2func(VARIABLES_MALLOC))==NABLA_ERROR)
      exit(NABLA_ERROR|
           fprintf(stderr,
                   "\n[variables] Error with variable %s\n",
                   var->name));
  }
}


// ****************************************************************************
// * Variables Postfix
// ****************************************************************************
void xHookVariablesFree(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * Free Variables\n\
\t// ********************************************************");
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next)
    if (genericVariable(nabla, var, witch2func(VARIABLES_FREE))==NABLA_ERROR)
      exit(NABLA_ERROR|fprintf(stderr, "\n[variables] Error with variable %s\n", var->name));
  fprintf(nabla->entity->src, "\n\treturn 0;\n}");
}



