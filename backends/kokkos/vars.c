///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
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
// * kHookVarDeclPrefix & kHookVarDeclPostfix
// ****************************************************************************
char* kHookVarDeclPrefix(nablaMain *nabla){
  //printf("\n\t[1;34m[kHookVarDeclPrefix][m");
  return "Kokkos::View<";
}
char* kHookVarDeclPostfix(nablaMain *nabla){
  return "*>&";
}


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
// ****************************************************************************
static NABLA_STATUS generateSingleVariableMalloc(nablaMain *nabla,
                                                 nablaVariable *var){
  const char *type=dimType(nabla,var->type);
  if (var->dim==0)
    fprintf(nabla->entity->src,
            "\n\tKokkos::View<%s*> %s_%s(\"%s_%s_label\",%s);",
            type,
            var->item,var->name,
            var->item,var->name,            
            itemUPCASE(var->item));
  if (var->dim==1)
    fprintf(nabla->entity->src,
            "\n\tKokkos::View<%s*> %s_%s(\"%s_%s_label\",%ld*%s);",
            type,
            var->item,var->name,
            var->item,var->name,
            var->size,
            itemUPCASE(var->item));
  return NABLA_OK;
}


// ****************************************************************************
// * Dump des options dans le header
// ****************************************************************************
void kHookVariablesPrefix(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\n\n\
// ********************************************************\n\
// * xHookVariablesPrefix\n\
// ********************************************************");
  nablaOption *opt;
  fprintf(nabla->entity->hdr,"\n// Options");
  for(opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->hdr,
            "\n#define %s %s",
            opt->name, opt->dflt);
 }


// ****************************************************************************
// * Malloc des variables
// ****************************************************************************
void kHookVariablesMalloc(nablaMain *nabla){
  if (isWithLibrary(nabla,with_real))
    xHookMesh1D(nabla);
  else if (isWithLibrary(nabla,with_real2))
    xHookMesh2D(nabla);
  else
    xHookMesh3D(nabla);
  
  fprintf(nabla->entity->src,"\n\
\t// ********************************************************\n\
\t// * Déclaration & Malloc des Variables\n\
\t// ********************************************************");
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (generateSingleVariableMalloc(nabla, var)==NABLA_ERROR)
      exit(NABLA_ERROR|
           fprintf(stderr,
                   "\n[variables] Error with variable %s\n",
                   var->name));
  }
}


// ****************************************************************************
// * Variables Postfix
// ****************************************************************************
void kHookVariablesFree(nablaMain *nabla){
  fprintf(nabla->entity->src, "\n\n\treturn 0;\n}");
}



