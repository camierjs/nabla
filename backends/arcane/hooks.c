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
#include "backends/arcane/arcane.h"

// ****************************************************************************
// * nHookXyz pre/post fix
// ****************************************************************************
char* nccArcSystemPrefix(void){ return "m_"; }


// ****************************************************************************
// * hookPrimaryExpressionToReturn
// ****************************************************************************
bool aHookPrimaryExpressionToReturn(nablaMain *nabla, nablaJob *job, astNode *n){
  const char* var=dfsFetchFirst(job->stdParamsNode,rulenameToId("direct_declarator"));
  dbg("\n\t[hookPrimaryExpressionToReturn] ?");
  if (var!=NULL && strcmp(n->children->token,var)==0){
    dbg("\n\t[hookPrimaryExpressionToReturn] primaryExpression hits returned argument");
    nprintf(nabla, NULL, "%s_per_thread[tid]",var);
    return true;
  }else{
    dbg("\n\t[hookPrimaryExpressionToReturn] ELSE");
    //nprintf(nabla, NULL, "%s",n->children->token);
  }
  return false;
}


// ****************************************************************************
// *
// ****************************************************************************
bool arcaneHookDfsVariable(nablaMain *nabla){
  if (isAnArcaneFamily(nabla)) return true;
  return false;
}
bool arcaneHookDfsExtra(nablaMain *nabla, nablaJob *job,bool type){
  if (isAnArcaneFamily(nabla)) return true;
  return false;
}
char* arcaneHookDfsArgType(nablaMain *nabla, nablaVariable *var){
  if (isAnArcaneFamily(nabla)){
    const int offset=var->in?6:0;
    char *str=(char*)calloc(1024,sizeof(char));
    snprintf(str,1024,"%sVariable%s%s",
             var->in?"const ":"",
             var->item,
             var->type);
    str[8+offset]-=32;
    str[12+offset]-=32;
    return str;
  }
  return var->type;
}




// ****************************************************************************
// * Dump des variables appelées
// ****************************************************************************
void aHookDfsForCalls(nablaMain *nabla,
                      nablaJob *fct,
                      astNode *n,
                      const char *namespace,
                      astNode *nParams){
  nMiddleFunctionDumpFwdDeclaration(nabla,fct,nParams,namespace);
}


// ****************************************************************************
// * aHookJobHit
// ****************************************************************************
bool aHookJobHit(nablaMain *nabla,bool is_an_entry_point){
  if (isAnArcaneFamily(nabla) && is_an_entry_point) return false;
  return true;
}


/*
 * nablaArcaneColor
 */
char *nablaArcaneColor(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_ALONE)==BACKEND_COLOR_ARCANE_ALONE)
    return "Module";
  if ((middlend->colors&BACKEND_COLOR_ARCANE_FAMILY)==BACKEND_COLOR_ARCANE_FAMILY)
    return "";
  if ((middlend->colors&BACKEND_COLOR_ARCANE_MODULE)==BACKEND_COLOR_ARCANE_MODULE)
    return "Module";
  if ((middlend->colors&BACKEND_COLOR_ARCANE_SERVICE)==BACKEND_COLOR_ARCANE_SERVICE)
    return "Service";
  exit(NABLA_ERROR|fprintf(stderr,"[nablaArcaneColor] Unable to switch!"));
  return NULL;
}
bool isAnArcaneAlone(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_ALONE)==BACKEND_COLOR_ARCANE_ALONE)
    return true;
  return false;
}
bool isAnArcaneModule(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_ALONE)==BACKEND_COLOR_ARCANE_ALONE)
    return true;
  if ((middlend->colors&BACKEND_COLOR_ARCANE_MODULE)==BACKEND_COLOR_ARCANE_MODULE)
    return true;
  return false;
}
bool isAnArcaneService(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_SERVICE)==BACKEND_COLOR_ARCANE_SERVICE)
    return true;
  return false;
}
bool isAnArcaneFamily(nablaMain *middlend){
  if ((middlend->colors&BACKEND_COLOR_ARCANE_FAMILY)==BACKEND_COLOR_ARCANE_FAMILY)
    return true;
  return false;
}


char* arcaneEntryPointPrefix(nablaMain *nabla,
                             nablaJob *entry_point){
  return "";
}


void arcaneIteration(nablaMain *nabla){
  nprintf(nabla, "/*ITERATION*/", "subDomain()->commonVariables().globalIteration()");
}

void arcaneError(nablaMain *nabla,nablaJob *job){nprintf(nabla,"/*ERROR*/","\nexit(-1);\n");}

void arcaneExit(nablaMain *nabla,nablaJob *job){
  // le token 'exit' doit être appelé depuis une fonction
  assert(job->is_a_function);
  nprintf(nabla, "/*EXIT*/","{\n\
if (m_hlt_dive.at(m_hlt_level)){\n\
   m_hlt_exit[m_hlt_level]=true;\n\
}else{\n\
   subDomain()->timeLoopMng()->stopComputeLoop(true);\n\
}}");
}


void arcaneTime(nablaMain *nabla){
  nprintf(nabla, "/*TIME*/", "subDomain()->commonVariables().globalTime()");
}


void arcaneFatal(nablaMain *nabla){
  dbg("\n[arcaneFatal]");
  nprintf(nabla, NULL, "throw FatalErrorException");
} 


void arcaneAddCallNames(nablaMain *nabla,
                        nablaJob *job,
                        astNode *n){
  dbg("\n[arcaneAddCallNames]");
  /*nothing to do*/
}


void arcaneAddArguments(nablaMain *nabla,nablaJob *job){
  dbg("\n[arcaneAddArguments]");
  /*nothing to do*/
}


void arcaneTurnTokenToOption(nablaMain *nabla,nablaOption *opt){
  nprintf(nabla, "/*tt2o arc*/", "options()->%s()", opt->name);
}


// ***************************************************************************** 
// * Traitement des tokens SYSTEM
// *****************************************************************************
//#warning nablaSystem and test error (ddfv alpha)
void arcaneHookSystem(astNode * n,nablaMain *arc, const char cnf, char enum_enum){
  char *itm=(cnf=='c')?"cell":(cnf=='n')?"node":"face";
  char *etm=(enum_enum=='c')?"c":(enum_enum=='n')?"n":"f";
  //if (n->tokenid == DIESE)         nprintf(arc, "/*nablaSystem*/", "%s.index()",etm);//asInteger
  if (n->tokenid == LID)           nprintf(arc, "/*nablaSystem*/", "[%s->localId()]",itm);//asInteger
  if (n->tokenid == SID)           nprintf(arc, "/*nablaSystem*/", "[subDomain()->subDomainId()]");
  if (n->tokenid == THIS)          nprintf(arc, "/*nablaSystem THIS*/", NULL);
  if (n->tokenid == NBNODE)        nprintf(arc, "/*nablaSystem NBNODE*/", NULL);
  if (n->tokenid == NBCELL)        nprintf(arc, "/*nablaSystem NBCELL*/", NULL);
  //if (n->tokenid == INODE)         nprintf(arc, "/*nablaSystem INODE*/", NULL);
  if (n->tokenid == BOUNDARY_CELL) nprintf(arc, "/*nablaSystem BOUNDARY_CELL*/", NULL);
  if (n->tokenid == FATAL)         nprintf(arc, "/*nablaSystem*/", "throw FatalErrorException");
  if (n->tokenid == BACKCELL)      nprintf(arc, "/*nablaSystem*/", "[%s->backCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == BACKCELLUID)   nprintf(arc, "/*nablaSystem*/", "[%s->backCell().uniqueId()]",itm);
  if (n->tokenid == FRONTCELL)     nprintf(arc, "/*nablaSystem*/", "[%s->frontCell()]",(enum_enum=='\0')?itm:etm);
  if (n->tokenid == FRONTCELLUID)  nprintf(arc, "/*nablaSystem*/", "[%s->frontCell().uniqueId()]",itm);
  if (n->tokenid == NEXTCELL)      nprintf(arc, "/*nablaSystem NEXTCELL*/", "[nextCell]");
  if (n->tokenid == PREVCELL)      nprintf(arc, "/*nablaSystem PREVCELL*/", "[prevCell]");
  if (n->tokenid == NEXTNODE)      nprintf(arc, "/*nablaSystem NEXTNODE*/", "[nextNode]");
  if (n->tokenid == PREVNODE)      nprintf(arc, "/*nablaSystem PREVNODE*/", "[prevNode]");
  if (n->tokenid == PREVLEFT)      nprintf(arc, "/*nablaSystem PREVLEFT*/", "[cn.previousLeft()]");
  if (n->tokenid == PREVRIGHT)     nprintf(arc, "/*nablaSystem PREVRIGHT*/", "[cn.previousRight()]");
  if (n->tokenid == NEXTLEFT)      nprintf(arc, "/*nablaSystem NEXTLEFT*/", "[cn.nextLeft()]");
  if (n->tokenid == NEXTRIGHT)     nprintf(arc, "/*nablaSystem NEXTRIGHT*/", "[cn.nextRight()]");
}

// ****************************************************************************
// * functionGlobalVar
// ****************************************************************************
char *functionGlobalVar(const nablaMain *arc,
                        const nablaJob *job,
                        const nablaVariable *var){
  if (job->item[0] != '\0') return NULL; // On est bien une fonction
  if (var->item[0] != 'g') return NULL;  // On a bien affaire à une variable globale
  const bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  const int scalar = var->dim==0;
  const int resolve = job->parse.isPostfixed!=2;
  dbg("\n\t\t[functionGlobalVar] name=%s, scalar=%d, resolve=%d",var->name, scalar,resolve);
  //nprintf(arc, "/*0*/", "%s",(left_of_assignment_operator)?"":"()"); // "()" permet de récupérer les m_global_...()
  if (left_of_assignment_operator || !scalar) return "/*global_*/";
  return "()";
}


// ****************************************************************************
// * arcaneHookFunctionName
// ****************************************************************************
void arcaneHookFunctionName(nablaMain *arc){
  nprintf(arc, NULL, "%s%s::", arc->name, "");//nablaArcaneColor(arc));
}

