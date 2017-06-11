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
#include "nabla.tab.h"
#include "backends/arcane/arcane.h"

extern bool adrs_it;

// ****************************************************************************
// * arcaneHookIsTest
// ****************************************************************************
void arcaneHookIsTest(nablaMain *nabla, nablaJob *job, node *n, int token){
  // Arcane, traite le test en 'objets'
  if (token!=IS) return;
  assert(n->children && n->children->next && n->children->next->token);
  const char *token2function = n->children->next->token;
  if (n->children->next->tokenid==OWN)
    nprintf(nabla, NULL, "/*IS*/.isOwn()");
  else
    nprintf(nabla, NULL, "/*IS*/.%s()", token2function);
//#warning On purge le token pour pas quil soit parsé
  n->children->next->token[0]=0;
}


// ****************************************************************************
// * nArcaneHookTokenPrefix
// ****************************************************************************
char* arcaneHookTokenPrefix(nablaMain *nabla){return sdup("m_");}

// ****************************************************************************
// * nArcaneHookTokenPostfix
// ****************************************************************************
char* arcaneHookTokenPostfix(nablaMain *nabla){return sdup("");}




//*****************************************************************************
// * Transformation de tokens en variables Arcane
// * selon les contextes dans le cas d'un '[Cell|node]Enumerator'
// ****************************************************************************
nablaVariable *arcaneHookTurnTokenToVariable(node * n,
                                             nablaMain *arc,
                                             nablaJob *job){
  dbg("\n\t\t[arcaneHookTurnTokenToVariable]");
  assert(job->item!=NULL);
  const char cnfg=job->item[0];
  //char enum_enum=job->parse.enum_enum;
  //bool left_of_assignment_operator=job->parse.left_of_assignment_operator;
  //int isPostfixed=job->parse.isPostfixed;
  //dbg("\n\t\t[arcaneHookTurnTokenToVariable] local variabled but var!");
  //nablaVariable *var=nMiddleVariableFind(arc->variables, n->token);
  nablaVariable *var=/*(job->nb_in_item_set==0)?
                       nMiddleVariableFind(arc->variables, n->token):*/
    nMiddleVariableFindWithSameJobItem(arc,job,arc->variables, n->token);
  //dbg("\n\t\t[arcaneHookTurnTokenToVariable] local variabled!");
      
  // Si on ne trouve pas de variable, on a rien à faire
  if (var == NULL){
    //dbg("\n\t\t[arcaneHookTurnTokenToVariable] Pas une variable!");
    //nprintf(arc,NULL,"/*tt2a (isPostfixed=%d)(isLeft=%d)*/",isPostfixed,left_of_assignment_operator);
    // Si on est dans une zone 'adrs_it' et que ce n'est PAS une variable, on rajoute l''&'
    if (adrs_it==true) nprintf(arc, NULL, "&");
    return NULL;
  }
  
  // Si on est dans une expression d'Aleph, on garde la référence à la variable  telle-quelle
  if (job->parse.alephKeepExpression==true){
    nprintf(arc, NULL, "m_%s_%s", var->item, var->name);
    return var;
  }

  // Si on est dans une zone 'adrs_it' et que c'est une variable, on rajoute l''adrs('
  if (adrs_it==true) nprintf(arc, NULL, "adrs(");
    
  //dbg("\n\t\t[arcaneHookTurnTokenToVariable] m_%s_%s token=%s", var->item, var->name, n->token);
  //nprintf(arc, NULL, "/*tt2a (isPostfixed=%d)(isLeft=%d)*/",isPostfixed,left_of_assignment_operator);
  if (var->gmpRank==-1){
    if (job->nb_in_item_set>0 && var->item[0]!='g'){
      nprintf(arc, NULL, "anyone[iitem]");
      return var;
    }else{
      nprintf(arc, NULL, "m_%s_%s", var->item, var->name);
    }
  }else{
    if (cnfg=='c')
      nprintf(arc, NULL, "m_gmp[%d][cell->localId()].get_mpz_t()", var->gmpRank);
    if (cnfg=='g')
      nprintf(arc, NULL, "m_gmp[%d].get_mpz_t()", var->gmpRank);
    return var;
  }
  
  // Lancement de tous les transformations connues au sein des Cells jobs
  nprintf(arc,NULL,cellJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobNodeVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobFaceVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobParticleVar(arc,job,var),NULL);
  nprintf(arc,NULL,cellJobGlobalVar(arc,job,var),NULL);

  // Lancement de tous les transformations connues au sein des Nodes jobs
  nprintf(arc,NULL,nodeJobNodeVar(arc,job,var),NULL);
  nprintf(arc,NULL,nodeJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,nodeJobFaceVar(arc,job,var),NULL);
  nprintf(arc,NULL,nodeJobGlobalVar(arc,job,var),NULL);

  // Lancement de tous les transformations connues au sein des Faces jobs
  nprintf(arc,NULL,faceJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,faceJobNodeVar(arc,job,var),NULL);
  nprintf(arc,NULL,faceJobFaceVar(arc,job,var),NULL);
  nprintf(arc,NULL,faceJobGlobalVar(arc,job,var),NULL);

  // Lancement de tous les transformations connues au sein des Particles jobs
  nprintf(arc,NULL,particleJobParticleVar(arc,job,var),NULL);
  nprintf(arc,NULL,particleJobCellVar(arc,job,var),NULL);
  nprintf(arc,NULL,particleJobGlobalVar(arc,job,var),NULL);


  // Lancement de tous les transformations connues au sein des fonctions
  nprintf(arc,NULL,functionGlobalVar(arc,job,var),NULL);
  
  if (adrs_it==true) nprintf(arc, NULL, ")");

  return var;
}

