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
// * nablaAtConstantParse
// ****************************************************************************
void nablaAtConstantParse(astNode * n, nablaMain *nabla, char *at){
  if (n->tokenid == '(') goto skip;
  if (n->tokenid == ')') goto skip;
  // Vérification que l'on ne déborde pas
  if (yyTranslate(n->tokenid)!=yyUndefTok()){
    // Si on tombe sur le "','", on sauve le 'when' dans l'entry_point
    if (yyNameTranslate(n->tokenid) == ',') {
      nablaStoreWhen(nabla,at);
      goto skip;
    }
  }
  if (n->token != NULL ){
    char *goto_end_of_at=at;
    while(*goto_end_of_at!=0)goto_end_of_at++;
    sprintf(goto_end_of_at, "%s", n->token);
    dbg("'%s' ", at);
  }
 skip:
  if (n->children != NULL) nablaAtConstantParse(n->children, nabla, at);
  if (n->next != NULL) nablaAtConstantParse(n->next, nabla, at);
}


/*****************************************************************************
 * nablaStoreWhen
 *****************************************************************************/
void nablaStoreWhen(nablaMain *nabla, char *at){
  nablaJob *entry_point=nablaJobLast(nabla->entity->jobs);
  entry_point->whens[entry_point->whenx]=atof(at);
  dbg("\n\t[nablaStoreWhen] Storing when @=%f ", entry_point->whens[entry_point->whenx]);
  entry_point->whenx+=1;
  *at=0;
}


/*****************************************************************************
 * nablaComparEntryPoints
 *****************************************************************************/
int nablaComparEntryPoints(const void *one, const void *two){
  nablaJob *first=(nablaJob*)one;
  nablaJob *second=(nablaJob*)two;
  if (first->whens[0]==second->whens[0]) return 0;
  if (first->whens[0]<second->whens[0]) return -1;
  return +1;
}


/*****************************************************************************
 * nablaNumberOfEntryPoints
 *****************************************************************************/
int nablaNumberOfEntryPoints(nablaMain *nabla){
  nablaJob *job;;
  int i,number_of_entry_points=0;
  for(job=nabla->entity->jobs;job!=NULL;job=job->next){
    if (!job->is_an_entry_point) continue;
    assert(job->whenx>=1);
    dbg("\n\t[nablaNumberOfEntryPoints] %s: whenx=%d @ ", job->name, job->whenx);
    for(i=0;i<job->whenx;++i)
      dbg("%f ", job->whens[i]);
    number_of_entry_points+=job->whenx; // On rajoute les différents whens
  }
  return number_of_entry_points;
}


/*****************************************************************************
 * nablaEntryPointsSort
 *****************************************************************************/
nablaJob* nablaEntryPointsSort(nablaMain *nabla,int number_of_entry_points){
  //bool initPhase=true;
  int i,j;
  nablaJob *job, *entry_points;
  
  dbg("\n[nablaEntryPointsSort] Sorting %d entry-points", number_of_entry_points);

  // On va rajouter le ComputeLoop[Begin||End]
  number_of_entry_points+=2;
  
  // On prépare le plan de travail de l'ensemble des entry_points
  entry_points =(nablaJob *)calloc(number_of_entry_points, sizeof(nablaJob));

  entry_points[0].item = strdup("\0");
  entry_points[0].is_an_entry_point=true;
  entry_points[0].is_a_function=true;
  entry_points[0].name = strdup("ComputeLoopBegin");
  entry_points[0].name_utf8 = strdup("ComputeLoopBegin");
  entry_points[0].whens[0] = ENTRY_POINT_compute_loop;
  entry_points[0].whenx = 1;

  entry_points[1].item = strdup("\0");
  entry_points[1].is_an_entry_point=true;
  entry_points[1].is_a_function=true;
  entry_points[1].name = strdup("ComputeLoopEnd");
  entry_points[1].name_utf8 = strdup("ComputeLoopEnd");
  entry_points[1].whens[0] = ENTRY_POINT_exit;
  entry_points[1].whenx = 1; 

  // On re scan pour remplir les duplicats
  for(i=2,job=nabla->entity->jobs;job!=NULL;job=job->next){
    if (!job->is_an_entry_point) continue;
    for(j=0;j<job->whenx;++j){
      dbg("\n\t[nablaEntryPointsSort] dumping #%d: %s @ %f", i, job->name, job->whens[j]);
      entry_points[i].item=job->item;
      entry_points[i].is_an_entry_point=true;
      entry_points[i].is_a_function=job->is_a_function;
      //assert(job->type!=NULL);
      assert(job->name!=NULL);
      entry_points[i].name=job->name;
      assert(job->name_utf8!=NULL);
      entry_points[i].name_utf8=job->name_utf8;
      entry_points[i].whens[0]=job->whens[j];
      // Pas utilisé, on passe après par la fonctionnccAxlGeneratorEntryPointWhere
      if (entry_points[i].whens[0]>ENTRY_POINT_compute_loop)
        entry_points[i].where=strdup("compute-loop");
      if (entry_points[i].whens[0]==-0.0)
        entry_points[i].where=strdup("build");
      if (entry_points[i].whens[0]<ENTRY_POINT_start_init)
        entry_points[i].where=strdup("init");
      assert(job->entity!=NULL);
      // On recopie les infos utiles pour le dump d'après
      entry_points[i].ifAfterAt=job->ifAfterAt;
      entry_points[i].entity=job->entity;
      entry_points[i].stdParamsNode=job->stdParamsNode;
      entry_points[i].nblParamsNode=job->nblParamsNode;
      entry_points[i].called_variables=job->called_variables;
      entry_points[i].reduction=job->reduction;
      entry_points[i].reduction_name=job->reduction_name;
      ++i;
    }
  }

  // On trie afin d'avoir tous les points d'entrée
  qsort(entry_points,number_of_entry_points,sizeof(nablaJob),nablaComparEntryPoints);

  if (nabla->optionDumpTree)
    timeTreeSave(nabla, entry_points, i);
  
  return entry_points;
}

