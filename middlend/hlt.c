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
//#include "nabla.tab.h"


// ****************************************************************************
// * dumpAtEndofAt
// ****************************************************************************
static void dumpAtEndofAt(nablaJob *job, const char *token){
  assert(atof(token)>=0.0);
  job->whens[job->when_index]+=
    job->when_sign*atof(token)*pow(10.0,-job->when_depth*NABLA_JOB_WHEN_HLT_FACTOR);
  dbg("\n\t\t[nMiddleAtConstantParse/dumpAtEndofAt] HLT level #%d => %.12e ",
      job->when_depth,
      job->whens[job->when_index]);
}

// ****************************************************************************
// * nablaAtConstantParse
// ****************************************************************************
void nMiddleAtConstantParse(nablaJob *job,node *n, nablaMain *nabla){
  // On évite les parenthèses rajoutée lors du parsing .y 'at_constant'
  if (n->tokenid == '-') {
    // On s'assure pour l'instant que les temps logiques
    // sont positifs dans les imbrications
    assert(job->when_depth==0);
    job->when_sign*=-1.0;
  }
  if (n->tokenid == '(') goto skip;
  if (n->tokenid == ')') goto skip;
  if (n->tokenid == '/') {
    // On s'enfonce dans la hiérarchie
    job->when_depth+=1;
    dbg("\n\t\t[nMiddleAtConstantParse] job->whens[%d]=%.12e",
        job->when_index,
        job->whens[job->when_index]);
    goto skip;
  }
  // Vérification que l'on ne déborde pas
  if (yyTranslate(n->tokenid)!=yyUndefTok()){
    // Si on tombe sur le "','", on sauve le 'when' dans l'entry_point
    if (yyNameTranslate(n->tokenid) == ',') {
      job->when_sign=1.0;
      nMiddleStoreWhen(job,nabla);
      goto skip;
    }
  }
  if (n->token != NULL )
    dumpAtEndofAt(job,n->token);
 skip:
  if (n->children != NULL) nMiddleAtConstantParse(job,n->children, nabla);
  if (n->next != NULL) nMiddleAtConstantParse(job,n->next, nabla);
}

// ****************************************************************************
// * nablaStoreWhen
// ****************************************************************************
void nMiddleStoreWhen(nablaJob *job,nablaMain *nabla){
  nablaJob *entry_point=job;
  dbg("\n\t\t[nablaStoreWhen] Storing when @=%.12e ",
      entry_point->whens[entry_point->when_index]);
  entry_point->when_index+=1;
  entry_point->whens[entry_point->when_index]=0.0;
}


// ****************************************************************************
// * nablaComparEntryPoints
// ****************************************************************************
int nMiddleComparEntryPoints(const void *one, const void *two){
  nablaJob *first=(nablaJob*)one;
  nablaJob *second=(nablaJob*)two;
  if (first->whens[0]==second->whens[0]) return 0;
  if (first->whens[0]<second->whens[0]) return -1;
  return +1;
}


// ****************************************************************************
// * nablaNumberOfEntryPoints
// ****************************************************************************
int nMiddleNumberOfEntryPoints(nablaMain *nabla){
  nablaJob *job;;
  int i,number_of_entry_points=0;
  for(job=nabla->entity->jobs;job!=NULL;job=job->next){
    if (!job->is_an_entry_point) continue;
    assert(job->when_index>=1);
    dbg("\n\t[nablaNumberOfEntryPoints] %s: when_index=%d @ ", job->name, job->when_index);
    for(i=0;i<job->when_index;++i)
      dbg("%.12e ", job->whens[i]);
    number_of_entry_points+=job->when_index; // On rajoute les différents whens
  }
  return number_of_entry_points;
}


// ****************************************************************************
// * nMiddleEntryPointsSort
// ****************************************************************************
nablaJob* nMiddleEntryPointsSort(nablaMain *nabla,
                                 int number_of_entry_points){
  int i,j;
  int hlt_max_depth=0;
  //int *hlt_idx_entry_points;
  int hlt_current_depth=0;
  //int hlt_dive_num=0;
  //double *hlt_dive_when=calloc(32,sizeof(double));
  nablaJob *job, *entry_points=NULL;
  
  dbg("\n\t\t[nMiddleEntryPointsSort] Sorting %d entry-points:", number_of_entry_points);
  // On va rajouter le ComputeLoop[Begin||End]
  number_of_entry_points+=2;

  // Première passe des job pour compter la profondeur max des imbrications des HLT
  // Cela va nous permettre de rajouter les hltDives associés
  for(job=nabla->entity->jobs;job!=NULL;job=job->next)
    if (hlt_max_depth<job->when_depth)
      hlt_max_depth=job->when_depth;
  dbg("\n\t\t[nMiddleEntryPointsSort] hlt_max_depth=%d", hlt_max_depth);
  number_of_entry_points+=hlt_max_depth;

  // On prépare le plan de travail de l'ensemble des entry_points
  entry_points =(nablaJob*)calloc(1+number_of_entry_points, sizeof(nablaJob));
  assert(entry_points);
  dbg("\n\t\t[nMiddleEntryPointsSort] calloc'ed!");

  // On rajoute les DIVES
  for(i=0,job=nabla->entity->jobs;job!=NULL;job=job->next){
    // Si on remonte, on met à jour la profondeur
    if (hlt_current_depth>job->when_depth)
      hlt_current_depth=job->when_depth;
    // Si on plonge, on rajoute notre DIVE
    if (job->when_depth>hlt_current_depth){
      char name[11+18+1];
      const double when=job->whens[0];
      //const unsigned long *adrs = (unsigned long*)&when;
      //snprintf(name,11+18,"hltDive_at_0x%lx",*adrs);
      snprintf(name,11+18,"hltDive%d",i);
      dbg("\n\t\t[nMiddleEntryPointsSort] Adding hltDive%d before %s @ %f, name=%s",
          i,job->name,when,name);
      //entry_points[i].scope = sdup(job->scope);
      entry_points[i].region = job->region?sdup(job->region):sdup("\0");
      entry_points[i].item = sdup("\0");
      entry_points[i].is_an_entry_point=true;
      entry_points[i].is_a_function=true;
      entry_points[i].name = sdup(name);
      entry_points[i].name_utf8 = sdup(name);
      entry_points[i].whens[0] = when;
      entry_points[i].when_index = 1;
      entry_points[i].when_depth = hlt_current_depth;
      i+=1;
      hlt_current_depth=job->when_depth;
    }
  }
  
  dbg("\n\t\t[nMiddleEntryPointsSort] Adding ComputeLoopBegin [%d]", i);
  entry_points[i].item = sdup("\0");
  entry_points[i].is_an_entry_point=true;
  entry_points[i].is_a_function=true;
  entry_points[i].name = sdup("ComputeLoopBegin");
  entry_points[i].name_utf8 = sdup("ComputeLoopBegin");
  entry_points[i].whens[0] = ENTRY_POINT_compute_loop;
  entry_points[i].when_index = 1;
  i+=1;
  
  dbg("\n\t\t[nMiddleEntryPointsSort] Adding ComputeLoopEnd [%d]", i);
  entry_points[i].item = sdup("\0");
  entry_points[i].is_an_entry_point=true;
  entry_points[i].is_a_function=true;
  entry_points[i].name = sdup("ComputeLoopEnd");
  entry_points[i].name_utf8 = sdup("ComputeLoopEnd");
  entry_points[i].whens[0] = ENTRY_POINT_exit;
  entry_points[i].when_index = 1; 

  // On re scan pour remplir les duplicats
  for(i+=1,job=nabla->entity->jobs;job!=NULL;job=job->next){
    if (!job->is_an_entry_point) continue;
    // Pour chaque temps logique séparé par une virgule
    for(j=0;j<job->when_index;++j){
      dbg("\n\t\t\t[nMiddleEntryPointsSort] Scanning&Dumping #%d: %s @ %.12e",
          i, job->name, job->whens[j]);
      const int HLT_depth=entry_points[i].when_depth;
      
      if (hlt_current_depth>HLT_depth) // Si on trouve une transition en EXIT
        hlt_current_depth=HLT_depth;   // On met à jour notre profondeur
      
      //entry_points[i].scope = job->scope;
      entry_points[i].region = job->region;
      entry_points[i].item=job->item;
      entry_points[i].is_an_entry_point=true;
      entry_points[i].is_a_function=job->is_a_function;
      //assert(job->type!=NULL);
      assert(job->name!=NULL);
      entry_points[i].name=job->name;
      assert(job->name_utf8!=NULL);
      entry_points[i].name_utf8=job->name_utf8;
      entry_points[i].whens[0]=job->whens[j];
      entry_points[i].when_depth=job->when_depth;
      // Pas utilisé, on passe après par la fonction nccAxlGeneratorEntryPointWhere
      if (entry_points[i].whens[0]>ENTRY_POINT_compute_loop)
        entry_points[i].where=sdup("compute-loop");
      if (entry_points[i].whens[0]==-0.0)
        entry_points[i].where=sdup("build");
      if (entry_points[i].whens[0]<ENTRY_POINT_start_init)
        entry_points[i].where=sdup("init");
      assert(job->entity!=NULL);
      // On recopie les infos utiles pour le dump d'après
      entry_points[i].ifAfterAt=job->ifAfterAt;
      entry_points[i].entity=job->entity;
      entry_points[i].stdParamsNode=job->stdParamsNode;
      entry_points[i].nblParamsNode=job->nblParamsNode;
      entry_points[i].called_variables=job->called_variables;
      entry_points[i].used_variables=job->used_variables;
      entry_points[i].used_options=job->used_options;
      entry_points[i].reduction=job->reduction;
      entry_points[i].reduction_name=job->reduction_name;
      entry_points[i].enum_enum=job->enum_enum;
      entry_points[i].exists=job->exists;
      i+=1;
    }
  }

  // On trie afin d'avoir tous les points d'entrée
  qsort(entry_points,number_of_entry_points,sizeof(nablaJob),nMiddleComparEntryPoints);

  if (nabla->optionDumpTree!=0)
    nMiddleTimeTreeSave(nabla, entry_points, i);
  
  return entry_points;
}

