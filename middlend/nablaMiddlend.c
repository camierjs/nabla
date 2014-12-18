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


// ****************************************************************************
// * Dump dans le header des 'includes'
// ****************************************************************************
NABLA_STATUS nablaCompoundJobEnd(nablaMain* nabla_main){
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header des 'includes'
 *****************************************************************************/
NABLA_STATUS nablaInclude(nablaMain *nabla, char *include){
  fprintf(nabla->entity->src, "%s\n", include);
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header les 'define's
 *****************************************************************************/
NABLA_STATUS nablaDefines(nablaMain *nabla, nablaDefine *defines){
  int i;
  FILE *target_file = isAnArcaneService(nabla)?nabla->entity->src:nabla->entity->hdr;
  fprintf(target_file,"\n\
\n// *****************************************************************************\
\n// * Defines\
\n// *****************************************************************************");
  for(i=0;defines[i].what!=NULL;i+=1)
    fprintf(target_file, "\n#define %s %s",defines[i].what,defines[i].with);
  fprintf(target_file, "\n");
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header les 'define's
 *****************************************************************************/
NABLA_STATUS nablaTypedefs(nablaMain *nabla, nablaTypedef *typedefs){
  int i;
  fprintf(nabla->entity->hdr,"\n\
\n// *****************************************************************************\
\n// * Typedefs\
\n// *****************************************************************************");
  for(i=0;typedefs[i].what!=NULL;i+=1)
    fprintf(nabla->entity->hdr, "\ntypedef %s %s;",typedefs[i].what,typedefs[i].with);
  fprintf(nabla->entity->hdr, "\n");
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump dans le header des 'forwards's
 *****************************************************************************/
NABLA_STATUS nablaForwards(nablaMain *nabla, char **forwards){
  int i;
  fprintf(nabla->entity->hdr,"\n\
\n// *****************************************************************************\
\n// * Forwards\
\n// *****************************************************************************");
  for(i=0;forwards[i]!=NULL;i+=1)
    fprintf(nabla->entity->hdr, "\n%s",forwards[i]);
  fprintf(nabla->entity->hdr, "\n");
  return NABLA_OK;
}


// ****************************************************************************
// * switchItemSupportTokenid
// ****************************************************************************
static char *switchItemSupportTokenid(int item_support_tokenid){
  if (item_support_tokenid==CELLS) return "cell";
  if (item_support_tokenid==FACES) return "face";
  if (item_support_tokenid==NODES) return "node";
  if (item_support_tokenid==PARTICLES) return "particle";
//#warning switchItemSupportTokenid should be deterministic
  return "global";
}


/*****************************************************************************
 * Fonction de parsing et d'application des actions correspondantes
 *****************************************************************************/
void nablaMiddlendParseAndHook(astNode * n, nablaMain *nabla){
    
  ///////////////////////////////
  // Déclaration des libraries //
  ///////////////////////////////
  if (n->ruleid == rulenameToId("with_library")){
    dbg("\n\t[nablaMiddlendParseAndHook] with_library hit!");
    nablaLibraries(n,nabla->entity);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }
  
  ///////////////////////////////////////////////////////
  // Règle de définitions des items sur leurs supports //
  ///////////////////////////////////////////////////////
  if (n->ruleid == rulenameToId("nabla_item_definition")){
    char *item_support=n->children->children->token;
    // Nodes|Cells|Global|Faces|Particles
    int item_support_tokenid=n->children->children->tokenid;
    dbg("\n\t[nablaMiddlendParseAndHook] rule %s,  support %s", n->rule, item_support);
    // On backup temporairement le support (kind) de l'item
    nabla->tmpVarKinds=strdup(switchItemSupportTokenid(item_support_tokenid));
    nablaItems(n->children->next,
               rulenameToId("nabla_item_declaration"),
               nabla);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }

  ////////////////////////////////////
  // Règle de définitions des includes
  /////////////////////////////////////
  if (n->tokenid == INCLUDES){
    nablaInclude(nabla, n->token);
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit INCLUDES %s", n->token);
  }

  ///////////////////////////////////
  // Règle de définitions des options
  ///////////////////////////////////
  if (n->ruleid == rulenameToId("nabla_options_definition")){
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit %s", n->rule);
    nablaOptions(n->children,
                 rulenameToId("nabla_option_declaration"), nabla);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }

  /////////////////////////////////////////////////
  // On a une définition d'une fonction standard //
  /////////////////////////////////////////////////
  if (n->ruleid == rulenameToId("function_definition")){
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit %s", n->rule);
    nabla->hook->function(nabla,n);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }
  
  ////////////////////////////////////////
  // On a une définition d'un job Nabla //
  ////////////////////////////////////////
  if (n->ruleid == rulenameToId("nabla_job_definition")){
    dbg("\n\t[nablaMiddlendParseAndHook] rule hit %s", n->rule);
    nabla->hook->job(nabla,n);
    dbg("\n\t[nablaMiddlendParseAndHook] done");
  }

  //////////////////
  // DFS standard //
  //////////////////
  if(n->children != NULL) nablaMiddlendParseAndHook(n->children, nabla);
  if(n->next != NULL) nablaMiddlendParseAndHook(n->next, nabla);
}


/*****************************************************************************
 * Rajout des variables globales utiles aux mots clefs systèmes
 * On rajoute en dur les variables time, deltat, coord
 *****************************************************************************/
static void nablaMiddlendVariableGlobalAdd(nablaMain *nabla){
  dbg("\n\t[nablaMiddlendVariableGlobalAdd] Adding global deltat, time");
  nablaVariable *deltat = nablaVariableNew(nabla);
  nablaVariableAdd(nabla, deltat);
  deltat->axl_it=false;
  deltat->item=strdup("global");
  deltat->type=strdup("real");
  deltat->name=strdup("deltat");
  nablaVariable *time = nablaVariableNew(nabla);
  nablaVariableAdd(nabla, time);
  time->axl_it=false;
  time->item=strdup("global");
  time->type=strdup("real");
  time->name=strdup("time");
  if ((nabla->colors&BACKEND_COLOR_OKINA_SOA)==BACKEND_COLOR_OKINA_SOA){
    dbg("\n\t[nablaMiddlendVariableGlobalAdd] Adding SoA variables coordx,y,z");
    nablaVariable *coordx= nablaVariableNew(nabla);
    nablaVariableAdd(nabla, coordx);
    coordx->axl_it=true;
    coordx->item=strdup("node");
    coordx->type=strdup("real");
    coordx->name=strdup("coordx");
    coordx->field_name=strdup("NodeCoordX");
    nablaVariable *coordy= nablaVariableNew(nabla);
    nablaVariableAdd(nabla, coordy);
    coordy->axl_it=true;
    coordy->item=strdup("node");
    coordy->type=strdup("real");
    coordy->name=strdup("coordy");
    coordy->field_name=strdup("NodeCoordY");
    nablaVariable *coordz= nablaVariableNew(nabla);
    nablaVariableAdd(nabla, coordz);
    coordz->axl_it=true;
    coordz->item=strdup("node");
    coordz->type=strdup("real");
    coordz->name=strdup("coordz");
    coordz->field_name=strdup("NodeCoordZ");
  }else{
    dbg("\n\t[nablaMiddlendVariableGlobalAdd] Adding AoS variables Real3 coord");
    nablaVariable *coord = nablaVariableNew(nabla);
    nablaVariableAdd(nabla, coord);
    coord->axl_it=true;
    coord->item=strdup("node");
    coord->type=strdup("real3");
    coord->name=strdup("coord");
    coord->field_name=strdup("NodeCoord");
  }
}


/*****************************************************************************
 * nablaMiddlendInit
 *****************************************************************************/
static nablaMain *nablaMiddlendInit(const char *nabla_entity_name){
  nablaMain *nabla=(nablaMain*)calloc(1,sizeof(nablaMain));
  nablaEntity *entity; 
  nabla->name=strdup(nabla_entity_name);
  dbg("\n\t[nablaMiddlendInit] setting nabla->name to '%s'", nabla->name);
  dbg("\n\t[nablaMiddlendInit] Création de notre premier entity");
  entity=nablaEntityNew(nabla);
  dbg("\n\t[nablaMiddlendInit] Rajout du 'main'");
  nablaEntityAddEntity(nabla, entity);
  dbg("\n\t[nablaMiddlendInit] Rajout du nom de l'entity '%s'", nabla_entity_name);  
  entity->name=strdup(nabla_entity_name);
  entity->name_upcase=toolStrUpCase(nabla_entity_name);  // On lui rajoute son nom
  dbg("\n\t[nablaMiddlendInit] Rajout du name_upcase de l'entity %s", entity->name_upcase);  
  entity->main=nabla;                        // On l'ancre à l'unique entity pour l'instant
  assert(nabla->name != NULL);
  dbg("\n\t[nablaMiddlendInit] Returning nabla");
  return nabla;
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
nablaJob* nablaEntryPointsSort(nablaMain *nabla){
  int i,j,number_of_entry_points=0;
  nablaJob *job, *entry_points;
  
  number_of_entry_points=nablaNumberOfEntryPoints(nabla);
  dbg("\n[nablaEntryPointsSort] found %d entry-points", number_of_entry_points);

  // On prépare le plan de travail de l'ensemble des entry_points
  entry_points=(nablaJob *)calloc(number_of_entry_points, sizeof(nablaJob));

  // On re scan pour remplir les duplicats
  for(i=0,job=nabla->entity->jobs;job!=NULL;job=job->next){
    if (!job->is_an_entry_point) continue;
    for(j=0;j<job->whenx;++j){
      dbg("\n\t[nablaEntryPointsSort] dumping #%d: %s @ %f", i, job->name, job->whens[j]);
      entry_points[i].item=job->item;
      entry_points[i].is_an_entry_point=true;
      entry_points[i].is_a_function=job->is_a_function;
      //assert(job->type!=NULL);
      entry_points[i].item=job->item;
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
      entry_points[i].min_assignment=job->min_assignment;
      ++i;
    }
  }

  // On trie afin d'avoir tous les points d'entrée
  qsort(entry_points,number_of_entry_points,sizeof(nablaJob),nablaComparEntryPoints);

  if (nabla->optionDumpTree)
    timeTreeSave(nabla, entry_points, i);
  
  return entry_points;
}


/// ***************************************************************************
// * nablaInsertSpace
// ****************************************************************************
void nablaInsertSpace( nablaMain *nabla, astNode * n){
  if (n->token!=NULL) {
    if (n->parent!=NULL){
      if (n->parent->rule!=NULL){
        if ( (n->parent->ruleid==rulenameToId("type_qualifier")) ||
             (n->parent->ruleid==rulenameToId("type_specifier")) ||
             (n->parent->ruleid==rulenameToId("jump_statement")) ||
             (n->parent->ruleid==rulenameToId("selection_statement")) ||
             (n->parent->ruleid==rulenameToId("storage_class_specifier"))
             ){
          nprintf(nabla, NULL, " ");
          //nprintf(nabla, NULL, "/*%s*/ ",n->parent->rule);
        }else{
          //nprintf(nabla, NULL, "/*%s*/",n->parent->rule);
        }
      }
    }
  }
}


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


static void nablaLibrariesSwitch(astNode * n, nablaEntity *entity){
  if (strncmp(n->children->token,"ℵ",3)==0){
    dbg("\n\t[nablaLibraries] ALEPH single_library hit!");
    entity->libraries|=(1<<aleph);
    return;
  }

  switch(n->children->token[2]){
    
  case ('e'):{ // AL[E]PH || ℵ -(p3c)-> Al[e]
    dbg("\n\t[nablaLibraries] ALEPH single_library hit!");
    entity->libraries|=(1<<aleph);
    break;
  }
    
  case ('i'):{ // MP[I] || MA[I]L
    switch (n->children->token[1]){
    case ('p'):{ // M[P]I
      dbg("\n\t[nablaLibraries] MPI single_library hit!");
      entity->libraries|=(1<<mpi);
      break;
    }
    case ('a'):{ // M[A]IL
      dbg("\n\t[nablaLibraries] MAIL single_library hit!");
      entity->libraries|=(1<<mail);
      break;
    }
    default: error(!0,0,"Could not switch M[p]i || M[a]il!");
    }
    break;
  }
      
  case ('p'):{ // GM[P]
    dbg("\n\t[nablaLibraries] GMP single_library hit!");
    entity->libraries|=(1<<gmp);
    break;
  }
      
  case ('r'):{ // CA[R]TESIAN || PA[R]TICLES
    switch (n->children->token[0]){
    case ('c'):{
      dbg("\n\t[nablaLibraries] CARTESIAN single_library hit!");
      entity->libraries|=(1<<cartesian);
      break;
    }
    case ('p'):{
      dbg("\n\t[nablaLibraries] PARTICLES single_library hit!");
      entity->libraries|=(1<<particles);
      break;
    }
    default: error(!0,0,"Could not switch CARTESIAN || PARTICLES!");
    }
    break;
  }
      
  case ('t'):{ // Ma[t]erials || Ma[t]hematica || df[t]
    if (n->children->token[0]=='d'){
      dbg("\n\t[nablaLibraries] DFT single_library hit!");
      entity->libraries|=(1<<dft);
      break;
    }
    switch (n->children->token[3]){
    case ('e'):{
      dbg("\n\t[nablaLibraries] MATERIALS single_library hit!");
      entity->libraries|=(1<<materials);
      break;
    }
    case('h'):{
      dbg("\n\t[nablaLibraries] MATHEMATICA single_library hit!");
      entity->libraries|=(1<<mathematica);
      break;
    }
    default: error(!0,0,"Could not switch Ma[t]erials || Ma[t]hematica!");
    }
    break;
  }
      
  case ('u'):{ // SL[U]RM
    dbg("\n\t[nablaLibraries] SLURM single_library hit!");
    entity->libraries|=(1<<slurm);
    break;
  }
 
  default:{
    dbg("\n\t[nablaLibraries] single_library token=%s",n->children->token);
    error(!0,0,"Could not find library!");
  }
  }
}
/*****************************************************************************
 * DFS scan for libraries
 *****************************************************************************/
void nablaLibraries(astNode * n, nablaEntity *entity){
  if (n->ruleid == rulenameToId("single_library"))
    nablaLibrariesSwitch(n,entity);
  if(n->children != NULL) nablaLibraries(n->children,  entity);
  if(n->next != NULL) nablaLibraries(n->next, entity);
}


/*****************************************************************************
 * nablaMiddlendSwitch
 *****************************************************************************/
int nablaMiddlendSwitch(astNode *root,
                        const bool optionDumpTree,
                        //const char *input,
                        const char *nabla_entity_name,
                        const BACKEND_SWITCH backend,
                        const BACKEND_COLORS colors,
                        char *interface_name,
                        char *interface_path,
                        char *service_name){
  nablaMain *nabla=nablaMiddlendInit(nabla_entity_name);
  dbg("\n\t[nablaMiddlendSwitch] On initialise le type de backend (= 0x%x) et de ses variantes (= 0x%x)",backend,colors);
  nabla->backend=backend;
  nabla->colors=colors;
  nabla->interface_name=interface_name;
  nabla->interface_path=interface_path;
  nabla->service_name=service_name;
  nabla->optionDumpTree=optionDumpTree;
  nabla->simd=NULL;
  nabla->parallel=NULL;  
  dbg("\n\t[nablaMiddlendSwitch] On rajoute les variables globales");
  nablaMiddlendVariableGlobalAdd(nabla);
  dbg("\n\t[nablaMiddlendSwitch] Now switching...");
  switch (backend){
  case BACKEND_ARCANE: return nccArcane(nabla,root,nabla_entity_name);
  case BACKEND_CUDA:   return nccCuda  (nabla,root,nabla_entity_name);
  case BACKEND_OKINA:  return nccOkina (nabla,root,nabla_entity_name);
  default:
    exit(NABLA_ERROR|fprintf(stderr, "\n[nablaMiddlendSwitch] Error while switching backend!\n"));
  }
  return NABLA_ERROR;
}
