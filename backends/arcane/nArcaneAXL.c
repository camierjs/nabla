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


/***************************************************************************** 
 * On prépare l'axl du main
 *****************************************************************************/
NABLA_STATUS nccAxlGenerateHeader(nablaMain *arc){
  fprintf(arc->axl,"\
<?xml version=\"1.0\" encoding=\"ISO-8859-1\" ?>\
\n<%s name=\"%s%s\" %s>\
\n\t<description>Descripteur de %s</description>",
          isAnArcaneModule(arc)?"module":"service",
          arc->entity->name,
          isAnArcaneService(arc)?"Service":"",
          isAnArcaneService(arc)?"type=\"caseoption\"":"version=\"1.0\"",
          arc->entity->name);
  if (isAnArcaneService(arc)){
    if (arc->interface_name)
      fprintf(arc->axl,"\n\t<interface name=\"%s\"/>",arc->interface_name);
    else
      return NABLA_ERROR;
  }
  return NABLA_OK;
}

/***************************************************************************** 
 * Dump d'une variables dans le fichier AXL
 *****************************************************************************/
static NABLA_STATUS nccAxlGenerateVariable(nablaMain *arc, nablaVariable *var){
  int typeDumped=0;
  if (var->item==NULL) return NABLA_ERROR;
  if (var->name==NULL) return NABLA_ERROR;
  if (var->type==NULL) return NABLA_ERROR;
  
  // On ne dump pas les variable mpz_t dans l'axl depuis qu'on passe
  // par la structure MultiArray2VariableRef m_gmp_backups
  if (var->gmpRank!=-1) return NABLA_OK;

  // On ne dump pas non plus les item particles
  if (var->item[0]=='p') return NABLA_OK;
  
  if (!var->axl_it) return NABLA_OK;
  //if (var->item[0]=='g') return NABLA_OK; // On ne veut pas dans le fichier AXL des globales
  
  fprintf(arc->axl,"\n\t\t\t<!-- %s %s -->",var->item, var->name);
  // Be carefull: Variables field-name and name are swaped
  fprintf(arc->axl,"\n\t\t\t\t<variable\n\t\t\tfield-name=\"%s_%s\"", var->item, var->name);
  if (var->field_name!=NULL)
    fprintf(arc->axl,"\n\t\t\t\tname=\"%s\"", var->field_name);
  else
    fprintf(arc->axl,"\n\t\t\t\tname=\"%s_%s\"", var->item, var->name);
  if (var->type[0]=='u')
    typeDumped+=fprintf(arc->axl,"\n\t\t\t\tdata-type=\"int64\""); // UniqueId
  // gmp mpInteger to byte
  if (strcmp(var->type, "mpinteger")==0)
    typeDumped+=fprintf(arc->axl,"\n\t\t\t\tdata-type=\"integer\"");
  if (typeDumped==0)
    fprintf(arc->axl,"\n\t\t\t\tdata-type=\"%s\"", var->type);
  fprintf(arc->axl,"\n\t\t\t\titem-kind=\"%s\"", (var->item[0]!='g')?var->item:"none");
  fprintf(arc->axl,"\n\t\t\t\tdim=\"%d\" dump=\"%s\" need-sync=\"true\" />",
          var->dim,
          var->dump?"true":"false");
  return NABLA_OK;
}


/***************************************************************************** 
 * Dump des options dans le fichier AXL
 * optional, multiple & mandatory
 *****************************************************************************/
static NABLA_STATUS nccAxlGenerateOption(nablaMain *arc, nablaOption *opt){
  if (opt->name==NULL) return NABLA_ERROR;
  if (opt->type==NULL) return NABLA_ERROR;
  fprintf(arc->axl,"\n\n\n\t\t\t<!-- - - - - - %s - - - - - -->",opt->name);
  fprintf(arc->axl,"\n\t\t\t<simple name=\"%s\" type=\"%s\"",opt->name,opt->type);
  if (opt->dflt!=NULL) fprintf(arc->axl," default=\"%s\"",opt->dflt);
  fprintf(arc->axl,">");
  fprintf(arc->axl,"\n\t\t\t\t<userclass>User</userclass>");
  fprintf(arc->axl,"\n\t\t\t\t<description></description>");
  fprintf(arc->axl,"\n\t\t\t</simple>");
  return NABLA_OK;
}


// *****************************************************************************
// * nccAxlGeneratorEntryPointWhere
// *****************************************************************************
static char *nccAxlGeneratorEntryPointWhere(double when){ 
  // 'build' appel du point d'entrée avant l'initialisation. Le jeu de données n'est pas encore lu.
  // Ce point d'entrée sert généralement à construire certains objets utiles au entity mais est peu utilisé par les entity numériques.
  if (when == ENTRY_POINT_build) return "build";

  // 'init' sert à initialiser les structures de données du entity qui ne sont pas conservées lors d'une protection.
  // A ce stade de l'initialisation, le jeu de données et le maillage ont déjà été lus.
  // L'initialisation sert également à vérifier certaines valeurs, calculer des valeurs initiales... 
  if (when == ENTRY_POINT_init) return "init";
  
  // 'start-init' sert à initialiser les variables et les valeurs uniquement lors du démarrage du cas (t=0)
  //if (when < ENTRY_POINT_start_init) return "start-init";
  if (when < ENTRY_POINT_start_init) return "init";
  
  // 'continue-init' sert à initialiser des structures spécifiques au mode reprise.
  if (when == ENTRY_POINT_continue_init) return "continue-init";

  // 'compute-loop' appel du point d'entrée tant que l'on itère
  if (when > ENTRY_POINT_compute_loop) return "compute-loop";

  // 'exit' sert, par exemple, à désallouer des structures de données lors de la sortie du code:
  // fin de simulation, arrêt avant reprise... 
  if (when == ENTRY_POINT_exit) return "compute-loop";

  return NULL;
}


// *****************************************************************************
// * nccAxlGeneratorEntryPointWhenName
// *****************************************************************************
static char* nccAxlGeneratorEntryPointWhenName(double when){
  char name[18+1];
  const register unsigned long *adrs = (unsigned long*)&when;
  snprintf(name,18,"0x%lx",*adrs);
  //dbg("\n\t\t[nccAxlGeneratorEntryPointWhenName] when=%f => %p",when, *adrs);
  dbg("\n\t\t[nccAxlGeneratorEntryPointWhenName] ENTRY_POINT_build=%f ENTRY_POINT_init=%f", ENTRY_POINT_build, ENTRY_POINT_init);
  return strdup(name);
}

  
/***************************************************************************** 
 * Dump de la structure des Variables et des EntryPoints dans le fichier AXL
 *****************************************************************************/
NABLA_STATUS nccAxlGenerator(nablaMain *arc){
  nablaVariable *var;
  nablaOption *opt;
  nablaJob *entry_point;
  int i,number_of_entry_points;
  bool is_into_compute_loop=false;
  
  dbg("\n\n[nccAxlGenerator]");
  fprintf(arc->axl,"\n\t\t<variables>");
  for(var=arc->variables;var!=NULL;var=var->next)
    if (nccAxlGenerateVariable(arc, var)!=NABLA_OK) return NABLA_ERROR;
  fprintf(arc->axl,"\n\t\t</variables>\n\n\t\t<entry-points>");

  number_of_entry_points=nMiddleNumberOfEntryPoints(arc);
  entry_point=nMiddleEntryPointsSort(arc,number_of_entry_points);

  // Et on rescan afin de dumper
  for(i=0;i<number_of_entry_points;++i){
    if (strcmp(entry_point[i].name,"ComputeLoopEnd")==0)continue;
    if (strcmp(entry_point[i].name,"ComputeLoopBegin")==0)continue;
    const double when=entry_point[i].whens[0];
    const char *where=nccAxlGeneratorEntryPointWhere(when);
    const char *whenName=nccAxlGeneratorEntryPointWhenName(when);
    dbg("\n\t[nccAxlGenerator] sorted #%d: %s @ %f(=%s) in '%s'", i,
        entry_point[i].name, when, whenName, where);
    // Si l'on passe pour la première fois la frontière du zéro, on l'écrit dans le .config
    if (when > ENTRY_POINT_continue_init && is_into_compute_loop==false){
      is_into_compute_loop=true;
      if (isAnArcaneModule(arc)==true)
        fprintf(arc->cfg,"\n\t\t\t</entry-points>\n\n\t\t\t<entry-points where=\"compute-loop\">");
    }
    // On remplit la ligne du fichier CONFIG
    if (isAnArcaneModule(arc)==true)
      fprintf(arc->cfg, "\n\t\t\t\t<entry-point name=\"%s.%s@%f\" />",
              arc->entity->name, entry_point[i].name, when);
    // On remplit la ligne du fichier AXL
    fprintf(arc->axl, "\n\t\t\t<entry-point method-name=\"%s_at_%s\" name=\"%s@%f\"",
            entry_point[i].name, whenName, entry_point[i].name, when);
    
    // Pourrait être le where, mais pas de doublons dans l'axl
    fprintf(arc->axl, " where=\"%s\" property=\"none\" />",where);

    dbg("\n\n[nccAxlGenerator] On remplit la ligne du fichier HDR");
    fprintf(arc->entity->hdr, "\n\tinline void %s_at_%s(){",
            entry_point[i].name, whenName);
    // S'il y a un 'if' après le '@', on le dump maintenant
    if (entry_point[i].ifAfterAt!=NULL){
      dbg("\n\t[nccAxlGenerator] dumpIfAfterAt!");
      fprintf(arc->entity->hdr, "if (");
      nMiddleDumpIfAfterAt(entry_point[i].ifAfterAt, arc);
      fprintf(arc->entity->hdr, ") ");
    }
    fprintf(arc->entity->hdr, "%s();}", entry_point[i].name);
  }
  fprintf(arc->axl,"\n\t\t</entry-points>");

  // Génération des options dans le fichier AXL
  fprintf(arc->axl,"\n\t\t<options>");
  for(opt=arc->options;opt!=NULL;opt=opt->next)
    if (nccAxlGenerateOption(arc, opt)!=NABLA_OK) return NABLA_ERROR;
  fprintf(arc->axl,"\n\t\t</options>");

  
  return NABLA_OK;
}
