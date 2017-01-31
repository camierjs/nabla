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


// ***************************************************************************** 
// * Dump des options
// *****************************************************************************/
static void legionHookVarsOptions(nablaMain *nabla){
  fprintf(nabla->entity->hdr,"\
\nc.printf(\"[33m[pennant_common] Configuration variables.[m\\n\");\
\nconfig_fields_input = terralib.newlist({\n\
  {field = \"bcx\", type = double[2], default_value = `array(0.0, 0.0), linked_field = \"bcx_n\"},\n\
  {field = \"bcx_n\", type = int64, default_value = 0, is_linked_field = true},\n\
  {field = \"bcy\", type = double[2], default_value = `array(0.0, 0.0), linked_field = \"bcy_n\"},\n\
  {field = \"bcy_n\", type = int64, default_value = 0, is_linked_field = true},\n\
  {field = \"meshparams\", type = double[4], default_value = `arrayof(double, 0, 0, 0, 0), linked_field = \"meshparams_n\"},\n\
  {field = \"meshparams_n\", type = int64, default_value = 0, is_linked_field = true},\n\
");
  for(nablaOption *opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->hdr,
            "\n\t{field = \"%s\", type = %s, default_value = %s},",
            opt->name,
            opt->type[0]=='r'?"double":
            opt->type[0]=='i'?"int64":
            opt->type[0]=='b'?"bool":
            "\n#error Unknown Type for option",
            opt->dflt);
  fprintf(nabla->entity->hdr,"\n}) -- end of config_defaults\
\nc.printf(\"[33m[pennant_common] Adding config_fields_all[m\\n\");\
\nconfig_fields_all:insertall(config_fields_input)");
  fprintf(nabla->entity->hdr,"\n\
\nc.printf(\"[33m[pennant_common] Configuring entries[m\\n\");\
\nconfig = terralib.types.newstruct(\"config\")\
\nconfig.entries:insertall(config_fields_all)\n");
}


// ****************************************************************************
// * 
// ****************************************************************************
static char *legionHookVarsType(nablaMain *nabla,
                     char *type){
  if (strncmp(type,"real2",5)==0) return "vec2";
  if (strncmp(type,"real",4)==0) return "double";
  if (strncmp(type,"integer",4)==0) return "uint8";
  return type;
}

// ***************************************************************************** 
// * Dump des variables
// *****************************************************************************
// * Attention read_input de legion_input utilise des offset en dur pour l'init!
// *****************************************************************************
static void legionHookVarsVariable(nablaMain *nabla){
  // Variables aux mailles = cells = zones
  fprintf(nabla->entity->hdr,"\n\n\
\nc.printf(\"[33m[pennant_common] fspace zone[m\\n\");\
\nfspace zone {");
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    if (var->dim!=0) continue;
    const char *type=legionHookVarsType(nabla,var->type);
    fprintf(nabla->entity->hdr,"\n\t%s :\t%s,",var->name,type);
  }
  fprintf(nabla->entity->hdr,"\n}");
  
  // Variables aux noeuds = points
  fprintf(nabla->entity->hdr,"\n\n\
\nc.printf(\"[33m[pennant_common] fspace point[m\\n\");\
\nfspace point {");
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    //if (strncmp(var->name,"coord",5)==0) continue;
    const char *type=legionHookVarsType(nabla,var->type);
    fprintf(nabla->entity->hdr,"\n\t%s :\t%s,",var->name,type);
  }
  fprintf(nabla->entity->hdr,"\n}");
  
  // Variables aux faces = sides
  fprintf(nabla->entity->hdr,"\n\n\
\nc.printf(\"[33m[pennant_common] fspace side[m\\n\");\
\nfspace side (rz : region(zone),\
\n\t\t\t\trpp : region(point),\
\n\t\t\t\trpg : region(point),\
\n\t\t\t\trs : region(side(rz, rpp, rpg, rs))) {");
fprintf(nabla->entity->hdr,"\
\n\tmapsz :  ptr(zone, rz),                      -- maps: side -> zone\
\n\tmapsp1 : ptr(point, rpp, rpg),               -- maps: side -> points 1 and 2\
\n\tmapsp2 : ptr(point, rpp, rpg),\
\n\tmapss3 : ptr(side(rz, rpp, rpg, rs), rs),    -- maps: side -> previous side\
\n\tmapss4 : ptr(side(rz, rpp, rpg, rs), rs),    -- maps: side -> next side");
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]=='c' && var->dim==0) continue;
    if (var->item[0]=='g') continue;
    if (var->item[0]=='n') continue;
    const char *type=legionHookVarsType(nabla,var->type);
    fprintf(nabla->entity->hdr,"\n\t%s :\t%s,",var->name,type);
  }
  fprintf(nabla->entity->hdr,"\n}");
}

// ****************************************************************************
// * Dump des options dans le header
// ****************************************************************************
void legionHookVarsPrefix(nablaMain *nabla){
  fprintf(nabla->entity->hdr, "\n\n-- legionHookVariablesPrefix");
  legionHookVarsOptions(nabla);
  legionHookVarsVariable(nabla);
}


// ****************************************************************************
// * Variables Postfix
// ****************************************************************************
void legionHookVarsFree(nablaMain *nabla){
  fprintf(nabla->entity->hdr, "\n-- legionHookVariablesFree");
}
