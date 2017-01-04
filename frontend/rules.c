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


// ****************************************************************************
// * Rules still used directly by Nabla
// ****************************************************************************
#define NB_USED_RULENAMES 44
const static char* usedRuleNames[NB_USED_RULENAMES]={
  "argument_expression_list",
  "assignment_expression",
  "assignment_operator",
  "at_constant",
  "compound_statement",
  "declaration",
  "direct_declarator",
  "expression",
  "function_definition",
  "is_test",
  "jump_statement",
  "nabla_direct_declarator",
  "nabla_item",
  "nabla_item_definition",
  "nabla_item_declaration",
  "nabla_items",
  "nabla_job_definition",
  "nabla_options_definition",
  "nabla_option_declaration",
  "nabla_parameter_declaration",
  "nabla_parameter_list",
  "nabla_reduction",
  "nabla_region",
  "nabla_scope",
  "nabla_system",
  "nabla_xyz_declaration",
  "nabla_xyz_direction",
  "parameter_declaration",
  "parameter_type_list",
  "postfix_expression",
  "preproc",
  "primary_expression",
  "selection_statement",
  "single_library",
  "storage_class_specifier",
  "type_qualifier",
  "type_specifier",
  "unary_expression",
  "with_library",
  "power_dimensions",
  "power_dimension",
  "power_function",
  "power_args",
  "forall_switch"
};
static int usedRuleNamesId[NB_USED_RULENAMES];


void iniUsedRuleNames(void){
  const int nb_used_rulename=NB_USED_RULENAMES;
  dbg("\n\t[iniUsedRuleNames] iniUsedRuleNames=%d",nb_used_rulename);
  for(int i=0;i<nb_used_rulename;i+=1){
    const int ruleid=rulenameToId(usedRuleNames[i]);
    dbg("\n\t[iniUsedRuleNames] #%d %s %d",i,usedRuleNames[i],ruleid);
    usedRuleNamesId[i]=ruleid;
  }
}


const int ruleToId(const int rule){
  assert(rule<NB_USED_RULENAMES);
  return usedRuleNamesId[rule]; 
}
