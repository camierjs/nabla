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
#ifndef _NABLA_FRONTEND_H_
#define _NABLA_FRONTEND_H_

#include "ast.h"
#include "dbg.h"


// ****************************************************************************
// * Rules still used directly by Nabla
// * Must be sync'ed with rules.c static usedRuleNames array
// ****************************************************************************
enum{
  rule_argument_expression_list=0,
  rule_assignment_expression,
  rule_assignment_operator,
  rule_at_constant,
  rule_compound_statement,
  rule_declaration,
  rule_direct_declarator,
  rule_expression,
  rule_function_definition,
  rule_is_test,
  rule_jump_statement,
  rule_nabla_direct_declarator,
  rule_nabla_item,
  rule_nabla_item_definition,
  rule_nabla_item_declaration,
  rule_nabla_items,
  rule_nabla_job_definition,
  rule_nabla_job_prefix,
  rule_nabla_options_definition,
  rule_nabla_option_declaration,
  rule_nabla_parameter_declaration,
  rule_nabla_parameter_list,
  rule_nabla_reduction,
  rule_nabla_region,
  rule_nabla_nesw,
  rule_nabla_scope,
  rule_nabla_system,
  rule_nabla_xyz_declaration,
  rule_nabla_xyz_direction,
  rule_parameter_declaration,
  rule_parameter_type_list,
  rule_postfix_expression,
  rule_preproc,
  rule_primary_expression,
  rule_selection_statement,
  rule_single_library,
  rule_storage_class_specifier,
  rule_type_qualifier,
  rule_type_specifier,
  rule_unary_expression,
  rule_with_library,
  rule_power_dimensions,
  rule_power_dimension,
  rule_power_function,
  rule_power_args,
  rule_forall_range,
  rule_forall_switch,
  nb_used_rulenames
} used_rulenames;


// ****************************************************************************
// * Structure used to pass a batch of actions for each ruleid found while DFS
// ****************************************************************************
typedef struct RuleActionStruct{
  int ruleid;
  void (*action)(node*,void*);
} RuleAction;


// ****************************************************************************
// * DFS functions
// ****************************************************************************
void dfsDumpToken(node*);

void scanTokensForActions(node*, RuleAction*, void*);
char *dfsFetchFirst(node*, int);
char *dfsFetchAll(node*,const int,int*,char*);
node *dfsFetch(node*,int);
node *dfsHit(node*,int);

node *dfsFetchTokenId(node*,int);
node *dfsFetchToken(node*, const char *);
node *dfsFetchRule(node*,int);
int dfsScanJobsCalls(void*,void*,node*);
void iniUsedRuleNames(void);

#endif // _NABLA_FRONTEND_H_
