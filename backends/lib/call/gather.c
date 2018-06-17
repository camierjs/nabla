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
// * Gather for Cells
// ****************************************************************************
static char* xCallGatherCells(nablaJob *job,
                              nablaVariable* var){
  const bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  const bool dim2D = (job->entity->libraries&(1<<with_real2))!=0;
  const bool is_int_t = strcmp(var->type,"integer")==0;
  const bool is_real_t = strcmp(var->type,"real")==0;
  const bool is_bool_t = strcmp(var->type,"bool")==0;
  const bool is_real3x3_t = strcmp(var->type,"real3x3")==0;
  char gather[1024];
  const bool has_non_null_koffset = var->koffset!=0;
  char str_pip_koffset[64];
  char str_uds_koffset[64];
  char iterator[2]={job->parse.enum_enum,0};
  
  if (var->item[0]=='n')
    sprintf(str_pip_koffset,"((n+NABLA_NODE_PER_CELL+(%d))%%%%NABLA_NODE_PER_CELL)",var->koffset);
  
  if (var->item[0]=='f')
    sprintf(str_pip_koffset,"((%s+4+(%d))%%4)",iterator,var->koffset);
  
  sprintf(str_uds_koffset,"_%02x",var->koffset);
  
  if (var->item[0]=='n')
    snprintf(gather, 1024, "\
/*const %s*/ %s gathered_%s_%s%s=rgather%sk(xs_cell_node[%s*NABLA_NB_CELLS+(c<<WARP_BIT)],%s_%s%s);\n\t\t\t",
             var->type,
             is_bool_t?"bool":is_int_t?"int":is_real_t?"real":is_real3x3_t?"real3x3":dim2D?"real2":dim1D?"real":"real3",
             var->item,
             var->name,
             has_non_null_koffset ? str_uds_koffset:"",             
             is_int_t?"N":is_real_t?"":dim1D?"":is_real3x3_t?"3x3":"3",
             has_non_null_koffset ? str_pip_koffset:iterator,
             var->item,
             var->name,
             strcmp(var->type,"real")==0?"":"");
  
  if (var->item[0]=='f')
    snprintf(gather, 1024, "\
/*const %s*/ %s gathered_%s_%s%s=rgather%sk(xs_cell_face[%s*NABLA_NB_CELLS+(c<<WARP_BIT)],%s_%s%s);\n\t\t\t",
             var->type,
             is_bool_t?"bool":is_int_t?"int":is_real_t?"real":is_real3x3_t?"real3x3":dim2D?"real2":dim1D?"real":"real3",
             var->item,
             var->name,
             has_non_null_koffset ? str_uds_koffset:"",             
             is_bool_t?"B":is_int_t?"N":is_real_t?"":is_real3x3_t?"3x3":"3",
             has_non_null_koffset ? str_pip_koffset:iterator,
             var->item,
             var->name,
             strcmp(var->type,"real")==0?"":"");
  
  return sdup(gather);
}


// ****************************************************************************
// * Gather for Nodes
// * En STD, le gather aux nodes est le même qu'aux cells
// ****************************************************************************
static char* xCallGatherNodes(nablaJob *job,
                              nablaVariable* var){
  const bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  const bool dim2D = (job->entity->libraries&(1<<with_real2))!=0;
  const bool is_int_t = strcmp(var->type,"integer")==0;
  const bool is_real_t = strcmp(var->type,"real")==0;
  const bool is_bool_t = strcmp(var->type,"bool")==0;
  const bool is_real3x3_t = strcmp(var->type,"real3x3")==0;
  char gather[1024];
  snprintf(gather, 1024, "\
%s gathered_%s_%s=rGatherAndZeroNegOnes(xs_node_cell[NABLA_NODE_PER_CELL*(n<<WARP_BIT)+c],%s %s_%s);\n\t\t\t",
           is_bool_t?"bool":is_int_t?"int":is_real_t?"real":is_real3x3_t?"real3x3":dim2D?"real2":dim1D?"real":"real3",
           var->item,
           var->name,
           var->dim==0?"":"xs_node_cell_corner[NABLA_NODE_PER_CELL*(n<<WARP_BIT)+c],",
           var->item,
           var->name);
  return sdup(gather);
}


// ****************************************************************************
// * Gather for Faces
// ****************************************************************************
static char* xCallGatherFaces(nablaJob *job,
                              nablaVariable* var){
  const bool dim1D = (job->entity->libraries&(1<<with_real))!=0;
  const bool dim2D = (job->entity->libraries&(1<<with_real2))!=0;
  const bool is_int_t = strcmp(var->type,"integer")==0;
  const bool is_real_t = strcmp(var->type,"real")==0;
  const bool is_bool_t = strcmp(var->type,"bool")==0;
  const bool is_real3x3_t = strcmp(var->type,"real3x3")==0;
  char gather[1024];
  
  if (var->item[0]=='n')
    snprintf(gather, 1024, "\
%s gathered_%s_%s=rGatherAndZeroNegOnes(xs_face_node[NABLA_NB_FACES*(n<<WARP_BIT)+f],%s %s_%s);\n\t\t\t",
             is_bool_t?"bool":is_int_t?"int":is_real_t?"real":is_real3x3_t?"real3x3":dim2D?"real2":dim1D?"real":"real3",
             var->item,
             var->name,
             var->dim==0?"":"xs_node_cell_corner[NABLA_NODE_PER_CELL*(n<<WARP_BIT)+f],",
             var->item, var->name);
  
  if (var->item[0]=='c')
    snprintf(gather, 1024, "\
%s gathered_%s_%s=rGatherAndZeroNegOnes(xs_face_cell[NABLA_NB_FACES*(c<<WARP_BIT)+f],%s %s_%s);\n\t\t\t",
             is_bool_t?"bool":is_int_t?"int":is_real_t?"real":is_real3x3_t?"real3x3":dim2D?"real2":dim1D?"real":"real3",
             var->item,
             var->name,
             var->dim==0?"":"/*xCallGatherFaces and var->dim==1*/",
             var->item, var->name);
  return sdup(gather);
}


// ****************************************************************************
// * Gather switch
// ****************************************************************************
char* xCallGather(nablaJob *job,nablaVariable* var){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return xCallGatherCells(job,var);
  if (itm=='n') return xCallGatherNodes(job,var);
  if (itm=='f') return xCallGatherFaces(job,var);
  nablaError("Could not distinguish job item in xGather for job '%s'!", job->name);
  return NULL;
}


// ****************************************************************************
// * 
// ****************************************************************************
static void xCallEnumEnumContinues(node *n,nablaJob *job){
  const nablaMain *nabla = job->entity->main;
  //dbg("\n\t\t\t\t[xCallEnumEnumContinues]");
  assert(job->enum_enum_node);
  node *forall_range = job->enum_enum_node;
  const node *nesw = dfsHit(forall_range,ruleToId(rule_nabla_nesw));
  const node *region = dfsHit(forall_range->children,ruleToId(rule_nabla_region));
  
  if (nesw && nesw->children->tokenid==NORTH)
    nprintf(nabla,NULL,"if (f!=2/*NORTH*/) continue;");
  if (nesw && nesw->children->tokenid==EAST)
    nprintf(nabla,NULL,"if (f!=1/*EAST*/) continue;");
  if (nesw && nesw->children->tokenid==SOUTH)
    nprintf(nabla,NULL,"if (f!=0/*SOUTH*/) continue;");
  if (nesw && nesw->children->tokenid==WEST)
    nprintf(nabla,NULL,"if (f!=3/*WEST*/) continue;");
  
  if (region && region->children->tokenid==OUTER)
    nprintf(nabla,NULL,"\n\t\t\tconst int fid=xs_cell_face[f*NABLA_NB_CELLS+(c<<WARP_BIT)];\
\n\t\t\tif (fid<msh->NABLA_NB_FACES_INNER) continue;\n\t\t\t");

}


// ****************************************************************************
// * Filtrage du GATHER
// ****************************************************************************
// * Une passe devrait être faite à priori afin de déterminer les contextes
// * d'utilisation: au sein d'un forall, postfixed ou pas, etc.
// * Et non pas que sur leurs déclarations en in et out
// ****************************************************************************
// * Idem, si on a différents ∀, pour l'instant ils gatherent trop!
// ****************************************************************************
char* xCallFilterGather(node *n,nablaJob *job){
  if (n!=NULL && job->enum_enum_node){
    // n==NULL quand on vient de xHookForAllPostfix, ce qu'on veut pas
    dbg("\n\t\t\t\t[xFilterGather] Launching xCallEnumEnumContinues");
    xCallEnumEnumContinues(n,job);
  }
  
  dbg("\n\t\t\t\t[xFilterGather]");
  char *gather_src_buffer=NULL;  
  if ((gather_src_buffer=calloc(NABLA_MAX_FILE_NAME,sizeof(char)))==NULL)
    nablaError("[xFilterGather] Could not calloc our gather_src_buffer!");

  dbg("\n\t\t\t\t[xFilterGather] Looping on all used_variables of this job:");
  for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
    dbg("\n\t\t\t\t\t[xFilterGather] variable: '%s-%d'", var->name, var->koffset);
    if (!var->is_gathered){
      dbg("\n\t\t\t\t\t[xFilterGather] not gathered");
      continue;
    }
    nprintf(job->entity->main, NULL, "/* '%s:k(%d)' is gathered",var->name,var->koffset);
    if (!dfsUsedInThisForallKoffset(job->entity->main,job,n,var->name,var->koffset)){
      nprintf(job->entity->main, NULL, " but NOT used InThisForall! */");
      continue;
    }
    nprintf(job->entity->main, NULL, " and IS used InThisForall! */");
    dbg("\n\t\t\t\t[xFilterGather] strcat");
    if (job->entity->main->call){
      strcat(gather_src_buffer,
             job->entity->main->call->simd->gather(job,var));
    }else
      strcat(gather_src_buffer,xCallGather(job,var));
  }
  dbg("\n\t\t\t\t[xFilterGather] gather_src_buffer='%s'",
      gather_src_buffer?gather_src_buffer:"NULL");
  dbg("\n\t\t\t\t[xFilterGather] done");
  char *rtn=sdup(gather_src_buffer);
  free(gather_src_buffer);
  return rtn;
}

