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

char* nccCudaIncludes(void){return "";}

char *nccCudaBits(void){return "Not relevant here";}


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* nccCudaPrevCell(void){
  return "gatherk_and_zero_neg_ones(cell_prev[direction*NABLA_NB_CELLS+tcid],";
}

// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* nccCudaNextCell(void){
  return "gatherk_and_zero_neg_ones(cell_next[direction*NABLA_NB_CELLS+tcid],";
}



// ****************************************************************************
// * Gather for Cells
// ****************************************************************************
static char* cudaGatherCells(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration) return "";
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
gather%sk(cell_node[n*NABLA_NB_CELLS+tcid],\n\t\t\t\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":"real3",
           strcmp(var->type,"real")==0?"":"3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item,
           var->name);
  return strdup(gather);
}

// ****************************************************************************
// * Gather for Nodes
// ****************************************************************************
static char* cudaGatherNodes(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration) return "";
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
//#warning continue node_cell_corner\n\
//if (node_cell_corner[8*tnid+i]==-1) continue;\n\
gatherFromNode_%sk%s(node_cell[8*tnid+i],\n\
%s\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":"real3",
           strcmp(var->type,"real")==0?"":"3",
           var->dim==0?"":"Array8",
           var->dim==0?"":"\t\t\t\t\t\tnode_cell_corner[8*tnid+i],\n\t\t\t",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item,
           var->name);
  return strdup(gather);
}


// ****************************************************************************
// * Gather switch
// ****************************************************************************
char* nccCudaGather(nablaJob *job,nablaVariable* var, enum_phase phase){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return cudaGatherCells(job,var,phase);
  if (itm=='n') return cudaGatherNodes(job,var,phase);
  error(!0,0,"Could not distinguish job item in okinaStdGather!");
  return NULL;
}


// ****************************************************************************
// * Scatter
// ****************************************************************************
char* nccCudaScatter(nablaVariable* var){
  char scatter[1024];
  snprintf(scatter, 1024, "\tscatter%sk(ia, &gathered_%s_%s, %s_%s);",
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           var->item, var->name);
  return strdup(scatter);
}


// ****************************************************************************
// * CUDA TYPEDEFS
// ****************************************************************************
nablaTypedef cudaTypedef[]={
  {"int","integer"},
  {NULL,NULL}
};



// ****************************************************************************
// * CUDA DEFINES
// ****************************************************************************
nablaDefine cudaDefines[]={
  {"Real3","real3"},
  {"Real","real"},
  //{"real","double"},
  {"ReduceMinToDouble(what)","reduce_min_kernel(global_device_shared_reduce_results,what)"},
  {"norm","fabs"},
  {"rabs","fabs"},
  {"square_root","sqrt"},
  {"cube_root","cbrt"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opScaMul(a,b)","dot3(a,b)"},
  {"opVecMul(a,b)","cross3(a,b)"},
  //{"opTernary(cond,ifStatement,elseStatement)","((cond)?ifStatement:elseStatement)"},
  {"knAt(a)",""},
  {"fatal(a,b)","cudaThreadExit()"},
  {"synchronize(a)",""},
  {"reduce(how,what)","what"},
  {"xyz","int"},
  {"GlobalIteration", "*global_iteration"},
  {"PAD_DIV(nbytes, align)", "(((nbytes)+(align)-1)/(align))"},
  {NULL,NULL}
};

// ****************************************************************************
// * Std or Mic FORWARDS
// ****************************************************************************
char* cudaForwards[]={
  "void gpuEnum(void);",
  NULL
};
