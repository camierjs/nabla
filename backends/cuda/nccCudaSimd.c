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
