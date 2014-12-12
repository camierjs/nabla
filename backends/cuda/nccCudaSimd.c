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


char *nccCudaBits(void){return "Not relevant here";}


// ****************************************************************************
// * Gather
// ****************************************************************************
char* nccCudaGather(nablaJob* job, nablaVariable* var, enum_phase phase){
  return "";
}


// ****************************************************************************
// * Scatter
// ****************************************************************************
char* nccCudaScatter(nablaVariable* var){
  return "";
}


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* nccCudaPrevCell(void){
  return "";//";
}

// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* nccCudaNextCell(void){
  return "";//";
}

char* nccCudaIncludes(void){return "";}


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
  {"real3","Real3"},
  {"Real","double"},
  {"real","double"},
  {"ReduceMinToDouble(what)","what"},
  // That's for now the way we talk to the host
  {"cuda_exit(what)","*global_deltat=-1.0"},
  {"norm","fabs"},
  {"rabs","fabs"},
  {"rsqrt","sqrt"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opScaMul(a,b)","dot(a,b)"},
  {"opVecMul(a,b)","cross(a,b)"},
  {"opTernary(cond,ifStatement,elseStatement)","(cond)?ifStatement:elseStatement"},
  {"knAt(a)",""},
  {"fatal(a,b)","cudaThreadExit()"},
  {"synchronize(a)",""},
  {"reducemin(a)","0.0"},
  {"mpi_reduce(how,what)","what"},//"mpi_reduce_min(global_min_array,what)"},
  {"xyz","int"},
  {"GlobalIteration", "*global_iteration"},
  {"PAD_DIV(nbytes, align)", "(((nbytes)+(align)-1)/(align))"},
  //{"exit", "cuda_exit(global_deltat)"},
  {NULL,NULL}
};

// ****************************************************************************
// * Std or Mic FORWARDS
// ****************************************************************************
char* cudaForwards[]={
  "inline void info(){}",
  "inline void debug(){}",
  "void gpuEnum(void);",
  NULL
};
