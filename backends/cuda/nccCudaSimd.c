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
  return "";//";
}

// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* nccCudaNextCell(void){
  return "";//";
}


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
  {"norm","fabs"},
  {"rabs","fabs"},
  {"rsqrt","sqrt"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opScaMul(a,b)","dot(a,b)"},
  {"opVecMul(a,b)","cross(a,b)"},
  {"opTernary(cond,ifStatement,elseStatement)","((cond)?ifStatement:elseStatement)"},
  {"knAt(a)",""},
  {"fatal(a,b)","cudaThreadExit()"},
  {"synchronize(a)",""},
  {"mpi_reduce(how,what)","cuda_reduce_min(global_min_array,what)"},
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
