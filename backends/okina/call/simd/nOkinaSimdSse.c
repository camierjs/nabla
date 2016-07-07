///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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

char* nOkinaSseIncludes(void){return "#include <immintrin.h>\n";}
char *nOkinaSseBits(void){return "128";}


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* nOkinaSsePrevCell(int direction){
  if (direction==DIR_X)
    return "gatherk_and_zero_neg_ones(\n\
			xs_cell_prev[MD_DirX*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			xs_cell_prev[MD_DirX*NABLA_NB_CELLS+(c<<WARP_BIT)+1],";
  if (direction==DIR_Y)
    return "gatherk_and_zero_neg_ones(\n\
			xs_cell_prev[MD_DirY*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			xs_cell_prev[MD_DirY*NABLA_NB_CELLS+(c<<WARP_BIT)+1],";
  if (direction==DIR_Z)
    return "gatherk_and_zero_neg_ones(\n\
			xs_cell_prev[MD_DirZ*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			xs_cell_prev[MD_DirZ*NABLA_NB_CELLS+(c<<WARP_BIT)+1],";
  assert(NULL);
  return NULL;
}


// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* nOkinaSseNextCell(int direction){
  if (direction==DIR_X)
    return "gatherk_and_zero_neg_ones(\n\
			xs_cell_next[MD_DirX*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			xs_cell_next[MD_DirX*NABLA_NB_CELLS+(c<<WARP_BIT)+1],";
  if (direction==DIR_Y)
    return "gatherk_and_zero_neg_ones(\n\
			xs_cell_next[MD_DirY*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			xs_cell_next[MD_DirY*NABLA_NB_CELLS+(c<<WARP_BIT)+1],";
  if (direction==DIR_Z)
    return "gatherk_and_zero_neg_ones(\n\
			xs_cell_next[MD_DirZ*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			xs_cell_next[MD_DirZ*NABLA_NB_CELLS+(c<<WARP_BIT)+1],";
  assert(NULL);
  return NULL;
}


// ****************************************************************************
// * Gather pour un job sur les cells
// ****************************************************************************
static char* nOkinaSseGatherCells(nablaJob *job,nablaVariable* var){ 
  char gather[1024];
  snprintf(gather,
           1024,
           "int __attribute__((unused)) cw,ia,ib;\n\t\t\t%s gathered_%s_%s;\n\t\t\t\
cw=(c<<WARP_BIT);\n\t\t\t\
gather%sk(ia=xs_cell_node[n*NABLA_NB_CELLS+cw+0],\n\t\t\t\
         ib=xs_cell_node[n*NABLA_NB_CELLS+cw+1],\n\t\t\t\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
             strcmp(var->type,"real")==0?"real":"real3",
             var->item, var->name,
             strcmp(var->type,"real")==0?"":"3",
             var->item, var->name,
             strcmp(var->type,"real")==0?"":"",
             var->item, var->name);
  return sdup(gather);
}


// ****************************************************************************
// * Gather pour un job sur les nodes
// * En SSE, le gather aux nodes N'est PAS le même qu'aux cells
// ****************************************************************************
static char* nOkinaSseGatherNodes(nablaJob *job,nablaVariable* var){ 
  char gather[1024];
  snprintf(gather, 1024, "int nw;\n\t\t\t%s gathered_%s_%s;\n\t\t\t\
nw=(n<<WARP_BIT);\n\t\t\t\
gatherFromNode_%sk%s(xs_node_cell[8*nw+c],\n\t\t\t\
%s\
         xs_node_cell[8*(nw+1)+c],\n\t\t\t\
%s\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"3",
           var->dim==0?"":"Array8",
           var->dim==0?"":"\t\t\txs_node_cell_corner[8*nw+c],\n\t\t\t",
           var->dim==0?"":"\t\t\txs_node_cell_corner[8*(nw+1)+c],\n\t\t\t",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item,
           var->name);
  return sdup(gather);
}


// ****************************************************************************
// * Gather switch
// ****************************************************************************
char* nOkinaSseGather(nablaJob *job,nablaVariable* var){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return nOkinaSseGatherCells(job,var);
  if (itm=='n') return nOkinaSseGatherNodes(job,var);
  nablaError("Could not distinguish job item in nOkinaSseGather!");
  return NULL;
}


// ****************************************************************************
// * Scatter
// ****************************************************************************
char* nOkinaSseScatter(nablaJob *job,nablaVariable* var){
  char scatter[1024];
  snprintf(scatter, 1024, "\tscatter%sk(ia,ib, &gathered_%s_%s, %s_%s);",
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           var->item, var->name);
  return sdup(scatter);
}

// ****************************************************************************
// * nOkinaSseUid
// ****************************************************************************
char* nOkinaSseUid(nablaMain *nabla, nablaJob *job){
  const char cnfgem=job->item[0];
  if (cnfgem=='c') return "integer(WARP_SIZE*c+0,WARP_SIZE*c+1)";
  if (cnfgem=='n') return "integer(WARP_SIZE*n+0,WARP_SIZE*n+1)";
  assert(false);
  return NULL;
}

// ****************************************************************************
// * Sse or Mic TYPEDEFS
// ****************************************************************************
const nWhatWith nOkinaSseTypedef[]={
  {"struct real3","Real3"},
  {NULL,NULL}
};




// ****************************************************************************
// * Sse or Mic DEFINES
// ****************************************************************************
const nWhatWith nOkinaSseDefines[]={
  {"__host__",""},
  {"Integer", "integer"},
  {"Real", "real"},
  {"WARP_SIZE", "(1<<WARP_BIT)"},
  {"WARP_ALIGN", "(8<<WARP_BIT)"},    
  {"NABLA_NB_GLOBAL","WARP_SIZE"},
  {"reducemin(a)","0.0"},
  {"rabs(a)","(opTernary(((a)<0.0),(-a),(a)))"},
  {"set(a,b)", "_mm_set_pd(a,b)"},
  {"set1(cst)", "_mm_set1_pd(cst)"},
  {"square_root(u)", "_mm_sqrt_pd(u)"},
  {"cube_root(u)", "_mm_cbrt_pd(u)"},
  {"store(u,_u)", "_mm_store_pd(u,_u)"},
  {"load(u)", "_mm_load_pd(u)"},
  {"zero", "_mm_setzero_pd"},
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_INI)"},
  {"DBG_OFF", "0x0000ul"},
  {"DBG_CELL_VOLUME", "0x0001ul"},
  {"DBG_CELL_CQS", "0x0002ul"},
  {"DBG_GTH", "0x0004ul"},
  {"DBG_NODE_FORCE", "0x0008ul"},
  {"DBG_INI_EOS", "0x0010ul"},
  {"DBG_EOS", "0x0020ul"},
  {"DBG_DENSITY", "0x0040ul"},
  {"DBG_MOVE_NODE", "0x0080ul"},
  {"DBG_INI", "0x0100ul"},
  {"DBG_INI_CELL", "0x0200ul"},
  {"DBG_INI_NODE", "0x0400ul"},
  {"DBG_LOOP", "0x0800ul"},
  {"DBG_FUNC_IN", "0x1000ul"},
  {"DBG_FUNC_OUT", "0x2000ul"},
  {"DBG_VELOCITY", "0x4000ul"},
  {"DBG_BOUNDARIES", "0x8000ul"},
  {"DBG_ALL", "0xFFFFul"},
  {"opAdd(u,v)", "(u+v)"},
  {"opSub(u,v)", "(u-v)"},
  {"opDiv(u,v)", "(u/v)"},
  {"opMul(u,v)", "(u*v)"},
  {"opMod(u,v)", "(u%v)"},
  {"opScaMul(u,v)","dot3(u,v)"},
  {"opVecMul(u,v)","cross(u,v)"},    
  {"dot", "dot3"},
  {"knAt(a)",""},
  {"fatal(a,b)","exit(-1)"},
  {"synchronize(a)",""},
  {"mpi_reduce(how,what)","how##ToDouble(what)"},
  {"reduce(how,what)","how##ToDouble(what)"},
  {"xyz","int"},
  {"GlobalIteration", "global_iteration"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {"MD_Plus","0"},
  {"MD_Negt","4"},
  {"MD_Shift","3"},
  {"MD_Mask","7"}, // [sign,..]
  {NULL,NULL}
};


// ****************************************************************************
// * Sse or Mic FORWARDS
// ****************************************************************************
const char* nOkinaSseForwards[]={
  "inline std::ostream& info(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline std::ostream& debug(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline int WARP_BASE(int a){ return (a>>WARP_BIT);}",
  "inline int WARP_OFFSET(int a){ return (a&(WARP_SIZE-1));}",
  "inline int WARP_NFFSET(int a){ return ((WARP_SIZE-1)-WARP_OFFSET(a));}",
  NULL
};
