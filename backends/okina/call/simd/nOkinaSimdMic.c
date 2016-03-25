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

char* nOkinaMicIncludes(void){return "#include <immintrin.h>\n";}
char *nOkinaMicBits(void){return "512";}


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* nOkinaMicPrevCell(int direction){
  if (direction==DIR_X)
    return "gatherk_and_zero_neg_ones(\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
			cell_prev[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
  if (direction==DIR_Y)
    return "gatherk_and_zero_neg_ones(\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
			cell_prev[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
  if (direction==DIR_Z)
    return "gatherk_and_zero_neg_ones(\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
			cell_prev[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
  assert(NULL);
  return NULL;
}

// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* nOkinaMicNextCell(int direction){
  if (direction==DIR_X)
    return "gatherk_and_zero_neg_ones(\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
			cell_next[DIR_X*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
  if (direction==DIR_Y)
    return "gatherk_and_zero_neg_ones(\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
			cell_next[DIR_Y*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
  if (direction==DIR_Z)
    return "gatherk_and_zero_neg_ones(\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
			cell_next[DIR_Z*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
  assert(NULL);
  return NULL;
}

// ****************************************************************************
// * Gather
// ****************************************************************************
char* nOkinaMicGatherCells(nablaJob *job,nablaVariable* var){
  char gather[1024];
  snprintf(gather, 1024, "int __attribute__((unused)) cw,ia,ib,ic,id,ie,iff,ig,ih;\n\
\n\t\t\t%s gathered_%s_%s;\n\t\t\t\
cw=(c<<WARP_BIT);\n\t\t\t\
gather%sk(ia=cell_node[n*NABLA_NB_CELLS+cw+0],\n\t\t\t\
          ib=cell_node[n*NABLA_NB_CELLS+cw+1],\n\t\t\t\
          ic=cell_node[n*NABLA_NB_CELLS+cw+2],\n\t\t\t\
          id=cell_node[n*NABLA_NB_CELLS+cw+3],\n\t\t\t\
          ie=cell_node[n*NABLA_NB_CELLS+cw+4],\n\t\t\t\
         iff=cell_node[n*NABLA_NB_CELLS+cw+5],\n\t\t\t\
          ig=cell_node[n*NABLA_NB_CELLS+cw+6],\n\t\t\t\
          ih=cell_node[n*NABLA_NB_CELLS+cw+7],\n\t\t\t\
          %s_%s%s,\n\t\t\t\
          &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item, var->name,
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item, var->name);
  return strdup(gather);
}

// ****************************************************************************
// * Gather pour un job sur les nodes
// ****************************************************************************
static char* nOkinaMicGatherNodes(nablaJob *job,nablaVariable* var){ 
  char gather[1024];
  snprintf(gather, 1024, "int nw;\n\t\t\t%s gathered_%s_%s;\n\t\t\t\
nw=(n<<WARP_BIT);\n\t\t\t\
gatherFromNode_%sk%s(node_cell[8*(nw+0)+c],%s\n\
\t\t\t\t\t\tnode_cell[8*(nw+1)+c],%s\n\
\t\t\t\t\t\tnode_cell[8*(nw+2)+c],%s\n\
\t\t\t\t\t\tnode_cell[8*(nw+3)+c],%s\n\
\t\t\t\t\t\tnode_cell[8*(nw+4)+c],%s\n\
\t\t\t\t\t\tnode_cell[8*(nw+5)+c],%s\n\
\t\t\t\t\t\tnode_cell[8*(nw+6)+c],%s\n\
\t\t\t\t\t\tnode_cell[8*(nw+7)+c],%s\n\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"":"3",
           var->dim==0?"":"Array8",
           var->dim==0?"":"node_cell_corner[8*(nw+0)+c],",
           var->dim==0?"":"node_cell_corner[8*(nw+1)+c],",
           var->dim==0?"":"node_cell_corner[8*(nw+2)+c],",
           var->dim==0?"":"node_cell_corner[8*(nw+3)+c],",
           var->dim==0?"":"node_cell_corner[8*(nw+4)+c],",
           var->dim==0?"":"node_cell_corner[8*(nw+5)+c],",
           var->dim==0?"":"node_cell_corner[8*(nw+6)+c],",
           var->dim==0?"":"node_cell_corner[8*(nw+7)+c],",
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
char* nOkinaMicGather(nablaJob *job,nablaVariable* var){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return nOkinaMicGatherCells(job,var);
  if (itm=='n') return nOkinaMicGatherNodes(job,var);
  nablaError("Could not distinguish job item in nOkinaMicGather!");
  return NULL;
}


// ****************************************************************************
// * MIC Scatter
// ****************************************************************************
char* nOkinaMicScatter(nablaJob *job,nablaVariable* var){
  char scatter[1024];
  snprintf(scatter, 1024, "\tscatter%s(ia,ib,ic,id,ie,iff,ig,ih, %s_%s, gathered_%s_%s);",
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           var->item, var->name);
  return strdup(scatter);
}


// ****************************************************************************
// * MIC TYPEDEFS
// ****************************************************************************
const nWhatWith nOkinaMicTypedef[]={
  //{"struct real3","Real3"},
  {NULL,NULL}
};


// ****************************************************************************
// * MIC DEFINES
// ****************************************************************************
const nWhatWith nOkinaMicDefines[]={
  {"__host__",""},
  {"integer", "Integer"},
  {"Real", "real"},
  {"Real3", "real3"},
  //{"norm","rabs"},
  {"WARP_SIZE", "(1<<WARP_BIT)"},
  {"WARP_ALIGN", "(8<<WARP_BIT)"},    
  {"NABLA_NB_GLOBAL","WARP_SIZE"},
  {"reducemin(a)","0.0"},
  {"rabs(a)","(opTernary(((a)<0.0),(-a),(a)))"},
  {"add(u,v)", "_mm512_add_pd(u,v)"},
  {"sub(u,v)", "_mm512_sub_pd(u,v)"},
  {"div(u,v)", "_mm512_div_pd(u,v)"},
  {"mul(u,v)", "_mm512_mul_pd(u,v)"},
  {"set(a,b,c,d,e,f,g,h)", "_mm512_set_pd(a,b,c,d,e,f,g,h)"},
  {"set1(cst)", "_mm512_set1_pd(cst)"},
  {"square_root(u)", "_mm512_sqrt_pd(u)"},
  {"cube_root(u)", "_mm512_cbrt_pd(u)"},
  {"store(addr,data)", "_mm512_store_pd(addr,data)"},
  {"load(u)", "_mm512_load_pd(u)"},
  {"zero", "_mm512_setzero_pd"},
  // DEBUG STUFFS
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_FUNC_IN)"},
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
// * MIC FORWARDS
// ****************************************************************************
const char* nOkinaMicForwards[]={
  "inline std::ostream& info(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline std::ostream& debug(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline int WARP_BASE(int a){ return (a>>WARP_BIT);}",
  "inline int WARP_OFFSET(int a){ return (a&(WARP_SIZE-1));}",
  "inline int WARP_NFFSET(int a){ return ((WARP_SIZE-1)-WARP_OFFSET(a));}",
  NULL
};
  
