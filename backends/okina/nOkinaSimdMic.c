/*****************************************************************************
 * CEA - DAM/DSSI/SNEC/LECM
 *****************************************************************************
 * File     : nOkinaSimd.c
 * Author   : Camier Jean-Sylvain
 * Created  : 2013.12.23
 * Updated  : 2013.12.23
 *****************************************************************************
 * Description:
 *****************************************************************************
 * Date			Author	Description
 * 2013.12.23	camierjs	Creation
 *****************************************************************************/
#include "nabla.h"

char* okinaMicIncludes(void){return "#include <immintrin.h>\n";}
char *okinaMicBits(void){return "512";}


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* okinaMicPrevCell(void){
  return "gatherk_and_zero_neg_ones(\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
}

// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* okinaMicNextCell(void){
  return "gatherk_and_zero_neg_ones(\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+0],\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+1],\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+2],\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+3],\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+4],\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+5],\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+6],\n\
cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+7],";
}

// ****************************************************************************
// * Gather
// ****************************************************************************
char* okinaMicGather(nablaJob *job,nablaVariable* var, enum_phase phase){
  if (phase==enum_phase_declaration)
    return strdup("register int cw=(c<<WARP_BIT);\n\t\t\tregister int ia,ib,ic,id,ie,iff,ig,ih;");
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t__declspec(align(64)) %s gathered_%s_%s=inlined_gather%sk(ia=cell_node[n*NABLA_NB_CELLS+cw+0],\n\t\t\t\
         ib=cell_node[n*NABLA_NB_CELLS+cw+1],\n\t\t\t\
         ic=cell_node[n*NABLA_NB_CELLS+cw+2],\n\t\t\t\
         id=cell_node[n*NABLA_NB_CELLS+cw+3],\n\t\t\t\
         ie=cell_node[n*NABLA_NB_CELLS+cw+4],\n\t\t\t\
         iff=cell_node[n*NABLA_NB_CELLS+cw+5],\n\t\t\t\
         ig=cell_node[n*NABLA_NB_CELLS+cw+6],\n\t\t\t\
         ih=cell_node[n*NABLA_NB_CELLS+cw+7],\n\t\t\t\
         %s_%s%s);\n\t\t\t\
         //gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item, var->name,
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           strcmp(var->type,"real")==0?"":"",
           var->item, var->name);
  return strdup(gather);
}


// ****************************************************************************
// * MIC Scatter
// ****************************************************************************
char* okinaMicScatter(nablaVariable* var){
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
nablaTypedef okinaMicTypedef[]={
  //{"struct real3","Real3"},
  {NULL,NULL}
};


// ****************************************************************************
// * MIC DEFINES
// ****************************************************************************
nablaDefine okinaMicDefines[]={
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
  {"rsqrt(u)", "_mm512_sqrt_pd(u)"},
  {"rcbrt(u)", "_mm512_cbrt_pd(u)"},
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
  {"knAt(a)","{}"},
  {"fatal(a,b)","exit(-1)"},
  {"synchronize(a)",""},
  {"mpi_reduce(how,what)","what"},
  {"xyz","int"},
  {"GlobalIteration", "global_iteration"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {NULL,NULL}
};


// ****************************************************************************
// * MIC FORWARDS
// ****************************************************************************
char* okinaMicForwards[]={
  "inline std::ostream& debug(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline int WARP_BASE(int a){ return (a>>WARP_BIT);}",
  "inline int WARP_OFFSET(int a){ return (a&(WARP_SIZE-1));}",
  "inline int WARP_NFFSET(int a){ return ((WARP_SIZE-1)-WARP_OFFSET(a));}",
  "static void nabla_ini_node_coords(void);",
  "static void verifCoords(void);",
  "static void micTestReal(void);",
  "static void micTestReal3(void);",
  NULL
};
  
