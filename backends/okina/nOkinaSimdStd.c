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

char* okinaStdIncludes(void){return "";}

char *okinaStdBits(void){return "64";}


// ****************************************************************************
// * Prev Cell
// ****************************************************************************
char* okinaStdPrevCell(void){
  return "gatherk_and_zero_neg_ones(cell_prev[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+0],";
}


// ****************************************************************************
// * Next Cell
// ****************************************************************************
char* okinaStdNextCell(void){
  return "gatherk_and_zero_neg_ones(cell_next[direction*NABLA_NB_CELLS+(c<<WARP_BIT)+0],";
}


// ****************************************************************************
// * Gather for Cells
// ****************************************************************************
static char* okinaStdGatherCells(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration){
    return strdup("register int __attribute__((unused)) cw,ia;");
  }
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
cw=(c<<WARP_BIT);\n\t\t\t\
gather%sk(ia=cell_node[n*NABLA_NB_CELLS+cw+0],\n\t\t\t\
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
// * En STD, le gather aux nodes est le même qu'aux cells
// ****************************************************************************
static char* okinaStdGatherNodes(nablaJob *job, nablaVariable* var, enum_phase phase){
  // Phase de déclaration
  if (phase==enum_phase_declaration){
    return strdup("int nw;");
  }
  // Phase function call
  char gather[1024];
  snprintf(gather, 1024, "\n\t\t\t%s gathered_%s_%s=%s(0.0);\n\t\t\t\
nw=(n<<WARP_BIT);\n\t\t\t\
gatherFromNode_%sk%s(node_cell[8*nw+c],\n\
%s\
         %s_%s%s,\n\t\t\t\
         &gathered_%s_%s);\n\t\t\t",
           strcmp(var->type,"real")==0?"real":"real3",
           var->item,
           var->name,
           strcmp(var->type,"real")==0?"real":"real3",
           strcmp(var->type,"real")==0?"":"3",
           var->dim==0?"":"Array8",
           var->dim==0?"":"\t\t\t\t\t\tnode_cell_corner[8*nw+c],\n\t\t\t",
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
char* okinaStdGather(nablaJob *job,nablaVariable* var, enum_phase phase){
  const char itm=job->item[0];  // (c)ells|(f)aces|(n)odes|(g)lobal
  if (itm=='c') return okinaStdGatherCells(job,var,phase);
  if (itm=='n') return okinaStdGatherNodes(job,var,phase);
  error(!0,0,"Could not distinguish job item in okinaStdGather!");
  return NULL;
}


// ****************************************************************************
// * Scatter
// ****************************************************************************
char* okinaStdScatter(nablaVariable* var){
  char scatter[1024];
  snprintf(scatter, 1024, "\tscatter%sk(ia, &gathered_%s_%s, %s_%s);",
           strcmp(var->type,"real")==0?"":"3",
           var->item, var->name,
           var->item, var->name);
  return strdup(scatter);
}




// ****************************************************************************
// * Std or Mic TYPEDEFS
// ****************************************************************************
nablaTypedef okinaStdTypedef[]={
  {"struct real3","Real3"},
  {NULL,NULL}
};



// ****************************************************************************
// * Std or Mic DEFINES
// ****************************************************************************
nablaDefine okinaStdDefines[]={
  {"integer", "Integer"},
  {"real", "Real"},
  {"WARP_SIZE", "(1<<WARP_BIT)"},
  {"WARP_ALIGN", "(8<<WARP_BIT)"},    
  {"NABLA_NB_GLOBAL","WARP_SIZE"},
  {"reducemin(a)","0.0"},
  {"rabs(a)","(opTernary(((a)<0.0),(-a),(a)))"},
  {"add(u,v)", "(u+v)"},
  //{"and(u,v)", "(u&v)"},
  //{"sub(u,v)", "(u-v)"},
  //{"div(u,v)", "(u/v)"},
  //{"mul(u,v)", "(u*v)"},
  {"set(a)", "a"},
  {"set1(cst)", "cst"},
  {"rsqrt(u)", "::sqrt(u)"},
  //{"shuffle(u,v,k)", "_mm256_shuffle_pd(u,v,k)"},
  {"store(u,_u)", "(*u=_u)"},
  {"load(u)", "(*u)"},
  {"zero()", "0.0"},
  // DEBUG STUFFS
  {"DBG_MODE", "(false)"},
  {"DBG_LVL", "(DBG_ALL)"},
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
  //#warning fatal returns
  {"fatal(a,b)","exit(-1)"},
  {"synchronize(a)","_Pragma(\"omp barrier\")"},
  //{"mpi_reduce(how,what)","mpi_reduce_min(tid,global_min_array,what)"},
  {"mpi_reduce(how,what)","how##ToDouble(what)"},
  {"xyz","int"},
  {"GlobalIteration", "global_iteration"},
  {"MD_DirX","0"},
  {"MD_DirY","1"},
  {"MD_DirZ","2"},
  {NULL,NULL}
};

// ****************************************************************************
// * Std or Mic FORWARDS
// ****************************************************************************
char* okinaStdForwards[]={
  "inline std::ostream& info(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline std::ostream& debug(){std::cout.flush();std::cout<<\"\\n\";return std::cout;}",
  "inline int WARP_BASE(int a){ return (a>>WARP_BIT);}",
  "inline int WARP_OFFSET(int a){ return (a&(WARP_SIZE-1));}",
  "inline int WARP_NFFSET(int a){ return ((WARP_SIZE-1)-WARP_OFFSET(a));}",
  "static void nabla_ini_node_coords(void);",
  "static void verifCoords(void);",
  NULL
};
