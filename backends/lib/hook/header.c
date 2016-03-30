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

// ****************************************************************************
// * xHookHeaderOpen
// ****************************************************************************
void xHookHeaderOpen(nablaMain *nabla){
  char hdrFileName[NABLA_MAX_FILE_NAME];
  sprintf(hdrFileName, "%s.h", nabla->name);
  if ((nabla->entity->hdr=fopen(hdrFileName, "w")) == NULL) exit(NABLA_ERROR);
}

// ****************************************************************************
// * xHookHeaderDump
// ****************************************************************************
void xHookHeaderDump(nablaMain *nabla){
  assert(nabla->entity->name);
  xDumpHeader(nabla);
}

// ****************************************************************************
// * hookHeaderPrefix
// ****************************************************************************
void xHookHeaderPrefix(nablaMain *nabla){
  assert(nabla->entity->name);
  fprintf(nabla->entity->hdr,
          "#ifndef __BACKEND_%s_H__\n#define __BACKEND_%s_H__",
          nabla->entity->name,
          nabla->entity->name);
}

// ****************************************************************************
// * ENUMERATES Work
// ****************************************************************************
static void xHeaderDefineEnumerates(nablaMain *nabla){
  // S'il n'y a pas de call, c'est qu'on ne le gère pas
  const char *parallel_prefix_for_loop=
    (nabla->call)?nabla->call->parallel?nabla->call->parallel->loop(nabla):"":"";
  fprintf(nabla->entity->hdr,"\n\n\
// *********************************************************\n\
// * Forward enumerates\n\
// *********************************************************\n\
#define FOR_EACH_PARTICLE(p) %sfor(int p=0;p<NABLA_NB_PARTICLES;p+=1)\n\
#define FOR_EACH_PARTICLE_WARP(p) %sfor(int p=0;p<NABLA_NB_PARTICLES;p+=1)\n\
\n\
#define FOR_EACH_CELL(c) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
#define FOR_EACH_CELL_WARP(c) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
#define FOR_EACH_OUTER_CELL_WARP(c) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=nxtOuterCellOffset(c))\n \
#define FOR_EACH_CELL_WARP_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<NABLA_NODE_PER_CELL;n+=1)\n\n\
#define FOR_EACH_CELL_WARP_NODE(n)\\\n\
  %sfor(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\\\n\
    for(int n=NABLA_NODE_PER_CELL-1;n>=0;--n)\n\
\n\
#define FOR_EACH_CELL_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
\n\
#define FOR_EACH_NODE_MSH(n) for(int n=0;n<msh.NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE(n) /*%s*/for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_WARP(n) %sfor(int n=0;n<NABLA_NB_NODES_WARP;n+=1)\n\
#define FOR_EACH_NODE_CELL(c)\
 for(int c=0,nc=NABLA_NODE_PER_CELL*n;c<NABLA_NODE_PER_CELL;c+=1,nc+=1)\n\n\
#define FOR_EACH_NODE_CELL_MSH(c)\
 for(int c=0,nc=msh.NABLA_NODE_PER_CELL*n;c<msh.NABLA_NODE_PER_CELL;c+=1,nc+=1)\n\n\
#define FOR_EACH_NODE_WARP_CELL(c)\\\n\
    for(int c=0;c<NABLA_NODE_PER_CELL;c+=1)\n\
\n\
#define FOR_EACH_FACE(f) %sfor(int f=0;f<NABLA_NB_FACES;f+=1)\n\
#define FOR_EACH_FACE_WARP(f) %sfor(int f=0;f<NABLA_NB_FACES;f+=1)\n\
#define FOR_EACH_INNER_FACE(f) %sfor(int f=0;f<NABLA_NB_FACES_INNER;f+=1)\n\
#define FOR_EACH_INNER_FACE_WARP(f) %sfor(int f=0;f<NABLA_NB_FACES_INNER;f+=1)\n\
#define FOR_EACH_OUTER_FACE(f)\
 %sfor(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)\n\
#define FOR_EACH_OUTER_FACE_WARP(f)\
 %sfor(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)\n\
// Pour l'instant en étant que multi-threadé, les 'own' sont les 'all'\n\
#define FOR_EACH_OWN_INNER_FACE(f) %sfor(int f=0;f<NABLA_NB_FACES_INNER;f+=1)\n\
#define FOR_EACH_OWN_INNER_FACE_WARP(f) %sfor(int f=0;f<NABLA_NB_FACES_INNER;f+=1)\n\
#define FOR_EACH_OWN_OUTER_FACE(f)\
 %sfor(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)\n\
#define FOR_EACH_OWN_OUTER_FACE_WARP(f)\
 %sfor(int f=NABLA_NB_FACES_INNER;f<NABLA_NB_FACES_INNER+NABLA_NB_FACES_OUTER;f+=1)\n\
\n\
#define FOR_EACH_FACE_CELL(c)\\\n\
    for(int c=0;c<NABLA_NODE_PER_FACE;c+=1)\n",
          parallel_prefix_for_loop, // FOR_EACH_PARTICLE
          parallel_prefix_for_loop, // FOR_EACH_PARTICLE_WARP
          parallel_prefix_for_loop, // FOR_EACH_CELL
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP
          parallel_prefix_for_loop, // FOR_EACH_OUTER_CELL_WARP
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_SHARED
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_NODE
          parallel_prefix_for_loop, // FOR_EACH_CELL_SHARED
          parallel_prefix_for_loop, // FOR_EACH_NODE
          parallel_prefix_for_loop, // FOR_EACH_NODE_WARP
          parallel_prefix_for_loop, // FOR_EACH_FACE
          parallel_prefix_for_loop, // FOR_EACH_FACE_WARP
          parallel_prefix_for_loop, // FOR_EACH_INNER_FACE
          parallel_prefix_for_loop, // FOR_EACH_INNER_FACE_WARP
          parallel_prefix_for_loop, // FOR_EACH_OUTER_FACE
          parallel_prefix_for_loop, // FOR_EACH_OUTER_FACE_WARP
          parallel_prefix_for_loop, // FOR_EACH_OWN_INNER_FACE
          parallel_prefix_for_loop, // FOR_EACH_OWN_INNER_FACE_WARP
          parallel_prefix_for_loop, // FOR_EACH_OWN_OUTER_FACE
          parallel_prefix_for_loop  // FOR_EACH_OWN_OUTER_FACE_WARP
          );
}

// ****************************************************************************
// * ENUMERATES Hooks
// ****************************************************************************
void xHookHeaderDefineEnumerates(nablaMain *nabla){
  // xHeaderDefineEnumerates a été déplacé à la fin du header,
  // dans le xHookHeaderPostfix
}

// ****************************************************************************
// * 
// ****************************************************************************
void xHookHeaderPostfix(nablaMain *nabla){
  xHookMeshStruct(nabla);
  xHeaderDefineEnumerates(nabla);
  fprintf(nabla->entity->hdr,
          "\n\n#endif // __BACKEND_%s_H__\n",
          nabla->entity->name);
}
