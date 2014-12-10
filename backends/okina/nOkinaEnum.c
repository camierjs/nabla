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


void okinaDefineEnumerates(nablaMain *nabla){
  const char *parallel_prefix_for_loop=nabla->parallel->loop(nabla);
  fprintf(nabla->entity->hdr,"\n\n\
/*********************************************************\n\
 * Forward enumerates\n\
 *********************************************************/\n\
#define FOR_EACH_CELL(c) %sfor(int c=0;c<NABLA_NB_CELLS;c+=1)\n\
#define FOR_EACH_CELL_NODE(n) for(int n=0;n<8;n+=1)\n\
\n\
#define FOR_EACH_CELL_WARP(c) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
#define FOR_EACH_CELL_WARP_SHARED(c,local) %sfor(int c=0;c<NABLA_NB_CELLS_WARP;c+=1)\n\
\n\
#define FOR_EACH_CELL_WARP_NODE(n)\\\n\
  %sfor(int cn=WARP_SIZE*c+WARP_SIZE-1;cn>=WARP_SIZE*c;--cn)\\\n\
    for(int n=8-1;n>=0;--n)\n\
\n\
#define FOR_EACH_NODE(n) /*%s*/for(int n=0;n<NABLA_NB_NODES;n+=1)\n\
#define FOR_EACH_NODE_CELL(c) for(int c=0,nc=8*n;c<8;c+=1,nc+=1)\n\
\n\
#define FOR_EACH_NODE_WARP(n) %sfor(int n=0;n<NABLA_NB_NODES_WARP;n+=1)\n\
\n\
#define FOR_EACH_NODE_WARP_CELL(c)\\\n\
    for(int c=0;c<8;c+=1)\n",
          parallel_prefix_for_loop, // FOR_EACH_CELL
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_SHARED
          parallel_prefix_for_loop, // FOR_EACH_CELL_WARP_NODE
          parallel_prefix_for_loop, // FOR_EACH_NODE
          parallel_prefix_for_loop  // FOR_EACH_NODE_WARP
          );
}
