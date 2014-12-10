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


char *nccCudaBits(void){return "";}


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
