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


/*****************************************************************************\
 * statics																							*
\*****************************************************************************/
static FILE *fTrace=NULL;
static unsigned long dbgFlg=NCC_DBG_STD_LVL;
static char *fTraceFile=NULL;


/*****************************************************************************\
 * toDbg																								*
\*****************************************************************************/
void dbgt(const bool flag, const int fd, const char *msg, ...){
  char bfrDbg[1024];
  if (flag != true) return;
  va_list args;
  va_start(args, msg);
  if (vsprintf(bfrDbg, msg, args) < 0) return;
  write(fd, bfrDbg, strlen(bfrDbg));/* fdDbg[1] */
  va_end(args);  
}


/*****************************************************************************\
 *																										*
\*****************************************************************************/
NABLA_STATUS dbg(const char *str, ...){
  va_list args;
  va_start(args, str);
  if (fTrace!=NULL){
    if (vfprintf(fTrace, str, args)<0)
      exit(printf("[dbg] vfprintf\n"));
    if (fflush(fTrace)!=0)
      exit(printf("[dbg] Could not flush to file\n"));
  }
  va_end(args);
  return NABLA_OK;
}


/*****************************************************************************\
 * Function to open specified dbg trace file												*
\*****************************************************************************/
void dbgOpenTraceFile(const char *file){
  fTraceFile=strdup(file);
  if ((fTrace=fopen(fTraceFile, "w")) == NULL)
    exit(printf("[dbgOpenTraceFile] Cannot open trace file\n"));
}


/*****************************************************************************\
 * Function to close specified dbg trace file											*
\*****************************************************************************/
void dbgCloseTraceFile(void){
  if (fTrace!=NULL)
    (void)fclose(fTrace);
}


/*****************************************************************************\
 * Function to get the set flag	 															*
\*****************************************************************************/
unsigned long dbgGet(void){
  return dbgFlg;
}


/*****************************************************************************\
 * Function to set the flag				 													*
\*****************************************************************************/
unsigned long dbgSet(unsigned long flg){
  return (dbgFlg=flg);
}
