/*****************************************************************************\
* File     : ncc_dbg.c   																		*
* Author   : Camier Jean-Sylvain																*
*******************************************************************************
* Description: 																					*
*******************************************************************************
* Date			Author	Description															*
* 05.01.2010	jscamier	Creation																*
\*****************************************************************************/
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
/*NABLA_STATUS dbg(unsigned long flg, const char *str, ...){*/
NABLA_STATUS dbg(const char *str, ...){
  /*if ((flg & dbgFlg)==flg){*/
  va_list args;

  va_start(args, str);
  if (fTrace!=NULL){
    if (vfprintf(fTrace, str, args)<0)
      exit(printf("[dbg] vfprintf\n"));
    if (fflush(fTrace)!=0)
      exit(printf("[dbg] Could not flush to file\n"));
  }
  va_end(args);
  //}
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
 * Function to close specified dbg trace file												*
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
 * Function to set the flag				 														*
\*****************************************************************************/
unsigned long dbgSet(unsigned long flg){
  return (dbgFlg=flg);
}
