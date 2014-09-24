#ifndef _KN_DBG_HPP_
#define _KN_DBG_HPP_

#include <stdarg.h>

/******************************************************************************
 * Outils de traces
 *****************************************************************************/

void dbg(const unsigned int flag, const char *format, ...){
  if (!DBG_MODE) return;
  if ((flag&DBG_LVL)==0) return;
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  fflush(stdout);
  va_end(args);
}

#define dbgFuncIn()  do{dbg(DBG_FUNC_IN,"\n\t > %%s",__FUNCTION__);}while(0)
#define dbgFuncOut() do{dbg(DBG_FUNC_OUT,"\n\t\t < %%s",__FUNCTION__);}while(0)



inline void dbgReal3(const unsigned int flag, real3& v){
  if (!DBG_MODE) return;
  if ((flag&DBG_LVL)==0) return;
  double x[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  double y[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  double z[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  store(x, v.x);
  store(y, v.y);
  store(z, v.z);
//  for(int i=WARP_SIZE-1;i>=0;--i)
  for(int i=0;i<WARP_SIZE;i+=1)
    printf("\n\t\t\t[%%.14f,%%.14f,%%.14f]", x[i], y[i], z[i]);
  fflush(stdout);
  }


inline void dbgReal(const unsigned int flag, real v){
  if (!DBG_MODE) return;
  if ((flag&DBG_LVL)==0) return;
  double x[WARP_SIZE] __attribute__ ((aligned(WARP_ALIGN)));
  store(x, v);
  printf("[");
//  for(int i=WARP_SIZE-1;i>=0;--i)
  for(int i=0;i<WARP_SIZE;i+=1)
    printf("%%.14f ", x[i]);
  printf("]");
  fflush(stdout);
}


#endif // _KN_DBG_HPP_
