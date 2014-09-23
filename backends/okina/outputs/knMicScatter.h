#ifndef _KN_MIC_SCATTER_H_
#define _KN_MIC_SCATTER_H_

/******************************************************************************
 * Scatter: (X is the data @ offset x)
 * scatter: |ABCD| and offsets:    a                 b       c   d
 * data:    |....|....|....|....|..A.|....|....|....|B...|...C|..D.|....|....|
 * ! à la séquence car quand c et d sont sur le même warp, ça percute
 ******************************************************************************/
inline void scatter(const int a,
                     const int b,
                     const int c,
                     const int d,
                     const int e,
                     const int f,
                     const int g,
                     const int h,
                     real* base,
                     real scatter){
  __m512i index= _mm512_set_epi32(0,0,0,0,0,0,0,0,h,g,f,e,d,c,b,a);
  _mm512_i32loscatter_pd(base,index,scatter,_MM_SCALE_8);
}

inline void scatter3(const int a,
                      const int b,
                      const int c,
                      const int d,
                      const int e,
                      const int f,
                      const int g,
                      const int h,
                      real3* base,
                      real3 sctr){
  scatter(a,b,c,d,e,f,g,h,&base->x,sctr.x);
  scatter(a,b,c,d,e,f,g,h,&base->y,sctr.y);
  scatter(a,b,c,d,e,f,g,h,&base->z,sctr.z);
}

#endif //  _KN_MIC_SCATTER_H_
