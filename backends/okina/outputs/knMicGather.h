#ifndef _KN_MIC_GATHER_H_
#define _KN_MIC_GATHER_H_


/******************************************************************************
 * Gather: (X is the data @ offset x)       a            b       c   d
 * data:   |....|....|....|....|....|....|..A.|....|....|B...|...C|..D.|....|      
 * gather: |ABCD|
 ******************************************************************************/
inline void gatherk(const int a,
                    const int b,
                    const int c,
                    const int d,
                    const int e,
                    const int f,
                    const int g,
                    const int h,
                    const real* addr,
                    real &gather){
  __m512i index= _mm512_set_epi32(0,0,0,0,0,0,0,0,h,g,f,e,d,c,b,a);
  gather = _mm512_i32logather_pd(index, addr, _MM_SCALE_8);
}

inline real inlined_gatherk(const int a,
                            const int b,
                            const int c,
                            const int d,
                            const int e,
                            const int f,
                            const int g,
                            const int h,
                            const real* addr){
  const __m512i index= _mm512_set_epi32(0,0,0,0,0,0,0,0,h,g,f,e,d,c,b,a);
  return _mm512_i32logather_pd(index, addr, _MM_SCALE_8);
}

inline __m512d gatherk_and_zero_neg_ones(const int a, const int b,
                                         const int c, const int d,
                                         const int e, const int f,
                                         const int g, const int h,
                                         real *data){
  double *p=(double*)data;
  const __m512d zero512=_mm512_set1_pd(0.0);
/*  const __m256d bData=_mm_movedup_pd(_mm_load_sd(&p[(b<0)?0:b])); 
  const __m512d ba=_mm512_castpd256_pd512(_mm_loadl_pd(bData,&p[(a<0)?0:a]));
  const  __m256d dData=_mm_movedup_pd(_mm_load_sd(&p[(d<0)?0:d]));
  const __m256d dc=_mm_loadl_pd(dData,&p[(c<0)?0:c]);
  const __m512d dcba=_mm512_insertf256_pd(ba,dc,0x01);
  const __m512d dcbat=opTernary(_mm512_cmp_pd(_mm512_set_pd(d,c,b,a), zero512, _CMP_GE_OQ), dcba, zero512);
*/  //debug()<<"a="<<a<<", b="<<b<<", c="<<c<<", d="<<d<<", dcba="<<dcba<<", dcbat="<<dcbat;
  //debug()<<"_mm512_set_pd(d,c,b,a)="<<_mm512_set_pd(d,c,b,a);
  return zero512;//dcbat;
}


/******************************************************************************
 * Gather avec des real3
 ******************************************************************************/
inline void gather3ki(const int a,
                      const int b,
                      const int c,
                      const int d,
                      const int e,
                      const int f,
                      const int g,
                      const int h,
                      const real3* addr,
                      real3 &gthr,
                      int i){
  __m512d x=inlined_gatherk(a,b,c,d,e,f,g,h,&addr->x);
  __m512d y=inlined_gatherk(a,b,c,d,e,f,g,h,&addr->y);
  __m512d z=inlined_gatherk(a,b,c,d,e,f,g,h,&addr->z);
  if (i==0) gthr.x=x;
  if (i==1) gthr.y=y;
  if (i==2) gthr.z=z;
}

inline void gather3k(const int a,
                     const int b,
                     const int c,
                     const int d,
                     const int e,
                     const int f,
                     const int g,
                     const int h,
                     real3* addr,
                     real3 &gthr){
  gather3ki(a,b,c,d,e,f,g,h,addr,gthr,0);
  gather3ki(a,b,c,d,e,f,g,h,addr,gthr,1);
  gather3ki(a,b,c,d,e,f,g,h,addr,gthr,2);  
  }

inline real3 inlined_gather3k(const int a,
                              const int b,
                              const int c,
                              const int d,
                              const int e,
                              const int f,
                              const int g,
                              const int h,
                              const real3* addr){
  return real3(inlined_gatherk(a,b,c,d,e,f,g,h,&addr->x),
               inlined_gatherk(a,b,c,d,e,f,g,h,&addr->y),
               inlined_gatherk(a,b,c,d,e,f,g,h,&addr->z));
}

#endif // _KN_MIC_GATHER_H_
