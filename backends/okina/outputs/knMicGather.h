#ifndef _KN_MIC_GATHER_H_
#define _KN_MIC_GATHER_H_

std::ostream& operator<<(std::ostream &os, const __m512d v);

/******************************************************************************
 * Gather: (X is the data @ offset x)       a            b       c   d
 * data:   |....|....|....|....|....|....|..A.|....|....|B...|...C|..D.|....|      
 * gather: |ABCD|
 ******************************************************************************/
inline void gatherk(const int a, const int b, const int c, const int d,
                    const int e, const int f, const int g, const int h,
                    const real* addr, real *gather){
  __m512i index= _mm512_set_epi32(0,0,0,0,0,0,0,0,h,g,f,e,d,c,b,a);
  *gather = _mm512_i32logather_pd(index, addr, _MM_SCALE_8);
}

inline real inlined_gatherk(const int a, const int b, const int c, const int d,
                            const int e, const int f, const int g, const int h,
                            const real* addr){
  const __m512i index= _mm512_set_epi32(0,0,0,0,0,0,0,0,h,g,f,e,d,c,b,a);
  return _mm512_i32logather_pd(index, addr, _MM_SCALE_8);
}

inline __m512d gatherk_and_zero_neg_ones(const int a, const int b,
                                         const int c, const int d,
                                         const int e, const int f,
                                         const int g, const int h,
                                         real *data){
  const double *p=(double*)data;
  const __m512d zero512=_mm512_set1_pd(0.0);
  const __m512d hgfedcba=_mm512_set_pd(p[(h<0)?0:h],
                                       p[(g<0)?0:g],
                                       p[(f<0)?0:f],
                                       p[(e<0)?0:e],
                                       p[(d<0)?0:d],
                                       p[(c<0)?0:c],
                                       p[(b<0)?0:b],
                                       p[(a<0)?0:a]);
  const __mmask8 mask = _mm512_cmplt_pd_mask(_mm512_set_pd(h,g,f,e,d,c,b,a),zero512);
  const __m512d hgfedcbat=opTernary(mask, zero512, hgfedcba);
  //info()<<"a="<<a<<", b="<<b<<", c="<<c<<", d="<<d<<", e="<<e<<", f="<<f<<", g="<<g<<", h="<<h <<", hgfedcba="<<hgfedcba<<", hgfedcbat="<<hgfedcbat;
  return hgfedcbat;
}

inline void gatherFromNode_k(const int a, const int b,
                             const int c, const int d,
                             const int e, const int f,
                             const int g, const int h,
                             real *data, real *gthr){
  *gthr=gatherk_and_zero_neg_ones(a,b,c,d,e,f,g,h,data);
}

/******************************************************************************
 * Gather avec des real3
 ******************************************************************************/
inline void gather3ki(const int a, const int b,
                      const int c, const int d,
                      const int e, const int f,
                      const int g, const int h,
                      const real3* data, real3 *gthr,
                      int i){
  const double *p =(double*)data;
  const double pa=p[8*(3*WARP_BASE(a)+i)+WARP_OFFSET(a)];
  const double pb=p[8*(3*WARP_BASE(b)+i)+WARP_OFFSET(b)];
  const double pc=p[8*(3*WARP_BASE(c)+i)+WARP_OFFSET(c)];
  const double pd=p[8*(3*WARP_BASE(d)+i)+WARP_OFFSET(d)];
  const double pe=p[8*(3*WARP_BASE(e)+i)+WARP_OFFSET(e)];
  const double pf=p[8*(3*WARP_BASE(f)+i)+WARP_OFFSET(f)];
  const double pg=p[8*(3*WARP_BASE(g)+i)+WARP_OFFSET(g)];
  const double ph=p[8*(3*WARP_BASE(h)+i)+WARP_OFFSET(h)];
  const __m512d hgfedcba=_mm512_set_pd(ph,pg,pf,pe,pd,pc,pb,pa);
  if (i==0) (*gthr).x=hgfedcba;
  if (i==1) (*gthr).y=hgfedcba;
  if (i==2) (*gthr).z=hgfedcba;
  //if (i==0) gthr->x=inlined_gatherk(a,b,c,d,e,f,g,h,&(addr->x));
  //if (i==1) gthr->y=inlined_gatherk(a,b,c,d,e,f,g,h,&(addr->y));
  //if (i==2) gthr->z=inlined_gatherk(a,b,c,d,e,f,g,h,&(addr->z));
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
                     real3* gthr){
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
// ******************************************************************************
// *
// ******************************************************************************
inline void gatherFromNode_3kiArray8(const int a, const int a_corner,
                                     const int b, const int b_corner,
                                     const int c, const int c_corner,
                                     const int d, const int d_corner,
                                     const int e, const int e_corner,
                                     const int f, const int f_corner,
                                     const int g, const int g_corner,
                                     const int h, const int h_corner,
                                     real3 *data, real3 *gthr, int i){
  const double *p=(double *)data; 
  const double pa=a<0?0.0:p[8*(3*8*WARP_BASE(a)+3*a_corner+i)+WARP_OFFSET(a)];
  const double pb=b<0?0.0:p[8*(3*8*WARP_BASE(b)+3*b_corner+i)+WARP_OFFSET(b)];
  const double pc=c<0?0.0:p[8*(3*8*WARP_BASE(c)+3*c_corner+i)+WARP_OFFSET(c)];
  const double pd=d<0?0.0:p[8*(3*8*WARP_BASE(d)+3*d_corner+i)+WARP_OFFSET(d)];
  const double pe=e<0?0.0:p[8*(3*8*WARP_BASE(e)+3*e_corner+i)+WARP_OFFSET(e)];
  const double pf=f<0?0.0:p[8*(3*8*WARP_BASE(f)+3*f_corner+i)+WARP_OFFSET(f)];
  const double pg=g<0?0.0:p[8*(3*8*WARP_BASE(g)+3*g_corner+i)+WARP_OFFSET(g)];
  const double ph=h<0?0.0:p[8*(3*8*WARP_BASE(h)+3*h_corner+i)+WARP_OFFSET(h)];
  const __m512d hgfedcba=_mm512_set_pd(ph,pg,pf,pe,pd,pc,pb,pa);
  if (i==0) (*gthr).x=hgfedcba;
  if (i==1) (*gthr).y=hgfedcba;
  if (i==2) (*gthr).z=hgfedcba;
}

inline void gatherFromNode_3kArray8(const int a, const int a_corner,
                                    const int b, const int b_corner,
                                    const int c, const int c_corner,
                                    const int d, const int d_corner,
                                    const int e, const int e_corner,
                                    const int f, const int f_corner,
                                    const int g, const int g_corner,
                                    const int h, const int h_corner,
                                    real3 *data, real3 *gthr){
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,
                           e,e_corner,f,f_corner,g,g_corner,h,h_corner,data,gthr,0);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,
                           e,e_corner,f,f_corner,g,g_corner,h,h_corner,data,gthr,1);
  gatherFromNode_3kiArray8(a,a_corner,b,b_corner,c,c_corner,d,d_corner,
                           e,e_corner,f,f_corner,g,g_corner,h,h_corner,data,gthr,2);
}

#endif // _KN_MIC_GATHER_H_
