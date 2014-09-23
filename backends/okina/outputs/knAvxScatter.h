#ifndef _KN_SCATTER_H_
#define _KN_SCATTER_H_


// *****************************************************************************
// * Scatter: (X is the data @ offset x)
// * scatter: |ABCD| and offsets:    a                 b       c   d
// * data:    |....|....|....|....|..A.|....|....|....|B...|...C|..D.|....|....|
// * ! à la séquence car quand c et d sont sur le même warp, ça percute
// ******************************************************************************
inline void scatterk(const int a, const int b,
                     const int c, const int d,
                     real *scatter, real *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  p[a]=s[0];
  p[b]=s[1];
  p[c]=s[2];
  p[d]=s[3];
}


// *****************************************************************************
// * Scatter for real3
// *****************************************************************************
inline void scatter3k(const int a, const int b,
                      const int c, const int d,
                      real3 *scatter, real3 *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  
  p[4*(3*WARP_BASE(a)+0)+WARP_OFFSET(a)]=s[0];
  p[4*(3*WARP_BASE(b)+0)+WARP_OFFSET(b)]=s[1];
  p[4*(3*WARP_BASE(c)+0)+WARP_OFFSET(c)]=s[2];
  p[4*(3*WARP_BASE(d)+0)+WARP_OFFSET(d)]=s[3];

  p[4*(3*WARP_BASE(a)+1)+WARP_OFFSET(a)]=s[4];
  p[4*(3*WARP_BASE(b)+1)+WARP_OFFSET(b)]=s[5];
  p[4*(3*WARP_BASE(c)+1)+WARP_OFFSET(c)]=s[6];
  p[4*(3*WARP_BASE(d)+1)+WARP_OFFSET(d)]=s[7];

  p[4*(3*WARP_BASE(a)+2)+WARP_OFFSET(a)]=s[8];
  p[4*(3*WARP_BASE(b)+2)+WARP_OFFSET(b)]=s[9];
  p[4*(3*WARP_BASE(c)+2)+WARP_OFFSET(c)]=s[10];
  p[4*(3*WARP_BASE(d)+2)+WARP_OFFSET(d)]=s[11];
}

#endif //  _KN_SCATTER_H_
