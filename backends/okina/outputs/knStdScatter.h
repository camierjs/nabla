#ifndef _KN_STD_SCATTER_H_
#define _KN_STD_SCATTER_H_


// *****************************************************************************
// * Scatter: (X is the data @ offset x)
// * scatter: |ABCD| and offsets:    a                 b       c   d
// * data:    |....|....|....|....|..A.|....|....|....|B...|...C|..D.|....|....|
// * ! à la séquence car quand c et d sont sur le même warp, ça percute
// ******************************************************************************
inline void scatterk(const int a, real *scatter, real *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  p[a]=s[0];
}


// *****************************************************************************
// * Scatter for real3
// *****************************************************************************
inline void scatter3k(const int a, real3 *scatter, real3 *data){
  double *s=(double *)scatter;
  double *p=(double *)data;
  p[3*a+0]=s[0];
  p[3*a+1]=s[1];
  p[3*a+2]=s[2];
}

#endif //  _KN_STD_SCATTER_H_
