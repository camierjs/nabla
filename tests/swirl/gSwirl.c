#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <stdlib.h>
#include <stdlib.h>


#define OPTIONS "\n\
options{\n\
  ℝ option_δt_initial = 0.1;\n\
  ℝ option_stoptime = 1.0;\n\
};\n"

#define JOB "\n∀ %s void _%d_(void)@%f{}"


// ****************************************************************************
// * Launch a system command
// ****************************************************************************
void sys(const char* command){
  int ret = system(command);
  assert(ret==0);
}

// ****************************************************************************
// * returns an int from [0,k[
// ****************************************************************************
int irand(int k){
  return k*drand48();
}


// ****************************************************************************
// * returns cells|nodes|faces|particles
// ****************************************************************************
char* cnfp(void){
  switch (irand(4)){
  case 0: return "cells";
  case 1: return "nodes";
  case 2: return "faces";
  case 3: return "particles";
  default: return NULL;
  }
  return NULL;
}


// ****************************************************************************
// * Génération d'une partie (init ou loop)
// ****************************************************************************
void generate(FILE* file,
              const char* label,
              const int stages,
              const int max_parallels,
              const double alpha,
              const double beta){
  for(int i=0;i<stages;i+=1){
    const double at=alpha*drand48()+beta;
    const int nb_parallels=1+irand(max_parallels);
    printf("[Init] %d/%d, @=%f #parallels:%d<%d\n",
           i+1,
           stages,
           at,
           nb_parallels,
           max_parallels);
    for(int j=0;j<nb_parallels;j+=1) // nombre de parellèles
      fprintf(file,JOB,cnfp(),rand(),at);
  }
}


// ****************************************************************************
// * MAIN
// ****************************************************************************
int main(int argc, char* argv[]){
  int n=(argc>=2)?atoi(argv[1]):4;
  int m=(argc>=3)?atoi(argv[2]):7;
  int p=(argc>=4)?atoi(argv[3]):8;
  int q=(argc>=5)?atoi(argv[4]):10;
  FILE* file=fopen("gSwirl.n", "w");
  assert(file);
  // seed for a new sequence of pseudo-randoms 
  srand((unsigned long)argv[0]);
  srand48(rand());
  
  // Options
  fprintf(file,OPTIONS);
  // Phase d'init: -1.0 pour le négatif
  //generate(file,"init",1,2,-1.0,-5.0);
  generate(file,"init",n,m,-1.0,-4.0);
  //generate(file,"loopIni",4,4,+1.0,0.0);
  generate(file,"loopCompute",p,q,+1.0,4.0);
  //generate(file,"loopEnd",2,4,+1.0,8.0);
  
  if (fclose(file)!=0) return EBADF;
  printf("done\n");
  sys("/tmp/nabla/nabla --tnl --lambda gSwirl --std --seq -i gSwirl.n");
  sys("/usr/bin/dot -Tsvg gSwirl.time.dot -o gSwirl.time.svg");
  return 0;
}
