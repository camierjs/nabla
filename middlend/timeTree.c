///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
#include "nabla.h"
#include "nabla.tab.h"

/******************************************************************************
 * +->m, .->d
 ******************************************************************************/
static char *strKillMinusDot(const char * str){
  char *p=sdup(str);
  char *bkp=p;
  for(;*p!=0;p++){
    if (*p==45) *p=109;
    if (*p==46) *p=100;
  }
  return bkp;
} 

// *****************************************************************************
// * Création du nom d'un when selon son temps logique
// *****************************************************************************
static char *whenName(const char *prefix, double at){
  char bfr[1024];
  sprintf(bfr, "%s%0.02f", prefix, at);
  return sdup(bfr);
}

// *****************************************************************************
// * Création du nom d'un cluster selon son temps logique
// *****************************************************************************
static char *clusterName(double at){
  return strKillMinusDot(whenName("cluster",at));
}


// *****************************************************************************
// * Couleur en fonction du support: \"#CCDDCC\"
// * palegoldenrod, palevioletred, palegreen, palevioletred &paleturquoise
// *****************************************************************************
static char *color(const char item){
  switch (item){
  case 'c': return "palegreen";
  case 'n': return "palegoldenrod";
  case 'f': return "paleturquoise";
  case 'p': return "orchid1";
  case '\0': return "palevioletred";
  default: return "red";
  }
  return "red";
}


// *****************************************************************************
// * Shape si c'est une fonction ou un job
// * hexagon, trapezium, ellipse
// *****************************************************************************
static char *shape(const bool is_a_function){
  if (is_a_function) return "doubleoctagon";
  return "ellipse";
}

// *****************************************************************************
// * Filled si c'est une fonction ou un job
// *****************************************************************************
static char *filled(const bool is_a_function){
  //if (is_a_function) return "dummy";
  return "filled";
}



// *****************************************************************************
// * timeTreeSaveMathematica
// *****************************************************************************
static void timeTreeSaveMathematica(FILE *fTreeOutput, nablaJob *job,
                                    int number_of_entry_points,
                                    int optionDumpTree){
  dbg("\n[timeTreeSaveNodes] number_of_entry_points=%d", number_of_entry_points);
  if (number_of_entry_points==0){
    dbg("\n[timeTreeSaveNodes] returning");
    return;
  }
  
  for(int i=0;i<number_of_entry_points;i+=1){
    const char *jobName=job[i].name;
    const char *jobWhen=whenName("", job[i].whens[0]);
    fprintf(fTreeOutput,"\n\tnode_%s_%s [style=%s, shape=%s, color=%s, label=\"%s\", fontsize=32];",
            jobName,
            strKillMinusDot(jobWhen),
            filled(job[i].is_a_function),
            shape(job[i].is_a_function),
            color(job[i].item[0]),
            (optionDumpTree==OPTION_TIME_DOT_MMA)?"":job[i].name_utf8);
  }

  fprintf(fTreeOutput,"\n");
  for(int i=0;i<number_of_entry_points;i+=1){
    int j,k,n;
    double jAt;//,kAt;
    const double iAt=job[i].whens[0];
    const char *iJobName=job[i].name;
    const char *iJobWhen=strKillMinusDot(whenName("", job[i].whens[0]));
    
    for(j=i;j<number_of_entry_points;j+=1){
      if (job[j].whens[0]==iAt) continue;
      jAt=job[j].whens[0];
      break;
    }
    for(k=j;k<number_of_entry_points;k+=1){
      if (job[k].whens[0]==jAt) continue;
      break;
    }
    for(n=j;n<k;n+=1){
      const char *nJobName=job[n].name;
      const char *nJobWhen=strKillMinusDot(whenName("", job[n].whens[0]));
      fprintf(fTreeOutput,"\n\tnode_%s_%s -> node_%s_%s [arrowsize=0.3,penwidth=0.15];",
              iJobName,iJobWhen,
              nJobName,nJobWhen);      
    }
  }
}


// *****************************************************************************
// * timeTreeSaveNodes
// * Les jobs en entrée sont triés et n'ont pas de siblings, tout a été mis à plat
// * Attention car toute la structure des jobs n'est pas à jour
// * On le fait en deux passes selon le bool subgraph
// *****************************************************************************
__attribute__((unused))
static void timeTreeSaveNodes(FILE *fTreeOutput, nablaJob *job,
                              int number_of_entry_points,
                              bool subgraph){
  bool initPhase=true;
  int i,iGraph=0;
  const double atIni = 123456789.123456789;
  double at=atIni;

  dbg("\n[timeTreeSaveNodes] number_of_entry_points=%d", number_of_entry_points);
  if (number_of_entry_points==0){
    dbg("\n[timeTreeSaveNodes] returning");
    return;
  }
  
  // InitPhase
  if (subgraph)
     fprintf(fTreeOutput,"subgraph clusterInitPhase{\n\t\tlabel=\"InitPhase\";\n\t\tsortv=%d;\n\
\t\tnode_InitPhase [style=filled, shape=tripleoctagon, color=gray, label=\"Init Phase\", fontsize=32];\n}",iGraph+=1);

  for(i=0;i<number_of_entry_points;i+=1){
    const char *jobName=job[i].name_utf8;
    const char *jobWhen=whenName("", job[i].whens[0]);
    const char *thisClusterName=clusterName(at);
    const char *nextClusterName=clusterName(job[i].whens[0]);
    // On scrute le passage en computeLoop
    if (job[i].whens[0]>0){
      // Si on vient de finir la phase d'init, on rajoute un graph
      if (initPhase){
         if (subgraph)
           fprintf(fTreeOutput,"\n\t}\n\n\nsubgraph clusterComputeLoop{\n\
\t\t//label=\"ComputeLoop\";\n\
\t\tsortv=%d;\n\
\t\tnode_ComputeLoop [style=filled, shape=tripleoctagon, color=gray, label=\"Compute Loop\", fontsize=32];\n",iGraph+=1);
      }
      initPhase=false;
    }
    // Changement d'instant logique:
    if (job[i].whens[0]!=at){
      // On termine le cluster courant, sauf au premier coup
      // On rajout un noeud pour enchaîner les clusters
      if (at==atIni){
        if (subgraph)
          fprintf(fTreeOutput,"\t//Start -> node_%s_%s [lhead=%s];\n",
                  jobName,strKillMinusDot(jobWhen), nextClusterName);
      }else{
        if (subgraph){
          fprintf(fTreeOutput, "\n\t}\n");
          //fprintf(fTreeOutput, "\n\t\t%s [shape=point,style=invis];\n\t}\n",thisClusterName);
        }else{
          fprintf(fTreeOutput,"\tnode_%s_%s -> node_%s_%s [ltail=%s,lhead=%s,style=invis];\n",
                  job[i-1].name_utf8,
                  strKillMinusDot(whenName("",job[i-1].whens[0])),
                  jobName,
                  strKillMinusDot(jobWhen),
                  thisClusterName,
                  nextClusterName);
          fprintf(fTreeOutput,"\t//%s -> %s;\n",
                  thisClusterName,nextClusterName);
          fprintf(fTreeOutput,"\t//%s -> %s [ltail=%s,lhead=%s,style=invis];\n\n",
                  thisClusterName, nextClusterName,
                  thisClusterName, nextClusterName);
        }
      }
      // Et création d'un nouveau subgraph
      // An input graph should not have a label, since this will be used in its layout.
      // Since gvpack ignores root graph labels, resulting layout may contain some extra space
      if (subgraph)
        fprintf(fTreeOutput,"\nsubgraph %s{\n\t\tlabel=\"@%s\";\n\t\tsortv=%d;",
                nextClusterName,jobWhen,iGraph+=1);
      // Mise à jour de notre temps logique courant
      at=job[i].whens[0];
    }
    { // Remplissage du subgraph courant
      //dbg("\n[timeTreeSaveNodes] Saving nodes %s %s", jobName, jobWhen);
      if (subgraph)
        fprintf(fTreeOutput,"\n\t\tnode_%s_%s [style=%s, shape=%s, color=%s, label=\"%s\", fontsize=32];",
                jobName,
                strKillMinusDot(jobWhen),
                filled(job[i].is_a_function),
                shape(job[i].is_a_function),
                color(job[i].item[0]),
                jobName);
    }
  }
  if (subgraph)
     fprintf(fTreeOutput,"\n\t\t//%s [shape=point,style=invis];\n\
\t}\n\t//node_%s_%s -> End [ltail=%s];\n",
          clusterName(job[number_of_entry_points-1].whens[0]),
          job[number_of_entry_points-1].name_utf8,
          strKillMinusDot(whenName("", job[number_of_entry_points-1].whens[0])),
          clusterName(job[number_of_entry_points-1].whens[0]));
   
  // EndOfComputeLoop
  if (subgraph)
    fprintf(fTreeOutput,"subgraph clusterEndOfComputeLoop{\n\t\tlabel=\"EndOfComputeLoop\";\n\t\tsortv=%d;\n\
\t\tnode_EndOfComputeLoop [style=filled, shape=tripleoctagon, color=gray, label=\"End Of Compute Loop\", fontsize=32];\n}",iGraph+=1);

}


/*****************************************************************************
 * timeTreeSaveEdges
 *****************************************************************************/
__attribute__((unused))
static NABLA_STATUS timeTreeSaveEdges(FILE *fTreeOutput, astNode *l, astNode *father){
  for(;l not_eq NULL;l=l->next){
    fprintf(fTreeOutput, "\n\tnode_%d -> node_%d;", father->id, l->id);
    timeTreeSaveEdges(fTreeOutput, l->children, l);
  }
  return NABLA_OK;
}


/*****************************************************************************
 * timeTreeSave
 *****************************************************************************/
NABLA_STATUS nMiddleTimeTreeSave(nablaMain *nabla,
                                 nablaJob *jobs,
                                 int number_of_entry_points){
  FILE *dotFile;
  char fileName[NABLA_MAX_FILE_NAME];
  dbg("\n[timeTreeSave] Saving time tree for %s", nabla->name);
  sprintf(fileName, "%s.time.dot", nabla->name);
  // Saving tree file
  if ((dotFile=fopen(fileName, "w")) == 0) return NABLA_ERROR|dbg("[timeTreeSave] fopen ERROR");
  fprintf(dotFile, "digraph {\n\tedge[arrowhead=open];");
  dbg("\n[timeTreeSave] timeTreeSaveNodes");
  timeTreeSaveMathematica(dotFile,jobs,number_of_entry_points, nabla->optionDumpTree);
  //timeTreeSaveNodes(dotFile,jobs,number_of_entry_points,true);
  //timeTreeSaveNodes(dotFile,jobs,number_of_entry_points,false);
  //timeTreeSaveEdges(dotFile, jobs);
  //fprintf(dotFile, "\n\t//Start [shape=Mdiamond];\n\t//End [shape=Msquare];");
  fprintf(dotFile, "\n\tnode_ComputeLoopEnd_inf -> node_ComputeLoopBegin_0d00 [penwidth=1.0];");
  fprintf(dotFile, "\n}\n");
  fclose(dotFile);
  return NABLA_OK;
}		



/*

// *****************************************************************************
// * getInOutPutsNodes
// *****************************************************************************
void getInOutPutsNodes(FILE *fOut, astNode *n, char *color){
  if (fOut==NULL) return;
  if (n->token not_eq NULL){
    if (n->tokenid == CELL){
      fprintf(fOut, "\n\tcell_");color="CCDDCC";
    }else if (n->tokenid == NODE){
      fprintf(fOut, "\n\tnode_");color="DDCCCC";
    }else if (n->tokenid == FACE){
      fprintf(fOut, "\n\tface_");color="CCCCDD";
    }else{
      fprintf(fOut, "%s[shape = circle, color=\"#%s\"];", n->token, color);
      dbg("\n[getInOutPutsNodes] %s", n->token);
    }
  }
  if(n->children != NULL) getInOutPutsNodes(fOut, n->children, color);
  if(n->next != NULL) getInOutPutsNodes(fOut, n->next, color);
}


// *****************************************************************************
// * getInOutPutsEdges
// *****************************************************************************
void getInOutPutsEdges(FILE *fOut, astNode *n, int inout, char *nName1, char* nName2){
  if (fOut==NULL) return;
  if (n->token != NULL){
    if (n->tokenid == CELL){
      if (inout == OUT) fprintf(fOut, "\n\t%sJob_%s -> c", nName1, nName2);
      else fprintf(fOut, "\n\tc");
    }else if (n->tokenid == NODE){
      if (inout == OUT) fprintf(fOut, "\n\t%sJob_%s -> n", nName1, nName2);
      else fprintf(fOut, "\n\tn");
    }else if (n->tokenid == FACE){
      if (inout == OUT) fprintf(fOut, "\n\t%sJob_%s -> f", nName1, nName2);
      else fprintf(fOut, "\n\tn");
    }else{
      if (inout == IN) fprintf(fOut, "%s -> %sJob_%s;", n->token, nName1, nName2);
      else fprintf(fOut, "%s", n->token);
    }
  }
  if(n->children != NULL) getInOutPutsEdges(fOut, n->children, inout, nName1, nName2);
  if(n->next != NULL) getInOutPutsEdges(fOut, n->next, inout, nName1, nName2);
}
*/
