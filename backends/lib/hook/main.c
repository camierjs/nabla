///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
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

extern char* nablaAlephHeader(nablaMain*);


// ****************************************************************************
// * Forward des fonctions utilisées dans le main
// ****************************************************************************
#define BACKEND_MAIN_FORWARDS "\n\
extern \"C\" int inOpt(char *file,int *argc, char **argv);\n\n\n\
extern void glvis(const int,const int,const int,const double,double*,double*);\n\n\n"

// ****************************************************************************
// * Backend PREFIX - Génération du 'main'
// * look at c++/4.7/bits/ios_base.h for cout options
// ****************************************************************************
#define BACKEND_MAIN_FUNCTION "\n\
// ******************************************************************************\n\
// * Main\n\
// ******************************************************************************\n\
int main(int argc, char *argv[]){\n"

#define BACKEND_MAIN_VARIABLES "\
\tfloat alltime=0.0;\n\
\tstruct timeval st, et;\n\
\t__attribute__((unused)) const int NABLA_NB_PARTICLES=(argc==1)?1024:atoi(argv[1]);\n\
\t// Initialisation des Swirls\n\
\tint hlt_level=0;\n\
\tbool visualization=false;\n\
\tbool* hlt_exit=(bool*)calloc(64,sizeof(bool));\n\
\t// Initialisation de la précision du cout\n\
\tstd::cout.precision(14);//21, 14 pour Arcane\n\
\t//std::cout.setf(std::ios::floatfield);\n\
\tstd::cout.setf(std::ios::scientific, std::ios::floatfield);\n\
\t// ********************************************************\n\
\t// Initialisation du temps et du deltaT\n\
\t// ********************************************************\n\
\tdouble global_time[1]={0.0};//{option_dtt_initial};// Arcane fait comme cela!;\n\
\tint global_iteration[1]={1};\n\
\treal global_greek_deltat[1] = {set1(0.0)};// @ 0;\n\
\t//printf(\"\\n\\33[7;32m[main] time=%%e, iteration is #%%d\\33[m\",global_time[0],global_iteration[0]);\n"

#define BACKEND_MAIN_OPTIONS_PREFIX "\n\
\t// ********************************************************\n\
\t// Parse command-line options\n\
\t// ********************************************************\n\
\tint o; int longindex=0;\n\
\tconst struct option longopts[]={\n\
\t\t{\"vis\",no_argument,NULL,0x5d8a73d0},\n\
\t\t{\"org\",required_argument,NULL,0x3c0f6f4c},\n"
#define BACKEND_MAIN_OPTIONS_POSTFIX "\
\t\t{NULL,0,NULL,0}\n\
\t};\n"
#define BACKEND_MAIN_OPTIONS_WHILE_PREFIX "\
\twhile ((o=getopt_long_only(argc, argv, \"\",longopts,&longindex))!=-1){\n\
\t\tswitch (o){\n\
\t\tcase 0x5d8a73d0:{\n\
\t\t\tvisualization=true;\n\
\t\t\tbreak;\n\
\t\t\t};\n\
\t\tcase 0x3c0f6f4c:{//optorg\n\
\t\t\tconst int org_optind=optind;\n\
\t\t\toptind=0;\n\
\t\t\tchar *org_optarg=strdup(optarg);\n\
\t\t\tprintf(\"[1;33morg input file '%%s'[0m\\n\",optarg);\n\
\t\t\t{\n\t\t\t\tint org_o,org_longindex=0;\n\
\t\t\t\tint org_argc=0;\n\
\t\t\t\tchar **org_argv=(char**)calloc(1024,sizeof(char*));\n\
\t\t\t\tint status=inOpt(optarg,&org_argc,org_argv);\n\
\t\t\t\tprintf(\"[1;33morg_argc=%%d[0m\\n\",org_argc);\n\
\t\t\t\tassert(status==0);\n\
\t\t\t\twhile ((org_o=getopt_long_only(org_argc,\n\
\t\t\t\t\t\t\t\t\t\t\t\t\t\t\torg_argv,\"\",\n\
\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tlongopts,&org_longindex))!=-1){\n\
\t\t\t\t\tswitch (org_o){\n"

#define BACKEND_MAIN_OPTIONS_WHILE_PREFIX_BIS "\
\t\t\t\t\tdefault: exit(fprintf(stderr, \"[nabla] Error in orgopt option line\\n\"));\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\toptind=org_optind;\n\
\t\t\toptarg=org_optarg;\n\
\t\t\tbreak;}\n"

#define BACKEND_MAIN_OPTIONS_WHILE_POSTFIX "\
\t\tcase '?':\n\
\t\t\tif ((optopt>(int)'A')&&(optopt<(int)'z'))\n\
\t\t\t\texit(fprintf (stderr, \"\\n[nabla] Unknown option `-%%c'.\\n\", optopt));\n\
\t\tdefault: exit(fprintf(stderr, \"[nabla] Error in command line\\n\"));\n\
\t\t}\n\
\t}"


static void dumpOptions(nablaMain *nabla,const int tabs){
  for(nablaOption *opt=nabla->options;opt!=NULL;opt=opt->next){
    const char *ascii_name=opt->name;
    for(int t=tabs;t>0;t-=1) fprintf(nabla->entity->src,"\t"); fprintf(nabla->entity->src, "\t\tcase 0x%X: //%s %s\n",*(unsigned int*)&opt, opt->type, ascii_name);
    for(int t=tabs;t>0;t-=1) fprintf(nabla->entity->src,"\t"); fprintf(nabla->entity->src, "\t\t\tif (!optarg) break;\n");
    for(int t=tabs;t>0;t-=1) fprintf(nabla->entity->src,"\t"); fprintf(nabla->entity->src, "\t\t\tprintf(\"[1;33m%s %s = %%s[0m\\n\", optarg);", opt->type, opt->name);
    if (opt->type[0]=='r') fprintf(nabla->entity->src, " %s=atof(optarg);\n",opt->name);
    if (opt->type[0]=='i') fprintf(nabla->entity->src, " %s=atol(optarg);\n",opt->name);
    if (opt->type[0]=='b') fprintf(nabla->entity->src, " %s=(0==strcmp(optarg,\"true\"));\n",opt->name);
    for(int t=tabs;t>0;t-=1) fprintf(nabla->entity->src,"\t"); fprintf(nabla->entity->src, "\t\t\tbreak;\n");
  }
}


NABLA_STATUS xHookMainPrefix(nablaMain *nabla){
  dbg("\n[hookMainPrefix]");
  if ((nabla->entity->libraries&(1<<with_aleph))!=0)
    fprintf(nabla->entity->hdr, "%s", nablaAlephHeader(nabla));
  fprintf(nabla->entity->src, BACKEND_MAIN_FORWARDS);
  fprintf(nabla->entity->src, BACKEND_MAIN_FUNCTION);
  fprintf(nabla->entity->src, BACKEND_MAIN_VARIABLES);
  fprintf(nabla->entity->src, BACKEND_MAIN_OPTIONS_PREFIX);
  for(nablaOption *opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->src, "\t\t{\"%s\",required_argument,NULL,(int)0x%X},\n",opt->name,*(unsigned int*)&opt);
  fprintf(nabla->entity->src, BACKEND_MAIN_OPTIONS_POSTFIX);
  fprintf(nabla->entity->src, BACKEND_MAIN_OPTIONS_WHILE_PREFIX);

  dumpOptions(nabla,3);
  
  fprintf(nabla->entity->src, BACKEND_MAIN_OPTIONS_WHILE_PREFIX_BIS);
  
  dumpOptions(nabla,0);
  
  fprintf(nabla->entity->src, BACKEND_MAIN_OPTIONS_WHILE_POSTFIX);
  return NABLA_OK;
}

// ****************************************************************************
// * Backend PREINIT - Génération du 'main'
// ****************************************************************************
#define BACKEND_MAIN_PREINIT "\
\tconst nablaMesh msh={\n\
\t\tNABLA_NODE_PER_CELL,\n\
\t\tNABLA_CELL_PER_NODE,\n\
\t\tNABLA_CELL_PER_FACE,\n\
\t\tNABLA_NODE_PER_FACE,\n\
\t\tNABLA_FACE_PER_CELL,\n\
\n\
\t\tNABLA_NB_NODES_X_AXIS,\n\
\t\tNABLA_NB_NODES_Y_AXIS,\n\
\t\tNABLA_NB_NODES_Z_AXIS,\n\
\n\
\t\tNABLA_NB_CELLS_X_AXIS,\n\
\t\tNABLA_NB_CELLS_Y_AXIS,\n\
\t\tNABLA_NB_CELLS_Z_AXIS,\n\
\n\
\t\tNABLA_NB_FACES_X_INNER,\n\
\t\tNABLA_NB_FACES_Y_INNER,\n\
\t\tNABLA_NB_FACES_Z_INNER,\n\
\t\tNABLA_NB_FACES_X_OUTER,\n\
\t\tNABLA_NB_FACES_Y_OUTER,\n\
\t\tNABLA_NB_FACES_Z_OUTER,\n\
\t\tNABLA_NB_FACES_INNER,\n\
\t\tNABLA_NB_FACES_OUTER,\n\
\t\tNABLA_NB_FACES,\n\
\n\
\t\tNABLA_NB_NODES_X_TICK,\n\
\t\tNABLA_NB_NODES_Y_TICK,\n\
\t\tNABLA_NB_NODES_Z_TICK,\n\
\n\
\t\tNABLA_NB_NODES,\n\
\t\tNABLA_NODES_PADDING,\n\
\t\tNABLA_NB_CELLS,\n\
\t\tNABLA_NB_NODES_WARP,\n\
\t\tNABLA_NB_CELLS_WARP};\n\
\tprintf(\"%%d noeuds, %%d mailles & %%d faces\",NABLA_NB_NODES,NABLA_NB_CELLS,NABLA_NB_FACES);\n \
\tnabla_ini_connectivity(msh,node_coord,\n\t\t\t\t\t\t\t\t\txs_cell_node,xs_cell_prev,xs_cell_next,xs_cell_face,\n\t\t\t\t\t\t\t\t\txs_node_cell,xs_node_cell_corner,xs_node_cell_and_corner,\n\t\t\t\t\t\t\t\t\txs_face_cell,xs_face_node);\n"
NABLA_STATUS xHookMainPreInit(nablaMain *nabla){
  int i;
  dbg("\n[hookMainPreInit]");
  
  fprintf(nabla->entity->src, "\n\n\t// BACKEND_MAIN_PREINIT");

  if (isWithLibrary(nabla,with_real))
    xHookMesh1DConnectivity(nabla);
  else if (isWithLibrary(nabla,with_real2))
    xHookMesh2DConnectivity(nabla);
  else
    xHookMesh3DConnectivity(nabla,"xs");

  // Puis le BACKEND_MAIN_PREINIT
  // qui lance le nabla_ini_connectivity
  fprintf(nabla->entity->src, BACKEND_MAIN_PREINIT);
  
  nprintf(nabla,NULL,"\n\
\t// ****************************************************************\n\
\t// Initialisation des variables\n\
\t// ****************************************************************");
  // Variables Particulaires
  i=0;
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='p') continue;
    i+=1;
  }
  if (i>0){
    nprintf(nabla,NULL,"/*i=%d*/",i);
    nprintf(nabla,NULL,"\n\tFOR_EACH_PARTICLE(p){");
    for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='p') continue;
      nprintf(nabla,NULL,"\n\t\t%s_%s[p]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
    }
    nprintf(nabla,NULL,"\n\t}");
  }
  
  // Variables aux noeuds
  i=0;
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='n') continue;
    if (strcmp(var->name, "coord")==0) continue;
    i+=1;
  }
  if (i>0){
    nprintf(nabla,NULL,"\n\tFOR_EACH_NODE_WARP(n){");
    for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='n') continue;
      if (strcmp(var->name, "coord")==0) continue;
      nprintf(nabla,NULL,"\n\t\t%s_%s[n]=",var->item,var->name);
      if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
      if (strcmp(var->type, "real3x3")==0) nprintf(nabla,NULL,"real3x3();");
      if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
      if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
    }
    nprintf(nabla,NULL,"\n\t}");
  }
  
  // Variables aux mailles
  i=0;
  for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
    if (var->item[0]!='c') continue;
    i+=1;
  }
  if (i>0){
    nprintf(nabla,NULL,"\n\tFOR_EACH_CELL_WARP(c){");
    for(nablaVariable *var=nabla->variables;var!=NULL;var=var->next){
      if (var->item[0]!='c') continue;
      if (var->dim==0){
        nprintf(nabla,NULL,"\n\t\t%s_%s[c]=",var->item,var->name);
        if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"zero();");
        if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
        if (strcmp(var->type, "integer")==0) nprintf(nabla,NULL,"0;");
        if (strcmp(var->type, "real3x3")==0) nprintf(nabla,NULL,"real3x3();");
      }else{
        nprintf(nabla,NULL,"\n\t\tFOR_EACH_CELL_WARP_NODE(n)");
        nprintf(nabla,NULL," %s_%s[n+NABLA_NODE_PER_CELL*c]=",var->item,var->name);
        if (strcmp(var->type, "real")==0) nprintf(nabla,NULL,"0.0;");
        if (strcmp(var->type, "real2")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "real3")==0) nprintf(nabla,NULL,"real3();");
        if (strcmp(var->type, "real3x3")==0) nprintf(nabla,NULL,"real3x3();");
        if (strcmp(var->type, "int")==0) nprintf(nabla,NULL,"0;");
      }
    }
    nprintf(nabla,NULL,"\n\t}");
  }
  return NABLA_OK;
}


// ****************************************************************************
// * lambdaMainVarInitKernel
// ****************************************************************************
NABLA_STATUS xHookMainVarInitKernel(nablaMain *nabla){
  nprintf(nabla,NULL,"\n// xHookMainVarInitKernel");
  return NABLA_OK;
}


// ****************************************************************************
// * hookMainVarInitKernel
// ****************************************************************************
NABLA_STATUS xHookMainVarInitCall(nablaMain *nabla){
  nprintf(nabla,NULL,"\n\t/*xHookMainVarInitCall*/");
  return NABLA_OK;
}



// ****************************************************************************
// * hookMain
// ****************************************************************************
NABLA_STATUS xHookMainHLT(nablaMain *n){
  nablaVariable *var;
  nablaJob *entry_points;
  int i,numParams,number_of_entry_points;
  bool is_into_compute_loop=false;
  double last_when;
  n->HLT_depth=0;
  
  dbg("\n[hookMain]");
  number_of_entry_points=nMiddleNumberOfEntryPoints(n);
  entry_points=nMiddleEntryPointsSort(n,number_of_entry_points);
  
  // Et on rescan afin de dumper, on rajoute les +2 ComputeLoop[End|Begin]
  for(i=0,last_when=entry_points[i].whens[0];i<number_of_entry_points+2;++i){
    if (strncmp(entry_points[i].name,"hltDive",7)==0) continue;
    if (strcmp(entry_points[i].name,"ComputeLoopEnd")==0) continue;
    if (strcmp(entry_points[i].name,"ComputeLoopBegin")==0) continue;
    
    dbg("%s\n\t[hookMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    
    // Si l'on passe pour la première fois la frontière du zéro, on écrit le code pour boucler
    if (entry_points[i].whens[0]>=0 && is_into_compute_loop==false){
      is_into_compute_loop=true;
      nprintf(n, NULL,"\n\tgettimeofday(&st, NULL);\n\
\t//if (visualization) glvis(X_EDGE_ELEMS,Y_EDGE_ELEMS,Z_EDGE_ELEMS,LENGTH,(double*)node_coord,(double*)cell_e);\n\
\twhile ((global_time[0]<option_stoptime) && (global_iteration[0]!=option_max_iterations)){");
    }
    
    if (entry_points[i].when_depth==(n->HLT_depth+1)){
      nprintf(n, NULL, "\n\n\t// DIVING in HLT! (depth=%d)",n->HLT_depth);
      nprintf(n, NULL, "\n\thlt_exit[hlt_level=%d]=true;\n\tdo{",n->HLT_depth);
      n->HLT_depth=entry_points[i].when_depth;
    }
    if (entry_points[i].when_depth==(n->HLT_depth-1)){
      nprintf(n, NULL, "\n\t// Poping from HLT!");
//#warning HWed 'redo_with_a_smaller_time_step'
      nprintf(n, NULL, "\n\t}while(hlt_exit[%d]);hlt_level-=1;\n",entry_points[i].when_depth);
      n->HLT_depth=entry_points[i].when_depth;
    }
    
    // \n ou if d'un IF after '@'
    if (entry_points[i].ifAfterAt!=NULL){
      dbg("\n\t[hookMain] dumpIfAfterAt!");
      nprintf(n, NULL, "\n\t\tif (");
      nMiddleDumpIfAfterAt(entry_points[i].ifAfterAt, n,false);
      nprintf(n, NULL, ") ");
    }else nprintf(n, NULL, "\n");
    
    // On provoque un parallel->sync
    // si l'on découvre un temps logique différent
    if (last_when!=entry_points[i].whens[0])
      nprintf(n, NULL, "%s",
              is_into_compute_loop?"\t\t":"\t");
    last_when=entry_points[i].whens[0];
    
    // Dump de la tabulation et du nom du point d'entrée
    //#warning f or s?
    nprintf(n, NULL, "%s%s(",
            is_into_compute_loop?"\t\t":"\t",
            ///*@%f*/entry_points[i].whens[0],
            //n->call->parallel->spawn(),
            entry_points[i].name);
        
    // On s'autorise un endroit pour insérer des arguments
    nMiddleArgsDumpFromDFS(n,&entry_points[i]);
    
    // Si on doit appeler des jobs depuis cette fonction @ée
    if (entry_points[i].called_variables != NULL){
      if (!entry_points[i].reduction)
         nMiddleArgsAddExtra(n,&numParams);
      //else nprintf(n, NULL, "NABLA_NB_CELLS_WARP,");
      // Et on rajoute les called_variables en paramètre d'appel
      dbg("\n\t[hookMain] Et on rajoute les called_variables en paramètre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }//else nprintf(n,NULL,"/*NULL_called_variables*/");
    nprintf(n, NULL, ");");
  }
  free(entry_points);
  return NABLA_OK;
}

// ****************************************************************************
// * Backend POSTINIT - Génération du 'main'
// ****************************************************************************
#define BACKEND_MAIN_POSTINIT "\n\t\t//BACKEND_MAIN_POSTINIT"
NABLA_STATUS xHookMainPostInit(nablaMain *nabla){
  dbg("\n[hookMainPostInit]");
  fprintf(nabla->entity->src, BACKEND_MAIN_POSTINIT);
  return NABLA_OK;
}

// ****************************************************************************
// * Backend POSTFIX - Génération du 'main'
// ****************************************************************************
#define BACKEND_MAIN_POSTFIX "\n\t\t//BACKEND_MAIN_POSTFIX\
\n\t\tglobal_time[0]+=*(double*)&global_greek_deltat[0];\
\n\t\tglobal_iteration[0]+=1;\
\n\t\t//printf(\"\\ntime=%%e, dt=%%e\\n\", global_time[0], *(double*)&global_greek_deltat[0]);\
\n\t}\
\n\tgettimeofday(&et, NULL);\n\
\n\tif (visualization) glvis(X_EDGE_ELEMS,Y_EDGE_ELEMS,Z_EDGE_ELEMS,LENGTH,(double*)node_coord,(double*)cell_p);\n\
\talltime = ((et.tv_sec-st.tv_sec)*1000.+ (et.tv_usec - st.tv_usec)/1000.0);\n\
\tprintf(\"\\n\\t\\33[7m[#%%04d] Elapsed time = %%12.6e(s)\\33[m\\n\", global_iteration[0]-1, alltime/1000.0);\n"


// ****************************************************************************
// * hookMainPostfix
// ****************************************************************************
NABLA_STATUS xHookMainPostfix(nablaMain *nabla){
  fprintf(nabla->entity->src, BACKEND_MAIN_POSTFIX);
  xHookMeshFreeConnectivity(nabla);
  return NABLA_OK;
}


