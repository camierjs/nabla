#include "nabla.h"

int yylineno;
extern FILE *yyin;
#define YYSTYPE astNode*
int yyparse (astNode **);
char *nabla_input_file=NULL; 


// *****************************************************************************
// * yyerror
// *****************************************************************************
void yyerror(astNode **root, char *error){
  fflush(stdout);
  printf("%s:%d: %s\n",nabla_input_file,yylineno, error);
}


/*****************************************************************************
 * Nabla Parsing
 *****************************************************************************/
NABLA_STATUS nablaParsing(const char *preprocessedNfile,
                          const char *nabla_entity_name,
                          const bool optionDumpTree,
                          char *npFileName,
                          const BACKEND_SWITCH backend,
                          const BACKEND_COLORS colors,
                          char *interface_name,
                          char *specific_path,
                          char *service_name){
  astNode *root=NULL;  
  
  if(!(yyin = fopen(npFileName,"r")))
    return NABLA_ERROR | dbg("\n[nablaParsing] Could not open '%s' file", preprocessedNfile);

  dbg("\n[nablaParsing] Starting parsing");
  if (yyparse(&root)){
    fclose(yyin);
    return NABLA_ERROR | dbg("\n[nablaParsing] Error while parsing!");
  }
  dbg("\n[nablaParsing] Closing & Quit");
  fclose (yyin);
  
  dbg("\n[nablaParsing] On scan l'arbre pour transformer les tokens en UTF8");
  dfsUtf8(root);
  
  if (optionDumpTree){
    dbg("\n[nablaParsing] On dump l'arbre créé");
    astTreeSave(nabla_entity_name, root);
  }
  
  // Initial files setup and checkup
  if (nabla_entity_name==NULL)
    return NABLA_ERROR | dbg("\n[nccParseur] No entity name has been set!");

  dbg("\n[nablaParsing] nabla_entity_name=%s", nabla_entity_name);
  dbg("\n[nablaParsing] nabla_input_file=%s", nabla_input_file);

  dbg("\n[nablaParsing] Now launching nablaMiddlendSwitch");
  return nablaMiddlendSwitch(root,
                             optionDumpTree,
                             preprocessedNfile,
                             nabla_entity_name,
                             backend,
                             colors,
                             interface_name,
                             specific_path,
                             service_name);
}


/*****************************************************************************
 * $(CPATH)/gcc -std=c99 -E -Wall -x c $(TGT).nabla -o $(TGT).n
 *****************************************************************************/
int sysPreprocessor(const char *nabla_entity_name,
                    const char *list_of_nabla_files,
                    const char *unique_temporary_file_name,
                    const int unique_temporary_file_fd){
  int i=0;
  char cat_sed_temporary_file_name[NABLA_MAX_FILE_NAME];
  char tok_command[NABLA_MAX_FILE_NAME];
  char cat_command[NABLA_MAX_FILE_NAME];
  char gcc_command[NABLA_MAX_FILE_NAME];
  char *nabla_file, *dup_list_of_nabla_files=strdup(list_of_nabla_files);
    
  // On crée un fichier temporaire où l'on va sed'er les includes, par exemple
  sprintf(cat_sed_temporary_file_name, "/tmp/nabla_%s_sed_XXXXXX", nabla_entity_name);
  mkstemp(cat_sed_temporary_file_name);
  dbg("\n[sysPreprocessor] cat_sed_temporary_file_name is %s",cat_sed_temporary_file_name);

  // Pour chaque fichier .n en entrée, on va le cat'er et insérer des délimiteurs
  cat_command[0]='\0';
  printf("Loading: ");
  for(i=0,nabla_file=strtok(dup_list_of_nabla_files, " ");
      nabla_file!=NULL;
      i+=1,nabla_file=strtok(NULL, " ")){
    printf("%s%s",i==0?"":", ",nabla_file);
    snprintf(tok_command,NABLA_MAX_FILE_NAME-1,          
             "%scat --squeeze-blank %s|sed -e 's/#include/ include/g' %s %s",
             i==0?"":" && ",
             nabla_file,
             i==0?">":">>",
             cat_sed_temporary_file_name);
    strcat(cat_command,tok_command);
    //printf("\ncat_command: %s", cat_command);
    dbg("\n\n[sysPreprocessor] cat_command is %s",cat_command);
  }
  free(dup_list_of_nabla_files);
  printf("\n");
  //printf("\nfinal_cat_command: %s\n", cat_command);

  // On lance la commande de cat précédemment créée
  if (system(cat_command)<0)
    exit(NABLA_ERROR|fprintf(stderr, "\n[nablaPreprocessor] Error in system cat command!\n"));

  // Et on lance la commande de préprocessing
  snprintf(gcc_command,NABLA_MAX_FILE_NAME-1, // -P option
           "/usr/local/opendev1/gcc/gcc/4.8.1/bin/gcc -P -std=c99 -E -Wall -x c %s>/proc/%d/fd/%d",
           cat_sed_temporary_file_name,
           getpid(),
           unique_temporary_file_fd);
  dbg("\n[sysPreprocessor] gcc_command=%s", gcc_command);
  return system(gcc_command);
}
 

/*****************************************************************************
 * nablaPreprocessor
 *****************************************************************************/
void nablaPreprocessor(char *nabla_entity_name,
                       char *list_of_nabla_files,
                       char *unique_temporary_file_name,
                       const int unique_temporary_file_fd){
  char *scanForSpaceToPutZero;
  // Saving list of ∇ files for yyerror
  //nabla_input_file=strdup(list_of_nabla_files);
  nabla_input_file=strdup(unique_temporary_file_name);
  printf("nabla:1: %s\n",nabla_input_file);
  //printf("%s:1:\n",unique_temporary_file_name);

  // Scanning list_of_nabla_files to get fetch the first only filename
  // It allows emacs to go to this file+line when a syntax error is discovered
  scanForSpaceToPutZero=nabla_input_file;
  while(*scanForSpaceToPutZero!=32){
    scanForSpaceToPutZero+=1;
    if (*scanForSpaceToPutZero==0) break;
  }
  if (*scanForSpaceToPutZero==32) *scanForSpaceToPutZero=0;
  dbg("\n[nablaPreprocessor] But giving nabla_input_file = %s, unique_temporary_file = %s",
      list_of_nabla_files, unique_temporary_file_name);
  
  if (sysPreprocessor(nabla_entity_name,
                      list_of_nabla_files,
                      unique_temporary_file_name,
                      unique_temporary_file_fd)!=0)
    exit(NABLA_ERROR|fprintf(stderr, "\n[nablaPreprocessor] Error in preprocessor stage\n"));
  dbg("\n[nablaPreprocessor] done!");
}


/*****************************************************************************
 * Main
 *****************************************************************************/
int main(int argc, char * argv[]){
  int c;
  BACKEND_SWITCH backend=BACKEND_VOID;
  BACKEND_COLORS backend_color=BACKEND_COLOR_VOID;
  bool optionDumpTree=false;
  char *nabla_entity_name=NULL;
  int longindex=0;
  char *preprocessedNfile=NULL;
  char *interface_name=NULL;
  char *specific_path=NULL;
  char *service_name=NULL;
  char unique_temporary_file_name[NABLA_MAX_FILE_NAME];
  int unique_temporary_file_fd=0;
  char *input_file_list=NULL;
  static struct option longopts[]={  
    {"arcane",no_argument,NULL,BACKEND_ARCANE},
       {"alone",required_argument,NULL,BACKEND_COLOR_ARCANE_ALONE},
       {"module",required_argument,NULL,BACKEND_COLOR_ARCANE_MODULE},
       {"service",required_argument,NULL,BACKEND_COLOR_ARCANE_SERVICE},

    {"cuda",required_argument,NULL,BACKEND_CUDA},

    {"okina",required_argument,NULL,BACKEND_OKINA},
       {"tiling",no_argument,NULL,BACKEND_COLOR_OKINA_TILING},
       {"std",no_argument,NULL,BACKEND_COLOR_OKINA_STD},
       {"sse",no_argument,NULL,BACKEND_COLOR_OKINA_SSE},
       {"avx",no_argument,NULL,BACKEND_COLOR_OKINA_AVX},
       {"avx2",no_argument,NULL,BACKEND_COLOR_OKINA_AVX2},
       {"mic",no_argument,NULL,BACKEND_COLOR_OKINA_MIC},
       {"cilk",no_argument,NULL,BACKEND_COLOR_OKINA_CILK},
       {"omp",no_argument,NULL,BACKEND_COLOR_OKINA_OpenMP},
       {"seq",no_argument,NULL,BACKEND_COLOR_OKINA_SEQ},
       {"soa",no_argument,NULL,BACKEND_COLOR_OKINA_SOA},
       {"aos",no_argument,NULL,BACKEND_COLOR_OKINA_AOS},
    {NULL,0,NULL,0}
  }; 
  
  // Setting null bytes ('\0') at the beginning of dest, before concatenation
  input_file_list=calloc(NABLA_MAX_FILE_NAME,sizeof(char));
 
  if (argc<=1)
    exit(fprintf(stderr, "[1;36mNAME[0m\n\
\t[1;36mnabla[0m - Numerical Analysis Based LAnguage's\n\
\t        Optimized Code Generator for Specific Architectures\n\
[1;36mSYNOPSIS[0m\n\
\t[1;36mnabla[0m [-t] [-v [4mlogfile[0m] [options] [4mpossible_target[0m -i [4minput file list[0m\n\
[1;36mWARNING[0m\n\
\tThe [1;36mnabla[0m generator is still under heavy development and\n\
\tdistributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;\n\
\twithout even the implied warranty of FITNESS FOR A PARTICULAR PURPOSE.\n\
[1;36mDESCRIPTION[0m\n\
\tThe purpose of the [4mnabla[0m utility is to generate automatically\n\
\tthe required files for each specified target.\n\
[1;36mOPTIONS[0m\n\
\t[1;36m-t[0m\t\tGenerate the intermediate AST dot files\n\
\t[1;36m-v [4mlogfile[0m\tGenerate intermediate debug info to [4mlogfile[0m\n\
[4mpossible_target[0m is:\n\
\t[1;36m--arcane [4mname[0m\tCode generation for the target ARCANE\n\
\t\t[36m--alone[0m\t\tGenerate a [4mstand-alone[0m application\n\
\t\t[36m--module[0m\tGenerate a [4mmodule[0m\n\
\t\t[36m--service[0m\tGenerate a [4mservice[0m\n\
\t\t\t[36m-I [4mname[0m\t\tInterface name to use\n\
\t\t\t[36m-p [4mpath[0m\t\tPath of the interface file\n\
\t\t\t[36m-n [4mname[0m\t\tService name that'll be generate\n\
\t[1;36m--cuda [4mname[0m\tCode generation for the target CUDA\n\
\t\t\tCUDA does not yet support all of the libraries you would like\n\
\t[1;36m--okina [4mname[0m\tCode generation for the target XeonPhi\n\
\t\t[36m--std[0m\t\tSTD 1=2^(WARP_BIT=0) vector depth\n\
\t\t[36m--sse[0m\t\tSSE 2=2^(WARP_BIT=1) vector depth\n\
\t\t[36m--avx[0m\t\tAVX 4=2^(WARP_BIT=2) vector depth\n\
\t\t[36m--avx2[0m\t\tAVX 4=2^(WARP_BIT=2) vector depth + gather & FMA\n\
\t\t[36m--mic[0m\t\tMIC 8=2^(WARP_BIT=3) vector depth\n\
\t\t[36m--cilk[0m\t\tCilk+ parallel implementation\n\
\t\t[36m--omp[0m\t\tOpenMP parallel implementation\n\
\t\t[36m--seq[0m\t\tNo parallel implementation\n\
[1;36mEMACS MODE[0m\n\
\t(setq load-path (cons \"/cea/BS/home/s3/camierjs/.emacs.d\" load-path))\n\
\t(autoload 'nabla-mode \"nabla-mode\" \"Nabla Mode\" t)\n\
\t(setq auto-mode-alist (cons '(\"\\.n\\'\" . nabla-mode) auto-mode-alist))\n\
\t(set-selection-coding-system 'utf-8)\n\
\t(setq locale-coding-system 'utf-8)\n\
\t(set-language-environment 'utf-8)\n\
[1;36mAUTHOR[0m\n\
\tJean-Sylvain Camier (#5568) <jean-sylvain.camier@cea.fr>\n\
[1;36mBUGS[0m\n\
\tTest bugs are to be reported to the above address.\n"));
//\t\t[36m--soa[0m\t\tSoA for coordx,coordy,coordz+Reals\n        \
//\t\t[36m--aos[0m\t\tAoS for coords+Real3s\n                     \
//\t\t[36m(--tiling[0m\tDiced domain decomposition approach)\n    \

  while ((c=getopt_long(argc, argv, "tv:I:p:n:i:",longopts,&longindex))!=-1){
    switch (c){
      
      // ************************************************************
      // * Standard OPTIONS
      // ************************************************************      
    case 't': // DUMP tree option
      optionDumpTree=true;
      dbg("\n[nabla] Command line specifies to dump the tree");
      break;

    case 'v': // DEBUG MANAGEMENT
      if (dbgSet(DBG_ALL) == DBG_OFF) return NABLA_ERROR;
      dbgOpenTraceFile(optarg);
      dbg("[nabla] Command line specifies debug file: %s", optarg);
      break;


      // ************************************************************
      // * BACKEND ARCANE avec ses variantes ALONE, MODULE ou SERVICE
      // ************************************************************      
    case BACKEND_ARCANE:
      backend=BACKEND_ARCANE;
      dbg("\n[nabla] Command line hits target ARCANE (%s)", longopts[longindex].name);
      break;
      
    case BACKEND_COLOR_ARCANE_ALONE:
      backend_color=BACKEND_COLOR_ARCANE_ALONE;
      dbg("\n[nabla] Command line specifies ARCANE's STAND-ALONE option");
      nabla_entity_name=strdup(optarg);
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, unique_temporary_file_name);
      dbg("\n[nabla] Command line specifies new ARCANE nabla_entity_name: %s", nabla_entity_name);
      break;
      
    case 'p': // specific path to source directory for a module or a service
      specific_path=strdup(optarg);
      dbg("\n[nabla] Command line specifies ARCANE's path: %s", specific_path);
      break;

    case BACKEND_COLOR_ARCANE_MODULE:
      backend_color=BACKEND_COLOR_ARCANE_MODULE;
      dbg("\n[nabla] Command line specifies ARCANE's MODULE option");
      nabla_entity_name=strdup(optarg);
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, unique_temporary_file_name);
      dbg("\n[nabla] Command line specifies new ARCANE nabla_entity_name: %s", nabla_entity_name);
      break;
      
    case BACKEND_COLOR_ARCANE_SERVICE:
      backend_color=BACKEND_COLOR_ARCANE_SERVICE;
      dbg("\n[nabla] Command line specifies ARCANE's SERVICE option");
      nabla_entity_name=strdup(optarg);
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, unique_temporary_file_name);
      dbg("\n[nabla] Command line specifies new ARCANE nabla_entity_name: %s", nabla_entity_name);
      break;
    case 'I': // Interface name
      interface_name=strdup(optarg);
      dbg("\n[nabla] Command line specifies ARCANE's SERVICE interface name: %s", interface_name);
      break;
    case 'n': // Service name
      service_name=strdup(optarg);
      dbg("\n[nabla] Command line specifies ARCANE's SERVICE service name: %s", service_name);
      break;
      

      // ************************************************************
      // * BACKEND OKINA avec ses variantes
      // ************************************************************
    case BACKEND_OKINA:
      backend=BACKEND_OKINA;
      backend_color=BACKEND_COLOR_VOID;
      dbg("\n[nabla] Command line hits long option %s", longopts[longindex].name);
      nabla_entity_name=strdup(optarg);
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, unique_temporary_file_name);
      dbg("\n[nabla] Command line specifies new OKINA nabla_entity_name: %s", nabla_entity_name);
      break;
    case BACKEND_COLOR_OKINA_TILING:
      backend_color=BACKEND_COLOR_OKINA_TILING;
      dbg("\n[nabla] Command line specifies OKINA's tiling option");
      break;
    case BACKEND_COLOR_OKINA_STD:
      backend_color|=BACKEND_COLOR_OKINA_STD;
      dbg("\n[nabla] Command line specifies OKINA's STD option");
      break;
    case BACKEND_COLOR_OKINA_SSE:
      backend_color|=BACKEND_COLOR_OKINA_SSE;
      dbg("\n[nabla] Command line specifies OKINA's SSE option");
      break;
    case BACKEND_COLOR_OKINA_AVX:
      backend_color|=BACKEND_COLOR_OKINA_AVX;
      dbg("\n[nabla] Command line specifies OKINA's AVX option");
      break;
    case BACKEND_COLOR_OKINA_AVX2:
      backend_color|=BACKEND_COLOR_OKINA_AVX2;
      dbg("\n[nabla] Command line specifies OKINA's AVX2 option");
      break;
    case BACKEND_COLOR_OKINA_MIC:
      backend_color|=BACKEND_COLOR_OKINA_MIC;
      dbg("\n[nabla] Command line specifies OKINA's MIC option");
      break;
    case BACKEND_COLOR_OKINA_CILK:
      backend_color|=BACKEND_COLOR_OKINA_CILK;
      dbg("\n[nabla] Command line specifies OKINA's CILK option");
      break;
    case BACKEND_COLOR_OKINA_OpenMP:
      backend_color|=BACKEND_COLOR_OKINA_OpenMP;
      dbg("\n[nabla] Command line specifies OKINA's OpenMP option");
      break;
    case BACKEND_COLOR_OKINA_SEQ:
      backend_color|=BACKEND_COLOR_OKINA_SEQ;
      dbg("\n[nabla] Command line specifies OKINA's SEQ option");
      break;
    case BACKEND_COLOR_OKINA_AOS:
      backend_color|=BACKEND_COLOR_OKINA_AOS;
      dbg("\n[nabla] Command line specifies OKINA's AoS option");
      break;
    case BACKEND_COLOR_OKINA_SOA:
      backend_color|=BACKEND_COLOR_OKINA_SOA;
      dbg("\n[nabla] Command line specifies OKINA's SoA option");
      break;

      // ************************************************************
      // * BACKEND CUDA avec pas de variantes pour l'instant
      // ************************************************************
    case BACKEND_CUDA:
      backend=BACKEND_CUDA;
      dbg("\n[nabla] Command line hits long option %s", longopts[longindex].name);
      nabla_entity_name=strdup(optarg);
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, unique_temporary_file_name);
      dbg("\n[nabla] Command line specifies new CUDA nabla_entity_name: %s", nabla_entity_name);
      break;

       
    case 'i': // INPUT FILES
      strcat(input_file_list,optarg);
      dbg("\n[nabla] first input_file_list: %s ", input_file_list);
      while (optind < argc){
        input_file_list=strcat(input_file_list," ");
        input_file_list=strcat(input_file_list,argv[optind++]);
        dbg("\n[nabla] next input_file_list: %s ", input_file_list);
      }
      break;
       
    case '?': // UNKNOWN OPTIONS
      dbg("\n[nabla] UNKNOWN OPTIONS");
      if ((optopt>(int)'A')&&(optopt<(int)'z')) fprintf (stderr, "\n[nabla] Unknown option `-%c'.\n", optopt);
      else fprintf (stderr, "\n[nabla] Unknown option character `\\%d'.\n", optopt);
      exit(NABLA_ERROR);
    default: exit(NABLA_ERROR|fprintf(stderr, "\n[nabla] Error in command line\n"));
    }
  }
  
  if (nabla_entity_name==NULL)
    exit(NABLA_ERROR|fprintf(stderr,"\n[nabla] Error with entity name!\n"));

  if (backend==BACKEND_VOID)
    exit(NABLA_ERROR|fprintf(stderr,"\n[nabla] Error with target switch!\n"));
 
  if (unique_temporary_file_fd==0)
    exit(NABLA_ERROR|fprintf(stderr,"\n[nabla] Error in unique temporary file\n"));
  
  if (input_file_list==NULL)
    exit(NABLA_ERROR|fprintf(stderr,"\n[nabla] Error in input_file_list\n"));

  // On a notre fichier temporaire et la listes des fichiers ∇ à parser
  nablaPreprocessor(nabla_entity_name,
                    input_file_list,
                    unique_temporary_file_name,
                    unique_temporary_file_fd);

  dbg("\n[nabla] Now triggering nablaParsing with these options");
  if (nablaParsing(preprocessedNfile,
                   nabla_entity_name?nabla_entity_name:argv[argc-1],
                   optionDumpTree,
                   unique_temporary_file_name,
                   backend,
                   backend_color,
                   interface_name,
                   specific_path,
                   service_name)!=NABLA_OK)
    exit(NABLA_ERROR|fprintf(stderr, "\n[nabla] nablaParsing error\n"));
  if (dbgGet()==DBG_OFF)
    unlink(unique_temporary_file_name);
  return NABLA_OK;
}

