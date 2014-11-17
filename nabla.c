#include "nabla.h"

extern int yylineno;
extern FILE *yyin;
#define YYSTYPE astNode*
int yyparse (astNode **);
extern char nabla_input_file[]; 
static void nabla_error_print_progname(void){/* No program  name here */}
static char *unique_temporary_file_name=NULL;


// *****************************************************************************
// * NABLA_MAN
// *****************************************************************************
#define NABLA_MAN "[1;36mNAME[0m\n\
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
\tTest bugs are to be reported to the above address.\n"
//\t\t[36m--soa[0m\t\tSoA for coordx,coordy,coordz+Reals\n
//\t\t[36m--aos[0m\t\tAoS for coords+Real3s\n
//\t\t[36m(--tiling[0m\tDiced domain decomposition approach)\n


// *****************************************************************************
// * nabla_unlink
// *****************************************************************************
static void nabla_unlink(void){
  // Unlinking temp files
  if (unique_temporary_file_name!=NULL)
    unlink(unique_temporary_file_name);
}


// *****************************************************************************
// * nabla_error
// * int vsprintf(char *str, const char *format, va_list ap);
// *****************************************************************************
void nabla_error(const char *format,...){
  //int n;
  //const int size = 8192;
  //char *error_msg;
  va_list args;
  
  //if ((error_msg = malloc(size))==NULL)    error(!0,0,"Could not even malloc for our error message!");

  va_start(args, format);
  //assert(error_msg);
  error_at_line(!0,0,nabla_input_file, yylineno-1, format,args);

  //n=vsnprintf(error_msg,size,format,args);
  va_end(args);

  // We do not want to unlink yet
  // nabla_unlink();
  //assert(nabla_input_file!=NULL);
  //printf("nabla_input_file=%s\n",nabla_input_file);
  //if (n > -1 && n < size)    error_at_line(!0,0,"strdup(nabla_input_file)", yylineno, error_msg);
  //free(error_msg);
}


// *****************************************************************************
// * yyerror
// *****************************************************************************
void yyerror(astNode **root, char *error){
  fflush(stdout);
  printf("%s:%d: %s\n",nabla_input_file,yylineno-1, error);
}


/*****************************************************************************
 * Nabla Parsing
 *****************************************************************************/
NABLA_STATUS nablaParsing(const char *nabla_entity_name,
                          const bool optionDumpTree,
                          char *npFileName,
                          const BACKEND_SWITCH backend,
                          const BACKEND_COLORS colors,
                          char *interface_name,
                          char *specific_path,
                          char *service_name){
  astNode *root=NULL;  
  if(!(yyin=fopen(npFileName,"r")))
    return NABLA_ERROR | dbg("\n[nablaParsing] Could not open '%s' file", npFileName);

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
  const int size = NABLA_MAX_FILE_NAME;
  int cat_sed_temporary_fd=0;
  char *cat_sed_temporary_file_name=NULL;
  char *tok_command=NULL;
  char *cat_command=NULL;
  char *gcc_command=NULL;
  char *nabla_file, *dup_list_of_nabla_files=strdup(list_of_nabla_files);
  
  if ((cat_sed_temporary_file_name = malloc(size))==NULL)
    error(!0,0,"[sysPreprocessor] Could not malloc cat_sed_temporary_file_name!");
  if ((tok_command = malloc(size))==NULL)
    error(!0,0,"[sysPreprocessor] Could not malloc tok_command!");
  if ((cat_command = malloc(size))==NULL)
    error(!0,0,"[sysPreprocessor] Could not malloc cat_command!");
  if ((gcc_command = malloc(size))==NULL)
    error(!0,0,"[sysPreprocessor] Could not malloc gcc_command!");

  // On crée un fichier temporaire où l'on va sed'er les includes, par exemple
  snprintf(cat_sed_temporary_file_name, size, "/tmp/nabla_%s_sed_XXXXXX", nabla_entity_name);
  cat_sed_temporary_fd=mkstemp(cat_sed_temporary_file_name);
  if (cat_sed_temporary_fd==-1)
    error(!0,0,"[sysPreprocessor] Could not mkstemp cat_sed_temporary_fd!");
  //printf("%s:1: is our temporary sed file\n",cat_sed_temporary_file_name);
  dbg("\n[sysPreprocessor] cat_sed_temporary_file_name is %s",cat_sed_temporary_file_name);

  // Pour chaque fichier .n en entrée, on va le cat'er et insérer des délimiteurs
  cat_command[0]='\0';
  //printf("Loading: ");
  for(i=0,nabla_file=strtok(dup_list_of_nabla_files, " ");
      nabla_file!=NULL;
      i+=1,nabla_file=strtok(NULL, " ")){
    //printf("%s%s",i==0?"":", ",nabla_file);
    // Une ligne de header du cat en cours
    snprintf(tok_command,size,
             "%secho '# 1 \"%s\"' %s %s",
             i==0?"":" && ",
             nabla_file,
             i==0?">":">>",
             cat_sed_temporary_file_name);
    strcat(cat_command,tok_command);
    snprintf(tok_command,size,
             " && cat %s|sed -e 's/#include/ include/g'>> %s",//--squeeze-blank
             nabla_file,
             cat_sed_temporary_file_name);
    strcat(cat_command,tok_command);
    //printf("\ncat_command: %s", cat_command);
    dbg("\n\n[sysPreprocessor] cat_command is %s",cat_command);
  }
  free(dup_list_of_nabla_files);
  //printf("\n");
  //printf("\nfinal_cat_command: %s\n", cat_command);

  // On lance la commande de cat précédemment créée
  if (system(cat_command)<0)
    exit(NABLA_ERROR|fprintf(stderr, "\n[nablaPreprocessor] Error in system cat command!\n"));

  // Et on lance la commande de préprocessing
  // -P Inhibit generation of linemarkers in the output from the preprocessor.
  // This might be useful when running the preprocessor on something that is not C code,
  // and will be sent to a program which might be confused by the linemarkers.
  // -C  Do not discard comments.
  // All comments are passed through to the output file, except for comments in processed directives,
  // which are deleted along with the directive.
  snprintf(gcc_command,size,
           "gcc -std=c99 -C -E -Wall -x c %s>/proc/%d/fd/%d",
           cat_sed_temporary_file_name,
           getpid(),
           unique_temporary_file_fd);
  dbg("\n[sysPreprocessor] gcc_command=%s", gcc_command);
  return system(gcc_command);
}
 


// ****************************************************************************
// * nablaPreprocessor
// ****************************************************************************
void nablaPreprocessor(char *nabla_entity_name,
                       char *list_of_nabla_files,
                       char *unique_temporary_file_name,
                       const int unique_temporary_file_fd){
  //char *scanForSpaceToPutZero;
  // Saving list of ∇ files for yyerror
  //nabla_input_file=strdup(list_of_nabla_files);
  //nabla_input_file=strdup(unique_temporary_file_name);
  printf("%s:1: is our temporary file\n",unique_temporary_file_name);
  //printf("%s:1: is our nabla_input_file\n",nabla_input_file);

  // Scanning list_of_nabla_files to fetch only the first filename
  // It allows emacs to go to this file+line when a syntax error is discovered
  /*scanForSpaceToPutZero=nabla_input_file;
  while(*scanForSpaceToPutZero!=32){
    scanForSpaceToPutZero+=1;
    if (*scanForSpaceToPutZero==0) break;
  }
  if (*scanForSpaceToPutZero==32) *scanForSpaceToPutZero=0;
  dbg("\n[nablaPreprocessor] But giving nabla_input_file = %s, unique_temporary_file = %s",
  list_of_nabla_files, unique_temporary_file_name);*/
  
  if (sysPreprocessor(nabla_entity_name,
                      list_of_nabla_files,
                      unique_temporary_file_name,
                      unique_temporary_file_fd)!=0)
    exit(NABLA_ERROR|fprintf(stderr, "\n[nablaPreprocessor] Error in preprocessor stage\n"));
  dbg("\n[nablaPreprocessor] done!");
}


// ***************************************************************************
// * Main
// ***************************************************************************
int main(int argc, char * argv[]){
  int c;
  BACKEND_SWITCH backend=BACKEND_VOID;
  BACKEND_COLORS backend_color=BACKEND_COLOR_VOID;
  bool optionDumpTree=false;
  char *nabla_entity_name=NULL;
  int longindex=0;
  char *interface_name=NULL;
  char *specific_path=NULL;
  char *service_name=NULL;
  int unique_temporary_file_fd=0;
  char *input_file_list=NULL;
  const struct option longopts[]={
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
       {"gcc",no_argument,NULL,BACKEND_COLOR_OKINA_GCC},
       {"icc",no_argument,NULL,BACKEND_COLOR_OKINA_ICC},
    {NULL,0,NULL,0}
  };

  // Set our nabla_error_print_progname for emacs to be able to visit
  error_print_progname=&nabla_error_print_progname;

  // Setting null bytes ('\0') at the beginning of dest, before concatenation
  input_file_list=calloc(NABLA_MAX_FILE_NAME,sizeof(char));

  // Check for at least several arguments
  if (argc<=1)
    exit(0&fprintf(stderr, NABLA_MAN));

  // Now switch the arguments
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
      // * INPUT FILES
      // ************************************************************      
    case 'i':
      strcat(input_file_list,optarg);
      dbg("\n[nabla] first input_file_list: %s ", input_file_list);
      while (optind < argc){
        input_file_list=strcat(input_file_list," ");
        input_file_list=strcat(input_file_list,argv[optind++]);
        dbg("\n[nabla] next input_file_list: %s ", input_file_list);
      }
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
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, &unique_temporary_file_name);
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
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, &unique_temporary_file_name);
      dbg("\n[nabla] Command line specifies new ARCANE nabla_entity_name: %s", nabla_entity_name);
      break;
      
    case BACKEND_COLOR_ARCANE_SERVICE:
      backend_color=BACKEND_COLOR_ARCANE_SERVICE;
      dbg("\n[nabla] Command line specifies ARCANE's SERVICE option");
      nabla_entity_name=strdup(optarg);
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, &unique_temporary_file_name);
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
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, &unique_temporary_file_name);
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
    case BACKEND_COLOR_OKINA_GCC:
      backend_color|=BACKEND_COLOR_OKINA_GCC;
      dbg("\n[nabla] Command line specifies OKINA's GCC option");
      break;
    case BACKEND_COLOR_OKINA_ICC:
      backend_color|=BACKEND_COLOR_OKINA_ICC;
      dbg("\n[nabla] Command line specifies OKINA's ICC option");
      break;

      // ************************************************************
      // * BACKEND CUDA avec pas de variantes pour l'instant
      // ************************************************************
    case BACKEND_CUDA:
      backend=BACKEND_CUDA;
      dbg("\n[nabla] Command line hits long option %s", longopts[longindex].name);
      nabla_entity_name=strdup(optarg);
      unique_temporary_file_fd=nablaMakeTempFile(nabla_entity_name, &unique_temporary_file_name);
      dbg("\n[nabla] Command line specifies new CUDA nabla_entity_name: %s", nabla_entity_name);
      break;
       
      // ************************************************************
      // * UNKNOWN OPTIONS
      // ************************************************************      
    case '?':
      dbg("\n[nabla] UNKNOWN OPTIONS");
      if ((optopt>(int)'A')&&(optopt<(int)'z'))
        fprintf (stderr, "\n[nabla] Unknown option `-%c'.\n", optopt);
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
    exit(NABLA_ERROR|fprintf(stderr,"\n[nabla] Error with unique temporary file\n"));
  
  if (input_file_list==NULL)
    exit(NABLA_ERROR|fprintf(stderr,"\n[nabla] Error in input_file_list\n"));

  // On a notre fichier temporaire et la listes des fichiers ∇ à parser
  nablaPreprocessor(nabla_entity_name,
                    input_file_list,
                    unique_temporary_file_name,
                    unique_temporary_file_fd);

  dbg("\n[nabla] Now triggering nablaParsing with these options");
  if (nablaParsing(nabla_entity_name?nabla_entity_name:argv[argc-1],
                   optionDumpTree,
                   unique_temporary_file_name,
                   backend,
                   backend_color,
                   interface_name,
                   specific_path,
                   service_name)!=NABLA_OK)
    exit(NABLA_ERROR|fprintf(stderr, "\n[nabla] nablaParsing error\n"));
  nabla_unlink();
  return NABLA_OK;
}

