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

extern FILE *yyin;
extern int yylineno;
#define YYSTYPE astNode*
int yyparse (astNode **);
extern char nabla_input_file[]; 
static char *unique_temporary_file_name=NULL;
extern char nabla_version[]; 
extern char nabla_license[]; 


// *****************************************************************************
// * NABLA_MAN
// \t[1;35m--lib [36;4mname[0m\tCode generation for C/C++ library [1;5;31m(WiP)[0m\n
// \t[1;35m--loci [36;4mname[0m\tCode generation for LOCI [1;5;31m(WiP)[0m\n
// \t[1;35m--mma [36;4mname[0m\tCode generation for Mathematica [1;5;31m(WiP)[0m\n
// \t[1;35m--uintah [36;4mname[0m\tCode generation for UINTAH [1;5;31m(WiP)[0m\n
// \t[1;35m--vhdl [36;4mname[0m\tCode generation for VHDL [1;5;31m(WiP)[0m\n
// *****************************************************************************
#define NABLA_MAN "[1;36mNAME[0m\n\
\t[1;36mnabla[0m - [1;35mN[0mumerical [1;35mA[0mnalysis [1;35mB[0mased [1;35mLA[0mnguage - Version: [1;35m%s[0m\n\
\t        Optimized Code Generator for Specific Compilers/Architectures\n\
[1;36mSYNOPSIS[0m\n\
\t[1;36mnabla[0m [-t[nl]] [-v [4mlogfile[0m] [1;4;35mTARGET[0m -i [4minput file list[0m\n\
[1;36mWARNING[0m\n\
\tThe [1;36mnabla[0m generator is still under heavy development and\n\
\tdistributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;\n\
\twithout even the implied warranty of FITNESS FOR A PARTICULAR PURPOSE.\n\
[1;36mDESCRIPTION[0m\n\
\tThe purpose of the [4mnabla[0m utility is to generate automatically\n\
\tthe required files for each specified target.\n\
[1;36mOPTIONS[0m\n\
\t[1;36m-t[0m\t\tGenerate the intermediate AST dot files\n\
\t[1;36m-tnl[0m\t\tGenerate the intermediate AST dot files without labels\n\
\t[1;36m-v [4mlogfile[0m\tGenerate debug info to [4mlogfile[0m\n\
\t[1;36m--version[0m\tOutput version information and exit\n\
\t[1;36m--license[0m\tOutput license information and exit\n\
[1;4;35mTARGET[0m can be:\n\
\t[1;35m--arcane [36;4mname[0m\tCode generation for the ARCANE middleware\n\
\t\t[36m--alone[0m\tGenerates a [4mstand-alone[0m application\n\
\t\t[36m--library[0m Generates a [4mlibrary[0m\n\
\t\t[36m--module[0m  Generates a [4mmodule[0m\n\
\t\t[36m--service[0m Generates a [4mservice[0m\n\
\t\t\t[36m-I [4mname[0m Interface name to use\n\
\t\t\t[36m-p [4mpath[0m Path of the interface file\n\
\t\t\t[36m-n [4mname[0m Service name that will be generate\n\
\t[1;35m--cuda [36;4mname[0m\tCode generation for CUDA\n\
\t[1;35m--raja [36;4mname[0m\tCode generation for RAJA\n\
\t[1;35m--kokkos [36;4mname[0m\tCode generation for KOKKOS\n\
\t[1;35m--legion [36;4mname[0m\tCode generation for LEGION\n\
\t[1;35m--lambda [36;4mname[0m\tCode generation for LAMBDA (C/C++11) stand-alone target\n\
\t[1;35m--okina [36;4mname[0m\tCode generation for OKINA  (C/C++11) stand-alone target\n\
\t\t[36m--std[0m\t\tStandard code generation with no explicit vectorization\n\
\t\t[36m--sse[0m\t\tExplicit code generation with SSE intrinsics\n\
\t\t[36m--avx[0m\t\tExplicit code generation with AVX intrinsics\n\
\t\t[36m--avx2[0m\t\tExplicit code generation with AVX2 intrinsics\n\
\t\t[36m--avx512[0m\tExplicit code generation with AVX152 intrinsics\n\
\t\t[36m--mic[0m\t\tExplicit code generation with MIC intrinsics\n\
\t\t[36m--seq[0m\tSequential code generation (default)\n\
\t\t[36m--omp[0m\tOpenMP parallel implementation\n\
\t\t[36m--cilk[0m\tCilk+ parallel implementation\n\
\t\t\t(still experimental with latest GNU GCC)\n\
\t\t[36m--gcc[0m\tGNU GCC pragma generation (default)\n\
\t\t[36m--icc[0m\tIntel ICC pragma generation\n\
\t\t[0mif [36;4mname[36;0m==mfem[0m, generates intrinsics classes for MFEM (mfem.org)\n\
[1;36mEMACS MODE[0m\n\
\tYou can find a nabla-mode.el file within the distribution.\n\
\tLoading emacs utf-8 locale coding system could be a good idea:\n\
\t\t(setq load-path (cons \"~/.emacs.d\" load-path))\n\
\t\t(autoload 'nabla-mode \"nabla-mode\" \"Nabla Mode\" t)\n\
\t\t(setq auto-mode-alist (cons '(\"\\.n\\'\" . nabla-mode) auto-mode-alist))\n\
\t\t(set-selection-coding-system 'utf-8)\n\
\t\t(setq locale-coding-system 'utf-8)\n\
\t\t(set-language-environment 'utf-8)\n\
[1;36mAUTHOR[0m\n\
\tJean-Sylvain Camier:\t<jean-sylvain.camier@cea.fr>\n\
\t\t\t\t<camierjs@nabla-lang.org>\n\
[1;36mBUGS[0m\n\
\tBugs are to be reported to the above address.\n"


// ****************************************************************************
// * nabla_error
// ****************************************************************************
void nablaErrorVariadic(const char *file,
                        const int line,
                        const char *format,...){
  va_list args;
  va_start(args, format);
  fflush(stdout);
  fflush(stderr);
  fprintf(stderr,"\r%s:%d:%d: error: ",file,line,yylineno-1);
  vfprintf(stderr,format,args);
  fprintf(stderr,"\n");
  va_end(args);
  exit(-1);
}


// ****************************************************************************
// * yyerror
// ****************************************************************************
void yyerror(astNode **root, char *error){
  fflush(stdout);
  printf("\r%s:%d: %s\n",nabla_input_file,yylineno-1, error);
}


// ****************************************************************************
// * Nabla Parsing
// ****************************************************************************
static NABLA_STATUS nablaParsing(const char *nabla_entity_name,
                                 const int optionDumpTree,
                                 char *npFileName,
                                 const NABLA_BACKEND backend,
                                 const BACKEND_OPTION option,
                                 const BACKEND_PARALLELISM parallelism,
                                 const BACKEND_COMPILER compiler,
                                 char *interface_name,
                                 char *specific_path,
                                 char *service_name){
  astNode *root=NULL;  
  if(!(yyin=fopen(npFileName,"r")))
    return NABLA_ERROR | dbg("\n\t[nablaParsing] Could not open '%s' file",
                             npFileName);
  dbg("\n* nablaParsing, options= 0x%0X",option);
  dbg("\n** [nablaParsing] Starting parsing");
  if (option!=BACKEND_OPTION_OKINA_MFEM){
    if (yyparse(&root)){
      fclose(yyin);
      return NABLA_ERROR | dbg("\n\t[nablaParsing] Error while parsing!");
    }
    dbg("\n\t[nablaParsing] Parsing done!");
    fclose(yyin);
  }
  
  if (optionDumpTree!=0){
    dbg("\n** [nablaParsing] On dump l'arbre créé (%s.dot)", nabla_entity_name);
    assert(root);
    astTreeSave(nabla_entity_name, root);
  }
  // Initial files setup and checkup
  if (nabla_entity_name==NULL)
    return NABLA_ERROR | dbg("\n[nccParseur] No entity name has been set!");
  
  dbg("\n** [nablaParsing] nabla_entity_name=%s", nabla_entity_name);
  dbg("\n** [nablaParsing] nabla_input_file=%s", nabla_input_file);
  dbg("\n** [nablaParsing] Now launching nablaMiddlendSwitch");
  return nMiddleSwitch(root,
                       optionDumpTree,
                       nabla_entity_name,
                       backend, option,
                       parallelism, compiler,
                       interface_name,
                       specific_path,
                       service_name);
}


// ****************************************************************************
// * $(CPATH)/gcc -std=c99 -E -Wall -x c $(TGT).nabla -o $(TGT).n
// ****************************************************************************
static int sysPreprocessor(const char *nabla_entity_name,
                           const char *list_of_nabla_files,
                           const char *unique_temporary_file_name,
                           const int unique_temporary_file_fd){
  const int size = NABLA_MAX_FILE_NAME;
  char *cat_sed_temporary_file_name=NULL;
  char *gcc_command=NULL;
  int cat_sed_temporary_fd=0;
  
  if ((cat_sed_temporary_file_name = (char*)calloc(size,sizeof(char)))==NULL)
    nablaError("[sysPreprocessor] Could not calloc cat_sed_temporary_file_name!");
  
  if ((gcc_command = (char*)calloc(size,sizeof(char)))==NULL)
    nablaError("[sysPreprocessor] Could not calloc gcc_command!");
  
  // On crée un fichier temporaire où l'on va:
  //    - cat'er les différents fichiers .n et en initialisant le numéro de ligne
  //    - sed'er les includes, par exemple
  snprintf(cat_sed_temporary_file_name,
           size,
           "/tmp/nabla_%s_sed_XXXXXX", nabla_entity_name);
  cat_sed_temporary_fd=mkstemp(cat_sed_temporary_file_name);
  if (cat_sed_temporary_fd==-1)
    nablaError("[sysPreprocessor] Could not mkstemp cat_sed_temporary_fd!");
  
  toolCatAndHackIncludes(list_of_nabla_files,
                         cat_sed_temporary_file_name);
  
  // Et on lance la commande de préprocessing
  // -P Inhibit generation of linemarkers in the output from the preprocessor.
  //    This might be useful when running the preprocessor on something that is not C code,
  //    and will be sent to a program which might be confused by the linemarkers.
  //
  // -C Do not discard comments.
  //    All comments are passed through to the output file, except for comments in processed directives,
  //    which are deleted along with the directive.
  //
  // Porting to GCC 4.8: to disable the stdc-predef.h preinclude: -ffreestanding or use the -P
  snprintf(gcc_command,size,
           // Il faut garder les linemarkers afin de metre à jour 'nabla_input_file' dans nabla.y
           "gcc -ffreestanding -std=c99 -C -E -Wall -x c %s -o %s",
           cat_sed_temporary_file_name, unique_temporary_file_name );
  dbg("\n\t[sysPreprocessor] gcc_command='%s'", gcc_command);
  // We flush all open output streams
  fflush(NULL);
  if (system(gcc_command)<0){
    free(gcc_command);
    exit(NABLA_ERROR|fprintf(stderr, "\n\t[sysPreprocessor] Error while preprocessing!\n"));
  }
  if (unlink(cat_sed_temporary_file_name)<0){
    free(cat_sed_temporary_file_name);
    exit(NABLA_ERROR|fprintf(stderr, "\n\t[sysPreprocessor] Error while unlinking sed file!\n"));
  }
  free(gcc_command);
  free(cat_sed_temporary_file_name);
  return NABLA_OK;
}
 

// ****************************************************************************
// * nablaPreprocessor
// ****************************************************************************
static void nablaPreprocessor(char *nabla_entity_name,
                              char *list_of_nabla_files,
                              char *unique_temporary_file_name,
                              const int unique_temporary_file_fd){
  //dbg("\r%s:1: is our preprocessing temporary file\n",unique_temporary_file_name);
  if (sysPreprocessor(nabla_entity_name,
                      list_of_nabla_files,
                      unique_temporary_file_name,
                      unique_temporary_file_fd)!=0)
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n\t[nablaPreprocessor] Error in preprocessor stage\n"));
  dbg("\n\t[nablaPreprocessor] done!");
}


// ****************************************************************************
// * Main
// ****************************************************************************
int main(int argc, char * argv[]){
  int c;
  int optionDumpTree=0;
  char *nabla_entity_name=NULL;
  int longindex=0;
  char *interface_name=NULL;
  char *specific_path=NULL;
  char *service_name=NULL;
  int unique_temporary_file_fd=0;
  char *input_file_list=NULL;
  NABLA_BACKEND backend=0;
  BACKEND_OPTION option=0;
  BACKEND_PARALLELISM parallelism=0;
  BACKEND_COMPILER compiler =0;
  const struct option longopts[]={
    // Standard options:
    {"help",no_argument,NULL,OPTION_HELP},
    {"version",no_argument,NULL,OPTION_VERSION},
    {"license",no_argument,NULL,OPTION_LICENSE},
    {"tnl",no_argument,NULL,OPTION_TIME_DOT_MMA},
    // Backend options:
    {"arcane",no_argument,NULL,BACKEND_ARCANE},
       {"alone",required_argument,NULL,BACKEND_OPTION_ARCANE_ALONE},
       {"family",required_argument,NULL,BACKEND_OPTION_ARCANE_FAMILY},
       {"module",required_argument,NULL,BACKEND_OPTION_ARCANE_MODULE},
       {"service",required_argument,NULL,BACKEND_OPTION_ARCANE_SERVICE},
    {"cuda",required_argument,NULL,BACKEND_CUDA},
    {"okina",required_argument,NULL,BACKEND_OKINA},
       {"std",no_argument,NULL,BACKEND_OPTION_OKINA_STD},
       {"sse",no_argument,NULL,BACKEND_OPTION_OKINA_SSE},
       {"avx",no_argument,NULL,BACKEND_OPTION_OKINA_AVX},
       {"avx2",no_argument,NULL,BACKEND_OPTION_OKINA_AVX2},
       {"avx512",no_argument,NULL,BACKEND_OPTION_OKINA_AVX512},
       {"mic",no_argument,NULL,BACKEND_OPTION_OKINA_MIC},
       {"cilk",no_argument,NULL,BACKEND_PARALLELISM_CILK},
       {"omp",no_argument,NULL,BACKEND_PARALLELISM_OMP},
       {"seq",no_argument,NULL,BACKEND_PARALLELISM_SEQ},
       {"gcc",no_argument,NULL,BACKEND_COMPILER_GCC},
       {"icc",no_argument,NULL,BACKEND_COMPILER_ICC},
    //{"mfem",no_argument,NULL,BACKEND_OPTION_OKINA_MFEM},
    {"lambda",required_argument,NULL,BACKEND_LAMBDA},
    {"raja",required_argument,NULL,BACKEND_RAJA},
    {"kokkos",required_argument,NULL,BACKEND_KOKKOS},
    {"legion",required_argument,NULL,BACKEND_LEGION},
    //{"loci",required_argument,NULL,BACKEND_LOCI},
    //{"uintah",required_argument,NULL,BACKEND_UINTAH},
    //{"mma",no_argument,NULL,BACKEND_MMA},
    //{"library",required_argument,NULL,BACKEND_LIBRARY},
    //{"vhdl",required_argument,NULL,BACKEND_VHDL},
    {NULL,0,NULL,0}
  };

  // Setting null bytes ('\0') at the beginning of dest, before concatenation
  input_file_list=calloc(NABLA_MAX_FILE_NAME,sizeof(char));
  // Check for at least several arguments
  if (argc<=1)
    exit(0&fprintf(stderr, NABLA_MAN,nabla_version));
  // Now switch the arguments
  while ((c=getopt_long(argc, argv, "tv:I:p:n:i:",longopts,&longindex))!=-1){
    switch (c){
      // ************************************************************
      // * Help, Version & License options
      // ************************************************************      
    case OPTION_HELP:
      printf(NABLA_MAN,nabla_version);
      return NABLA_OK;
    case OPTION_VERSION:
      printf("Nabla version is %s\n",nabla_version);
      return NABLA_OK;
    case OPTION_LICENSE:
      printf(nabla_license);
      return NABLA_OK;
      // ************************************************************
      // * Standard OPTIONS: t, tnl, v
      // ************************************************************      
    case 't': // DUMP tree option
      optionDumpTree=OPTION_TIME_DOT_STD;
      dbg("\n\t[nabla] Command line specifies to dump the tree");
      break;
    case OPTION_TIME_DOT_MMA:
      optionDumpTree=OPTION_TIME_DOT_MMA;
      dbg("\n\t[nabla] Command line specifies to dump the MMA tree");
      break;
    case 'v': // DEBUG MANAGEMENT
      if (dbgSet(DBG_ALL) == DBG_OFF) return NABLA_ERROR;
      dbgOpenTraceFile(optarg);
      dbg("* Command line"); // org mode first item
      dbg("\n\t[nabla] Command line specifies debug file: %s", optarg);
      break;      
      // ************************************************************
      // * INPUT FILES
      // ************************************************************      
    case 'i':
      strcat(input_file_list,optarg);
      dbg("\n\t[nabla] first input_file_list: %s ", input_file_list);
      while (optind < argc){
        input_file_list=strcat(input_file_list," ");
        input_file_list=strcat(input_file_list,argv[optind++]);
        dbg("\n\t[nabla] next input_file_list: %s ", input_file_list);
      }
      break;
      // ************************************************************
      // * BACKEND ARCANE avec ses variantes:
      // *    - ALONE, FAMILY, MODULE ou SERVICE
      // *    - p(ath), I(nterface), n(name)
      // ************************************************************      
    case BACKEND_ARCANE:
      backend=BACKEND_ARCANE;
      dbg("\n\t[nabla] Command line hits target ARCANE (%s)",
          longopts[longindex].name);
      break;
    case BACKEND_OPTION_ARCANE_ALONE:
      option=BACKEND_OPTION_ARCANE_ALONE;
      dbg("\n\t[nabla] Command line specifies ARCANE's STAND-ALONE option");
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new ARCANE nabla_entity_name: %s",
          nabla_entity_name);
      break;
    case BACKEND_OPTION_ARCANE_FAMILY:
      option=BACKEND_OPTION_ARCANE_FAMILY;
      dbg("\n\t[nabla] Command line specifies ARCANE's FAMILY option");
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new ARCANE nabla_entity_name: %s",
          nabla_entity_name);
      break;
    case BACKEND_OPTION_ARCANE_MODULE:
      option=BACKEND_OPTION_ARCANE_MODULE;
      dbg("\n\t[nabla] Command line specifies ARCANE's MODULE option");
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new ARCANE nabla_entity_name: %s",
          nabla_entity_name);
      break;
    case BACKEND_OPTION_ARCANE_SERVICE:
      option=BACKEND_OPTION_ARCANE_SERVICE;
      dbg("\n\t[nabla] Command line specifies ARCANE's SERVICE option");
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new ARCANE nabla_entity_name: %s",
          nabla_entity_name);
      break;
    case 'p': // specific path to source directory for a module or a service
      specific_path=optarg;
      dbg("\n\t[nabla] Command line specifies ARCANE's path: %s",
          specific_path);
      break;
    case 'I': // Interface name
      interface_name=optarg;
      dbg("\n\t[nabla] Command line specifies ARCANE's SERVICE interface name: %s",
          interface_name);
      break;
    case 'n': // Service name
      service_name=optarg;
      dbg("\n\t[nabla] Command line specifies ARCANE's SERVICE service name: %s",
          service_name);
      break;
      // ************************************************************
      // * BACKEND OKINA avec ses variantes:
      // *    - STD, SSE, AVX, AVX2, AVX512 & MIC
      // *    - CILK, OpenMP, SEQ
      // *    - ICC, GCC
      // ************************************************************
    case BACKEND_OKINA:
      backend=BACKEND_OKINA;
      dbg("\n\t[nabla] Command line hits long option %s",
          longopts[longindex].name);
      nabla_entity_name=optarg;
      if (strncmp(nabla_entity_name,"mfem",4)==0){
        dbg("\n\t[nabla] Command line hits MFEM");
        assert(option==0);
        option=BACKEND_OPTION_OKINA_MFEM;
      }
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new OKINA nabla_entity_name: %s",
          nabla_entity_name);
      break;
    case BACKEND_OPTION_OKINA_STD:
      assert(option==0);
      option=BACKEND_OPTION_OKINA_STD;
      dbg("\n\t[nabla] Command line specifies OKINA's STD option");
      break;
    case BACKEND_OPTION_OKINA_SSE:
      assert(option==0);
      option=BACKEND_OPTION_OKINA_SSE;
      dbg("\n\t[nabla] Command line specifies OKINA's SSE option");
      break;
    case BACKEND_OPTION_OKINA_AVX:
      assert(option==0);
      option=BACKEND_OPTION_OKINA_AVX;
      dbg("\n\t[nabla] Command line specifies OKINA's AVX option");
      break;
    case BACKEND_OPTION_OKINA_AVX2:
      assert(option==0);
      option=BACKEND_OPTION_OKINA_AVX2;
      dbg("\n\t[nabla] Command line specifies OKINA's AVX2 option");
      break;
    case BACKEND_OPTION_OKINA_AVX512:
      assert(option==0);
      option=BACKEND_OPTION_OKINA_AVX512;
      dbg("\n\t[nabla] Command line specifies OKINA's AVX512 option");
      break;
    case BACKEND_OPTION_OKINA_MIC:
      assert(option==0);
      option=BACKEND_OPTION_OKINA_MIC;
      dbg("\n\t[nabla] Command line specifies OKINA's MIC option");
      break;
    case BACKEND_PARALLELISM_CILK:
      assert(parallelism==0);
      parallelism=BACKEND_PARALLELISM_CILK;
      dbg("\n\t[nabla] Command line specifies OKINA's CILK option");
      break;
    case BACKEND_PARALLELISM_OMP:
      assert(parallelism==0);
      parallelism=BACKEND_PARALLELISM_OMP;
      dbg("\n\t[nabla] Command line specifies OKINA's OpenMP option");
      break;
    case BACKEND_PARALLELISM_SEQ:
      assert(parallelism==0);
      parallelism=BACKEND_PARALLELISM_SEQ;
      dbg("\n\t[nabla] Command line specifies OKINA's SEQ option");
      break;
    case BACKEND_COMPILER_GCC:
      assert(compiler==0);
      compiler=BACKEND_COMPILER_GCC;
      dbg("\n\t[nabla] Command line specifies OKINA's GCC option");
      break;
    case BACKEND_COMPILER_ICC:
      assert(compiler==0);
      compiler=BACKEND_COMPILER_ICC;
      dbg("\n\t[nabla] Command line specifies OKINA's ICC option");
      break;
      // ************************************************************
      // * BACKEND CUDA avec aucune variantes pour l'instant
      // ************************************************************
    case BACKEND_CUDA:
      backend=BACKEND_CUDA;
      dbg("\n\t[nabla] Command line hits long option %s",
          longopts[longindex].name);
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new CUDA nabla_entity_name: %s",
          nabla_entity_name);
      break;
      // ************************************************************
      // * BACKEND LAMBDA avec aucune variantes pour l'instant
      // ************************************************************
    case BACKEND_LAMBDA:
      backend=BACKEND_LAMBDA;
      dbg("\n\t[nabla] Command line hits long option %s",
          longopts[longindex].name);
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new LAMBDA nabla_entity_name: %s",
          nabla_entity_name);
      break;    
      // ************************************************************
      // * BACKEND KOKKOS
      // ************************************************************
    case BACKEND_KOKKOS:
      backend=BACKEND_KOKKOS;
      dbg("\n\t[nabla] Command line hits long option %s",
          longopts[longindex].name);
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new KOKKOS nabla_entity_name: %s",
          nabla_entity_name);
      break;      
      // ************************************************************
      // * BACKEND LEGION
      // ************************************************************
    case BACKEND_LEGION:
      backend=BACKEND_LEGION;
      dbg("\n\t[nabla] Command line hits long option %s",
          longopts[longindex].name);
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      dbg("\n\t[nabla] Command line specifies new LEGION nabla_entity_name: %s",
          nabla_entity_name);
      break;      
      // ************************************************************
      // * BACKEND LIBRARY en cours de construction
      // ************************************************************
      /*case BACKEND_LIBRARY:
      backend=BACKEND_LIBRARY;
      dbg("\n\t[nabla] LIBRARY BACKEND WIP!");
      exit(NABLA_ERROR);*/ 
      // ************************************************************
      // * BACKEND RAJA en cours de construction
      // ************************************************************
    case BACKEND_RAJA:
      backend=BACKEND_RAJA;
      dbg("\n\t[nabla] RAJA BACKEND WIP!");
      nabla_entity_name=optarg;
      unique_temporary_file_fd=toolMkstemp(nabla_entity_name,
                                           &unique_temporary_file_name);
      break;      
      // ************************************************************
      // * BACKEND LOCI en cours de construction
      // ************************************************************
      /*case BACKEND_LOCI:
      backend=BACKEND_LOCI;
      dbg("\n\t[nabla] LOCI BACKEND WIP!");
      exit(NABLA_ERROR);*/
       // ************************************************************
      // * BACKEND UINTAH en cours de construction
      // ************************************************************
      /*case BACKEND_UINTAH:
      backend=BACKEND_UINTAH;
      dbg("\n\t[nabla] UINTAH BACKEND WIP!");
      exit(NABLA_ERROR);*/
      // ************************************************************
      // * BACKEND MMA en cours de construction
      // ************************************************************
      /*case BACKEND_MMA:
      backend=BACKEND_MMA;
      dbg("\n\t[nabla] MMA BACKEND WIP!");
      exit(NABLA_ERROR);*/
      // ************************************************************
      // * BACKEND VHDL en cours de construction
      // ************************************************************
      /*case BACKEND_VHDL:
      backend=BACKEND_VHDL;
      dbg("\n\t[nabla] VHDLBACKEND WIP!");
      exit(NABLA_ERROR);*/
      // ************************************************************
      // * UNKNOWN OPTIONS
      // ************************************************************      
    case '?':
      dbg("\n\t[nabla] UNKNOWN OPTIONS");
      if ((optopt>(int)'A')&&(optopt<(int)'z'))
        fprintf (stderr, "\n\t[nabla] Unknown option `-%c'.\n", optopt);
      else fprintf (stderr, "\n\t[nabla] Unknown option character `\\%d'.\n", optopt);
      exit(NABLA_ERROR);
    default: exit(NABLA_ERROR|
                  fprintf(stderr,
                          "\n\t[nabla] Error in command line\n"));
    }
  }
  
  if (nabla_entity_name==NULL)
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n\t[nabla] Error with entity name!\n"));

  if (backend==0)
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n\t[nabla] Error with target switch!\n"));
 
  if (unique_temporary_file_fd==0)
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n\t[nabla] Error with unique temporary file\n"));
  
  if (input_file_list==NULL)
    exit(NABLA_ERROR|
         fprintf(stderr,
                 "\n\t[nabla] Error in input_file_list\n"));

  // On a notre fichier temporaire et la listes des fichiers ∇ à parser
  nablaPreprocessor(nabla_entity_name,
                    input_file_list,
                    unique_temporary_file_name,
                    unique_temporary_file_fd);
  free(input_file_list);
  
  dbg("\n\t[nabla] Now triggering nablaParsing with these options");
  if (nablaParsing(nabla_entity_name?nabla_entity_name:argv[argc-1],
                   optionDumpTree,
                   unique_temporary_file_name,
                   backend,option,parallelism,compiler,
                   interface_name,specific_path,
                   service_name)!=NABLA_OK){
    toolUnlink(unique_temporary_file_name);
    exit(NABLA_ERROR);
  }
  toolUnlink(unique_temporary_file_name);
  return NABLA_OK;
}

