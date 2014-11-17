// ****************************************************************************
// * IVDEP Pragma
// ****************************************************************************
char *nccOkinaPragmaIccIvdep(void){ return "\\\n_Pragma(\"ivdep\")"; }
char *nccOkinaPragmaGccIvdep(void){ return "__declspec(align(64)) "; }



char *nccOkinaPragmaIccAlign(void){ return "\\\n_Pragma(\"ivdep\")"; }
char *nccOkinaPragmaGccAlign(void){ return "__attribute__ ((aligned(WARP_ALIGN))) "; }

