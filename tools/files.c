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

// *****************************************************************************
// * nablaMakeTempFile
// *****************************************************************************
int toolMkstemp(const char *entity_name, char **unique_temporary_file_name){
  const int size = NABLA_MAX_FILE_NAME;
  if ((*unique_temporary_file_name=calloc(size,sizeof(char)))==NULL)
    nablaError("[nablaMakeTempFile] Could not calloc our unique_temporary_file_name!");
  const int n=snprintf(*unique_temporary_file_name, size, "/tmp/nabla_%s_XXXXXX", entity_name);
  if (n > -1 && n < size)
    return mkstemp(*unique_temporary_file_name);
  nablaError("[nablaMakeTempFile] Error in snprintf into unique_temporary_file_name!");
  return -1;
}


// ****************************************************************************
// * nToolUnlink: deletes a name from the filesystem
// ****************************************************************************
void toolUnlink(char *pathname){
  if (pathname==NULL) return;
  if (unlink(pathname)!=0)
    nablaError("Error while removing '%s' file", pathname);
  free(pathname);
}



// ****************************************************************************
// * nToolFileCatAndHackIncludes
// ****************************************************************************
int toolCatAndHackIncludes(const char *list_of_nabla_files,
                           const char *cat_sed_temporary_file_name){
  size_t size;
//#warning BUFSIZ pour les anciens compilos/stations ?
  char *buf=(char*)calloc(BUFSIZ+1,sizeof(char));
  char *pointer_that_matches=NULL;
  char *nabla_file_name=NULL;
  char *dup_list_of_nabla_files=sdup(list_of_nabla_files);
  FILE *cat_sed_temporary_file=NULL;
  
  printf("\r%s:1: is our temporary sed file\n",cat_sed_temporary_file_name);
  dbg("\n\t[nToolFileCatAndHackIncludes] cat_sed_temporary_file_name is %s",
      cat_sed_temporary_file_name);
  
  cat_sed_temporary_file=fopen(cat_sed_temporary_file_name,"w");
  
  for(nabla_file_name=strtok(dup_list_of_nabla_files, " ");      
      /*test*/ nabla_file_name!=NULL;
      /*incr*/ nabla_file_name=strtok(NULL, " ")){
    fprintf(cat_sed_temporary_file,"# 1 \"%s\"\n",nabla_file_name);
    FILE *nabla_FILE=fopen(nabla_file_name,"r");
    assert(nabla_FILE);
    // Now copying .n file to our tmp one
    while ((size=fread(buf, 1, BUFSIZ, nabla_FILE))){
      //printf("\n\tbuf: '%s'",buf);
      // On recherche les '#include' pour les transformer en ' include'
      while ((pointer_that_matches=strstr(buf,"#include"))!=NULL){
        dbg("\n\t[nToolFileCatAndHackIncludes] '#include' FOUND! Hacking!");
        *pointer_that_matches=' ';
      }
      fwrite(buf, 1, size, cat_sed_temporary_file);
    }
    assert(fclose(nabla_FILE)==0);
  }
  assert(fclose(cat_sed_temporary_file)==0);
  free(buf);
  // done by sfree: free(dup_list_of_nabla_files);
  return NABLA_OK;
}
 
