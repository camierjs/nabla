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
#include "nabla.tab.h"


// ****************************************************************************
// * isWithLibrary
// ****************************************************************************
bool isWithLibrary(nablaMain *nabla, with_library lib){
  const int this_lib = 1<<lib;
  return ((nabla->entity->libraries & this_lib)==this_lib);
}


// ****************************************************************************
// * nMiddleLibrariesSwitch
// ****************************************************************************
static void nMiddleLibrariesSwitch(astNode * n, nablaEntity *entity){

  //dbg("\n\t[nMiddleLibrariesSwitch] token= %s",n->children->token);
  if (strncmp(n->children->token,"ℵ",3)==0){
    dbg("\n\t[nMiddleLibrariesSwitch] ALEPH single_library hit!");
    entity->libraries|=(1<<with_aleph);
    dbg("\n\t[nMiddleLibrariesSwitch] Alephlibrary=0x%X",entity->libraries);
    return;
  }

  if (strncmp(n->children->token,"Real2",5)==0){
    dbg("\n\t[nMiddleLibrariesSwitch] Real2 single_library hit!");
    entity->libraries|=(1<<with_real2);
    dbg("\n\t[nMiddleLibrariesSwitch] Real2 library=0x%X",entity->libraries);
    return;
  }
  
  if (strncmp(n->children->token,"Real",4)==0){
    dbg("\n\t[nMiddleLibrariesSwitch] Real single_library hit!");
    entity->libraries|=(1<<with_real);
    dbg("\n\t[nMiddleLibrariesSwitch] Real library=0x%X",entity->libraries);
    return;
  }

  switch(n->children->token[2]){
    
    /*case ('e'):{ // AL[E]PH || ℵ -(p3c)-> Al[e]
    dbg("\n\t[nablaLibraries] ALEPH single_library hit!");
    entity->libraries|=(1<<with_aleph);
    break;
    }*/
    
  case ('i'):{ // MP[I] || MA[I]L
    switch (n->children->token[1]){
    case ('p'):{ // M[P]I
      dbg("\n\t[nablaLibraries] MPI single_library hit!");
      entity->libraries|=(1<<with_mpi);
      break;
    }
    case ('a'):{ // M[A]IL
      dbg("\n\t[nablaLibraries] MAIL single_library hit!");
      entity->libraries|=(1<<with_mail);
      break;
    }
    default: nablaError("Could not switch M[p]i || M[a]il!");
    }
    break;
  }
      
  case ('p'):{ // GM[P]
    dbg("\n\t[nablaLibraries] GMP single_library hit!");
    entity->libraries|=(1<<with_gmp);
    break;
  }
      
  case ('r'):{ // CA[R]TESIAN || PA[R]TICLES
    switch (n->children->token[0]){
    case ('c'):{
      dbg("\n\t[nablaLibraries] CARTESIAN single_library hit!");
      entity->libraries|=(1<<with_cartesian);
      break;
    }
    case ('p'):{
      dbg("\n\t[nablaLibraries] PARTICLES single_library hit!");
      entity->libraries|=(1<<with_particles);
      break;
    }
    default: nablaError("Could not switch CARTESIAN || PARTICLES!");
    }
    break;
  }
      
  case ('t'):{ // Ma[t]erials || Ma[t]hematica || df[t]
    if (n->children->token[0]=='d'){
      dbg("\n\t[nablaLibraries] DFT single_library hit!");
      entity->libraries|=(1<<with_dft);
      break;
    }
    switch (n->children->token[3]){
    case ('e'):{
      dbg("\n\t[nablaLibraries] MATERIALS single_library hit!");
      entity->libraries|=(1<<with_materials);
      break;
    }
    case('h'):{
      dbg("\n\t[nablaLibraries] MATHEMATICA single_library hit!");
      entity->libraries|=(1<<with_mathematica);
      break;
    }
    default: nablaError("Could not switch Ma[t]erials || Ma[t]hematica!");
    }
    break;
  }
      
  case ('u'):{ // SL[U]RM
    dbg("\n\t[nablaLibraries] SLURM single_library hit!");
    entity->libraries|=(1<<with_slurm);
    break;
  }
 
  default:{
    dbg("\n\t[nablaLibraries] single_library token=%s",n->children->token);
    nablaError("Could not find library!");
  }
  }
}


/*****************************************************************************
 * DFS scan for libraries
 *****************************************************************************/
void nMiddleLibraries(astNode * n, nablaEntity *entity){
  if (n->ruleid == rulenameToId("single_library"))
    nMiddleLibrariesSwitch(n,entity);
  if(n->children != NULL) nMiddleLibraries(n->children,  entity);
  if(n->next != NULL) nMiddleLibraries(n->next, entity);
}
