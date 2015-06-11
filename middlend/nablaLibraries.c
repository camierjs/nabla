///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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

static void nablaLibrariesSwitch(astNode * n, nablaEntity *entity){
  if (strncmp(n->children->token,"ℵ",3)==0){
    dbg("\n\t[nablaLibraries] ALEPH single_library hit!");
    entity->libraries|=(1<<aleph);
    return;
  }

  if (strncmp(n->children->token,"Real",4)==0){
    dbg("\n\t[nablaLibraries] Real single_library hit!");
    entity->libraries|=(1<<real);
    return;
  }

  switch(n->children->token[2]){
    
  case ('e'):{ // AL[E]PH || ℵ -(p3c)-> Al[e]
    dbg("\n\t[nablaLibraries] ALEPH single_library hit!");
    entity->libraries|=(1<<aleph);
    break;
  }
    
  case ('i'):{ // MP[I] || MA[I]L
    switch (n->children->token[1]){
    case ('p'):{ // M[P]I
      dbg("\n\t[nablaLibraries] MPI single_library hit!");
      entity->libraries|=(1<<mpi);
      break;
    }
    case ('a'):{ // M[A]IL
      dbg("\n\t[nablaLibraries] MAIL single_library hit!");
      entity->libraries|=(1<<mail);
      break;
    }
    default: nablaError("Could not switch M[p]i || M[a]il!");
    }
    break;
  }
      
  case ('p'):{ // GM[P]
    dbg("\n\t[nablaLibraries] GMP single_library hit!");
    entity->libraries|=(1<<gmp);
    break;
  }
      
  case ('r'):{ // CA[R]TESIAN || PA[R]TICLES
    switch (n->children->token[0]){
    case ('c'):{
      dbg("\n\t[nablaLibraries] CARTESIAN single_library hit!");
      entity->libraries|=(1<<cartesian);
      break;
    }
    case ('p'):{
      dbg("\n\t[nablaLibraries] PARTICLES single_library hit!");
      entity->libraries|=(1<<particles);
      break;
    }
    default: nablaError("Could not switch CARTESIAN || PARTICLES!");
    }
    break;
  }
      
  case ('t'):{ // Ma[t]erials || Ma[t]hematica || df[t]
    if (n->children->token[0]=='d'){
      dbg("\n\t[nablaLibraries] DFT single_library hit!");
      entity->libraries|=(1<<dft);
      break;
    }
    switch (n->children->token[3]){
    case ('e'):{
      dbg("\n\t[nablaLibraries] MATERIALS single_library hit!");
      entity->libraries|=(1<<materials);
      break;
    }
    case('h'):{
      dbg("\n\t[nablaLibraries] MATHEMATICA single_library hit!");
      entity->libraries|=(1<<mathematica);
      break;
    }
    default: nablaError("Could not switch Ma[t]erials || Ma[t]hematica!");
    }
    break;
  }
      
  case ('u'):{ // SL[U]RM
    dbg("\n\t[nablaLibraries] SLURM single_library hit!");
    entity->libraries|=(1<<slurm);
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
void nablaLibraries(astNode * n, nablaEntity *entity){
  if (n->ruleid == rulenameToId("single_library"))
    nablaLibrariesSwitch(n,entity);
  if(n->children != NULL) nablaLibraries(n->children,  entity);
  if(n->next != NULL) nablaLibraries(n->next, entity);
}
