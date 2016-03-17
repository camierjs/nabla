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

#define HOOK(h,f) if (nabla->hook->h->f) nabla->hook->h->f(nabla)

// ****************************************************************************
// * nMiddleBackendAnimate
// ****************************************************************************
//#warning nMiddleBackendAnimate a simplifier
NABLA_STATUS animate(nablaMain *nabla,
                     astNode *root,
                     struct hookStruct *hooks){
  // Initialisation des hooks avant tout
  nabla->hook=hooks;

  // Partie des hooks à remonter à termes dans le middlend
  HOOK(vars,init);
  HOOK(source,open);
  HOOK(source,include);

  // Le header
  HOOK(header,open);
  HOOK(header,prefix);
  HOOK(header,include);
  HOOK(mesh,prefix);
  HOOK(header,enums);
  HOOK(header,dump);

  // Parse du code préprocessé et lance les hooks associés
  // On en profite pour dumper dans le header les forwards des fonctions
  nMiddleGrammar(root,nabla);

  // On a besoin d'avoir parsé pour le core afin d'avoir renseigné les librairies
  HOOK(mesh,core);

  // Rapidement on place dans le header les variables et options
  // qui pourront etre utilisées par d'autres dump
  HOOK(vars,prefix);

  HOOK(main,varInitKernel);
  HOOK(main,prefix);
  HOOK(vars,malloc);
  
  HOOK(main,preInit);
  HOOK(main,varInitCall);
  HOOK(main,main);
  HOOK(main,postInit);
  
  // Partie POSTFIX  
  HOOK(header,postfix);
  HOOK(mesh,postfix);
  HOOK(main,postfix);
  HOOK(vars,free);
  
  dbg("\n\t[nMiddleBackendAnimate] Deleting kernel names");
  toolUnlinkKtemp(nabla->entity->jobs);
  
  dbg("\n\t[nMiddleBackendAnimate] done, NABLA_OK!");
  return NABLA_OK;
}

