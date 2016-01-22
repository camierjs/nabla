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
// * Diffraction
// ****************************************************************************
void nOkinaHookDiffraction(nablaMain *nabla, nablaJob *job, astNode **n){
  // On backup les statements qu'on rencontre pour éventuellement les diffracter (real3 => _x, _y & _z)
  // Et on amorce la diffraction
  if ((*n)->ruleid == rulenameToId("expression_statement")
      && (*n)->children->ruleid == rulenameToId("expression")
      //&& (*n)->children->children->ruleid == rulenameToId("expression")
      //&& job->parse.statementToDiffract==NULL
      //&& job->parse.diffractingXYZ==0
      ){
    //dbg("\n[nOkinaHookJobDiffractStatement] amorce la diffraction");
//#warning Diffracting is turned OFF
      job->parse.statementToDiffract=NULL;//*n;
      // We're juste READY, not diffracting yet!
      job->parse.diffractingXYZ=0;      
      nprintf(nabla, "/* DiffractingREADY */",NULL);
  }
  
  // On avance la diffraction
  if ((*n)->tokenid == ';'
      && job->parse.diffracting==true
      && job->parse.statementToDiffract!=NULL
      && job->parse.diffractingXYZ>0
      && job->parse.diffractingXYZ<3){
    dbg("\n[nOkinaHookJobDiffractStatement] avance dans la diffraction");
    job->parse.isDotXYZ=job->parse.diffractingXYZ+=1;
    (*n)=job->parse.statementToDiffract;
    nprintf(nabla, NULL, ";\n\t");
    //nprintf(nabla, "\t/*<REdiffracting>*/", "/*diffractingXYZ=%d*/", job->parse.diffractingXYZ);
  }

  // On flush la diffraction 
  if ((*n)->tokenid == ';' 
      && job->parse.diffracting==true
      && job->parse.statementToDiffract!=NULL
      && job->parse.diffractingXYZ>0
      && job->parse.diffractingXYZ==3){
    dbg("\n[nOkinaHookJobDiffractStatement] Flush de la diffraction");
    job->parse.diffracting=false;
    job->parse.statementToDiffract=NULL;
    job->parse.isDotXYZ=job->parse.diffractingXYZ=0;
    nprintf(nabla, "/*<end of diffracting>*/",NULL);
  }
  //dbg("\n[nOkinaHookJobDiffractStatement] return from token %s", (*n)->token?(*n)->token:"Null");
}
