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


// *****************************************************************************
// * 
// *****************************************************************************
void nArcaneHLTInit(nablaMain *arc){
  nablaJob *hltInitFunction=nMiddleJobNew(arc->entity);
  hltInitFunction->is_an_entry_point=true;
  hltInitFunction->is_a_function=true;
  hltInitFunction->scope  = strdup("NoScope");
  hltInitFunction->region = strdup("NoRegion");
  hltInitFunction->item   = strdup("\0");
  hltInitFunction->return_type  = strdup("void");
  hltInitFunction->name   = strdup("hltIni");
  hltInitFunction->name_utf8 = strdup("hltIni");
  hltInitFunction->xyz    = strdup("NoXYZ");
  hltInitFunction->direction  = strdup("NoDirection");
  sprintf(&hltInitFunction->at[0],"-huge_valf");
  hltInitFunction->when_index  = 1;
  hltInitFunction->whens[0] = ENTRY_POINT_init;
  nMiddleJobAdd(arc->entity, hltInitFunction);
}

  
// *****************************************************************************
// * nccAxlGeneratorHLTEntryPoint
// *****************************************************************************
void nArcaneHLTEntryPoint(nablaMain *arc,
                          nablaJob *entry_point,
                          int number_of_entry_points,
                          double hlt_dive_when){
  FILE *hdr=arc->entity->hdr;
  fprintf(hdr, "\n\t//nccAxlGeneratorHLTEntryPoint");
  // Voici les booléens d'exit & probe du dive HLT
  fprintf(hdr, "\n\tBool m_hlt_dive=false;");
  fprintf(hdr, "\n\tBool m_hlt_exit=false;");
  // Voici les points d'entrées de notre dive
  fprintf(hdr, "\n\tArray<IEntryPoint*> m_hlt_entry_points;");
  // Voici la fonction qui sera appelée lors du premier 'dive'
  fprintf(hdr, "\n\tvoid hltDive_at_%s(){\n\
      //info()<<\"[1;33mm_hlt_entry_points size=\"<<m_hlt_entry_points.size()<<\"[m\";\n\
      m_hlt_dive = true;\n\
      m_hlt_exit = false;\n\
      for(;!m_hlt_exit;){\n\
         for(Integer i=0, s=m_hlt_entry_points.size(); i<s; ++i){\n\
            info()<<\"[1;33m\"<<\"\tHLT launching: '\"<<m_hlt_entry_points.at(i)->name()<<\"'[m\";\n\
            m_hlt_entry_points.at(i)->executeEntryPoint();\n\
            //traceMng()->flush();\n\
         }\n\
      }\n\
     m_hlt_dive = false;\n\
   }", nccAxlGeneratorEntryPointWhenName(hlt_dive_when));
  // Et voici la fonction d'init du HLT
  fprintf(hdr, "\n\
   void hltIni(){\n\
      __attribute__((unused)) IEntryPointMng *entry_point_mng=subDomain()->entryPointMng();\n\
      __attribute__((unused)) IEntryPoint* entry_point;");
  for(int i=0;i<number_of_entry_points+2;i+=1){
    if (strcmp(entry_point[i].name,"ComputeLoopEnd")==0)continue;
    if (strcmp(entry_point[i].name,"ComputeLoopBegin")==0)continue;
    const int HLT_depth=entry_point[i].when_depth;
    if (HLT_depth==0) continue;
    const double when=entry_point[i].whens[0];
    //const char *whenName=nccAxlGeneratorEntryPointWhenName(when);  
    fprintf(hdr, "\
      entry_point=entry_point_mng->findEntryPoint(StringBuilder(\"%s@%f\"));\n\
      ARCANE_ASSERT((entry_point!=0),(\"nccAxlGeneratorHLTEntryPoint\"));\n\
      m_hlt_entry_points.add(entry_point);\n",
            entry_point[i].name,when);
  }
  fprintf(hdr, "}\n");
}
