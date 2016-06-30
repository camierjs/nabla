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
// * nArcaneHLTIni
// *****************************************************************************
static void nArcaneHLTIni(nablaMain *arc,
                          nablaJob *entry_point,
                          int number_of_entry_points,
                          double *hlt_dive_when){
  FILE *hdr=arc->entity->hdr;
  // Et voici la fonction d'init du HLT
  fprintf(hdr, "\n\
   void hltIni(){\n\
      m_hlt_dive.resize(%d);\n\
      m_hlt_dive.fill(false);\n\
      m_hlt_exit.resize(%d);\n\
      m_hlt_exit.fill(false);\n\
      m_hlt_entry_points.resize(%d);\n\
      __attribute__((unused)) IEntryPointMng *entry_point_mng=subDomain()->entryPointMng();\n\
      __attribute__((unused)) IEntryPoint* entry_point;\n",
          NABLA_JOB_WHEN_MAX,
          NABLA_JOB_WHEN_MAX,
          NABLA_JOB_WHEN_MAX);
  for(int i=0;i<number_of_entry_points+2;i+=1){
    if (strcmp(entry_point[i].name,"ComputeLoopEnd")==0)continue;
    if (strcmp(entry_point[i].name,"ComputeLoopBegin")==0)continue;
    const int HLT_depth=entry_point[i].when_depth;
    if (HLT_depth==0) continue;
    assert(HLT_depth<NABLA_JOB_WHEN_MAX);
    const double when=entry_point[i].whens[0];
    //const int depth=entry_point[i].when_depth;
    //if (HLT_depth>hlt_current_depth){
    fprintf(hdr, "\
      entry_point=entry_point_mng->findEntryPoint(StringBuilder(\"%s@%f\"));\n\
      m_hlt_entry_points[%d].push_back(entry_point);//hlt_dive_when[0]=%f\n",
            entry_point[i].name,when,HLT_depth-1,hlt_dive_when[0]);
  }
  fprintf(hdr, "\t}\n");
}


// *************************************************************
// *************************************************************
static char* tab(int k){
  char *rtrn,*tabs=(char *)calloc(k+1,sizeof(char));
  for(int i=0;i<k;i+=1) tabs[i]='\t';
  tabs[k]=0;
  rtrn=strdup(tabs);
  free(tabs);
  return rtrn;
}

// *****************************************************************************
// * nArcaneHLTEntryPoint
// *****************************************************************************
void nArcaneHLTEntryPoint(nablaMain *arc,
                          nablaJob *entry_point,
                          int number_of_entry_points,
                          double *hlt_dive_when){
  dbg("\n[nArcaneHLTEntryPoint]");
  FILE *hdr=arc->entity->hdr;
  fprintf(hdr, "\n\t//nArcaneHLTEntryPoint");
  // Voici les booléens d'exit & probe du dive HLT
  fprintf(hdr, "\n\tint m_hlt_level=0;");
  fprintf(hdr, "\n\tBoolArray m_hlt_dive;");
  fprintf(hdr, "\n\tBoolArray m_hlt_exit;");
  // Voici les points d'entrées de notre dive
  fprintf(hdr, "\n\tstd::vector<std::vector<IEntryPoint*> > m_hlt_entry_points;");

  dbg("\n[nArcaneHLTEntryPoint] Dump de la fonction HLT d'initialisation");
  nArcaneHLTIni(arc,entry_point,number_of_entry_points,hlt_dive_when);

  // Voici les fonction qui seront appelées lors des 'DIVE'
  for(int i=0;hlt_dive_when[i]!=0.0;i+=1){
    char *tabs=tab(1+i);
    fprintf(hdr, "\n\tvoid hltDive%d(){\n\
      const int bkp_level=m_hlt_level;\n\
      m_hlt_level=%d;\n\
      const int level=m_hlt_level;\n\
      //info()<<\"%s[1;33mm_hlt_entry_points.at(\"<<level<<\") size=\"<<\
m_hlt_entry_points.at(level).size()<<\"[m\";\n\
      m_hlt_dive[level] = true;\n\
      m_hlt_exit[level] = false;\n\
      for(;!m_hlt_exit.at(level);){\n\
         info();\n\
         for(Integer i=0, s=m_hlt_entry_points.at(level).size(); i<s; ++i){\n\
            //info()<<\"%s[1;33m\"<<\"HLT launching: '\"<<\
m_hlt_entry_points.at(level).at(i)->name()<<\"'[m\";\n \
            m_hlt_entry_points.at(level).at(i)->executeEntryPoint();\n\
            //info()<<\"%s[1;33m\"<<\"HLT m_hlt_exit.at(\"<<level<<\"): '\"<<m_hlt_exit.at(level)<<\"'[m\";\n \
            if (m_hlt_exit.at(level)) break;\n\
         }\n\
      }\n\
     //info()<<\"[1;33m\"<<\"END \"<<level<<\"[m\";\n\
     m_hlt_dive[level] = false;\n\
     m_hlt_level=bkp_level;\n\
   }",i,i,tabs,tabs,tabs);//nccAxlGeneratorEntryPointWhenName(hlt_dive_when[i])
    free(tabs);
  }
}
