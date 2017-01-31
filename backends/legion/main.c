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


// ****************************************************************************
// * legionHookMainPrefix
// ****************************************************************************
NABLA_STATUS legionHookMainPrefix(nablaMain *nabla){
  fprintf(nabla->entity->src, "\n\n\n\
task continue_simulation(cycle : int64, cstop : int64,\n\
                         time : double, tstop : double)\n\
  return (cycle < cstop and time < tstop)\n\
end\n\
\n\n\
-- ******************************************************************************\n\
-- * Initialize\n\
-- ******************************************************************************\n\
task initialize(rz_all : region(zone), rz_all_p : partition(disjoint, rz_all),\n\
                rp_all : region(point),\n\
                rp_all_private : region(point),\n\
                rp_all_private_p : partition(disjoint, rp_all_private),\n\
                rp_all_ghost : region(point),\n\
                rp_all_ghost_p : partition(aliased, rp_all_ghost),\n\
                rp_all_shared_p : partition(disjoint, rp_all_ghost),\n\
                rs_all : region(side(wild, wild, wild, wild)),\n\
                rs_all_p : partition(disjoint, rs_all),\n\
                conf : config)\n\
where\n\
  reads writes(rz_all, rp_all_private, rp_all_ghost, rs_all),\n\
  rp_all_private * rp_all_ghost\n\
do\n\
\tc.printf(\"\\t[32;1m[initialize][m\\n\");");
  return NABLA_OK;
}


// ***************************************************************************** 
// * Dump des options
// *****************************************************************************/
static void legionHookMainOptions(nablaMain *nabla){
  fprintf(nabla->entity->src,"\n\t-- legionHookMainOptions");
  for(nablaOption *opt=nabla->options;opt!=NULL;opt=opt->next)
    fprintf(nabla->entity->src,"\n\tvar %s = %s",opt->name,opt->dflt);
}

// ****************************************************************************
// * legionHookMainPreInit
// ****************************************************************************
NABLA_STATUS legionHookMainPreInit(nablaMain *nabla){
  legionHookMainOptions(nabla);
  fprintf(nabla->entity->src,"\n\t-- legionHookMainPreInit");
  return NABLA_OK;
}


// ****************************************************************************
// * legionHookMainHLT
// ****************************************************************************
NABLA_STATUS legionHookMainHLT(nablaMain *n){
  nablaVariable *var;
  nablaJob *entry_points;
  int i,numParams,number_of_entry_points;
  bool is_into_compute_loop=false;
  n->HLT_depth=0;
  
  dbg("\n[legionHookMainHLT]");
  number_of_entry_points=nMiddleNumberOfEntryPoints(n);
  entry_points=nMiddleEntryPointsSort(n,number_of_entry_points);
  
  // Et on rescan afin de dumper, on rajoute les +2 ComputeLoop[End|Begin]
  for(i=0;i<number_of_entry_points+2;++i){
    if (strncmp(entry_points[i].name,"hltDive",7)==0) continue;
    if (strcmp(entry_points[i].name,"ComputeLoopEnd")==0) continue;
    if (strcmp(entry_points[i].name,"ComputeLoopBegin")==0) continue;
    
    dbg("%s\n\t[hookMain] sorted #%d: %s @ %f in '%s'", (i==0)?"\n":"",i,
        entry_points[i].name,
        entry_points[i].whens[0],
        entry_points[i].where);
    
    // Si l'on passe pour la première fois la frontière du zéro, on écrit le code pour boucler
    if (entry_points[i].whens[0]>=0 && !is_into_compute_loop){
      is_into_compute_loop=true;
      nprintf(n,NULL, "\nend -- Initialize\n\n\n\
-- ******************************************************************************\n\
-- * Simulate\n\
-- ******************************************************************************\n\
task simulate(rz_all : region(zone), rz_all_p : partition(disjoint, rz_all),\n\
              rp_all : region(point),\n\
              rp_all_private : region(point),\n\
              rp_all_private_p : partition(disjoint, rp_all_private),\n\
              rp_all_ghost : region(point),\n\
              rp_all_ghost_p : partition(aliased, rp_all_ghost),\n\
              rp_all_shared_p : partition(disjoint, rp_all_ghost),\n\
              rs_all : region(side(wild, wild, wild, wild)),\n\
              rs_all_p : partition(disjoint, rs_all),\n\
              conf : config)\n\
where\n\
  reads writes(rz_all, rp_all_private, rp_all_ghost, rs_all),\n\
  rp_all_private * rp_all_ghost\n\
do\n\
\tc.printf(\"\\t[32;1m[simulate][m\\n\");");
      legionHookMainOptions(n);
      nprintf(n,NULL,"\n\tvar time = 0.0");
      nprintf(n,NULL,"\n\tvar cycle : int64 = 0");
      nprintf(n,NULL, "\n\twhile continue_simulation(cycle, cstop, time, tstop) do");
      nprintf(n,NULL, "\n\tc.printf(\"\\n[32;1m[cycle] %%d[m\",cycle)");
    }
    
    if (entry_points[i].when_depth==(n->HLT_depth+1))
      n->HLT_depth=entry_points[i].when_depth;
    
    if (entry_points[i].when_depth==(n->HLT_depth-1))
      n->HLT_depth=entry_points[i].when_depth;
        
    // \n ou if d'un IF after '@'
    if (entry_points[i].ifAfterAt!=NULL){
      dbg("\n\t[hookMain] dumpIfAfterAt!");
      nprintf(n, NULL, "\n\tif (");
      nMiddleDumpIfAfterAt(entry_points[i].ifAfterAt, n,false);
      nprintf(n, NULL, ") then");
    }else nprintf(n, NULL, "\n");
        
    // Dump de la tabulation et du nom du point d'entrée
    if (!entry_points[i].is_a_function){
      nprintf(n, NULL, "\tfor i=0,conf.npieces do\n\t\t%s(",
              entry_points[i].name);
      //if (entry_points[i].item[0]=='c') nprintf(n, NULL, "rz_all_p[i])\n\tend");
      if (entry_points[i].item[0]=='c' ||
          entry_points[i].item[0]=='n' ||
          entry_points[i].item[0]=='f'){
        nprintf(n, NULL, "rz_all_p[i],rp_all_private_p[i],rp_all_ghost_p[i],rs_all_p[i]");
        if (entry_points[i].used_options!=NULL){
          for(nablaOption *opt= entry_points[i].used_options;opt!=NULL;opt=opt->next)
            nprintf(n,NULL,", %s",opt->name);
        }
        nprintf(n,NULL,")\n\tend");
      }
    }else{
      nprintf(n, NULL, "\t-- Function: %s", entry_points[i].name);
    }
      
    // On s'autorise un endroit pour insérer des arguments
    //nMiddleArgsDumpFromDFS(n,&entry_points[i]);
    
    // Si on doit appeler des jobs depuis cette fonction @ée
    if (entry_points[i].called_variables != NULL){
      if (!entry_points[i].reduction)
         nMiddleArgsAddExtra(n,&numParams);
      //else nprintf(n, NULL, "NABLA_NB_CELLS_WARP,");
      // Et on rajoute les called_variables en paramètre d'appel
      dbg("\n\t[hookMain] Et on rajoute les called_variables en paramètre d'appel");
      for(var=entry_points[i].called_variables;var!=NULL;var=var->next){
        nprintf(n, NULL, ",\n\t\t/*used_called_variable*/%s_%s",var->item, var->name);
      }
    }//else nprintf(n,NULL,"/*NULL_called_variables*/");
    
    if (entry_points[i].ifAfterAt!=NULL)
      nprintf(n, NULL, " end");
  }
  free(entry_points);
  return NABLA_OK;
}


// ****************************************************************************
// * legionHookMainPostfix
// ****************************************************************************
NABLA_STATUS legionHookMainPostfix(nablaMain *nabla){
  fprintf(nabla->entity->src, "\n\t-- legionHookMainPostfix");
  fprintf(nabla->entity->src, "\n\tcycle += 1");
  fprintf(nabla->entity->src, "\n\tend -- while");
  fprintf(nabla->entity->src, "\nend -- main");
  return NABLA_OK;
}
