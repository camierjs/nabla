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
// *
// ****************************************************************************
char* legionHookCallPrefix(nablaMain *nabla,const char *type){
  return "task";
}


// ****************************************************************************
// *
// ****************************************************************************
void legionHookCallAddExtraParameters(nablaMain *nabla, nablaJob *job, int *numParams){
  //if (job->item[0]=='c') nprintf(nabla,NULL,"rz : region(zone)");
  if (job->item[0]=='c' || job->item[0]=='n' || job->item[0]=='f')
  nprintf(nabla,NULL,"rz : region(zone),\
\n\t\t\t\t\trpp : region(point),\
\n\t\t\t\t\trpg : region(point),\
\n\t\t\t\t\trs : region(side(rz, rpp, rpg, rs))");  
}


// ****************************************************************************
// *
// ****************************************************************************
char* legionHookCallITask(nablaMain *nabla, nablaJob *job){
  int r=0;
  int w=0;
  int c[2]={0,0};
  int n[2]={0,0};
  int f[2]={0,0};
  int s[2]={0,0};
  
  if (job->used_options!=NULL){
    for(nablaOption *opt=job->used_options;opt!=NULL;opt=opt->next)
      nprintf(nabla,NULL,",\n\t\t\t\t%s : %s",opt->name,
              opt->type[0]=='r'?"double":
              opt->type[0]=='i'?"int64":
              opt->type[0]=='b'?"bool":
              "\n#error Unknown Type for option");
  }
  nprintf(nabla,NULL,")");
  nprintf(nabla,NULL,"\nwhere");
  
  for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
    if (var->in  && var->item[0]!='g') { r+=1; nprintf(nabla,NULL,"\n-- r+=1 with %s",var->name);}
    if (var->out && var->item[0]!='g') { w+=1; nprintf(nabla,NULL,"\n-- w+=1 with %s",var->name);}
    if (var->in  && var->item[0]=='c' && var->dim==0){ c[0]+=1; nprintf(nabla,NULL,", c[0]+=1");}
    if (var->out && var->item[0]=='c' && var->dim==0){ c[1]+=1; nprintf(nabla,NULL,", c[1]+=1");}
    // Les variables de dim>0 sont portées par les 'sides'
    if (var->in  && var->item[0]=='c' && var->dim==1 && var->vitem=='n'){ s[0]+=1; nprintf(nabla,NULL,", s[0]+=1");}
    if (var->out && var->item[0]=='c' && var->dim==1 && var->vitem=='n'){ s[1]+=1; nprintf(nabla,NULL,", s[1]+=1");}
    if (var->in  && var->item[0]=='c' && var->dim==1 && var->vitem=='f'){ s[0]+=1; nprintf(nabla,NULL,", s[0]+=1");}
    if (var->out && var->item[0]=='c' && var->dim==1 && var->vitem=='f'){ s[1]+=1; nprintf(nabla,NULL,", s[1]+=1");}
    if (var->in  && var->item[0]=='n'){ n[0]+=1; nprintf(nabla,NULL,", n[0]+=1");}
    if (var->out && var->item[0]=='n'){ n[1]+=1; nprintf(nabla,NULL,", n[1]+=1");}
    if (var->in  && var->item[0]=='f'){ f[0]+=1; nprintf(nabla,NULL,", f[0]+=1");}
    if (var->out && var->item[0]=='f'){ f[1]+=1; nprintf(nabla,NULL,", f[1]+=1");}
  }

  // S'il y a des variables en lecture, on les annonce
  if (r>0){
    nprintf(nabla,NULL,"\n\treads(");
    
    // Première passe aux mailles
    if (c[0]>0){
      bool virgule=true;
      nprintf(nabla,NULL,"rz.{");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->in) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='c') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    
    // Deuxième passe aux noeuds
    if (n[0]>0){
      bool virgule=true;
      nprintf(nabla,NULL,"%srpp.{",c[0]>0?",":"");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->in) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='n') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    // re-Deuxième passe aux noeuds
    if (n[0]>0){
      bool virgule=true;
      nprintf(nabla,NULL,",rpg.{");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->in) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='n') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    
    // Troisième passe aux faces
    if ((f[0]+s[0])>0){
      bool virgule=true;
      nprintf(nabla,NULL,"%srs.{",c[0]+n[0]>0?",":"");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->in) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='f') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      // On revient pour les variables tableaux aux mailles
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->in) continue;
        if (var->dim==0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='c') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    nprintf(nabla,NULL,")%s",w>0?",":"");
  }
  
  // S'il y a des variables en écriture, on les annonce
  if (w>0){
    nprintf(nabla,NULL,"\n\twrites(");
    // Première passe aux mailles
    if (c[1]>0){
      bool virgule=true;
      nprintf(nabla,NULL,"rz.{");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->out) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='c') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    
    // Deuxième passe aux noeuds
    if (n[1]>0){
      bool virgule=true;
      nprintf(nabla,NULL,"%srpp.{",c[1]>0?",":"");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->out) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='n') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    // re-Deuxième passe aux noeuds
    if (n[1]>0){
      bool virgule=true;
      nprintf(nabla,NULL,",rpg.{");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->out) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='n') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    
    // Troisième passe aux faces
    if ((f[1]+s[1])>0){
      bool virgule=true;
      nprintf(nabla,NULL,"%srs.{",(c[1]+n[1])>0?",":"");
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->out) continue;
        if (var->dim>0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='f') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      // On revient pour reprendre les variables aux mailles de dim>0     
      for(nablaVariable *var=job->used_variables;var!=NULL;var=var->next){
        if (!var->out) continue;
        if (var->dim==0) continue;
        if (var->koffset!=0) continue;
        if (var->item[0]!='c') continue;
        nprintf(nabla,NULL,"%s%s",!virgule?",":"",var->name);
        virgule=false;
      }
      nprintf(nabla,NULL,"}");
    }
    nprintf(nabla,NULL,")");
  }
  return "\ndo";
}

// ****************************************************************************
// *
// ****************************************************************************
char* legionHookCallOTask(nablaMain *nabla, nablaJob *job){
  return "\nend\n";
}
