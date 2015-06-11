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


// ****************************************************************************
// * nccArcLibParticlesHeader
// * Let's be careful to quote subjects containing spaces.
// ****************************************************************************
char* nccArcLibParticlesHeader(void){
  return "\
\n#include <arcane/IMesh.h>\
\n#include <arcane/ItemVector.h>\
\n#include <arcane/IItemFamily.h>\
\n#include <arcane/ServiceBuilder.h>\
\n#include <arcane/IParticleFamily.h>\
\n#include <arcane/IParticleExchanger.h>\
\n#include <arcane/utils/ITraceMng.h>\
\n#include <arcane/ObserverPool.h>\
\n#include <arcane/IVariableMng.h>\
\n#include <arcane/MultiArray2VariableRef.h>";
}


// ****************************************************************************
// * nccArcLibParticlesPrivates
// ****************************************************************************
char* nccArcLibParticlesPrivates(nablaEntity *entity){
  nablaVariable *var;
  const size_t local_size=1024;
  const size_t globl_size=8192;
  char *type,varToCat[local_size];
  char *privates=malloc(globl_size);
  if (privates==NULL)
    nablaError("[nccArcLibParticlesPrivates] cannot malloc!");
  privates[0]=0;
  strcat(privates,"\nprivate:\t//Particles stuff\
\n\tvoid particlesInit();\
\n\tvoid moveAtomInBox(int, Int64);\
\n\tvoid moveAtomInBox(ParticleEnumerator, Int64);\
\n\tvoid particleAddToCell(Int64,Int64,Real3);\
\n\tvoid particleSyncToCell();\
\n\tvoid particleParallelExchange();\
\n\tinline ParticleVectorView cellParticles(Int32 cell_lid){\
\n\t\treturn m_nabla_particles_vector.at(cell_lid).view();\
\n\t};\
\n\tIItemFamily *m_particle_family;\
\n\tArray<bool> m_nabla_particles_own;\
\n\tArray<Real3> m_nabla_particles_coords;\
\n\tArray<Int32> m_nabla_particles_local_ids;\
\n\tArray<Int64> m_nabla_particles_unique_ids;\
\n\tArray<Int32> m_nabla_particles_cells_local_ids;\
\n\tArray<ParticleVector> m_nabla_particles_vector;\
\n\t//VariableMultiArray2Integer m_nabla_particle_ids;");
  for(var=entity->main->variables;var!=NULL;var=var->next){
    if (var->item[0]!='p') continue;
    varToCat[0]=0;
    // On transforme la premiere majuscule
    type=strdup(var->type); type[0]-=32;
    dbg("\n\t\t[nccArcLibParticlesPrivates] m_%s_%s", var->item, var->name);
    snprintf(varToCat,local_size,"\n\tVariableParticle%s m_particle_%s;", type, var->name);
    //dbg("\n\t\t[nccArcLibParticlesPrivates] varToCat=%s", varToCat);
    strcat(privates,varToCat);
    //dbg("\n\t\t[nccArcLibParticlesPrivates] privates=%s", privates);    
  }
  return strdup(privates);
}


// ****************************************************************************
// * nccArcLibParticlesConstructor
// ****************************************************************************
char* nccArcLibParticlesConstructor(nablaEntity *entity){
  nablaVariable *var;
  const size_t local_size=1024;
  const size_t globl_size=8192;
  char VariableBuildInfo[local_size];
  char *particleVars=malloc(globl_size);
  dbg("\n\t\t[nccArcLibParticlesConstructor]");
  if (particleVars==NULL)
    nablaError("[nccArcLibParticlesConstructor] cannot malloc!");
  particleVars[0]=0;
  strcat(particleVars,
         "\n\t\t//,m_nabla_particle_ids(VariableBuildInfo(mbi.subDomain(),\"NablaParticlesIds\"))");
  for(var=entity->main->variables;var!=NULL;var=var->next){
    if (var->item[0]!='p') continue;
    dbg("\n\t\t[nccArcLibParticlesConstructor] var->name=%s",var->name);
    snprintf(VariableBuildInfo,local_size,
            "\n\t\t\t\t\t\t,m_particle_%s(VariableBuildInfo(mbi.mesh(),\"%s\",\"particles\"))",
            var->name,var->name);
    strcat(particleVars,VariableBuildInfo);
  }
  return strdup(particleVars);
}


// ****************************************************************************
// * nccArcLibParticlesIni
// ****************************************************************************
void nccArcLibParticlesIni(nablaMain *arc){
  fprintf(arc->entity->src, "\n\
\n// ****************************************************************************\
\n// * particlesInit\
\n// ****************************************************************************\
\nvoid %s%s::particlesInit(){\
\n\tinfo()<<\"\\33[7m[particlesInit]\"<<\"\\33[m\";\
\n\tm_particle_family->setHasUniqueIdMap(false);\
\n\t// Resize en fonction du nombre de cells\
\n\tm_nabla_particles_vector.resize(allCells().size());\
\n\tinfo()<<\"[particlesInit] \\33[32mresized to \"<<allCells().size()<<\"\\33[m\";traceMng()->flush();\
\n\tENUMERATE_CELL(cell,allCells()){\
\n\t\t//info()<<\"[particlesInit] \\33[32minit #\"<<cell->localId()<<\"\\33[m\";traceMng()->flush();\
\n\t\tm_nabla_particles_vector[cell->localId()]=ParticleVector();\
\n\t}\
\n\tinfo()<<\"[particlesInit] \\33[32mdone\\33[m\";\
\n\ttraceMng()->flush();\
\n}",arc->name,nablaArcaneColor(arc));
  nablaJob *particlesInit=nablaJobNew(arc->entity);
  particlesInit->is_an_entry_point=true;
  particlesInit->is_a_function=true;
  particlesInit->scope  = strdup("NoGroup");
  particlesInit->region = strdup("NoRegion");
  particlesInit->item   = strdup("\0");
  particlesInit->rtntp  = strdup("void");
  particlesInit->name   = strdup("particlesInit");
  particlesInit->name_utf8   = strdup("particlesInit");
  particlesInit->xyz    = strdup("NoXYZ");
  particlesInit->drctn  = strdup("NoDirection");
  sprintf(&particlesInit->at[0],"-huge_valf");
  particlesInit->whenx  = 1;
  particlesInit->whens[0] = ENTRY_POINT_init;
  nablaJobAdd(arc->entity, particlesInit);

  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * particleAddToCell\
\n// ****************************************************************************\
\nvoid %s%s::particleAddToCell(Int64 particle_uid, Int64 cell_uid, Real3 r){\
\n\t//info()<<\"[particleAddToCell] particle #\"<<particle_uid<<\" in cell #\"<<cell_uid<<\"\";\
\n\tInt32Array lid(1);\
\n\tInt64Array uid(1);\
\n\tuid[0]=cell_uid;\
\n\tdefaultMesh()->itemFamily(IK_Cell)->itemsUniqueIdToLocalId(lid.view(),uid.constView(),false);\
\n\tif (lid[0]==NULL_ITEM_ID) return;\
\n\t\t//info()<<\"[particleAddToCell]\\33[36m in here!\\33[m\";traceMng()->flush();\
\n\t\t m_nabla_particles_own.add(defaultMesh()->cellFamily()->itemsInternal()[lid[0]]->isOwn());\
\n\t\t m_nabla_particles_unique_ids.add(particle_uid);\
\n\t\t // On add pour avoir la bonne taille lors de l'appel à addParticles\
\n\t\t m_nabla_particles_local_ids.add(0);\
\n\t\t m_nabla_particles_coords.add(r);\
\n\t\t m_nabla_particles_cells_local_ids.add(lid[0]);\
\n}",arc->name,nablaArcaneColor(arc));
  
  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * moveAtomInBox\
\n// ****************************************************************************\
\nvoid %s%s::moveAtomInBox(int puid, Int64 jBox){\
\n\tInt32Array clid(1);\
\n//\tInt32Array plid(1);\
\n\tInt64Array uid(1);\
\n\tuid[0]=jBox;\
\n\tinfo()<<\"[moveAtomInBox]\\33[36m puid=\"<<puid<<\", jBox=\"<<jBox<<\"\\33[m\";traceMng()->flush();\
\n\tdefaultMesh()->itemFamily(IK_Cell)->itemsUniqueIdToLocalId(clid.view(),uid.constView(),true);\
\n\tuid[0]=puid;\
\n\tinfo()<<\"[moveAtomInBox]\\33[36m Got cell's uid \\33[m\";traceMng()->flush();\
\n//\tdefaultMesh()->itemFamily(IK_Particle)->itemsUniqueIdToLocalId(plid.view(),uid.constView(),true);\
\n//\tinfo()<<\"[moveAtomInBox]\\33[36m Got particle's uid \\33[m\";traceMng()->flush();\
\n//\tCell cell=defaultMesh()->itemFamily(IK_Cell)->itemsInternal()[clid[0]];\
\n//\tParticle particle=defaultMesh()->itemFamily(IK_Particle)->itemsInternal()[puid];//lid[0]];\
\n//\tif (particle.hasCell()==false) fatal()<<\"moveAtomInBox !particle.hasCell()\";\
\n//\tinfo()<<\"[moveAtomInBox]\\33[36m particle \"<<particle.localId()\
\n//\t\t<<\" frome cell #\"<<particle.cell().localId()\
\n//\t\t<<\" to cell lid=#\"<<clid[0]<<\"\\33[m\";traceMng()->flush();\
\n//\tm_particle_family->toParticleFamily()->setParticleCell(particle, cell);\
\n}",arc->name,nablaArcaneColor(arc));
  
  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * moveAtomInBox\
\n// ****************************************************************************\
\nvoid %s%s::moveAtomInBox(ParticleEnumerator particle, Int64 jBox){\
\n//\tInt32Array clid(1);\
\n//\tInt64Array uid(1);\
\n//\tuid[0]=jBox;\
\n//\tdefaultMesh()->itemFamily(IK_Cell)->itemsUniqueIdToLocalId(clid.view(),uid.constView(),true);\
\n//\tif (particle->hasCell()==false) fatal()<<\"moveAtomInBox !particle->hasCell()\";\
\n//\tinfo()<<\"[moveAtomInBox]\\33[36m particle \"<<particle->localId()\
\n//\t\t<<\" frome cell #\"<<particle->cell().localId()\
\n//\t\t<<\" to cell lid=#\"<<clid[0]<<\"\\33[m\";traceMng()->flush();\
\n\tCell cell=defaultMesh()->itemFamily(IK_Cell)->itemsInternal()[jBox];\
\n\tm_particle_family->toParticleFamily()->setParticleCell(*particle, cell);\
\n}",arc->name,nablaArcaneColor(arc));

  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * particleSyncToCell\
\n// ****************************************************************************\
\nvoid %s%s::particleSyncToCell(){\
\n\tinfo()<<\"\\33[7m[particleSyncToCell]\"<<\"\\33[m\";\
\n\tm_particle_family->toParticleFamily()->addParticles(m_nabla_particles_unique_ids.view(),\
\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t m_nabla_particles_cells_local_ids,\
\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t m_nabla_particles_local_ids);\
\n\t//On flush l'update\
\n\tm_particle_family->endUpdate();\
\n\tinfo()<<\"[particleSyncToCell]\\33[36m endUpdate\\33[m\";traceMng()->flush();\
\n\tinfo()<<\"[particleSyncToCell]\\33[36m m_particle_family->allItems().size()=\"<<m_particle_family->allItems().size()<<\"\\33[m\";traceMng()->flush();\
\n\t// ET on set les coords et remplit les vecteurs des mailles\
\n\tENUMERATE_PARTICLE(particle,m_particle_family->allItems()){\
\n\t\tm_particle_r[particle]=m_nabla_particles_coords[m_nabla_particles_local_ids.at(particle->localId())];\
\n\t\tm_nabla_particles_vector[particle->cell().localId()].add(particle->localId());\
\n\t}\
\n\tinfo()<<\"[particleSyncToCell]\\33[32m done\\33[m\";traceMng()->flush();\
\n\t//particleParallelExchange();\
\n}",arc->name,nablaArcaneColor(arc));
  
  fprintf(arc->entity->src, "\
\n\n// ****************************************************************************\
\n// * particleParallelExchange\
\n// ****************************************************************************\
\nvoid %s%s::particleParallelExchange(){\
\n\tinfo()<<\"\\33[7m[particleParallelExchange]\"<<\"\\33[m\";\
\nServiceBuilder<IParticleExchanger> serviceBuilder(subDomain());\
\n\tInt32Array particles_local_id;\
\n\tIParallelMng* pm = subDomain()->parallelMng();\
\n\tif (!pm->isParallel()) return;\
\n\tInteger comm_rank = pm->commRank();\
\n\tIParticleExchanger* pe = serviceBuilder.createInstance(\"BasicParticleExchanger\");\
\n\tpe->initialize(m_particle_family);\
\n\tInt32Array particles_sub_domain_to_send;\
\n\tInt32Array incoming_particles_local_id;\
\n\tParticleVectorView particles_view = m_particle_family->allItems().view();\
\n\tInteger total_nb_particle = pm->reduce(Parallel::ReduceMax,particles_view.size());\
\n\tpe->beginNewExchange(total_nb_particle);\
\n\tbool is_finished = false;\
\n\twhile (!is_finished){\
\n\tInteger nb_remaining_particle = particles_view.size();\
\n\tparticles_local_id.clear();\
\n\tparticles_sub_domain_to_send.clear();\
\n\tif (nb_remaining_particle>10){\
\n\t\tInteger index = 0;\
\n\t\tENUMERATE_PARTICLE(ipart,particles_view){\
\n\t\t\t++index;\
\n\t\t\tParticle p = *ipart;\
\n\t\t\tCell cell = p.cell();\
\n\t\t\tInteger nb_face = cell.nbFace();\
\n\t\t\tFace face = cell.face(index%%nb_face);\
\n\t\t\tCell opposite_cell = (face.backCell()==cell) ? face.frontCell() : face.backCell();\
\n\t\t\tif (opposite_cell.null()) continue;\
\n\t\t\tif (opposite_cell.owner()==comm_rank) continue;\
\n\t\t\tparticles_local_id.add(p.localId());\
\n\t\t\tparticles_sub_domain_to_send.add(opposite_cell.owner());\
\n\t\t}\
\n\t}\
\n\tInteger nb_particle_tracking_finished = nb_remaining_particle - particles_view.size();\
\n\tincoming_particles_local_id.clear();\
\n\tinfo() << \" Particles to send: \" << particles_local_id.size();\
\n\tis_finished = pe->exchangeItems(nb_particle_tracking_finished,\
\n\t\t\t\t\t\t\tparticles_local_id,\
\n\t\t\t\t\t\t\tparticles_sub_domain_to_send,\
\n\t\t\t\t\t\t\t&incoming_particles_local_id,\
\n\t\t\t\t\t\t\t0);\
\n\t\tinfo() << \"Nb Particules: \" << m_particle_family->nbItem();\
\n\t\tparticles_view = m_particle_family->view(incoming_particles_local_id);\
\n\t}\n}",arc->name,nablaArcaneColor(arc));
}


// ****************************************************************************
// * nccArcLibParticlesDelete
// ****************************************************************************
char *nccArcLibParticlesDelete(void){ return ""; }

