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
#include <map>
#include "Aleph.h"
#include <iterator>
ISubDomain* subDomain(void);

#define ALEPH_INDEX_NOT_USED (-1)

// ****************************************************************************
// * AlephIndexing
// ****************************************************************************
AlephIndexing::AlephIndexing(AlephKernel *kernel):
  TraceAccessor(kernel->parallel()->traceMng()),
  m_kernel(kernel),
  m_sub_domain(kernel->subDomain()),
  m_current_idx(0),
  m_known_items_own(0){
  debug()<<"\33[1;33m[AlephIndexing::AlephIndexing] NEW"<<"\33[m";
  m_known_items_all_address.resize(0);
}


// ****************************************************************************
// * updateKnownItems
// ****************************************************************************
int AlephIndexing::updateKnownItems(vector<int> *var_idx,
                                    const int itm){
  debug()<<"\t\33[33m[updateKnownItems] itm="<<itm<<"\33[m";
  // Dans tous les cas l'adresse est celle-ci
  m_known_items_all_address.push_back(&(*var_idx)[itm]);
  // Si l'item ciblé n'est pas à nous, on ne doit pas le compter
  if (true){//itm.isOwn()){
    // On met à jour la case mémoire, et on le rajoute aux owned
    (*var_idx)[itm]=m_current_idx;
    m_known_items_own+=1;
  }else{
    debug()<< "\t\t\33[33m[AlephIndexing::updateKnownItems] is NOT ours"<<"\33[m";
    (*var_idx)[itm]=m_current_idx;
  }
  // Maintenant, on peut incrémenter l'index de ligne
  m_current_idx+=1;
  // Et on retourne le numéro demandé de la ligne
  debug()<< "\t\t\33[33m[AlephIndexing::updateKnownItems] returning \33[1;32m"<<m_current_idx-1<<"\33[m";
  return m_current_idx-1;
}


// ****************************************************************************
// * findWhichLidFromMapMap
// ****************************************************************************
int AlephIndexing::findWhichLidFromMapMap(double *var, const int itm){
  VarMapIdx::const_iterator iVarMap = m_var_map_idx.find(var);
  // Si la variable n'est même pas encore connue
  // On rajoute une entrée map(map(m_current_idx))
  if (iVarMap==m_var_map_idx.end()){
    debug()<<"\t\33[33m[findWhichLidFromMapMap] Unknown variable *var @"<<var<<"\33[m";
    traceMng()->flush();
    debug()<<"\t\33[33m[findWhichLidFromMapMap] On rajoute à notre map la variable '_idx' de cette variable\33[m";
    //string var_idx_name("var");//var->name());
    //var_idx_name+="_idx";
    //#warning Cell|Node|Faces Variable? for size!
    vector<int> *var_idx=new vector<int>(subDomain()->defaultMesh()->size()+1);
    // var->itemKind());
    // On rajoute à notre map la variable '_idx' de cette variable
    m_var_map_idx.insert(std::make_pair(var,var_idx));
    // On flush tous les indices potentiels de cette variable
    var_idx->assign(var_idx->size(),ALEPH_INDEX_NOT_USED);
    return updateKnownItems(var_idx,itm);
  }
  debug()<<"\t\33[33m[findWhichLidFromMapMap] KNOWN variable *var @"<<var<<"\33[m";
  vector<int> *var_idx = iVarMap->second;
  // Si cet item n'est pas connu de cette variable, on rajoute une entrée
  if ((*var_idx)[itm]==ALEPH_INDEX_NOT_USED){
    debug()<<"\t\33[33m[findWhichLidFromMapMap] Cet item n'est pas connu de cette variable, on rajoute une entrée\33[m";
    traceMng()->flush();
    return updateKnownItems(var_idx,itm);
  }
  debug()<<"\t\33[33m[AlephIndexing::findWhichLidFromMapMap] hits row #\33[1;32m"<<(*var_idx)[itm]<<"\33[m";
  traceMng()->flush();
  return (*var_idx)[itm];
}


// ****************************************************************************
// * get qui trig findWhichLidFromMapMap
// ****************************************************************************
int AlephIndexing::get(double *variable,
                       int* itm){
  return get(variable, *itm);
}
int AlephIndexing::get(double *variable,
                       int itm){
  double *var=variable;//variable.variable();
  if (m_kernel->isInitialized())
    return  (m_var_map_idx.find(var)->second)->at(itm)-m_kernel->topology()->part()[m_kernel->rank()];
  // On teste de bien travailler sur une variables scalaire
  //if (var->dimension()!=1) throw std::invalid_argument(A_FUNCINFO);
  // On vérifie que le type d'item est bien connu
  //if (var->itemKind()>=IK_Unknown) throw std::invalid_argument(A_FUNCINFO);
  //debug()<<"\33[1;33m[AlephIndexing::get] Valid couple, now looking for known idx (uid="<<itm->uniqueId()<<")\33[m";
  return findWhichLidFromMapMap(var,itm);
}


// ****************************************************************************
// * buildIndexesFromAddress
// ****************************************************************************
void AlephIndexing::buildIndexesFromAddress(void){
  const int topology_row_offset=m_kernel->topology()->part()[m_kernel->rank()];
  VarMapIdx::const_iterator iVarIdx=m_var_map_idx.begin();
  debug()<<"\33[1;7;33m[buildIndexesFromAddress] Re-inexing variables with offset "<<topology_row_offset<<"\33[m";
  // On ré-indice et synchronise toutes les variables qu'on a pu voir passer
  for(;iVarIdx!=m_var_map_idx.end(); ++iVarIdx){
    //items group = iVarIdx->first->itemGroup();
    vector<int> *var_idx = iVarIdx->second;
    for(int i=0;i!=var_idx->size();++i){
      // Si cet item n'est pas utilisé, on s'en occupe pas
      if (var_idx->at(i)==ALEPH_INDEX_NOT_USED) continue;
      // Sinon on rajoute l'offset
      var_idx->at(i)+=topology_row_offset;
    }
    debug()<<"\t\33[1;7;33m[buildIndexesFromAddress] Synchronizing idx for variable *or not*\33[m";
    //iVarIdx->second->synchronize();
  }
}


// ****************************************************************************
// * localKnownItems
// * Consolidation en nombre des m_known_items_own fonction des items
// ****************************************************************************
int AlephIndexing::localKnownItems(void){
  return m_known_items_own;
}


// ****************************************************************************
// * nowYouCanBuildTheTopology
// ****************************************************************************
void AlephIndexing::nowYouCanBuildTheTopology(AlephMatrix *fromThisMatrix,
                                              AlephVector *fromThisX,
                                              AlephVector *fromThisB){
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] \33[m";
  int lki=localKnownItems();
  // ReduceSum sur l'ensemble de la topologie
  int gki= m_kernel->parallel()->reduce(Parallel::ReduceSum,lki);
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] Working with lki="
         <<lki<<", gki="<<gki<<"\33[m";
  // Initialisation du kernel d'Aleph en fonction les locals et globals known items
  m_kernel->initialize(gki,lki);
  // A partir d'ici, le kernel est initialisé, on utilise directement les m_arguments_queue
  // LA topologie a été remplacée par une nouvelle
  debug()<<"\33[1;7;33m[AlephIndexing::nowYouCanBuildTheTopology] Kernel is now initialized, rewinding Aleph operations!\33[m";
  // Si on est en parallèle, il faut consolider les indices suivant la nouvelle topologie
  if (m_kernel->isParallel())
    buildIndexesFromAddress();
  // On peut maintenant créer le triplet (matrice,lhs,rhs)
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] asking kernel for a Matrix\33[m";
  AlephMatrix *firstMatrix=m_kernel->createSolverMatrix();  
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] asking kernel for a RHS Vector\33[m";
  AlephVector *firstRhsVector=m_kernel->createSolverVector();
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] asking kernel for a LHS Vector\33[m";
  AlephVector *firstLhsVector=m_kernel->createSolverVector();
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstMatrix->create()\33[m";
  firstMatrix->create();
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstRhsVector->create()\33[m";
  firstRhsVector->create();
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstLhsVector->create()\33[m";
  firstLhsVector->create();
  // Et on revient pour rejouer les setValues de la matrice avec les indices consolidés
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] reSetValues fromThisMatrix\33[m";
  fromThisMatrix->reSetValuesIn(firstMatrix,
                                m_known_items_all_address);
  // Et on revient pour rejouer les addValues de la matrice avec les indices consolidés
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] reAddValues fromThisMatrix\33[m";
  fromThisMatrix->reAddValuesIn(firstMatrix,
                                m_known_items_all_address);
  // On reprovoque l'assemblage
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstMatrix->assemble()\33[m";
  firstMatrix->assemble();
  // Et on fait le même processus pour les lhs et rhs
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstRhsVector reSetLocalComponents/assemble\33[m";
  firstRhsVector->reSetLocalComponents(fromThisB);
  firstRhsVector->assemble();
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstLhsVector reSetLocalComponents/assemble\33[m";
  firstLhsVector->reSetLocalComponents(fromThisX);
  firstLhsVector->assemble();
  debug()<<"\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] nothing more to do here!\33[m";
}


// ****************************************************************************
// * ~AlephIndexing
// ****************************************************************************
AlephIndexing::~AlephIndexing(){
  debug()<<"\t\33[1;33m[AlephIndexing::~AlephIndexing] deleting each new'ed VarMapIdx..."<<"\33[m";
  VarMapIdx::const_iterator iVarIdx = m_var_map_idx.begin();
  for(;iVarIdx!=m_var_map_idx.end();++iVarIdx)
    delete iVarIdx->second;
  debug()<<"\t\33[1;33m[AlephIndexing::~AlephIndexing] done!"<<"\33[m";
  traceMng()->flush();
}

