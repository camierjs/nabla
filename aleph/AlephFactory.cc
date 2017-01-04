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
#include "Aleph.h"
#include <map>
#include <string>

#include "IAlephFactory.h"
extern IAlephFactoryImpl* fetchHypre(void);


/******************************************************************************
 * IAlephFactory::IAlephFactory
 *****************************************************************************/
AlephFactory::AlephFactory(ITraceMng *tm): IAlephFactory(tm){
  // Liste des implémentations possibles.
  // 0 est le choix automatique qui doit aller vers une des bibliothèques suivantes:
  //m_impl_map.insert(std::make_pair(1,new FactoryImpl("Sloop")));
  m_impl_map.insert(std::make_pair(2,new FactoryImpl("Hypre")));
  //m_impl_map.insert(std::make_pair(3,new FactoryImpl("Trilinos")));
  //m_impl_map.insert(std::make_pair(4,new FactoryImpl("Cuda")));
  //m_impl_map.insert(std::make_pair(5,new FactoryImpl("PETSc")));
  //ServiceBuilder<IAlephFactoryImpl> sb(app);
  // Pour chaque implémentation possible,
  // créé la fabrique correspondante si elle est disponible.
  for(FactoryImplMap::iterator i = m_impl_map.begin(); i!=m_impl_map.end(); ++i ){
    FactoryImpl *implementation=i->second;
    const string name=i->second->m_name;
    const string implName = name+"AlephFactoryImpl";//implementation->m_name+"AlephFactory";
    debug()<<"\33[1;34m\t[AlephFactory] Adding "<<implName<<" library..."<<"\33[0m";
    //IAlephFactoryImpl *factory = sb.createInstance(name+"AlephFactory",SB_AllowNull);
    //implementation->m_factory = factory;
//#error dynamic_cast instance
    implementation->m_factory = fetchHypre();//dynamic_cast<IAlephFactoryImpl*>(new HypreAlephFactoryImpl());
  }
  debug()<<"\33[1;34m\t[AlephFactory] done"<<"\33[0m";
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
AlephFactory::~AlephFactory(){
  debug()<<"\33[1;34m\t[~AlephFactory] Destruction des fabriques"<<"\33[0m";
  FactoryImplMap::iterator i = m_impl_map.begin();
  for(;i!=m_impl_map.end();++i)
    delete i->second->m_factory;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
IAlephFactoryImpl* AlephFactory::_getFactory(int solver_index){
  FactoryImplMap::const_iterator ci = m_impl_map.find(solver_index);
  if (ci==m_impl_map.end())
    throw std::logic_error("[AlephFactory::_getFactory] Invalid solver index for aleph factory");
  FactoryImpl *implementation=ci->second;
  IAlephFactoryImpl* factory = implementation->m_factory;
  if (!factory)
    throw std::logic_error("[AlephFactory::_getFactory] Implementation not available");
  // Si la fabrique de l'implémentation considérée n'a pas
  // été initialisée, on le fait maintenant
  if (!implementation->m_initialized){
    debug()<< "\33[1;34m\t\t[_getFactory] initializing solver_index="
          << solver_index << " ..."<<"\33[0m";
    implementation->m_initialized=true;
    factory->initialize();
  }
  return factory;
}


/******************************************************************************
 * AlephFactory::GetTopology
 *****************************************************************************/
IAlephTopology* AlephFactory::GetTopology(AlephKernel *kernel,
                                          int index,
                                          int nb_row_size){
  debug()<<"\33[1;34m\t\t[IAlephFactory::GetTopology] Switch="<<kernel->underlyingSolver()<<"\33[0m";
  return _getFactory(kernel->underlyingSolver())->createTopology(traceMng(),
                                                                 kernel,
                                                                 index,
                                                                 nb_row_size);
}


/******************************************************************************
 * AlephFactory::GetVector
 *****************************************************************************/
IAlephVector* AlephFactory::GetVector(AlephKernel *kernel,
                                      int index){
  debug()<<"\33[1;34m\t\t[AlephFactory::GetVector] Switch="<<kernel->underlyingSolver()<<"\33[0m";
  return _getFactory(kernel->underlyingSolver())->createVector(traceMng(),
                                                               kernel,
                                                               index);
}

/******************************************************************************
 * AlephFactory::GetMatrix
 *****************************************************************************/
IAlephMatrix* AlephFactory::GetMatrix(AlephKernel *kernel,
                                      int index){
  debug()<<"\33[1;34m\t\t[AlephFactory::GetMatrix] Switch="<<kernel->underlyingSolver()<<"\33[0m";
  return _getFactory(kernel->underlyingSolver())->createMatrix(traceMng(),
                                                               kernel,
                                                               index);
};

