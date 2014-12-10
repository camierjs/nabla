// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
#ifndef ALEPH_IALEPHFACTORY_H
#define ALEPH_IALEPHFACTORY_H


#include "AlephStd.h"

#include "Aleph.h"
#include "IAlephFactory.h"
#include "AlephInterface.h"

#include <map>
#include <string>

class AlephFactory: public IAlephFactory{
 private:
  struct FactoryImpl{
   public:
    FactoryImpl(const String& name) : m_factory(0),
                                      m_name(name),
                                      m_initialized(false){}
   public:
    IAlephFactoryImpl* m_factory;
    String m_name;
    Bool m_initialized;
  };
 public:
  AlephFactory(ITraceMng *tm);
  ~AlephFactory();
 public:
  IAlephTopology* GetTopology(AlephKernel *kernel, Integer index, Integer nb_row_size);
  IAlephVector* GetVector(AlephKernel *kernel, Integer index);
  IAlephMatrix* GetMatrix(AlephKernel *kernel, Integer index);
 private:
  typedef std::map<Integer,FactoryImpl*> FactoryImplMap;
  FactoryImplMap m_impl_map;
  IAlephFactoryImpl* _getFactory(Integer solver_index);
};

#endif 
