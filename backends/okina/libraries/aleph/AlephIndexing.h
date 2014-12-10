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
#ifndef ALEPH_INDEXING_H
#define ALEPH_INDEXING_H

class AlephIndexing: public TraceAccessor{
 public:
  AlephIndexing(AlephKernel*);
  ~AlephIndexing();
 public:
  Int32 updateKnownItems(VariableItemInt32*,const Item &);
  Int32 findWhichLidFromMapMap(IVariable*,const Item &);
  Integer get(const VariableRef&, const ItemEnumerator&);
  Integer get(const VariableRef&, const Item&);
  void buildIndexesFromAddress(void);
  void nowYouCanBuildTheTopology(AlephMatrix*,AlephVector*,AlephVector*);
 private:
  Integer localKnownItems(void);
 private:
  AlephKernel *m_kernel;
  ISubDomain *m_sub_domain;
  Integer m_current_idx;
  Int32 m_known_items_own;
  Array<Int32*> m_known_items_all_address;
  typedef std::map<IVariable*,VariableItemInt32*> VarMapIdx;
  VarMapIdx m_var_map_idx;
};

#endif  

