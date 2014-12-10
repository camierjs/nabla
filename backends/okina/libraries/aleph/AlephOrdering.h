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
#ifndef ALEPH_ORDERING_H
#define ALEPH_ORDERING_H

class AlephOrdering: public TraceAccessor{
public:
  AlephOrdering(AlephKernel*);
  AlephOrdering(AlephKernel*,Integer,Integer,bool=false);
  ~AlephOrdering();
public:
  inline Integer swap(Integer i){
    if (m_do_swap) return m_swap.at(i);
    return i;
  }
private:
  void initCellOrder(void);
  void initTwiceCellOrder(void);
  void initFaceOrder(void);
  void initCellFaceOrder(void);
  void initCellNodeOrder(void);
  void initTwiceCellNodeOrder(void);
private:
  bool m_do_swap;
  AlephKernel* m_kernel;
private:
  Array<Int64> m_swap;
};

#endif  

