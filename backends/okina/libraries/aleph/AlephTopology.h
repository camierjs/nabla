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
#ifndef ALEPH_TOPOLOGY_H
#define ALEPH_TOPOLOGY_H 

#include "Aleph.h"

class IAlephTopology;

class AlephTopology: public TraceAccessor{
 public:
  AlephTopology(AlephKernel*);
  AlephTopology(ITraceMng*,AlephKernel*,Integer,Integer);
  virtual ~AlephTopology();
 public:
  void create(Integer);
  void setRowNbElements( IntegerConstArrayView row_nb_element);
  IntegerConstArrayView ptr_low_up_array();
  IntegerConstArrayView part();
  IParallelMng* parallelMng();
  void rowRange(Integer& min_row,Integer& max_row);
 private:
  inline void checkForInit(){
    if (m_has_been_initialized==false)
      throw FatalErrorException("AlephTopology::create","Has not been yet initialized!");
  }
 public:
  Integer rowLocalRange(const Integer);
  AlephKernel* kernel(void){return m_kernel;}
  Integer nb_row_size(void){/*checkForInit();*/ return m_nb_row_size;}
  Integer nb_row_rank(void){checkForInit(); return m_nb_row_rank;}
  Integer gathered_nb_row(Integer i){checkForInit(); return m_gathered_nb_row[i];}
  ArrayView<Integer> gathered_nb_row_elements(void){checkForInit(); return m_gathered_nb_row_elements;}
  ArrayView<Integer> gathered_nb_setValued(void){checkForInit(); return m_gathered_nb_setValued;}
  Integer gathered_nb_setValued(Integer i){checkForInit(); return m_gathered_nb_setValued[i];}
  bool hasSetRowNbElements(void){return m_has_set_row_nb_elements;}

 private:
  AlephKernel* m_kernel;
  Integer m_nb_row_size; // Nombre de lignes de la matrice réparties sur l'ensemble 
  Integer m_nb_row_rank; // Nombre de lignes de la matrice vue de mon rang
  Array<Integer> m_gathered_nb_row; // Indices des lignes par CPU
  Array<Integer> m_gathered_nb_row_elements;  // nombre d'éléments par ligne
  Array<Integer> m_gathered_nb_setValued;     // nombre d'éléments setValué par CPU
  bool m_created;
  bool m_has_set_row_nb_elements;
  bool m_has_been_initialized;
};

#endif  

