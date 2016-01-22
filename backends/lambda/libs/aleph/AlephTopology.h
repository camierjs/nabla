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
#ifndef ALEPH_TOPOLOGY_H
#define ALEPH_TOPOLOGY_H 

class IAlephTopology;

class AlephTopology: public TraceAccessor{
 public:
  AlephTopology(AlephKernel*);
  AlephTopology(ITraceMng*,AlephKernel*,int,int);
  virtual ~AlephTopology();
 public:
  void create(int);
  void setRowNbElements( vector<int> row_nb_element);
  vector<int> ptr_low_up_array();
  vector<int> part();
  IParallelMng* parallelMng();
  void rowRange(int& min_row,int& max_row);
 private:
  inline void checkForInit(){
    if (m_has_been_initialized==false)
      throw std::logic_error("[AlephTopology::create] Has not been yet initialized!");
  }
 public:
  int rowLocalRange(const int);
  AlephKernel* kernel(void){return m_kernel;}
  int nb_row_size(void){/*checkForInit();*/ return m_nb_row_size;}
  int nb_row_rank(void){checkForInit(); return m_nb_row_rank;}
  int gathered_nb_row(int i){checkForInit(); return m_gathered_nb_row[i];}
  vector<int> gathered_nb_row_elements(void){checkForInit(); return m_gathered_nb_row_elements;}
  vector<int> gathered_nb_setValued(void){checkForInit(); return m_gathered_nb_setValued;}
  int gathered_nb_setValued(int i){checkForInit(); return m_gathered_nb_setValued[i];}
  bool hasSetRowNbElements(void){return m_has_set_row_nb_elements;}

 private:
  AlephKernel* m_kernel;
  int m_nb_row_size; // Nombre de lignes de la matrice réparties sur l'ensemble 
  int m_nb_row_rank; // Nombre de lignes de la matrice vue de mon rang
  vector<int> m_gathered_nb_row; // Indices des lignes par CPU
  vector<int> m_gathered_nb_row_elements;  // nombre d'éléments par ligne
  vector<int> m_gathered_nb_setValued;     // nombre d'éléments setValué par CPU
  bool m_created;
  bool m_has_set_row_nb_elements;
  bool m_has_been_initialized;
};

#endif  

