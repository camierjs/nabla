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
#ifndef ALEPH_KAPPA_H
#define ALEPH_KAPPA_H 

class AlephKernel;
class AlephFactory;

class AlephKappaService
: public TraceAccessor{
 public:
  AlephKappaService(ITraceMng*);
  ~AlephKappaService();
  virtual void build(void){}
 public:
  //! Exécute l'opération du service
  virtual void execute(void);
  //! Vrai si le service est actif
  virtual bool isActive(void) const { return true;}
  virtual void setParallelMng(IParallelMng *wpm){ m_world_parallel = wpm; }

 private:
  AlephKernel *m_kernel;
  IApplication* m_application;
  IParallelMng* m_world_parallel;
  Integer m_world_rank;
  Integer m_size;
  Integer m_world_size;
  AlephFactory *m_factory;
  Integer m_underlying_solver;
  Integer m_solver_size;
  bool m_reorder;
};

#endif  

