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
//#include "nabla.h"

#include "mfem.hpp"
#include "assert.h"

#include <iostream>
//#include <stdio.h>
#include <stdarg.h>
using namespace std;
using namespace mfem;

// ****************************************************************************
// * dbg
// ****************************************************************************
const static bool debug_mode = false;
void dbg(const char *format, ...){
  if (!debug_mode) return;
  va_list args;
  va_start(args,format);
  vprintf(format,args);
  fflush(NULL);
  va_end(args);
}

// ****************************************************************************
// * glvis1D
// ****************************************************************************
void glvis1D(const int nx,
             const double sx,
             double *coord,double *data){}


// ****************************************************************************
// * Global variables
// ****************************************************************************
bool init=true;
Mesh *mesh;
FiniteElementCollection *fec;
FiniteElementSpace *fespace;
GridFunction *x;
socketstream *sol_sock;

// ****************************************************************************
// * glvis2DQud
// ****************************************************************************
void glvis2DQud(const int nx,const int ny,
                const double sx,const double sy,
                double *coord,double *data){
  if (init){
    dbg("\n[35m[glvis2DQud][m");
    const int dim = 2;
    const int order = 1;
    //const int nb_coords = (nx+1)*(ny+1);
    //Mesh(int nx, int ny, Element::Type type, int generate_edges = 0,double sx = 1.0, double sy = 1.0)
    mesh = new Mesh(nx,ny,Element::QUADRILATERAL,0,sx,sy);
    dbg("\n[35m[glvis2DQud] fec[m");
    fec = new L2_FECollection(order, dim);
    dbg("\n[35m[glvis2DQud] fespace[m");
    fespace = new FiniteElementSpace(mesh, fec);
    dbg("\n[35m[glvis2DQud] GridFunction[m");
    x=new GridFunction(fespace);
    dbg("\n[35m[glvis2DQud] x->VectorDim=%d[m",x->VectorDim());
    assert(fespace->Conforming()==true);
    dbg("\n[35m[glvis2DQud] socketstream[m");
    sol_sock=new socketstream("localhost", 19916);
    //sol_sock->precision(8);
    init=false;
    dbg("\n[35m[glvis2DQud] Init done![m");
  }

  // On assigne les coordonées
  dbg("\n[35;1m[glvis2DQud] On assigne les coords #NV=%d[m", mesh->GetNV());
  for(int i=0;i<mesh->GetNV();i++){
    double *v=mesh->GetVertex(i);
    dbg("\n\t[35m[glvis2DQud] #%d %f,%f[m", i, coord[3*i+0],coord[3*i+1]);
    v[0]=coord[3*i+0];
    v[1]=coord[3*i+1];
  }
  
  Array<int> dofs;
  dbg("\n[35;1m[glvis2DQud] On place les dofs aux mailles #NE=%d[m", mesh->GetNE());
  for(int i=0;i<mesh->GetNE();i+=1){
    fespace->GetElementDofs(i,dofs);
    dbg("\n\t[35m[glvis2DQud] i=%d, dofs.Size()=%d[m",i,dofs.Size());
    //cout<<"\n\t[35m[glvis2DQud] dofs="<<dofs<<"\n[m";
    for(int k=0;k<dofs.Size();k+=1){
      dbg("\n\t\t[35m[glvis2DQud] %d[m",dofs[k]);
      (*x)[dofs[k]]=data[i];
    }
  }
  
  *sol_sock << "solution\n" << *mesh << *x << flush;

  //delete x;
  //delete fespace;
  //delete fec;
  //delete mesh;
}



// ****************************************************************************
// * glvis3DHex
// ****************************************************************************
void glvis3DHex(const int nx, // Nombre de mailles en X,Y & Z
                const int ny,
                const int nz,
                const double sx, // Longueur des cotés en X,Y & Z
                const double sy,
                const double sz,
                double *coord, double *data){
  if (init){
    dbg("\n[glvis3DHex] init!");
    const int dim = 3;
    const int order = 1;
    mesh = new Mesh(nx,ny,nz,Element::HEXAHEDRON,0,sx,sy,sz);
    fec = new L2_FECollection(order, dim);
    fespace = new FiniteElementSpace(mesh, fec);
    x=new GridFunction(fespace);
    //dbg("\n[35m[glvis3DHex] x->VectorDim=%d[m\n",x->VectorDim());
    assert(fespace->Conforming()==true);
    sol_sock=new socketstream("localhost", 19916);
    init=false;
  }
  
  // On assigne les coordonées
  for(int i=0;i<mesh->GetNV();i++){
    double *v=mesh->GetVertex(i);
    v[0]=coord[3*i+0];
    v[1]=coord[3*i+1];
    v[2]=coord[3*i+2];
  }
 
  Array<int> dofs;
  for(int i=0;i<mesh->GetNE();i+=1){
    fespace->GetElementDofs(i,dofs);
    for(int dof=0;dof<dofs.Size();dof+=1)
      (*x)[dofs[dof]]=data[i];
  }
  
  *sol_sock << "solution\n" << *mesh << *x << flush;
  
  //delete fespace;
  //if (order > 0) { delete fec; }
  //delete mesh;
}
