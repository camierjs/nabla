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

using namespace std;
using namespace mfem;

// ****************************************************************************
// * glvis1D
// ****************************************************************************
void glvis1D(const int nx,
             const double sx,
             double *coord,double *data){}


// ****************************************************************************
// * glvis2DQud
// ****************************************************************************
void glvis2DQud(const int nx,const int ny,
                const double sx,const double sy,
                double *coord,double *data){
  const int dim = 2;
  const int order = 1;
  const int nb_coords = (nx+1)*(ny+1);
  Mesh *mesh = new Mesh(nx,ny,Element::QUADRILATERAL,0,sx,sy);

  // Dump des coords que l'on récupère
  for(int i=0;i<nb_coords;i+=1){
    double *v=&coord[3*i];
    printf("\n\tcoord[%d]=(%f,%f,%f)",i,v[0],v[1],v[2]);
  }

  for(int i=0;i<mesh->GetNV();i++){
    double *v=mesh->GetVertex(i);
    v[0]=coord[3*i+0];
    v[1]=coord[3*i+1];
    v[2]=coord[3*i+2];
    printf("\n\t    v[%d]=(%f,%f,%f)",i,v[0],v[1],v[2]);
  }
  
  FiniteElementCollection *fec = new L2_FECollection(order, dim);
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
  
  cout<<"\nfespace size is "<< fespace->GetNDofs();
  GridFunction x(fespace);
  assert(fespace->Conforming()==true);
  
  for(int i=0;i<mesh->GetNE();i+=1){
    const Element *element = mesh->GetElement(i);
    const int nv=element->GetNVertices();
    const int *iVertices=element->GetVertices();
    //printf("\n\te[%d] has %d vertices",i,nv);
    //for(int v=0;v<element->GetNVertices();v+=1) printf(" %d",iVertices[v]);
    // Now working with DOFs
    //printf("\n\tfespace->GetNDofs()=%d",fespace->GetNDofs());
    Array<int> dofs;
    fespace->GetElementDofs(i,dofs);
    for(int dof=0;dof<dofs.Size();dof+=1){
      //printf("\n\t\tdof[%d] is #%d",dof,dofs[dof]);
      // On les met tous à la même valeure pour l'instant
      x[dofs[dof]]=data[i];
    }
  }

  socketstream sol_sock("localhost", 19916);
  sol_sock << "solution\n" << *mesh << x << flush;
  
  delete fespace;
  delete fec;
  delete mesh;
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
  const int dim = 3;
  const int order = 1;
  //const char *mesh_file = "/home/nabla/nabla/tpl/mfem/data/inline-hex.mesh";
  Mesh *mesh = new Mesh(nx,ny,nz,Element::HEXAHEDRON,0,sx,sy,sz);
  
  // Dump des coords que l'on récupère
  /*for(int i=0;i<nv;i+=1){
    double *v=&node_coord[3*i];
    printf("\n\tnode_coord[%d]=(%f,%f,%f)",i,v[0],v[1],v[2]);
    }*/
    
  for(int i=0;i<mesh->GetNV();i++){
    double *v=mesh->GetVertex(i);
    //printf("\n\tv[%d]=(%f,%f,%f)",i,v[0],v[1],v[2]);
    //v->SetCoords(&node_coord[3*i]);
    v[0]=coord[3*i+0];
    v[1]=coord[3*i+1];
    v[2]=coord[3*i+2];
    //printf("\n\tv[%d]=(%f,%f,%f)",i,v[0],v[1],v[2]);
  }

  FiniteElementCollection *fec = new L2_FECollection(order, dim);
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
  
  cout<<"\nfespace size is "<< fespace->GetNDofs();
  GridFunction x(fespace);
  assert(fespace->Conforming()==true);
    
  for(int i=0;i<mesh->GetNE();i+=1){
    const Element *element = mesh->GetElement(i);
    const int nv=element->GetNVertices();
    const int *iVertices=element->GetVertices();
    //printf("\n\te[%d] has %d vertices",i,nv);
    //for(int v=0;v<element->GetNVertices();v+=1) printf(" %d",iVertices[v]);
    // Now working with DOFs
    //printf("\n\tfespace->GetNDofs()=%d",fespace->GetNDofs());
    Array<int> dofs;
    fespace->GetElementDofs(i,dofs);
    for(int dof=0;dof<dofs.Size();dof+=1){
      //printf("\n\t\tdof[%d] is #%d",dof,dofs[dof]);
      // On les met tous à la même valeure pour l'instant
      x[dofs[dof]]=data[i];
    }
  }

  {
    ofstream mesh_ofs("displaced.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    x.Save(sol_ofs);
  }
  
  {
    socketstream sol_sock("localhost", 19916);
    sol_sock << "solution\n" << *mesh << x << flush;
  }
  
  delete fespace;
  if (order > 0) { delete fec; }
  delete mesh;
}
