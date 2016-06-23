//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   bool linear_only = false;
   bool assemble_only = false;
   bool mfem_debug = false;
   int ref_levels = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&linear_only, "-lin", "--linear_only", "-no-lin",
                  "--no-linear_only",
                  "Enable or disable linear_only.");
   args.AddOption(&assemble_only, "-asm", "--assemble_only", "-no-asm",
                  "--no-assemble_only",
                  "Enable or disable assemble_only.");
   args.AddOption(&mfem_debug, "-dbg", "--mfem_debug", "-no-dbg",
                  "--no-mfem_debug",
                  "Enable or disable mfem_debug.");
   args.AddOption(&ref_levels, "-l", "--level", 
                  "Enable or disable linear quit.");
   args.Parse();
   if (!args.Good()){
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (mfem_debug) mfemDbgOn();

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   if (ref_levels==-1)
     ref_levels = (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
   for (int l = 0; l < ref_levels; l++){
     cout<<"[33m[main] mesh->UniformRefinement();[0m\n";
     mesh->UniformRefinement();
   }   

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   cout<<"[33m[main] mesh->GetNE()="<<mesh->GetNE()<<"[0m\n";
   cout<<"[33m[main] dim="<<dim<<"[0m\n";
   FiniteElementCollection *fec;
   if (order > 0)
   {
     cout<<"[33m[main] H1_FECollection[0m\n";
      fec = new H1_FECollection(order, dim);
  }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
       cout<<"[33m[main] Using isoparametric FEs[0m\n";
     cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      cout<<"[33m[main] H1_FECollection ORDER 1[0m\n";
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout<<"[33m[main] number of degrees of freedom = "<<fespace->GetNDofs()<<"[0m\n";
   cout<<"[33m[main] vector dimension = "<<fespace->GetVDim()<<"[0m\n";
   cout<<"[33m[main] order of the 0'th finite element = "<<fespace->GetOrder(0)<<"[0m\n";
   cout<<"[33m[main] GetNE="<<fespace->GetNE()<<"[0m\n";
   cout<<"[33m[main] GetNBE="<<fespace->GetNBE()<<"[0m\n";
   //cout<<"[33m[main] ="<<fespace->()<<"[0m\n";
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   cout<<"[33m[main] b = new LinearForm[0m\n";
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   cout<<"[33m[main] b->AddDomainIntegrator[0m\n";
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   cout<<"[33m[main] b->Assemble[0m\n";
   b->Assemble();

   if (!linear_only){
     // 7. Define the solution vector x as a finite element grid function
     //    corresponding to fespace. Initialize x with initial guess of zero,
     //    which satisfies the boundary conditions.
     cout<<"[33m[main] GridFunction: fespace[0m\n";
     GridFunction x(fespace);
     x = 0.0;

     // 8. Set up the bilinear form a(.,.) on the finite element space
     //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
     //    domain integrator.
     cout<<"[33m[main] a = new BilinearForm[0m\n";
     BilinearForm *a = new BilinearForm(fespace);
     //a->AddDomainIntegrator(new DiffusionIntegrator(one));
     cout<<"[33m[main] a->AddDomainIntegrator(new MassIntegrator)[0m\n";
     a->AddDomainIntegrator(new MassIntegrator(one));

     // 9. Assemble the bilinear form and the corresponding linear system,
     //    applying any necessary transformations such as: eliminating boundary
     //    conditions, applying conforming constraints for non-conforming AMR,
     //    static condensation, etc.
     if (static_cond) { a->EnableStaticCondensation(); }
     cout<<"[33m[main] a->Assemble[0m\n";
     a->Assemble();

   
     if (!assemble_only){
       SparseMatrix A;
       Vector B, X;
       cout<<"[33m[main] a->FormLinearSystem[0m\n";
       a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

       cout << "Size of linear system: " << A.Height() << endl;

       cout<<"[33m[main] Solving![0m\n";
#ifndef MFEM_USE_SUITESPARSE
       // 10. Define a simple symmetric Gauss-Seidel preconditioner and use it to
       //     solve the system A X = B with PCG.
       GSSmoother M(A);
       PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#else
       // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
       UMFPackSolver umf_solver;
       umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
       umf_solver.SetOperator(A);
       umf_solver.Mult(B, X);
#endif

       // 11. Recover the solution as a finite element grid function.
       a->RecoverFEMSolution(X, *b, x);

       // 12. Save the refined mesh and the solution. This output can be viewed later
       //     using GLVis: "glvis -m refined.mesh -g sol.gf".
       ofstream mesh_ofs("refined.mesh");
       mesh_ofs.precision(8);
       mesh->Print(mesh_ofs);
       ofstream sol_ofs("sol.gf");
       sol_ofs.precision(8);
       x.Save(sol_ofs);

       // 13. Send the solution by socket to a GLVis server.
       if (visualization){
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << *mesh << x << flush;
       }
     }
     delete a;
   }
   
   // 14. Free the used memory.
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
