#include "stdinc.h"
#include "linalg.h"
#include "ioc.h"

namespace mfem
{
  class STable3DNode
  {
  public:
    STable3DNode * Prev;
    int Column, Floor, Number;
  };
  class STable3D
  {
  private:
    int Size, NElem;
    STable3DNode **Rows;
      MemAlloc < STable3DNode, 1024 > NodesMem;
  public:
      explicit STable3D (int nr);
    int Push (int r, int c, int f);
    int operator () (int r, int c, int f) const;
    int Index (int r, int c, int f) const;
    int Push4 (int r, int c, int f, int t);
    int operator () (int r, int c, int f, int t) const;
    int NumberOfElements ()
    {
      return NElem;
    };
     ~STable3D ();
  };
}

namespace mfem
{
  class IntegrationPoint
  {
  public:
    double x, y, z, weight;
    void Init ()
    {
      x = y = z = weight = 0.0;
    }
    void Set (const double *p, const int dim)
    {
      x = p[0];
      if (dim > 1)
	{
	  y = p[1];
	  if (dim > 2)
	    z = p[2];
	}
    }
    void Get (double *p, const int dim) const
    {
      p[0] = x;
      if (dim > 1)
	{
	  p[1] = y;
	  if (dim > 2)
	    p[2] = z;
	}
    }
    void Set (const double x1, const double x2, const double x3,
	      const double w)
    {
      x = x1;
      y = x2;
      z = x3;
      weight = w;
    }
    void Set3w (const double *p)
    {
      x = p[0];
      y = p[1];
      z = p[2];
      weight = p[3];
    }
    void Set3 (const double x1, const double x2, const double x3)
    {
      x = x1;
      y = x2;
      z = x3;
    }
    void Set3 (const double *p)
    {
      x = p[0];
      y = p[1];
      z = p[2];
    }
    void Set2w (const double x1, const double x2, const double w)
    {
      x = x1;
      y = x2;
      weight = w;
    }
    void Set2w (const double *p)
    {
      x = p[0];
      y = p[1];
      weight = p[2];
    }
    void Set2 (const double x1, const double x2)
    {
      x = x1;
      y = x2;
    }
    void Set2 (const double *p)
    {
      x = p[0];
      y = p[1];
    }
    void Set1w (const double x1, const double w)
    {
      x = x1;
      weight = w;
    }
    void Set1w (const double *p)
    {
      x = p[0];
      weight = p[1];
    }
  };
  class IntegrationRule:public Array < IntegrationPoint >
  {
  private:
    friend class IntegrationRules;
    void GaussianRule ();
    void UniformRule ();
    void GrundmannMollerSimplexRule (int s, int n = 3);
    void AddTriMidPoint (const int off, const double weight)
    {
      IntPoint (off).Set2w (1. / 3., 1. / 3., weight);
    }
    void AddTriPoints3 (const int off, const double a, const double b,
			const double weight)
    {
      IntPoint (off + 0).Set2w (a, a, weight);
      IntPoint (off + 1).Set2w (a, b, weight);
      IntPoint (off + 2).Set2w (b, a, weight);
    }
    void AddTriPoints3 (const int off, const double a, const double weight)
    {
      AddTriPoints3 (off, a, 1. - 2. * a, weight);
    }
    void AddTriPoints3b (const int off, const double b, const double weight)
    {
      AddTriPoints3 (off, (1. - b) / 2., b, weight);
    }
    void AddTriPoints3R (const int off, const double a, const double b,
			 const double c, const double weight)
    {
      IntPoint (off + 0).Set2w (a, b, weight);
      IntPoint (off + 1).Set2w (c, a, weight);
      IntPoint (off + 2).Set2w (b, c, weight);
    }
    void AddTriPoints3R (const int off, const double a, const double b,
			 const double weight)
    {
      AddTriPoints3R (off, a, b, 1. - a - b, weight);
    }
    void AddTriPoints6 (const int off, const double a, const double b,
			const double c, const double weight)
    {
      IntPoint (off + 0).Set2w (a, b, weight);
      IntPoint (off + 1).Set2w (b, a, weight);
      IntPoint (off + 2).Set2w (a, c, weight);
      IntPoint (off + 3).Set2w (c, a, weight);
      IntPoint (off + 4).Set2w (b, c, weight);
      IntPoint (off + 5).Set2w (c, b, weight);
    }
    void AddTriPoints6 (const int off, const double a, const double b,
			const double weight)
    {
      AddTriPoints6 (off, a, b, 1. - a - b, weight);
    }
    void AddTetPoints3 (const int off, const double a, const double b,
			const double weight)
    {
      IntPoint (off + 0).Set (a, a, b, weight);
      IntPoint (off + 1).Set (a, b, a, weight);
      IntPoint (off + 2).Set (b, a, a, weight);
    }
    void AddTetPoints6 (const int off, const double a, const double b,
			const double c, const double weight)
    {
      IntPoint (off + 0).Set (a, b, c, weight);
      IntPoint (off + 1).Set (a, c, b, weight);
      IntPoint (off + 2).Set (b, c, a, weight);
      IntPoint (off + 3).Set (b, a, c, weight);
      IntPoint (off + 4).Set (c, a, b, weight);
      IntPoint (off + 5).Set (c, b, a, weight);
    }
    void AddTetMidPoint (const int off, const double weight)
    {
      IntPoint (off).Set (0.25, 0.25, 0.25, weight);
    }
    void AddTetPoints4 (const int off, const double a, const double weight)
    {
      IntPoint (off).Set (a, a, a, weight);
      AddTetPoints3 (off + 1, a, 1. - 3. * a, weight);
    }
    void AddTetPoints4b (const int off, const double b, const double weight)
    {
      const double a = (1. - b) / 3.;
      IntPoint (off).Set (a, a, a, weight);
      AddTetPoints3 (off + 1, a, b, weight);
    }
    void AddTetPoints6 (const int off, const double a, const double weight)
    {
      const double b = 0.5 - a;
      AddTetPoints3 (off, a, b, weight);
      AddTetPoints3 (off + 3, b, a, weight);
    }
    void AddTetPoints12 (const int off, const double a, const double bc,
			 const double weight)
    {
      const double cb = 1. - 2 * a - bc;
      AddTetPoints3 (off, a, bc, weight);
      AddTetPoints3 (off + 3, a, cb, weight);
      AddTetPoints6 (off + 6, a, bc, cb, weight);
    }
    void AddTetPoints12bc (const int off, const double b, const double c,
			   const double weight)
    {
      const double a = (1. - b - c) / 2.;
      AddTetPoints3 (off, a, b, weight);
      AddTetPoints3 (off + 3, a, c, weight);
      AddTetPoints6 (off + 6, a, b, c, weight);
    }
  public:
  IntegrationRule ():Array < IntegrationPoint > ()
    {
    }
    explicit IntegrationRule (int NP):Array < IntegrationPoint > (NP)
    {
      for (int i = 0; i < this->Size (); i++)
	(*this)[i].Init ();
    }
    IntegrationRule (IntegrationRule & irx, IntegrationRule & iry);
    int GetNPoints () const
    {
      return Size ();
    }
    IntegrationPoint & IntPoint (int i)
    {
      return (*this)[i];
    }
    const IntegrationPoint & IntPoint (int i) const
    {
      return (*this)[i];
    }
     ~IntegrationRule ()
    {
    }
  };
  class IntegrationRules
  {
  private:
    int own_rules, refined;
      Array < IntegrationRule * >PointIntRules;
      Array < IntegrationRule * >SegmentIntRules;
      Array < IntegrationRule * >TriangleIntRules;
      Array < IntegrationRule * >SquareIntRules;
      Array < IntegrationRule * >TetrahedronIntRules;
      Array < IntegrationRule * >CubeIntRules;
    void AllocIntRule (Array < IntegrationRule * >&ir_array, int Order)
    {
      if (ir_array.Size () <= Order)
	ir_array.SetSize (Order + 1, __null);
    }
    bool HaveIntRule (Array < IntegrationRule * >&ir_array, int Order)
    {
      return (ir_array.Size () > Order && ir_array[Order] != __null);
    }
    IntegrationRule *GenerateIntegrationRule (int GeomType, int Order);
    IntegrationRule *PointIntegrationRule (int Order);
    IntegrationRule *SegmentIntegrationRule (int Order);
    IntegrationRule *TriangleIntegrationRule (int Order);
    IntegrationRule *SquareIntegrationRule (int Order);
    IntegrationRule *TetrahedronIntegrationRule (int Order);
    IntegrationRule *CubeIntegrationRule (int Order);
    void DeleteIntRuleArray (Array < IntegrationRule * >&ir_array);
  public:
    explicit IntegrationRules (int Ref = 0);
    const IntegrationRule & Get (int GeomType, int Order);
    void Set (int GeomType, int Order, IntegrationRule & IntRule);
    void SetOwnRules (int o)
    {
      own_rules = o;
    }
    ~IntegrationRules ();
  };
  extern IntegrationRules IntRules;
  extern IntegrationRules RefinedIntRules;
}

namespace mfem
{
  class Geometry
  {
  public:
    enum Type
    { POINT, SEGMENT, TRIANGLE, SQUARE, TETRAHEDRON, CUBE };
    static const int NumGeom = 6;
    static const int NumBdrArray[];
    static const char *Name[NumGeom];
    static const double Volume[NumGeom];
  private:
      IntegrationRule * GeomVert[NumGeom];
    IntegrationPoint GeomCenter[NumGeom];
    DenseMatrix *PerfGeomToGeomJac[NumGeom];
  public:
      Geometry ();
     ~Geometry ();
    const IntegrationRule *GetVertices (int GeomType);
    const IntegrationPoint & GetCenter (int GeomType)
    {
      return GeomCenter[GeomType];
    }
    DenseMatrix *GetPerfGeomToGeomJac (int GeomType)
    {
      return PerfGeomToGeomJac[GeomType];
    }
    void GetPerfPointMat (int GeomType, DenseMatrix & pm);
    void JacToPerfJac (int GeomType, const DenseMatrix & J,
		       DenseMatrix & PJ) const;
    int NumBdr (int GeomType)
    {
      return NumBdrArray[GeomType];
    }
  };
  extern Geometry Geometries;
  class RefinedGeometry
  {
  public:
    int Times, ETimes;
    IntegrationRule RefPts;
      Array < int >RefGeoms, RefEdges;
      RefinedGeometry (int NPts, int NRefG, int NRefE):RefPts (NPts),
      RefGeoms (NRefG), RefEdges (NRefE)
    {
    }
  };
  class GeometryRefiner
  {
  private:
    int type;
    RefinedGeometry *RGeom[Geometry::NumGeom];
    IntegrationRule *IntPts[Geometry::NumGeom];
  public:
      GeometryRefiner ();
    void SetType (const int t)
    {
      type = t;
    }
    RefinedGeometry *Refine (int Geom, int Times, int ETimes = 1);
    const IntegrationRule *RefineInterior (int Geom, int Times);
    ~GeometryRefiner ();
  };
  extern GeometryRefiner GlobGeometryRefiner;
}

namespace mfem
{
  class FunctionSpace
  {
  public:
    enum
    {
      Pk,
      Qk,
      rQk
    };
  };
  class ElementTransformation;
  class Coefficient;
  class VectorCoefficient;
  class KnotVector;
  class FiniteElement
  {
  protected:
    int Dim, GeomType, Dof, Order, FuncSpace, RangeType, MapType;
    IntegrationRule Nodes;
  public:
    enum
    { SCALAR, VECTOR };
    enum
    { VALUE, INTEGRAL, H_DIV, H_CURL };
      FiniteElement (int D, int G, int Do, int O, int F = FunctionSpace::Pk);
    int GetDim () const
    {
      return Dim;
    }
    int GetGeomType () const
    {
      return GeomType;
    }
    int GetDof () const
    {
      return Dof;
    }
    int GetOrder () const
    {
      return Order;
    }
    int Space () const
    {
      return FuncSpace;
    }
    int GetRangeType () const
    {
      return RangeType;
    }
    int GetMapType () const
    {
      return MapType;
    }
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const = 0;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const = 0;
    const IntegrationRule & GetNodes () const
    {
      return Nodes;
    }
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const;
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void CalcCurlShape (const IntegrationPoint & ip,
				DenseMatrix & curl_shape) const;
    virtual void GetFaceDofs (int face, int **dofs, int *ndofs) const;
    virtual void CalcHessian (const IntegrationPoint & ip,
			      DenseMatrix & h) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    virtual void Project (Coefficient & coeff,
			  ElementTransformation & Trans, Vector & dofs) const;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const;
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const;
    virtual void ProjectCurl (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & curl) const;
    virtual void ProjectDiv (const FiniteElement & fe,
			     ElementTransformation & Trans,
			     DenseMatrix & div) const;
      virtual ~ FiniteElement ()
    {
    }
  };
  class NodalFiniteElement:public FiniteElement
  {
  protected:
    void NodalLocalInterpolation (ElementTransformation & Trans,
				  DenseMatrix & I,
				  const NodalFiniteElement & fine_fe) const;
    mutable Vector c_shape;
  public:
    NodalFiniteElement (int D, int G, int Do, int O, int F = FunctionSpace::Pk):
    FiniteElement (D, G, Do, O, F),
      c_shape (Do)
    {
    }
    void SetMapType (int M)
    {
      if (!(M == VALUE || M == INTEGRAL))
	{
	  {
	    std::ostringstream s;
	    s << std::setprecision (16);
	    s << std::setiosflags (std::ios_base::scientific);
	    s << "Verification failed: (" << "M == VALUE || M == INTEGRAL" <<
	      ") is false: " << "unknown MapType" << '\n';
	    s << " ... at line " << 219;
	    s << " in " << __PRETTY_FUNCTION__ << " of file " <<
	      "../fem/../mesh/../fem/fe.hpp" << ".";
	    s << std::ends;
	    if (!(0))
	      mfem::mfem_error (s.str ().c_str ());
	    else
	      mfem::mfem_warning (s.str ().c_str ());
	  };
	};
      MapType = M;
    }
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      NodalLocalInterpolation (Trans, I, *this);
    }
    virtual void Project (Coefficient & coeff,
			  ElementTransformation & Trans, Vector & dofs) const;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const;
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const;
    virtual void ProjectDiv (const FiniteElement & fe,
			     ElementTransformation & Trans,
			     DenseMatrix & div) const;
  };
  class PositiveFiniteElement:public FiniteElement
  {
  public:
  PositiveFiniteElement (int D, int G, int Do, int O, int F = FunctionSpace::Pk):FiniteElement (D, G, Do, O,
		   F)
    {
    }
    using FiniteElement::Project;
    virtual void Project (Coefficient & coeff,
			  ElementTransformation & Trans, Vector & dofs) const;
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const;
  };
  class VectorFiniteElement:public FiniteElement
  {
  private:
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  protected:
      mutable DenseMatrix Jinv;
    mutable DenseMatrix vshape;
    void CalcVShape_RT (ElementTransformation & Trans,
			DenseMatrix & shape) const;
    void CalcVShape_ND (ElementTransformation & Trans,
			DenseMatrix & shape) const;
    void Project_RT (const double *nk, const Array < int >&d2n,
		     VectorCoefficient & vc, ElementTransformation & Trans,
		     Vector & dofs) const;
    void Project_RT (const double *nk, const Array < int >&d2n,
		     const FiniteElement & fe, ElementTransformation & Trans,
		     DenseMatrix & I) const;
    void ProjectGrad_RT (const double *nk, const Array < int >&d2n,
			 const FiniteElement & fe,
			 ElementTransformation & Trans,
			 DenseMatrix & grad) const;
    void ProjectCurl_RT (const double *nk, const Array < int >&d2n,
			 const FiniteElement & fe,
			 ElementTransformation & Trans,
			 DenseMatrix & curl) const;
    void Project_ND (const double *tk, const Array < int >&d2t,
		     VectorCoefficient & vc, ElementTransformation & Trans,
		     Vector & dofs) const;
    void Project_ND (const double *tk, const Array < int >&d2t,
		     const FiniteElement & fe, ElementTransformation & Trans,
		     DenseMatrix & I) const;
    void ProjectGrad_ND (const double *tk, const Array < int >&d2t,
			 const FiniteElement & fe,
			 ElementTransformation & Trans,
			 DenseMatrix & grad) const;
    void LocalInterpolation_RT (const double *nk, const Array < int >&d2n,
				ElementTransformation & Trans,
				DenseMatrix & I) const;
    void LocalInterpolation_ND (const double *tk, const Array < int >&d2t,
				ElementTransformation & Trans,
				DenseMatrix & I) const;
  public:
    VectorFiniteElement (int D, int G, int Do, int O, int M, int F = FunctionSpace::Pk):
    FiniteElement (D, G, Do, O, F), Jinv (D), vshape (Do,
						      D)
    {
      RangeType = VECTOR;
      MapType = M;
    }
  };
  class PointFiniteElement:public NodalFiniteElement
  {
  public:
    PointFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class Linear1DFiniteElement:public NodalFiniteElement
  {
  public:
    Linear1DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class Linear2DFiniteElement:public NodalFiniteElement
  {
  public:
    Linear2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs = 0.0;
      dofs (vertex) = 1.0;
    }
  };
  class BiLinear2DFiniteElement:public NodalFiniteElement
  {
  public:
    BiLinear2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void CalcHessian (const IntegrationPoint & ip,
			      DenseMatrix & h) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs = 0.0;
      dofs (vertex) = 1.0;
    }
  };
  class GaussLinear2DFiniteElement:public NodalFiniteElement
  {
  public:
    GaussLinear2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class GaussBiLinear2DFiniteElement:public NodalFiniteElement
  {
  private:
    static const double p[2];
  public:
      GaussBiLinear2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class P1OnQuadFiniteElement:public NodalFiniteElement
  {
  public:
    P1OnQuadFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs = 1.0;
    }
  };
  class Quad1DFiniteElement:public NodalFiniteElement
  {
  public:
    Quad1DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class QuadPos1DFiniteElement:public FiniteElement
  {
  public:
    QuadPos1DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class Quad2DFiniteElement:public NodalFiniteElement
  {
  public:
    Quad2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void CalcHessian (const IntegrationPoint & ip,
			      DenseMatrix & h) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class GaussQuad2DFiniteElement:public NodalFiniteElement
  {
  private:
    static const double p[2];
    DenseMatrix A;
    mutable DenseMatrix D;
    mutable Vector pol;
  public:
      GaussQuad2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class BiQuad2DFiniteElement:public NodalFiniteElement
  {
  public:
    BiQuad2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class BiQuadPos2DFiniteElement:public FiniteElement
  {
  public:
    BiQuadPos2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (Coefficient & coeff, ElementTransformation & Trans,
			  Vector & dofs) const;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs = 0.;
      dofs (vertex) = 1.;
    }
  };
  class GaussBiQuad2DFiniteElement:public NodalFiniteElement
  {
  public:
    GaussBiQuad2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class BiCubic2DFiniteElement:public NodalFiniteElement
  {
  public:
    BiCubic2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void CalcHessian (const IntegrationPoint & ip,
			      DenseMatrix & h) const;
  };
  class Cubic1DFiniteElement:public NodalFiniteElement
  {
  public:
    Cubic1DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class Cubic2DFiniteElement:public NodalFiniteElement
  {
  public:
    Cubic2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void CalcHessian (const IntegrationPoint & ip,
			      DenseMatrix & h) const;
  };
  class Cubic3DFiniteElement:public NodalFiniteElement
  {
  public:
    Cubic3DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class P0TriangleFiniteElement:public NodalFiniteElement
  {
  public:
    P0TriangleFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs (0) = 1.0;
    }
  };
  class P0QuadFiniteElement:public NodalFiniteElement
  {
  public:
    P0QuadFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs (0) = 1.0;
    }
  };
  class Linear3DFiniteElement:public NodalFiniteElement
  {
  public:
    Linear3DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs = 0.0;
      dofs (vertex) = 1.0;
    }
    virtual void GetFaceDofs (int face, int **dofs, int *ndofs) const;
  };
  class Quadratic3DFiniteElement:public NodalFiniteElement
  {
  public:
    Quadratic3DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class TriLinear3DFiniteElement:public NodalFiniteElement
  {
  public:
    TriLinear3DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs = 0.0;
      dofs (vertex) = 1.0;
    }
  };
  class CrouzeixRaviartFiniteElement:public NodalFiniteElement
  {
  public:
    CrouzeixRaviartFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs = 1.0;
    }
  };
  class CrouzeixRaviartQuadFiniteElement:public NodalFiniteElement
  {
  public:
    CrouzeixRaviartQuadFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class P0SegmentFiniteElement:public NodalFiniteElement
  {
  public:
    P0SegmentFiniteElement (int Ord = 0);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class RT0TriangleFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[3][2];
  public:
      RT0TriangleFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RT0QuadFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[4][2];
  public:
      RT0QuadFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RT1TriangleFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[8][2];
  public:
      RT1TriangleFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RT1QuadFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[12][2];
  public:
      RT1QuadFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RT2TriangleFiniteElement:public VectorFiniteElement
  {
  private:
    static const double M[15][15];
  public:
      RT2TriangleFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
  };
  class RT2QuadFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[24][2];
    static const double pt[4];
    static const double dpt[3];
  public:
      RT2QuadFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class P1SegmentFiniteElement:public NodalFiniteElement
  {
  public:
    P1SegmentFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class P2SegmentFiniteElement:public NodalFiniteElement
  {
  public:
    P2SegmentFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class Lagrange1DFiniteElement:public NodalFiniteElement
  {
  private:
    Vector rwk;
    mutable Vector rxxk;
  public:
      Lagrange1DFiniteElement (int degree);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class P1TetNonConfFiniteElement:public NodalFiniteElement
  {
  public:
    P1TetNonConfFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class P0TetFiniteElement:public NodalFiniteElement
  {
  public:
    P0TetFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs (0) = 1.0;
    }
  };
  class P0HexFiniteElement:public NodalFiniteElement
  {
  public:
    P0HexFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const
    {
      dofs (0) = 1.0;
    }
  };
  class LagrangeHexFiniteElement:public NodalFiniteElement
  {
  private:
    Lagrange1DFiniteElement * fe1d;
    int dof1d;
    int *I, *J, *K;
    mutable Vector shape1dx, shape1dy, shape1dz;
    mutable DenseMatrix dshape1dx, dshape1dy, dshape1dz;
  public:
      LagrangeHexFiniteElement (int degree);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
     ~LagrangeHexFiniteElement ();
  };
  class RefinedLinear1DFiniteElement:public NodalFiniteElement
  {
  public:
    RefinedLinear1DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class RefinedLinear2DFiniteElement:public NodalFiniteElement
  {
  public:
    RefinedLinear2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class RefinedLinear3DFiniteElement:public NodalFiniteElement
  {
  public:
    RefinedLinear3DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class RefinedBiLinear2DFiniteElement:public NodalFiniteElement
  {
  public:
    RefinedBiLinear2DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class RefinedTriLinear3DFiniteElement:public NodalFiniteElement
  {
  public:
    RefinedTriLinear3DFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class Nedelec1HexFiniteElement:public VectorFiniteElement
  {
  private:
    static const double tk[12][3];
  public:
      Nedelec1HexFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_ND (Trans, shape);
    };
    virtual void CalcCurlShape (const IntegrationPoint & ip,
				DenseMatrix & curl_shape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class Nedelec1TetFiniteElement:public VectorFiniteElement
  {
  private:
    static const double tk[6][3];
  public:
      Nedelec1TetFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_ND (Trans, shape);
    };
    virtual void CalcCurlShape (const IntegrationPoint & ip,
				DenseMatrix & curl_shape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RT0HexFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[6][3];
  public:
      RT0HexFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RT1HexFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[36][3];
  public:
      RT1HexFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RT0TetFiniteElement:public VectorFiniteElement
  {
  private:
    static const double nk[4][3];
  public:
      RT0TetFiniteElement ();
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    };
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const;
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const;
  };
  class RotTriLinearHexFiniteElement:public NodalFiniteElement
  {
  public:
    RotTriLinearHexFiniteElement ();
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class Poly_1D
  {
  public:
    class Basis
    {
    private:
      int mode;
      DenseMatrix A;
      mutable Vector x, w;
    public:
        Basis (const int p, const double *nodes, const int _mode = 1);
      void Eval (const double x, Vector & u) const;
      void Eval (const double x, Vector & u, Vector & d) const;
    };
  private:
    Array < double *>open_pts, closed_pts;
    Array < Basis * >open_basis, closed_basis;
    static Array2D < int >binom;
    static void CalcMono (const int p, const double x, double *u);
    static void CalcMono (const int p, const double x, double *u, double *d);
    static void CalcLegendre (const int p, const double x, double *u);
    static void CalcLegendre (const int p, const double x, double *u,
			      double *d);
    static void CalcChebyshev (const int p, const double x, double *u);
    static void CalcChebyshev (const int p, const double x, double *u,
			       double *d);
  public:
    Poly_1D ()
    {
    }
    static const int *Binom (const int p);
    const double *OpenPoints (const int p);
    const double *ClosedPoints (const int p);
    Basis & OpenBasis (const int p);
    Basis & ClosedBasis (const int p);
    static void CalcBasis (const int p, const double x, double *u)
    {
      CalcChebyshev (p, x, u);
    }
    static void CalcBasis (const int p, const double x, double *u, double *d)
    {
      CalcChebyshev (p, x, u, d);
    }
    static double CalcDelta (const int p, const double x)
    {
      return pow (x, (double) p);
    }
    static void UniformPoints (const int p, double *x);
    static void GaussPoints (const int p, double *x);
    static void GaussLobattoPoints (const int p, double *x);
    static void ChebyshevPoints (const int p, double *x);
    static void CalcBinomTerms (const int p, const double x, const double y,
				double *u);
    static void CalcBinomTerms (const int p, const double x, const double y,
				double *u, double *d);
    static void CalcDBinomTerms (const int p, const double x, const double y,
				 double *d);
    static void CalcBernstein (const int p, const double x, double *u)
    {
      CalcBinomTerms (p, x, 1. - x, u);
    }
    static void CalcBernstein (const int p, const double x, double *u,
			       double *d)
    {
      CalcBinomTerms (p, x, 1. - x, u, d);
    }
    ~Poly_1D ();
  };
  extern Poly_1D poly1d;
  class H1_SegmentElement:public NodalFiniteElement
  {
  private:
    Poly_1D::Basis & basis1d;
    mutable Vector shape_x, dshape_x;
  public:
      H1_SegmentElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class H1_QuadrilateralElement:public NodalFiniteElement
  {
  private:
    Poly_1D::Basis & basis1d;
    mutable Vector shape_x, shape_y, dshape_x, dshape_y;
      Array < int >dof_map;
  public:
      H1_QuadrilateralElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
    const Array < int >&GetDofMap () const
    {
      return dof_map;
    }
  };
  class H1_HexahedronElement:public NodalFiniteElement
  {
  private:
    Poly_1D::Basis & basis1d;
    mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
      Array < int >dof_map;
  public:
      H1_HexahedronElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
    const Array < int >&GetDofMap () const
    {
      return dof_map;
    }
  };
  class H1Pos_SegmentElement:public PositiveFiniteElement
  {
  private:
    mutable Vector shape_x, dshape_x;
  public:
    H1Pos_SegmentElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class H1Pos_QuadrilateralElement:public PositiveFiniteElement
  {
  private:
    mutable Vector shape_x, shape_y, dshape_x, dshape_y;
    Array < int >dof_map;
  public:
      H1Pos_QuadrilateralElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class H1Pos_HexahedronElement:public PositiveFiniteElement
  {
  private:
    mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
    Array < int >dof_map;
  public:
      H1Pos_HexahedronElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class H1_TriangleElement:public NodalFiniteElement
  {
  private:
    mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
    mutable DenseMatrix du;
    DenseMatrix T;
  public:
      H1_TriangleElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class H1_TetrahedronElement:public NodalFiniteElement
  {
  private:
    mutable Vector shape_x, shape_y, shape_z, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
    mutable DenseMatrix du;
    DenseMatrix T;
  public:
      H1_TetrahedronElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class L2_SegmentElement:public NodalFiniteElement
  {
  private:
    int type;
      Poly_1D::Basis * basis1d;
    mutable Vector shape_x, dshape_x;
  public:
      L2_SegmentElement (const int p, const int _type = 0);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2Pos_SegmentElement:public PositiveFiniteElement
  {
  private:
    mutable Vector shape_x, dshape_x;
  public:
    L2Pos_SegmentElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2_QuadrilateralElement:public NodalFiniteElement
  {
  private:
    int type;
      Poly_1D::Basis * basis1d;
    mutable Vector shape_x, shape_y, dshape_x, dshape_y;
  public:
      L2_QuadrilateralElement (const int p, const int _type = 0);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2Pos_QuadrilateralElement:public PositiveFiniteElement
  {
  private:
    mutable Vector shape_x, shape_y, dshape_x, dshape_y;
  public:
    L2Pos_QuadrilateralElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2_HexahedronElement:public NodalFiniteElement
  {
  private:
    int type;
      Poly_1D::Basis * basis1d;
    mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
  public:
      L2_HexahedronElement (const int p, const int _type = 0);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2Pos_HexahedronElement:public PositiveFiniteElement
  {
  private:
    mutable Vector shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
  public:
    L2Pos_HexahedronElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2_TriangleElement:public NodalFiniteElement
  {
  private:
    int type;
    mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
    mutable DenseMatrix du;
    DenseMatrix T;
  public:
      L2_TriangleElement (const int p, const int _type = 0);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2Pos_TriangleElement:public PositiveFiniteElement
  {
  private:
    mutable Vector dshape_1d;
  public:
    L2Pos_TriangleElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2_TetrahedronElement:public NodalFiniteElement
  {
  private:
    int type;
    mutable Vector shape_x, shape_y, shape_z, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_z, dshape_l, u;
    mutable DenseMatrix du;
    DenseMatrix T;
  public:
      L2_TetrahedronElement (const int p, const int _type = 0);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class L2Pos_TetrahedronElement:public PositiveFiniteElement
  {
  private:
    mutable Vector dshape_1d;
  public:
    L2Pos_TetrahedronElement (const int p);
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
    virtual void ProjectDelta (int vertex, Vector & dofs) const;
  };
  class RT_QuadrilateralElement:public VectorFiniteElement
  {
  private:
    static const double nk[8];
      Poly_1D::Basis & cbasis1d, &obasis1d;
    mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
    mutable Vector dshape_cx, dshape_cy;
      Array < int >dof_map, dof2nk;
  public:
      RT_QuadrilateralElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    }
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_RT (nk, dof2nk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_RT (nk, dof2nk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_RT (nk, dof2nk, fe, Trans, I);
    }
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const
    {
      ProjectGrad_RT (nk, dof2nk, fe, Trans, grad);
    }
  };
  class RT_HexahedronElement:public VectorFiniteElement
  {
    static const double nk[18];
      Poly_1D::Basis & cbasis1d, &obasis1d;
    mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
    mutable Vector dshape_cx, dshape_cy, dshape_cz;
      Array < int >dof_map, dof2nk;
  public:
      RT_HexahedronElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    }
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_RT (nk, dof2nk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_RT (nk, dof2nk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_RT (nk, dof2nk, fe, Trans, I);
    }
    virtual void ProjectCurl (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & curl) const
    {
      ProjectCurl_RT (nk, dof2nk, fe, Trans, curl);
    }
  };
  class RT_TriangleElement:public VectorFiniteElement
  {
    static const double nk[6], c;
    mutable Vector shape_x, shape_y, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_l;
    mutable DenseMatrix u;
    mutable Vector divu;
      Array < int >dof2nk;
    DenseMatrix T;
  public:
      RT_TriangleElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    }
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_RT (nk, dof2nk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_RT (nk, dof2nk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_RT (nk, dof2nk, fe, Trans, I);
    }
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const
    {
      ProjectGrad_RT (nk, dof2nk, fe, Trans, grad);
    }
  };
  class RT_TetrahedronElement:public VectorFiniteElement
  {
    static const double nk[12], c;
    mutable Vector shape_x, shape_y, shape_z, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
    mutable DenseMatrix u;
    mutable Vector divu;
      Array < int >dof2nk;
    DenseMatrix T;
  public:
      RT_TetrahedronElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_RT (Trans, shape);
    }
    virtual void CalcDivShape (const IntegrationPoint & ip,
			       Vector & divshape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_RT (nk, dof2nk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_RT (nk, dof2nk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_RT (nk, dof2nk, fe, Trans, I);
    }
    virtual void ProjectCurl (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & curl) const
    {
      ProjectCurl_RT (nk, dof2nk, fe, Trans, curl);
    }
  };
  class ND_HexahedronElement:public VectorFiniteElement
  {
    static const double tk[18];
      Poly_1D::Basis & cbasis1d, &obasis1d;
    mutable Vector shape_cx, shape_ox, shape_cy, shape_oy, shape_cz, shape_oz;
    mutable Vector dshape_cx, dshape_cy, dshape_cz;
      Array < int >dof_map, dof2tk;
  public:
      ND_HexahedronElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_ND (Trans, shape);
    }
    virtual void CalcCurlShape (const IntegrationPoint & ip,
				DenseMatrix & curl_shape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_ND (tk, dof2tk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_ND (tk, dof2tk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_ND (tk, dof2tk, fe, Trans, I);
    }
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const
    {
      ProjectGrad_ND (tk, dof2tk, fe, Trans, grad);
    }
  };
  class ND_QuadrilateralElement:public VectorFiniteElement
  {
    static const double tk[8];
      Poly_1D::Basis & cbasis1d, &obasis1d;
    mutable Vector shape_cx, shape_ox, shape_cy, shape_oy;
    mutable Vector dshape_cx, dshape_cy;
      Array < int >dof_map, dof2tk;
  public:
      ND_QuadrilateralElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_ND (Trans, shape);
    }
    virtual void CalcCurlShape (const IntegrationPoint & ip,
				DenseMatrix & curl_shape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_ND (tk, dof2tk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_ND (tk, dof2tk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_ND (tk, dof2tk, fe, Trans, I);
    }
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const
    {
      ProjectGrad_ND (tk, dof2tk, fe, Trans, grad);
    }
  };
  class ND_TetrahedronElement:public VectorFiniteElement
  {
    static const double tk[18], c;
    mutable Vector shape_x, shape_y, shape_z, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_z, dshape_l;
    mutable DenseMatrix u;
      Array < int >dof2tk;
    DenseMatrix T;
  public:
      ND_TetrahedronElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_ND (Trans, shape);
    }
    virtual void CalcCurlShape (const IntegrationPoint & ip,
				DenseMatrix & curl_shape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_ND (tk, dof2tk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_ND (tk, dof2tk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_ND (tk, dof2tk, fe, Trans, I);
    }
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const
    {
      ProjectGrad_ND (tk, dof2tk, fe, Trans, grad);
    }
  };
  class ND_TriangleElement:public VectorFiniteElement
  {
    static const double tk[8], c;
    mutable Vector shape_x, shape_y, shape_l;
    mutable Vector dshape_x, dshape_y, dshape_l;
    mutable DenseMatrix u;
      Array < int >dof2tk;
    DenseMatrix T;
  public:
      ND_TriangleElement (const int p);
    virtual void CalcVShape (const IntegrationPoint & ip,
			     DenseMatrix & shape) const;
    virtual void CalcVShape (ElementTransformation & Trans,
			     DenseMatrix & shape) const
    {
      CalcVShape_ND (Trans, shape);
    }
    virtual void CalcCurlShape (const IntegrationPoint & ip,
				DenseMatrix & curl_shape) const;
    virtual void GetLocalInterpolation (ElementTransformation & Trans,
					DenseMatrix & I) const
    {
      LocalInterpolation_ND (tk, dof2tk, Trans, I);
    }
    using FiniteElement::Project;
    virtual void Project (VectorCoefficient & vc,
			  ElementTransformation & Trans, Vector & dofs) const
    {
      Project_ND (tk, dof2tk, vc, Trans, dofs);
    }
    virtual void Project (const FiniteElement & fe,
			  ElementTransformation & Trans,
			  DenseMatrix & I) const
    {
      Project_ND (tk, dof2tk, fe, Trans, I);
    }
    virtual void ProjectGrad (const FiniteElement & fe,
			      ElementTransformation & Trans,
			      DenseMatrix & grad) const
    {
      ProjectGrad_ND (tk, dof2tk, fe, Trans, grad);
    }
  };
  class NURBSFiniteElement:public FiniteElement
  {
  protected:
    mutable Array < KnotVector * >kv;
    mutable int *ijk, patch, elem;
    mutable Vector weights;
  public:
      NURBSFiniteElement (int D, int G, int Do, int O,
			  int F):FiniteElement (D, G, Do, O, F)
    {
      ijk = __null;
      patch = elem = -1;
      kv.SetSize (Dim);
      weights.SetSize (Dof);
      weights = 1.0;
    }
    void Reset () const
    {
      patch = elem = -1;
    }
    void SetIJK (int *IJK) const
    {
      ijk = IJK;
    }
    int GetPatch () const
    {
      return patch;
    }
    void SetPatch (int p) const
    {
      patch = p;
    }
    int GetElement () const
    {
      return elem;
    }
    void SetElement (int e) const
    {
      elem = e;
    }
    Array < KnotVector * >&KnotVectors () const
    {
      return kv;
    }
    Vector & Weights () const
    {
      return weights;
    }
  };
  class NURBS1DFiniteElement:public NURBSFiniteElement
  {
  protected:
    mutable Vector shape_x;
  public:
    NURBS1DFiniteElement (int p):NURBSFiniteElement (1, Geometry::SEGMENT,
						     p + 1, p,
						     FunctionSpace::Qk),
      shape_x (p + 1)
    {
    }
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class NURBS2DFiniteElement:public NURBSFiniteElement
  {
  protected:
    mutable Vector u, shape_x, shape_y, dshape_x, dshape_y;
  public:
    NURBS2DFiniteElement (int p):NURBSFiniteElement (2, Geometry::SQUARE,
						     (p + 1) * (p + 1), p,
						     FunctionSpace::Qk),
      u (Dof), shape_x (p + 1), shape_y (p + 1), dshape_x (p + 1),
      dshape_y (p + 1)
    {
    }
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
  class NURBS3DFiniteElement:public NURBSFiniteElement
  {
  protected:
    mutable Vector u, shape_x, shape_y, shape_z, dshape_x, dshape_y, dshape_z;
  public:
    NURBS3DFiniteElement (int p):NURBSFiniteElement (3, Geometry::CUBE,
						     (p + 1) * (p + 1) * (p +
									  1),
						     p, FunctionSpace::Qk),
      u (Dof), shape_x (p + 1), shape_y (p + 1), shape_z (p + 1),
      dshape_x (p + 1), dshape_y (p + 1), dshape_z (p + 1)
    {
    }
    virtual void CalcShape (const IntegrationPoint & ip,
			    Vector & shape) const;
    virtual void CalcDShape (const IntegrationPoint & ip,
			     DenseMatrix & dshape) const;
  };
}

namespace mfem
{
  class ElementTransformation
  {
  protected:
    int JacobianIsEvaluated;
    int WeightIsEvaluated;
    const IntegrationPoint *IntPoint;
  public:
    int Attribute, ElementNo;
      ElementTransformation ();
    void SetIntPoint (const IntegrationPoint * ip)
    {
      IntPoint = ip;
      WeightIsEvaluated = JacobianIsEvaluated = 0;
    }
    const IntegrationPoint & GetIntPoint ()
    {
      return *IntPoint;
    }
    virtual void Transform (const IntegrationPoint &, Vector &) = 0;
    virtual void Transform (const IntegrationRule &, DenseMatrix &) = 0;
    virtual const DenseMatrix & Jacobian () = 0;
    virtual double Weight () = 0;
    virtual int Order () = 0;
    virtual int OrderJ () = 0;
    virtual int OrderW () = 0;
    virtual int OrderGrad (const FiniteElement * fe) = 0;
    virtual int GetSpaceDim () = 0;
    virtual ~ ElementTransformation ()
    {
    }
  };
  class IsoparametricTransformation:public ElementTransformation
  {
  private:
    DenseMatrix dshape, dFdx;
    double Wght;
    Vector shape;
    const FiniteElement *FElem;
    DenseMatrix PointMat;
  public:
    void SetFE (const FiniteElement * FE)
    {
      FElem = FE;
    };
    DenseMatrix & GetPointMat ()
    {
      return PointMat;
    };
    void SetIdentityTransformation (int GeomType);
    virtual void Transform (const IntegrationPoint &, Vector &);
    virtual void Transform (const IntegrationRule &, DenseMatrix &);
    virtual const DenseMatrix & Jacobian ();
    virtual double Weight ();
    virtual int Order ()
    {
      return FElem->GetOrder ();
    }
    virtual int OrderJ ();
    virtual int OrderW ();
    virtual int OrderGrad (const FiniteElement * fe);
    virtual int GetSpaceDim ()
    {
      return PointMat.Height ();
    }
    virtual ~ IsoparametricTransformation ()
    {
    }
  };
  class IntegrationPointTransformation
  {
  public:
    IsoparametricTransformation Transf;
    void Transform (const IntegrationPoint &, IntegrationPoint &);
    void Transform (const IntegrationRule &, IntegrationRule &);
  };
  class FaceElementTransformations
  {
  public:
    int Elem1No, Elem2No, FaceGeom;
    ElementTransformation *Elem1, *Elem2, *Face;
    IntegrationPointTransformation Loc1, Loc2;
  };
}

namespace mfem
{
  class Mesh;
  class Coefficient
  {
  protected:
    double time;
  public:
      Coefficient ()
    {
      time = 0.;
    }
    void SetTime (double t)
    {
      time = t;
    }
    double GetTime ()
    {
      return time;
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip) = 0;
    double Eval (ElementTransformation & T,
		 const IntegrationPoint & ip, double t)
    {
      SetTime (t);
      return Eval (T, ip);
    }
    virtual ~ Coefficient ()
    {
    }
  };
  class ConstantCoefficient:public Coefficient
  {
  public:
    double constant;
    explicit ConstantCoefficient (double c = 1.0)
    {
      constant = c;
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip)
    {
      return (constant);
    }
  };
  class PWConstCoefficient:public Coefficient
  {
  private:
    Vector constants;
  public:
  explicit PWConstCoefficient (int NumOfSubD = 0):constants (NumOfSubD)
    {
      constants = 0.0;
    }
    PWConstCoefficient (Vector & c)
    {
      constants.SetSize (c.Size ());
      constants = c;
    }
    double &operator () (int i)
    {
      return constants (i - 1);
    }
    void operator= (double c)
    {
      constants = c;
    }
    int GetNConst ()
    {
      return constants.Size ();
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip);
  };
  class FunctionCoefficient:public Coefficient
  {
  protected:
    double (*Function) (Vector &);
    double (*TDFunction) (Vector &, double);
  public:
      FunctionCoefficient (double (*f) (Vector &))
    {
      Function = f;
      TDFunction = __null;
    }
    FunctionCoefficient (double (*tdf) (Vector &, double))
    {
      Function = __null;
      TDFunction = tdf;
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip);
  };
  class GridFunction;
  class GridFunctionCoefficient:public Coefficient
  {
  private:
    GridFunction * GridF;
    int Component;
  public:
      GridFunctionCoefficient (GridFunction * gf, int comp = 1)
    {
      GridF = gf;
      Component = comp;
    }
    void SetGridFunction (GridFunction * gf)
    {
      GridF = gf;
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip);
  };
  class TransformedCoefficient:public Coefficient
  {
  private:
    Coefficient * Q1;
    Coefficient *Q2;
    double (*Transform1) (double);
    double (*Transform2) (double, double);
  public:
      TransformedCoefficient (Coefficient * q, double (*F) (double)):Q1 (q),
      Transform1 (F)
    {
      Q2 = 0;
      Transform2 = 0;
    }
    TransformedCoefficient (Coefficient * q1, Coefficient * q2,
			    double (*F) (double, double)):Q1 (q1), Q2 (q2),
      Transform2 (F)
    {
      Transform1 = 0;
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip);
  };
  class DeltaCoefficient:public Coefficient
  {
  private:
    double center[3], scale, tol;
    Coefficient *weight;
  public:
      DeltaCoefficient ();
      DeltaCoefficient (double x, double y, double s)
    {
      center[0] = x;
      center[1] = y;
      center[2] = 0.;
      scale = s;
      tol = 1e-12;
      weight = __null;
    }
    DeltaCoefficient (double x, double y, double z, double s)
    {
      center[0] = x;
      center[1] = y;
      center[2] = z;
      scale = s;
      tol = 1e-12;
      weight = __null;
    }
    void SetTol (double _tol)
    {
      tol = _tol;
    }
    void SetWeight (Coefficient * w)
    {
      weight = w;
    }
    const double *Center ()
    {
      return center;
    }
    double Scale ()
    {
      return scale;
    }
    double Tol ()
    {
      return tol;
    }
    Coefficient *Weight ()
    {
      return weight;
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip)
    {
      mfem_error ("DeltaCoefficient::Eval");
      return 0.;
    }
    virtual ~ DeltaCoefficient ()
    {
      delete weight;
    }
  };
  class RestrictedCoefficient:public Coefficient
  {
  private:
    Coefficient * c;
    Array < int >active_attr;
  public:
      RestrictedCoefficient (Coefficient & _c, Array < int >&attr)
    {
      c = &_c;
      attr.Copy (active_attr);
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip)
    {
      return active_attr[T.Attribute - 1] ? c->Eval (T, ip, GetTime ()) : 0.0;
    }
  };
  class VectorCoefficient
  {
  protected:
    int vdim;
    double time;
  public:
      VectorCoefficient (int vd)
    {
      vdim = vd;
      time = 0.;
    }
    void SetTime (double t)
    {
      time = t;
    }
    double GetTime ()
    {
      return time;
    }
    int GetVDim ()
    {
      return vdim;
    }
    virtual void Eval (Vector & V, ElementTransformation & T,
		       const IntegrationPoint & ip) = 0;
    virtual void Eval (DenseMatrix & M, ElementTransformation & T,
		       const IntegrationRule & ir);
    virtual ~ VectorCoefficient ()
    {
    }
  };
  class VectorConstantCoefficient:public VectorCoefficient
  {
  private:
    Vector vec;
  public:
    VectorConstantCoefficient (const Vector & v):VectorCoefficient (v.
								    Size ()),
      vec (v)
    {
    }
    using VectorCoefficient::Eval;
    virtual void Eval (Vector & V, ElementTransformation & T,
		       const IntegrationPoint & ip)
    {
      V = vec;
    }
  };
  class VectorFunctionCoefficient:public VectorCoefficient
  {
  private:
    void (*Function) (const Vector &, Vector &);
    void (*TDFunction) (const Vector &, double, Vector &);
    Coefficient *Q;
  public:
    VectorFunctionCoefficient (int dim, void (*F) (const Vector &, Vector &), Coefficient * q = __null):VectorCoefficient (dim),
      Q
      (q)
    {
      Function = F;
      TDFunction = __null;
    }
  VectorFunctionCoefficient (int dim, void (*TDF) (const Vector &, double, Vector &), Coefficient * q = __null):VectorCoefficient (dim),
      Q
      (q)
    {
      Function = __null;
      TDFunction = TDF;
    }
    using VectorCoefficient::Eval;
    virtual void Eval (Vector & V, ElementTransformation & T,
		       const IntegrationPoint & ip);
    virtual ~ VectorFunctionCoefficient ()
    {
    }
  };
  class VectorArrayCoefficient:public VectorCoefficient
  {
  private:
    Array < Coefficient * >Coeff;
  public:
    explicit VectorArrayCoefficient (int dim);
      Coefficient & GetCoeff (int i)
    {
      return *Coeff[i];
    }
    Coefficient **GetCoeffs ()
    {
      return Coeff;
    }
    void Set (int i, Coefficient * c)
    {
      Coeff[i] = c;
    }
    double Eval (int i, ElementTransformation & T, IntegrationPoint & ip)
    {
      return Coeff[i]->Eval (T, ip, GetTime ());
    }
    using VectorCoefficient::Eval;
    virtual void Eval (Vector & V, ElementTransformation & T,
		       const IntegrationPoint & ip);
    virtual ~ VectorArrayCoefficient ();
  };
  class VectorGridFunctionCoefficient:public VectorCoefficient
  {
  private:
    GridFunction * GridFunc;
  public:
    VectorGridFunctionCoefficient (GridFunction * gf);
    virtual void Eval (Vector & V, ElementTransformation & T,
		       const IntegrationPoint & ip);
    virtual void Eval (DenseMatrix & M, ElementTransformation & T,
		       const IntegrationRule & ir);
      virtual ~ VectorGridFunctionCoefficient ()
    {
    }
  };
  class VectorRestrictedCoefficient:public VectorCoefficient
  {
  private:
    VectorCoefficient * c;
    Array < int >active_attr;
  public:
      VectorRestrictedCoefficient (VectorCoefficient & vc,
				   Array <
				   int >&attr):VectorCoefficient (vc.
								  GetVDim ())
    {
      c = &vc;
      attr.Copy (active_attr);
    }
    virtual void Eval (Vector & V, ElementTransformation & T,
		       const IntegrationPoint & ip);
    virtual void Eval (DenseMatrix & M, ElementTransformation & T,
		       const IntegrationRule & ir);
  };
  class MatrixCoefficient
  {
  protected:
    int vdim;
    double time;
  public:
      explicit MatrixCoefficient (int dim)
    {
      vdim = dim;
      time = 0.;
    }
    void SetTime (double t)
    {
      time = t;
    }
    double GetTime ()
    {
      return time;
    }
    int GetVDim ()
    {
      return vdim;
    }
    virtual void Eval (DenseMatrix & K, ElementTransformation & T,
		       const IntegrationPoint & ip) = 0;
    virtual ~ MatrixCoefficient ()
    {
    }
  };
  class MatrixFunctionCoefficient:public MatrixCoefficient
  {
  private:
    void (*Function) (const Vector &, DenseMatrix &);
    void (*TDFunction) (const Vector &, double, DenseMatrix &);
  public:
      MatrixFunctionCoefficient (int dim,
				 void (*F) (const Vector &,
					    DenseMatrix
					    &)):MatrixCoefficient (dim)
    {
      Function = F;
      TDFunction = __null;
    }
    MatrixFunctionCoefficient (int dim,
			       void (*TDF) (const Vector &, double,
					    DenseMatrix
					    &)):MatrixCoefficient (dim)
    {
      Function = __null;
      TDFunction = TDF;
    }
    virtual void Eval (DenseMatrix & K, ElementTransformation & T,
		       const IntegrationPoint & ip);
    virtual ~ MatrixFunctionCoefficient ()
    {
    }
  };
  class MatrixArrayCoefficient:public MatrixCoefficient
  {
  private:
    Array < Coefficient * >Coeff;
  public:
    explicit MatrixArrayCoefficient (int dim);
      Coefficient & GetCoeff (int i, int j)
    {
      return *Coeff[i * vdim + j];
    }
    void Set (int i, int j, Coefficient * c)
    {
      Coeff[i * vdim + j] = c;
    }
    double Eval (int i, int j, ElementTransformation & T,
		 IntegrationPoint & ip)
    {
      return Coeff[i * vdim + j]->Eval (T, ip, GetTime ());
    }
    virtual void Eval (DenseMatrix & K, ElementTransformation & T,
		       const IntegrationPoint & ip);
    virtual ~ MatrixArrayCoefficient ();
  };
  double ComputeLpNorm (double p, Coefficient & coeff, Mesh & mesh,
			const IntegrationRule * irs[]);
  double ComputeLpNorm (double p, VectorCoefficient & coeff, Mesh & mesh,
			const IntegrationRule * irs[]);
}

namespace mfem
{
  class Vertex
  {
  protected:
    double coord[3];
  public:
      Vertex ()
    {
    }
    Vertex (double *xx, int dim);
      Vertex (double x, double y)
    {
      coord[0] = x;
      coord[1] = y;
      coord[2] = 0.;
    }
    Vertex (double x, double y, double z)
    {
      coord[0] = x;
      coord[1] = y;
      coord[2] = z;
    }
    inline double *operator () () const
    {
      return (double *) coord;
    }
    inline double &operator () (int i)
    {
      return coord[i];
    }
    inline const double &operator () (int i) const
    {
      return coord[i];
    }
    void SetCoords (const double *p)
    {
      coord[0] = p[0];
      coord[1] = p[1];
      coord[2] = p[2];
    }
    ~Vertex ()
    {
    }
  };
}

namespace mfem
{
  class Mesh;
  class Element
  {
  protected:
    int attribute, base_geom;
  public:
    enum Type
    { POINT, SEGMENT, TRIANGLE, QUADRILATERAL, TETRAHEDRON,
      HEXAHEDRON, BISECTED, QUADRISECTED, OCTASECTED
    };
    explicit Element (int bg = Geometry::POINT)
    {
      attribute = -1;
      base_geom = bg;
    }
    virtual void SetVertices (const int *ind);
    virtual int GetType () const = 0;
    int GetGeometryType () const
    {
      return base_geom;
    }
    virtual void GetVertices (Array < int >&v) const = 0;
    virtual int *GetVertices () = 0;
    const int *GetVertices () const
    {
      return const_cast < Element * >(this)->GetVertices ();
    }
    virtual int GetNVertices () const = 0;
    virtual int GetNEdges () const = 0;
    virtual const int *GetEdgeVertices (int) const = 0;
    virtual int GetNFaces (int &nFaceVertices) const = 0;
    virtual const int *GetFaceVertices (int fi) const = 0;
    virtual void MarkEdge (DenseMatrix & pmat)
    {
    }
    virtual void MarkEdge (const DSTable & v_to_v, const int *length)
    {
    }
    virtual int NeedRefinement (DSTable & v_to_v, int *middle) const
    {
      return 0;
    }
    inline int GetAttribute () const
    {
      return attribute;
    }
    inline void SetAttribute (const int attr)
    {
      attribute = attr;
    }
    virtual int GetRefinementFlag ()
    {
      return 0;
    }
    virtual Element *Duplicate (Mesh * m) const = 0;
    virtual ~ Element ()
    {
    }
  };
  class RefinedElement:public Element
  {
  public:
    enum
    { COARSE = 0, FINE = 1 };
    static int State;
    Element *CoarseElem, *FirstChild;
      RefinedElement ()
    {
    }
    RefinedElement (Element * ce):Element (ce->GetGeometryType ())
    {
      attribute = ce->GetAttribute ();
      CoarseElem = ce;
    }
    void SetCoarseElem (Element * ce)
    {
      base_geom = ce->GetGeometryType ();
      attribute = ce->GetAttribute ();
      CoarseElem = ce;
    }
    Element *IAm ()
    {
      if (State == RefinedElement::COARSE)
	return CoarseElem;
      return FirstChild;
    }
    const Element *IAm () const
    {
      if (State == RefinedElement::COARSE)
	return CoarseElem;
      return FirstChild;
    }
    virtual void SetVertices (const int *ind)
    {
      IAm ()->SetVertices (ind);
    }
    virtual void GetVertices (Array < int >&v) const
    {
      IAm ()->GetVertices (v);
    }
    virtual int *GetVertices ()
    {
      return IAm ()->GetVertices ();
    }
    virtual int GetNVertices () const
    {
      return IAm ()->GetNVertices ();
    }
    virtual int GetNEdges () const
    {
      return (IAm ()->GetNEdges ());
    }
    virtual const int *GetEdgeVertices (int ei) const
    {
      return (IAm ()->GetEdgeVertices (ei));
    }
    virtual int GetNFaces (int &nFaceVertices) const
    {
      return IAm ()->GetNFaces (nFaceVertices);
    }
    virtual const int *GetFaceVertices (int fi) const
    {
      return IAm ()->GetFaceVertices (fi);
    };
    virtual void MarkEdge (DenseMatrix & pmat)
    {
      IAm ()->MarkEdge (pmat);
    }
    virtual void MarkEdge (const DSTable & v_to_v, const int *length)
    {
      IAm ()->MarkEdge (v_to_v, length);
    }
    virtual int NeedRefinement (DSTable & v_to_v, int *middle) const
    {
      return IAm ()->NeedRefinement (v_to_v, middle);
    }
  };
  class BisectedElement:public RefinedElement
  {
  public:
    int SecondChild;
      BisectedElement ()
    {
    }
    BisectedElement (Element * ce):RefinedElement (ce)
    {
    }
    virtual int GetType () const
    {
      return Element::BISECTED;
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      mfem_error ("BisectedElement::Duplicate()");
      return __null;
    }
  };
  class QuadrisectedElement:public RefinedElement
  {
  public:
    int Child2, Child3, Child4;
      QuadrisectedElement (Element * ce):RefinedElement (ce)
    {
    }
    virtual int GetType () const
    {
      return Element::QUADRISECTED;
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      mfem_error ("QuadrisectedElement::Duplicate()");
      return __null;
    }
  };
  class OctasectedElement:public RefinedElement
  {
  public:
    int Child[7];
      OctasectedElement (Element * ce):RefinedElement (ce)
    {
    }
    virtual int GetType () const
    {
      return Element::OCTASECTED;
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      mfem_error ("OctasectedElement::Duplicate()");
      return __null;
    }
  };
  extern MemAlloc < BisectedElement, 1024 > BEMemory;
}

namespace mfem
{
  class Point:public Element
  {
  protected:
    int indices[1];
  public:
      Point ():Element (Geometry::POINT)
    {
    }
    Point (const int *ind, int attr = -1);
    virtual int GetType () const
    {
      return Element::POINT;
    }
    virtual void GetVertices (Array < int >&v) const;
    virtual int *GetVertices ()
    {
      return indices;
    }
    virtual int GetNVertices () const
    {
      return 1;
    }
    virtual int GetNEdges () const
    {
      return (0);
    }
    virtual const int *GetEdgeVertices (int ei) const
    {
      return __null;
    }
    virtual int GetNFaces (int &nFaceVertices) const
    {
      nFaceVertices = 0;
      return 0;
    }
    virtual const int *GetFaceVertices (int fi) const
    {
      return __null;
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      return new Point (indices, attribute);
    }
    virtual ~ Point ()
    {
    }
  };
  extern PointFiniteElement PointFE;
}

namespace mfem
{
  class Segment:public Element
  {
  protected:
    int indices[2];
  public:
      Segment ():Element (Geometry::SEGMENT)
    {
    }
    Segment (const int *ind, int attr = 1);
      Segment (int ind1, int ind2, int attr = 1);
    virtual void SetVertices (const int *ind);
    virtual int GetType () const
    {
      return Element::SEGMENT;
    }
    virtual void GetVertices (Array < int >&v) const;
    virtual int *GetVertices ()
    {
      return indices;
    }
    virtual int GetNVertices () const
    {
      return 2;
    }
    virtual int GetNEdges () const
    {
      return (0);
    }
    virtual const int *GetEdgeVertices (int ei) const
    {
      return __null;
    }
    virtual int GetNFaces (int &nFaceVertices) const
    {
      nFaceVertices = 0;
      return 0;
    }
    virtual const int *GetFaceVertices (int fi) const
    {
      return __null;
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      return new Segment (indices, attribute);
    }
    virtual ~ Segment ()
    {
    }
  };
  extern Linear1DFiniteElement SegmentFE;
}

namespace mfem
{
  class Triangle:public Element
  {
  protected:
    int indices[3];
    static const int edges[3][2];
  public:
      Triangle ():Element (Geometry::TRIANGLE)
    {
    }
    Triangle (const int *ind, int attr = 1);
      Triangle (int ind1, int ind2, int ind3, int attr = 1);
    int NeedRefinement (DSTable & v_to_v, int *middle) const;
    virtual void SetVertices (const int *ind);
    virtual void MarkEdge (DenseMatrix & pmat);
    virtual void MarkEdge (const DSTable & v_to_v, const int *length);
    virtual int GetType () const
    {
      return Element::TRIANGLE;
    }
    virtual void GetVertices (Array < int >&v) const;
    virtual int *GetVertices ()
    {
      return indices;
    }
    virtual int GetNVertices () const
    {
      return 3;
    }
    virtual int GetNEdges () const
    {
      return (3);
    }
    virtual const int *GetEdgeVertices (int ei) const
    {
      return (edges[ei]);
    }
    virtual int GetNFaces (int &nFaceVertices) const
    {
      nFaceVertices = 0;
      return 0;
    }
    virtual const int *GetFaceVertices (int fi) const
    {
      {
	std::ostringstream s;
	s << std::setprecision (16);
	s << std::setiosflags (std::ios_base::scientific);
	s << "MFEM abort: " << "not implemented" << '\n';
	s << " ... at line " << 70;
	s << " in " << __PRETTY_FUNCTION__ << " of file " <<
	  "../fem/../mesh/triangle.hpp" << ".";
	s << std::ends;
	if (!(0))
	  mfem::mfem_error (s.str ().c_str ());
	else
	  mfem::mfem_warning (s.str ().c_str ());
      };
      return __null;
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      return new Triangle (indices, attribute);
    }
    virtual ~ Triangle ()
    {
    }
  };
  extern Linear2DFiniteElement TriangleFE;
}

namespace mfem
{
  class Quadrilateral:public Element
  {
  protected:
    int indices[4];
    static const int edges[4][2];
  public:
      Quadrilateral ():Element (Geometry::SQUARE)
    {
    }
    Quadrilateral (const int *ind, int attr = 1);
      Quadrilateral (int ind1, int ind2, int ind3, int ind4, int attr = 1);
    int GetType () const
    {
      return Element::QUADRILATERAL;
    }
    virtual void SetVertices (const int *ind);
    virtual void GetVertices (Array < int >&v) const;
    virtual int *GetVertices ()
    {
      return indices;
    }
    virtual int GetNVertices () const
    {
      return 4;
    }
    virtual int GetNEdges () const
    {
      return (4);
    }
    virtual const int *GetEdgeVertices (int ei) const
    {
      return (edges[ei]);
    }
    virtual int GetNFaces (int &nFaceVertices) const
    {
      nFaceVertices = 0;
      return 0;
    }
    virtual const int *GetFaceVertices (int fi) const
    {
      return __null;
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      return new Quadrilateral (indices, attribute);
    }
    virtual ~ Quadrilateral ()
    {
    }
  };
  extern BiLinear2DFiniteElement QuadrilateralFE;
}

namespace mfem
{
  class Hexahedron:public Element
  {
  protected:
    int indices[8];
  public:
    static const int edges[12][2];
    static const int faces[6][4];
      Hexahedron ():Element (Geometry::CUBE)
    {
    }
    Hexahedron (const int *ind, int attr = 1);
      Hexahedron (int ind1, int ind2, int ind3, int ind4,
		  int ind5, int ind6, int ind7, int ind8, int attr = 1);
    int GetType () const
    {
      return Element::HEXAHEDRON;
    }
    virtual void GetVertices (Array < int >&v) const;
    virtual int *GetVertices ()
    {
      return indices;
    }
    virtual int GetNVertices () const
    {
      return 8;
    }
    virtual int GetNEdges () const
    {
      return 12;
    }
    virtual const int *GetEdgeVertices (int ei) const
    {
      return edges[ei];
    }
    virtual int GetNFaces (int &nFaceVertices) const
    {
      nFaceVertices = 4;
      return 6;
    }
    virtual const int *GetFaceVertices (int fi) const
    {
      return faces[fi];
    }
    virtual Element *Duplicate (Mesh * m) const
    {
      return new Hexahedron (indices, attribute);
    }
    virtual ~ Hexahedron ()
    {
    }
  };
  extern TriLinear3DFiniteElement HexahedronFE;
}

namespace mfem
{
  class Tetrahedron:public Element
  {
  protected:
    int indices[4];
    static const int edges[6][2];
    int refinement_flag;
  public:
    enum
    { TYPE_PU = 0, TYPE_A = 1, TYPE_PF = 2, TYPE_O = 3, TYPE_M = 4 };
      Tetrahedron ():Element (Geometry::TETRAHEDRON)
    {
      refinement_flag = 0;
    }
    Tetrahedron (const int *ind, int attr = 1);
      Tetrahedron (int ind1, int ind2, int ind3, int ind4, int attr = 1);
    void ParseRefinementFlag (int refinement_edges[2], int &type, int &flag);
    void CreateRefinementFlag (int refinement_edges[2], int type, int flag =
			       0);
    virtual int GetRefinementFlag ()
    {
      return refinement_flag;
    }
    void SetRefinementFlag (int rf)
    {
      refinement_flag = rf;
    }
    virtual int NeedRefinement (DSTable & v_to_v, int *middle) const;
    virtual void SetVertices (const int *ind);
    virtual void MarkEdge (DenseMatrix & pmat)
    {
    }
    virtual void MarkEdge (const DSTable & v_to_v, const int *length);
    virtual int GetType () const
    {
      return Element::TETRAHEDRON;
    }
    virtual void GetVertices (Array < int >&v) const;
    virtual int *GetVertices ()
    {
      return indices;
    }
    virtual int GetNVertices () const
    {
      return 4;
    }
    virtual int GetNEdges () const
    {
      return (6);
    }
    virtual const int *GetEdgeVertices (int ei) const
    {
      return (edges[ei]);
    }
    virtual int GetNFaces (int &nFaceVertices) const
    {
      nFaceVertices = 3;
      return 4;
    }
    virtual const int *GetFaceVertices (int fi) const
    {
      {
	std::ostringstream s;
	s << std::setprecision (16);
	s << std::setiosflags (std::ios_base::scientific);
	s << "MFEM abort: " << "not implemented" << '\n';
	s << " ... at line " << 90;
	s << " in " << __PRETTY_FUNCTION__ << " of file " <<
	  "../fem/../mesh/tetrahedron.hpp" << ".";
	s << std::ends;
	if (!(0))
	  mfem::mfem_error (s.str ().c_str ());
	else
	  mfem::mfem_warning (s.str ().c_str ());
      };
      return __null;
    }
    virtual Element *Duplicate (Mesh * m) const;
    virtual ~ Tetrahedron ()
    {
    }
  };
  extern Linear3DFiniteElement TetrahedronFE;
}

namespace mfem
{
  class IdGenerator
  {
  public:
  IdGenerator (int first_id = 0):next (first_id)
    {
    }
    int Get ()
    {
      if (reusable.Size ())
	{
	  int id = reusable.Last ();
	    reusable.DeleteLast ();
	    return id;
	}
      return next++;
    }
    void Reuse (int id)
    {
      reusable.Append (id);
    }
  private:
    int next;
    Array < int >reusable;
  };
  template < typename Derived > struct Hashed2
  {
    int id;
    int p1, p2;
    Derived *next;
      Hashed2 (int id):id (id)
    {
    }
  };
  template < typename Derived > struct Hashed4
  {
    int id;
    int p1, p2, p3;
    Derived *next;
      Hashed4 (int id):id (id)
    {
    }
  };
  template < typename ItemT > class HashTable
  {
  public:
    HashTable (int init_size = 32 * 1024);
    ~HashTable ();
    ItemT *Get (int p1, int p2);
    ItemT *Get (int p1, int p2, int p3, int p4);
    ItemT *Peek (int p1, int p2) const;
    ItemT *Peek (int p1, int p2, int p3, int p4) const;
    template < typename OtherT > ItemT * Get (OtherT * i1, OtherT * i2)
    {
      return Get (i1->id, i2->id);
    }
    template < typename OtherT >
      ItemT * Get (OtherT * i1, OtherT * i2, OtherT * i3, OtherT * i4)
    {
      return Get (i1->id, i2->id, i3->id, i4->id);
    }
    template < typename OtherT > ItemT * Peek (OtherT * i1, OtherT * i2) const
    {
      return Peek (i1->id, i2->id);
    }
    template < typename OtherT >
      ItemT * Peek (OtherT * i1, OtherT * i2, OtherT * i3, OtherT * i4) const
    {
      return Peek (i1->id, i2->id, i3->id, i4->id);
    }
    ItemT *Peek (int id) const
    {
      return id_to_item[id];
    }
    void Delete (ItemT * item);
    void Reparent (ItemT * item, int new_p1, int new_p2);
    void Reparent (ItemT * item, int new_p1, int new_p2, int new_p3,
		   int new_p4);
    class Iterator
    {
    public:
      Iterator (HashTable < ItemT > &table):hash_table (table), cur_id (-1),
	cur_item (__null)
      {
	next ();
      }
      operator  ItemT *() const
      {
	return cur_item;
      }
      ItemT & operator* () const
      {
	return *cur_item;
      }
      ItemT *operator-> () const
      {
	return cur_item;
      }
      Iterator & operator++ ()
      {
	next ();
	return *this;
      }
    protected:
      HashTable < ItemT > &hash_table;
      int cur_id;
      ItemT *cur_item;
      void next ();
    };
    long MemoryUsage () const;
  protected:
    ItemT ** table;
    int mask;
    int num_items;
    inline int hash (int p1, int p2) const
    {
      return (984120265 * p1 + 125965121 * p2) & mask;
    }
    inline int hash (int p1, int p2, int p3) const
    {
      return (984120265 * p1 + 125965121 * p2 + 495698413 * p3) & mask;
    }
    inline int hash (const Hashed2 < ItemT > *item) const
    {
      return hash (item->p1, item->p2);
    }
    inline int hash (const Hashed4 < ItemT > *item) const
    {
      return hash (item->p1, item->p2, item->p3);
    }
    ItemT *SearchList (ItemT * item, int p1, int p2) const;
    ItemT *SearchList (ItemT * item, int p1, int p2, int p3) const;
    void Insert (int idx, ItemT * item);
    void Unlink (ItemT * item);
    void Rehash ();
    IdGenerator id_gen;
    Array < ItemT * >id_to_item;
  };
  template < typename ItemT > HashTable < ItemT >::HashTable (int init_size)
  {
    mask = init_size - 1;
    if (init_size & mask)
      mfem_error ("HashTable(): init_size size must be a power of two.");
    table = new ItemT *[init_size];
    memset (table, 0, init_size * sizeof (ItemT *));
    num_items = 0;
  }
  template < typename ItemT > HashTable < ItemT >::~HashTable ()
  {
    for (Iterator it (*this); it; ++it)
      delete it;
    delete[]table;
  }
  namespace internal
  {
    inline void sort3 (int &a, int &b, int &c)
    {
      if (a > b)
	std::swap (a, b);
      if (a > c)
	std::swap (a, c);
      if (b > c)
	std::swap (b, c);
    }
    inline void sort4 (int &a, int &b, int &c, int &d)
    {
      if (a > b)
	std::swap (a, b);
      if (a > c)
	std::swap (a, c);
      if (a > d)
	std::swap (a, d);
      sort3 (b, c, d);
    }
  }
  template < typename ItemT >
    ItemT * HashTable < ItemT >::Peek (int p1, int p2) const
  {
    if (p1 > p2)
      std::swap (p1, p2);
    return SearchList (table[hash (p1, p2)], p1, p2);
  }
  template < typename ItemT >
    ItemT * HashTable < ItemT >::Peek (int p1, int p2, int p3, int p4) const
  {
    internal::sort4 (p1, p2, p3, p4);
    return SearchList (table[hash (p1, p2, p3)], p1, p2, p3);
  }
  template < typename ItemT >
    void HashTable < ItemT >::Insert (int idx, ItemT * item)
  {
    item->next = table[idx];
    table[idx] = item;
    num_items++;
  }
  template < typename ItemT >
    ItemT * HashTable < ItemT >::Get (int p1, int p2)
  {
    if (p1 > p2)
      std::swap (p1, p2);
    int idx = hash (p1, p2);
    ItemT *node = SearchList (table[idx], p1, p2);
    if (node)
      return node;
    ItemT *newitem = new ItemT (id_gen.Get ());
    newitem->p1 = p1;
    newitem->p2 = p2;
    Insert (idx, newitem);
    if (id_to_item.Size () <= newitem->id)
      {
	id_to_item.SetSize (newitem->id + 1, __null);
      }
    id_to_item[newitem->id] = newitem;
    Rehash ();
    return newitem;
  }
  template < typename ItemT >
    ItemT * HashTable < ItemT >::Get (int p1, int p2, int p3, int p4)
  {
    internal::sort4 (p1, p2, p3, p4);
    int idx = hash (p1, p2, p3);
    ItemT *node = SearchList (table[idx], p1, p2, p3);
    if (node)
      return node;
    ItemT *newitem = new ItemT (id_gen.Get ());
    newitem->p1 = p1;
    newitem->p2 = p2;
    newitem->p3 = p3;
    Insert (idx, newitem);
    if (id_to_item.Size () <= newitem->id)
      {
	id_to_item.SetSize (newitem->id + 1, __null);
      }
    id_to_item[newitem->id] = newitem;
    Rehash ();
    return newitem;
  }
  template < typename ItemT >
    ItemT * HashTable < ItemT >::SearchList (ItemT * item, int p1,
					     int p2) const
  {
    while (item != __null)
      {
	if (item->p1 == p1 && item->p2 == p2)
	  return item;
	item = item->next;
      }
    return __null;
  }
  template < typename ItemT >
    ItemT * HashTable < ItemT >::SearchList (ItemT * item, int p1, int p2,
					     int p3) const
  {
    while (item != __null)
      {
	if (item->p1 == p1 && item->p2 == p2 && item->p3 == p3)
	  return item;
	item = item->next;
      }
    return __null;
  }
  template < typename ItemT > void HashTable < ItemT >::Rehash ()
  {
    const int fill_factor = 2;
    int old_size = mask + 1;
    if (num_items > old_size * fill_factor)
      {
	delete[]table;
	int new_size = 2 * old_size;
	table = new ItemT *[new_size];
	memset (table, 0, new_size * sizeof (ItemT *));
	mask = new_size - 1;
	num_items = 0;
	for (Iterator it (*this); it; ++it)
	  Insert (hash (it), it);
      }
  }
  template < typename ItemT > void HashTable < ItemT >::Unlink (ItemT * item)
  {
    ItemT **ptr = table + hash (item);
    while (*ptr)
      {
	if (*ptr == item)
	  {
	    *ptr = item->next;
	    num_items--;
	    return;
	  }
	ptr = &((*ptr)->next);
      }
    mfem_error ("HashTable<>::Unlink: item not found!");
  }
  template < typename ItemT > void HashTable < ItemT >::Delete (ItemT * item)
  {
    Unlink (item);
    id_to_item[item->id] = __null;
    id_gen.Reuse (item->id);
    delete item;
  }
  template < typename ItemT >
    void HashTable < ItemT >::Reparent (ItemT * item, int new_p1, int new_p2)
  {
    Unlink (item);
    if (new_p1 > new_p2)
      std::swap (new_p1, new_p2);
    item->p1 = new_p1;
    item->p2 = new_p2;
    int new_idx = hash (new_p1, new_p2);
    Insert (new_idx, item);
  }
  template < typename ItemT >
    void HashTable < ItemT >::Reparent (ItemT * item,
					int new_p1, int new_p2, int new_p3,
					int new_p4)
  {
    Unlink (item);
    internal::sort4 (new_p1, new_p2, new_p3, new_p4);
    item->p1 = new_p1;
    item->p2 = new_p2;
    item->p3 = new_p3;
    int new_idx = hash (new_p1, new_p2, new_p3);
    Insert (new_idx, item);
  }
  template < typename ItemT > void HashTable < ItemT >::Iterator::next ()
  {
    while (cur_id < hash_table.id_to_item.Size () - 1)
      {
	++cur_id;
	cur_item = hash_table.id_to_item[cur_id];
	if (cur_item)
	  return;
      }
    cur_item = __null;
  }
  template < typename ItemT > long HashTable < ItemT >::MemoryUsage () const
  {
    return sizeof (*this) +
      ((mask + 1) + id_to_item.Capacity ()) * sizeof (ItemT *) +
      num_items * sizeof (ItemT);
  }
}
namespace mfem
{
  class SparseMatrix;
  class Mesh;
  class IsoparametricTransformation;
  class FiniteElementSpace;
  struct Refinement
  {
    int index;
    int ref_type;
    Refinement (int index, int type = 7):index (index), ref_type (type)
    {
    }
  };
  class NCMesh
  {
  public:
    NCMesh (const Mesh * mesh);
    int Dimension () const
    {
      return Dim;
    }
    void Refine (const Array < Refinement > &refinements);
    void LimitNCLevel (int max_level);
    SparseMatrix *GetInterpolation (FiniteElementSpace * space,
				    SparseMatrix ** cR_ptr = __null);
    struct FineTransform
    {
      int coarse_index;
      DenseMatrix point_matrix;
      bool IsIdentity () const
      {
	return !point_matrix.Data ();
      }
    };
    void MarkCoarseLevel ()
    {
      leaf_elements.Copy (coarse_elements);
    }
    void ClearCoarseLevel ()
    {
      coarse_elements.DeleteAll ();
    }
    FineTransform *GetFineTransforms ();
    int GetEdgeMaster (int v1, int v2) const;
    long MemoryUsage ();
    ~NCMesh ();
  protected:
    void GetVerticesElementsBoundary (Array < mfem::Vertex > &vertices,
				      Array < mfem::Element * >&elements,
				      Array < mfem::Element * >&boundary);
    void SetEdgeIndicesFromMesh (Mesh * mesh);
    void SetFaceIndicesFromMesh (Mesh * mesh);
    friend class Mesh;
  protected:
    int Dim;
    struct RefCount
    {
      int ref_count;
        RefCount ():ref_count (0)
      {
      }
      int Ref ()
      {
	return ++ref_count;
      }
      int Unref ()
      {
	int ret = --ref_count;
	if (!ret)
	  delete this;
	return ret;
      }
    };
    struct Vertex:public RefCount
    {
      double pos[3];
      int index;
        Vertex ()
      {
      }
      Vertex (double x, double y, double z):index (-1)
      {
	pos[0] = x, pos[1] = y, pos[2] = z;
      }
    };
    struct Edge:public RefCount
    {
      int attribute;
      int index;
        Edge ():attribute (-1), index (-1)
      {
      }
      bool Boundary () const
      {
	return attribute >= 0;
      }
    };
    struct Node:public Hashed2 < Node >
    {
      Vertex *vertex;
      Edge *edge;
        Node (int id):Hashed2 < Node > (id), vertex (__null), edge (__null)
      {
      }
      void RefVertex ();
      void RefEdge ();
      void UnrefVertex (HashTable < Node > &nodes);
      void UnrefEdge (HashTable < Node > &nodes);
      ~Node ();
    };
    struct Element;
    struct Face:public RefCount, public Hashed4 < Face >
    {
      int attribute;
      int index;
      Element *elem[2];
        Face (int id):Hashed4 < Face > (id), attribute (-1), index (-1)
      {
	elem[0] = elem[1] = __null;
      }
      bool Boundary () const
      {
	return attribute >= 0;
      }
      void RegisterElement (Element * e);
      void ForgetElement (Element * e);
      Element *GetSingleElement () const;
      int Unref ()
      {
	return --ref_count;
      }
    };
    struct Element
    {
      int geom;
      int attribute;
      int ref_type;
      int index;
      union
      {
	Node *node[8];
	Element *child[8];
      };
        Element (int geom, int attr);
    };
    Array < Element * >root_elements;
    Array < Element * >leaf_elements;
    Array < Element * >coarse_elements;
    Array < int >vertex_nodeId;
    HashTable < Node > nodes;
    HashTable < Face > faces;
    struct RefStackItem
    {
      Element *elem;
      int ref_type;
        RefStackItem (Element * elem, int type):elem (elem), ref_type (type)
      {
      }
    };
    Array < RefStackItem > ref_stack;
    void Refine (Element * elem, int ref_type);
    void UpdateVertices ();
    void GetLeafElements (Element * e);
    void UpdateLeafElements ();
    void DeleteHierarchy (Element * elem);
    Element *NewHexahedron (Node * n0, Node * n1, Node * n2, Node * n3,
			    Node * n4, Node * n5, Node * n6, Node * n7,
			    int attr,
			    int fattr0, int fattr1, int fattr2,
			    int fattr3, int fattr4, int fattr5);
    Element *NewQuadrilateral (Node * n0, Node * n1, Node * n2, Node * n3,
			       int attr,
			       int eattr0, int eattr1, int eattr2,
			       int eattr3);
    Element *NewTriangle (Node * n0, Node * n1, Node * n2, int attr,
			  int eattr0, int eattr1, int eattr2);
    Vertex *NewVertex (Node * v1, Node * v2);
    Node *GetMidEdgeVertex (Node * v1, Node * v2);
    Node *GetMidEdgeVertexSimple (Node * v1, Node * v2);
    Node *GetMidFaceVertex (Node * e1, Node * e2, Node * e3, Node * e4);
    int FaceSplitType (Node * v1, Node * v2, Node * v3, Node * v4,
		       Node * mid[4] = __null);
    void ForceRefinement (Node * v1, Node * v2, Node * v3, Node * v4);
    void CheckAnisoFace (Node * v1, Node * v2, Node * v3, Node * v4,
			 Node * mid12, Node * mid34, int level = 0);
    void CheckIsoFace (Node * v1, Node * v2, Node * v3, Node * v4,
		       Node * e1, Node * e2, Node * e3, Node * e4,
		       Node * midf);
    void RefElementNodes (Element * elem);
    void UnrefElementNodes (Element * elem);
    void RegisterFaces (Element * elem);
    Node *PeekAltParents (Node * v1, Node * v2);
    bool NodeSetX1 (Node * node, Node ** n);
    bool NodeSetX2 (Node * node, Node ** n);
    bool NodeSetY1 (Node * node, Node ** n);
    bool NodeSetY2 (Node * node, Node ** n);
    bool NodeSetZ1 (Node * node, Node ** n);
    bool NodeSetZ2 (Node * node, Node ** n);
    struct Dependency
    {
      int dof;
      double coef;
        Dependency (int dof, double coef):dof (dof), coef (coef)
      {
      }
    };
    typedef Array < Dependency > DepList;
    struct DofData
    {
      bool finalized;
      DepList dep_list;
        DofData ():finalized (false)
      {
      }
      bool Independent () const
      {
	return !dep_list.Size ();
      }
    };
    DofData *dof_data;
    FiniteElementSpace *space;
    static int find_node (Element * elem, Node * node);
    void ReorderFacePointMat (Node * v0, Node * v1, Node * v2, Node * v3,
			      Element * elem, DenseMatrix & pm);
    void AddDependencies (Array < int >&master_dofs, Array < int >&slave_dofs,
			  DenseMatrix & I);
    void ConstrainEdge (Node * v0, Node * v1, double t0, double t1,
			Array < int >&master_dofs, int level);
    struct PointMatrix;
    void ConstrainFace (Node * v0, Node * v1, Node * v2, Node * v3,
			const PointMatrix & pm,
			Array < int >&master_dofs, int level);
    void ProcessMasterEdge (Node * node[2], Node * edge);
    void ProcessMasterFace (Node * node[4], Face * face);
    bool DofFinalizable (DofData & vd);
    struct Point
    {
      int dim;
      double coord[3];
        Point ()
      {
	dim = 0;
      }
      Point (double x, double y)
      {
	dim = 2;
	coord[0] = x;
	coord[1] = y;
      }
      Point (double x, double y, double z)
      {
	dim = 3;
	coord[0] = x;
	coord[1] = y;
	coord[2] = z;
      }
      Point (const Point & p0, const Point & p1)
      {
	dim = p0.dim;
	for (int i = 0; i < dim; i++)
	  coord[i] = (p0.coord[i] + p1.coord[i]) * 0.5;
      }
      Point (const Point & p0, const Point & p1, const Point & p2,
	     const Point & p3)
      {
	dim = p0.dim;
	for (int i = 0; i < dim; i++)
	  coord[i] = (p0.coord[i] + p1.coord[i] + p2.coord[i] + p3.coord[i])
	    * 0.25;
      }
      Point & operator= (const Point & src)
      {
	dim = src.dim;
	for (int i = 0; i < dim; i++)
	  coord[i] = src.coord[i];
	return *this;
      }
    };
    struct PointMatrix
    {
      int np;
      Point points[8];
        PointMatrix (const Point & p0, const Point & p1, const Point & p2)
      {
	np = 3;
	points[0] = p0;
	points[1] = p1;
	points[2] = p2;
      }
      PointMatrix (const Point & p0, const Point & p1, const Point & p2,
		   const Point & p3)
      {
	np = 4;
	points[0] = p0;
	points[1] = p1;
	points[2] = p2;
	points[3] = p3;
      }
      PointMatrix (const Point & p0, const Point & p1, const Point & p2,
		   const Point & p3, const Point & p4, const Point & p5,
		   const Point & p6, const Point & p7)
      {
	np = 8;
	points[0] = p0;
	points[1] = p1;
	points[2] = p2;
	points[3] = p3;
	points[4] = p4;
	points[5] = p5;
	points[6] = p6;
	points[7] = p7;
      }
      Point & operator ()(int i)
      {
	return points[i];
      }
      const Point & operator () (int i) const
      {
	return points[i];
      }
      void GetMatrix (DenseMatrix & point_matrix) const;
    };
    void GetFineTransforms (Element * elem, int coarse_index,
			    FineTransform * transforms,
			    const PointMatrix & pm);
    int GetEdgeMaster (Node * n) const;
    void FaceSplitLevel (Node * v1, Node * v2, Node * v3, Node * v4,
			 int &h_level, int &v_level);
    void CountSplits (Element * elem, int splits[3]);
    int CountElements (Element * elem);
  };
}

namespace mfem
{
  class KnotVector;
  class NURBSExtension;
  class FiniteElementSpace;
  class GridFunction;
  struct Refinement;
  class Mesh
  {
    friend class NURBSExtension;
  protected:
    int Dim;
    int spaceDim;
    int NumOfVertices, NumOfElements, NumOfBdrElements;
    int NumOfEdges, NumOfFaces;
    int State, WantTwoLevelState;
    int meshgen;
    int c_NumOfVertices, c_NumOfElements, c_NumOfBdrElements;
    int f_NumOfVertices, f_NumOfElements, f_NumOfBdrElements;
    int c_NumOfEdges, c_NumOfFaces;
    int f_NumOfEdges, f_NumOfFaces;
      Array < Element * >elements;
      Array < Vertex > vertices;
      Array < Element * >boundary;
      Array < Element * >faces;
    class FaceInfo
    {
    public:int Elem1No, Elem2No, Elem1Inf, Elem2Inf;
    };
      Array < FaceInfo > faces_info;
    Table *el_to_edge;
    Table *el_to_face;
    Table *el_to_el;
      Array < int >be_to_edge;
    Table *bel_to_edge;
      Array < int >be_to_face;
    mutable Table *face_edge;
    mutable Table *edge_vertex;
    Table *c_el_to_edge, *f_el_to_edge, *c_bel_to_edge, *f_bel_to_edge;
      Array < int >fc_be_to_edge;
    Table *c_el_to_face, *f_el_to_face;
      Array < FaceInfo > fc_faces_info;
    IsoparametricTransformation Transformation, Transformation2;
    IsoparametricTransformation FaceTransformation, EdgeTransformation;
    FaceElementTransformations FaceElemTr;
    GridFunction *Nodes;
    int own_nodes;
    Mesh *nc_coarse_level;
    static const int tet_faces[4][3];
    static const int hex_faces[6][4];
    static const int tri_orientations[6][3];
    static const int quad_orientations[8][4];
    friend class Tetrahedron;
      MemAlloc < Tetrahedron, 1024 > TetMemory;
      MemAlloc < BisectedElement, 1024 > BEMemory;
    void Init ();
    void InitTables ();
    void DeleteTables ();
    void DeleteCoarseTables ();
    Element *ReadElementWithoutAttr (std::istream &);
    static void PrintElementWithoutAttr (const Element *, std::ostream &);
    Element *ReadElement (std::istream &);
    static void PrintElement (const Element *, std::ostream &);
    void SetMeshGen ();
    double GetLength (int i, int j) const;
    void GetElementJacobian (int i, DenseMatrix & J);
    void MarkForRefinement ();
    void MarkTriMeshForRefinement ();
    void GetEdgeOrdering (DSTable & v_to_v, Array < int >&order);
    void MarkTetMeshForRefinement ();
    void PrepareNodeReorder (DSTable ** old_v_to_v, Table ** old_elem_vert);
    void DoNodeReorder (DSTable * old_v_to_v, Table * old_elem_vert);
    STable3D *GetFacesTable ();
    STable3D *GetElementToFaceTable (int ret_ftbl = 0);
    void RedRefinement (int i, const DSTable & v_to_v,
			int *edge1, int *edge2, int *middle)
    {
      UniformRefinement (i, v_to_v, edge1, edge2, middle);
    }
    void GreenRefinement (int i, const DSTable & v_to_v,
			  int *edge1, int *edge2, int *middle)
    {
      Bisection (i, v_to_v, edge1, edge2, middle);
    }
    void Bisection (int i, const DSTable &, int *, int *, int *);
    void Bisection (int i, const DSTable &, int *);
    void UniformRefinement (int i, const DSTable &, int *, int *, int *);
    void AverageVertices (int *indexes, int n, int result);
    void UpdateNodes ();
    virtual void QuadUniformRefinement ();
    virtual void HexUniformRefinement ();
    virtual void NURBSUniformRefinement ();
    virtual void LocalRefinement (const Array < int >&marked_el, int type =
				  3);
    void NonconformingRefinement (const Array < Refinement > &refinements,
				  int nc_limit = 0);
    void LoadPatchTopo (std::istream & input, Array < int >&edge_to_knot);
    void UpdateNURBS ();
    void PrintTopo (std::ostream & out, const Array < int >&e_to_k) const;
    void BisectTriTrans (DenseMatrix & pointmat, Triangle * tri, int child);
    void BisectTetTrans (DenseMatrix & pointmat, Tetrahedron * tet,
			 int child);
    int GetFineElemPath (int i, int j);
    int GetBisectionHierarchy (Element * E);
    void GetLocalPtToSegTransformation (IsoparametricTransformation &, int);
    void GetLocalSegToTriTransformation (IsoparametricTransformation & loc,
					 int i);
    void GetLocalSegToQuadTransformation (IsoparametricTransformation & loc,
					  int i);
    void GetLocalTriToTetTransformation (IsoparametricTransformation & loc,
					 int i);
    void GetLocalQuadToHexTransformation (IsoparametricTransformation & loc,
					  int i);
    static int GetTriOrientation (const int *base, const int *test);
    static int GetQuadOrientation (const int *base, const int *test);
    static void GetElementArrayEdgeTable (const Array <
					  Element * >&elem_array,
					  const DSTable & v_to_v,
					  Table & el_to_edge);
    void GetVertexToVertexTable (DSTable &) const;
    int GetElementToEdgeTable (Table &, Array < int >&);
    void AddPointFaceElement (int lf, int gf, int el);
    void AddSegmentFaceElement (int lf, int gf, int el, int v0, int v1);
    void AddTriangleFaceElement (int lf, int gf, int el,
				 int v0, int v1, int v2);
    void AddQuadFaceElement (int lf, int gf, int el,
			     int v0, int v1, int v2, int v3);
    bool FaceIsTrueInterior (int FaceNo) const
    {
      return FaceIsInterior (FaceNo) || (faces_info[FaceNo].Elem2Inf >= 0);
    }
    inline static void ShiftL2R (int &, int &, int &);
    inline static void Rotate3 (int &, int &, int &);
    void FreeElement (Element * E);
    void GenerateFaces ();
    void InitMesh (int _Dim, int _spaceDim, int NVert, int NElem,
		   int NBdrElem);
    void Make3D (int nx, int ny, int nz, Element::Type type,
		 int generate_edges, double sx, double sy, double sz);
    void Make2D (int nx, int ny, Element::Type type, int generate_edges,
		 double sx, double sy);
    void Make1D (int n, double sx = 1.0);
    Mesh (NCMesh & ncmesh);
    void Swap (Mesh & other, bool non_geometry = false);
  public:
    enum
    { NORMAL, TWO_LEVEL_COARSE, TWO_LEVEL_FINE };
    Array < int >attributes;
    Array < int >bdr_attributes;
    NURBSExtension *NURBSext;
    NCMesh *ncmesh;
    Mesh ()
    {
      Init ();
      InitTables ();
      meshgen = 0;
      Dim = 0;
    }
    Mesh (int _Dim, int NVert, int NElem, int NBdrElem = 0, int _spaceDim =
	  -1)
    {
      if (_spaceDim == -1)
	_spaceDim = _Dim;
      InitMesh (_Dim, _spaceDim, NVert, NElem, NBdrElem);
    }
    Element *NewElement (int geom);
    void AddVertex (const double *);
    void AddTri (const int *vi, int attr = 1);
    void AddTriangle (const int *vi, int attr = 1);
    void AddQuad (const int *vi, int attr = 1);
    void AddTet (const int *vi, int attr = 1);
    void AddHex (const int *vi, int attr = 1);
    void AddHexAsTets (const int *vi, int attr = 1);
    void AddElement (Element * elem)
    {
      elements[NumOfElements++] = elem;
    }
    void AddBdrSegment (const int *vi, int attr = 1);
    void AddBdrTriangle (const int *vi, int attr = 1);
    void AddBdrQuad (const int *vi, int attr = 1);
    void AddBdrQuadAsTriangles (const int *vi, int attr = 1);
    void GenerateBoundaryElements ();
    void FinalizeTriMesh (int generate_edges = 0, int refine = 0,
			  bool fix_orientation = true);
    void FinalizeQuadMesh (int generate_edges = 0, int refine = 0,
			   bool fix_orientation = true);
    void FinalizeTetMesh (int generate_edges = 0, int refine = 0,
			  bool fix_orientation = true);
    void FinalizeHexMesh (int generate_edges = 0, int refine = 0,
			  bool fix_orientation = true);
    void SetAttributes ();
    Mesh (int nx, int ny, int nz, Element::Type type, int generate_edges = 0,
	  double sx = 1.0, double sy = 1.0, double sz = 1.0)
    {
      Make3D (nx, ny, nz, type, generate_edges, sx, sy, sz);
    }
    Mesh (int nx, int ny, Element::Type type, int generate_edges = 0,
	  double sx = 1.0, double sy = 1.0)
    {
      Make2D (nx, ny, type, generate_edges, sx, sy);
    }
    explicit Mesh (int n, double sx = 1.0)
    {
      Make1D (n, sx);
    }
    Mesh (std::istream & input, int generate_edges = 0, int refine = 1,
	  bool fix_orientation = true);
    Mesh (Mesh * mesh_array[], int num_pieces);
    void Load (std::istream & input, int generate_edges = 0, int refine = 1,
	       bool fix_orientation = true);
    inline int MeshGenerator ()
    {
      return meshgen;
    }
    inline int GetNV () const
    {
      return NumOfVertices;
    }
    inline int GetNE () const
    {
      return NumOfElements;
    }
    inline int GetNBE () const
    {
      return NumOfBdrElements;
    }
    inline int GetNEdges () const
    {
      return NumOfEdges;
    }
    inline int GetNFaces () const
    {
      return NumOfFaces;
    }
    int GetNumFaces () const;
    inline int EulerNumber () const
    {
      return NumOfVertices - NumOfEdges + NumOfFaces - NumOfElements;
    }
    inline int EulerNumber2D () const
    {
      return NumOfVertices - NumOfEdges + NumOfElements;
    }
    int Dimension () const
    {
      return Dim;
    }
    int SpaceDimension () const
    {
      return spaceDim;
    }
    const double *GetVertex (int i) const
    {
      return vertices[i] ();
    }
    double *GetVertex (int i)
    {
      return vertices[i] ();
    }
    const Element *GetElement (int i) const
    {
      return elements[i];
    }
    Element *GetElement (int i)
    {
      return elements[i];
    }
    const Element *GetBdrElement (int i) const
    {
      return boundary[i];
    }
    Element *GetBdrElement (int i)
    {
      return boundary[i];
    }
    const Element *GetFace (int i) const
    {
      return faces[i];
    }
    int GetFaceBaseGeometry (int i) const;
    int GetElementBaseGeometry (int i) const
    {
      return elements[i]->GetGeometryType ();
    }
    int GetBdrElementBaseGeometry (int i) const
    {
      return boundary[i]->GetGeometryType ();
    }
    void GetElementVertices (int i, Array < int >&dofs) const
    {
      elements[i]->GetVertices (dofs);
    }
    void GetBdrElementVertices (int i, Array < int >&dofs) const
    {
      boundary[i]->GetVertices (dofs);
    }
    void GetElementEdges (int i, Array < int >&, Array < int >&) const;
    void GetBdrElementEdges (int i, Array < int >&, Array < int >&) const;
    void GetFaceEdges (int i, Array < int >&, Array < int >&) const;
    void GetFaceVertices (int i, Array < int >&vert) const
    {
      if (Dim == 1)
	{
	  vert.SetSize (1);
	  vert[0] = i;
	}
      else
	  faces[i]->GetVertices (vert);
    }
    void GetEdgeVertices (int i, Array < int >&vert) const;
    Table *GetFaceEdgeTable () const;
    Table *GetEdgeVertexTable () const;
    void GetElementFaces (int i, Array < int >&, Array < int >&) const;
    void GetBdrElementFace (int i, int *, int *) const;
    int GetBdrElementEdgeIndex (int i) const;
    int GetElementType (int i) const;
    int GetBdrElementType (int i) const;
    void GetPointMatrix (int i, DenseMatrix & pointmat) const;
    void GetBdrPointMatrix (int i, DenseMatrix & pointmat) const;
    static FiniteElement *GetTransformationFEforElementType (int);
    void GetElementTransformation (int i, IsoparametricTransformation * ElTr);
    ElementTransformation *GetElementTransformation (int i);
    void GetElementTransformation (int i, const Vector & nodes,
				   IsoparametricTransformation * ElTr);
    ElementTransformation *GetBdrElementTransformation (int i);
    void GetBdrElementTransformation (int i,
				      IsoparametricTransformation * ElTr);
    void GetFaceTransformation (int i, IsoparametricTransformation * FTr);
    ElementTransformation *GetFaceTransformation (int FaceNo);
    void GetEdgeTransformation (int i, IsoparametricTransformation * EdTr);
    ElementTransformation *GetEdgeTransformation (int EdgeNo);
    FaceElementTransformations *GetFaceElementTransformations (int FaceNo,
							       int mask = 31);
    FaceElementTransformations *GetInteriorFaceTransformations (int FaceNo)
    {
      if (faces_info[FaceNo].Elem2No < 0)
	return __null;
      return GetFaceElementTransformations (FaceNo);
    }
    FaceElementTransformations *GetBdrFaceTransformations (int BdrElemNo);
    bool FaceIsInterior (int FaceNo) const
    {
      return (faces_info[FaceNo].Elem2No >= 0);
    }
    void GetFaceElements (int Face, int *Elem1, int *Elem2);
    void GetFaceInfos (int Face, int *Inf1, int *Inf2);
    void CheckElementOrientation (bool fix_it = true);
    void CheckBdrElementOrientation (bool fix_it = true);
    int GetAttribute (int i) const
    {
      return elements[i]->GetAttribute ();
    }
    int GetBdrAttribute (int i) const
    {
      return boundary[i]->GetAttribute ();
    }
    const Table & ElementToElementTable ();
    const Table & ElementToFaceTable () const;
    const Table & ElementToEdgeTable () const;
    Table *GetVertexToElementTable ();
    Table *GetFaceToElementTable () const;
    virtual void ReorientTetMesh ();
    int *CartesianPartitioning (int nxyz[]);
    int *GeneratePartitioning (int nparts, int part_method = 1);
    void CheckPartitioning (int *partitioning);
    void CheckDisplacements (const Vector & displacements, double &tmax);
    void MoveVertices (const Vector & displacements);
    void GetVertices (Vector & vert_coord) const;
    void SetVertices (const Vector & vert_coord);
    void GetNode (int i, double *coord);
    void SetNode (int i, const double *coord);
    void MoveNodes (const Vector & displacements);
    void GetNodes (Vector & node_coord) const;
    void SetNodes (const Vector & node_coord);
    GridFunction *GetNodes ()
    {
      return Nodes;
    }
    void NewNodes (GridFunction & nodes, bool make_owner = false);
    void SwapNodes (GridFunction * &nodes, int &own_nodes_);
    void GetNodes (GridFunction & nodes) const;
    void SetNodalFESpace (FiniteElementSpace * nfes);
    void SetNodalGridFunction (GridFunction * nodes, bool make_owner = false);
    const FiniteElementSpace *GetNodalFESpace ();
    void UniformRefinement ();
    void GeneralRefinement (Array < Refinement > &refinements,
			    int nonconforming = -1, int nc_limit = 0);
    void GeneralRefinement (Array < int >&el_to_refine,
			    int nonconforming = -1, int nc_limit = 0);
    void KnotInsert (Array < KnotVector * >&kv);
    void DegreeElevate (int t);
    void UseTwoLevelState (int use)
    {
      if (!use && State != Mesh::NORMAL)
	SetState (Mesh::NORMAL);
      WantTwoLevelState = use;
    }
    void SetState (int s);
    int GetState () const
    {
      return State;
    }
    int GetNumFineElems (int i);
    int GetRefinementType (int i);
    int GetFineElem (int i, int j);
    ElementTransformation *GetFineElemTrans (int i, int j);
    virtual void PrintXG (std::ostream & out = std::cout) const;
    virtual void Print (std::ostream & out = std::cout) const;
    void PrintVTK (std::ostream & out);
    void PrintVTK (std::ostream & out, int ref, int field_data = 0);
    void GetElementColoring (Array < int >&colors, int el0 = 0);
    void PrintWithPartitioning (int *partitioning,
				std::ostream & out, int elem_attr = 0) const;
    void PrintElementsWithPartitioning (int *partitioning,
					std::ostream & out,
					int interior_faces = 0);
    void PrintSurfaces (const Table & Aface_face, std::ostream & out) const;
    void ScaleSubdomains (double sf);
    void ScaleElements (double sf);
    void Transform (void (*f) (const Vector &, Vector &));
    double GetElementSize (int i, int type = 0);
    double GetElementSize (int i, const Vector & dir);
    double GetElementVolume (int i);
    void PrintCharacteristics (Vector * Vh = __null, Vector * Vk = __null);
    void MesquiteSmooth (const int mesquite_option = 0);
    virtual ~ Mesh ();
  };
  std::ostream & operator<< (std::ostream & out, const Mesh & mesh);
  class NodeExtrudeCoefficient:public VectorCoefficient
  {
  private:
    int n, layer;
    double p[2], s;
    Vector tip;
  public:
      NodeExtrudeCoefficient (const int dim, const int _n, const double _s);
    void SetLayer (const int l)
    {
      layer = l;
    }
    using VectorCoefficient::Eval;
    virtual void Eval (Vector & V, ElementTransformation & T,
		       const IntegrationPoint & ip);
    virtual ~ NodeExtrudeCoefficient ()
    {
    }
  };
  Mesh *Extrude1D (Mesh * mesh, const int ny, const double sy,
		   const bool closed = false);
  inline void Mesh::ShiftL2R (int &a, int &b, int &c)
  {
    int t = a;
    a = c;
    c = b;
    b = t;
  }
  inline void Mesh::Rotate3 (int &a, int &b, int &c)
  {
    if (a < b)
      {
	if (a > c)
	  ShiftL2R (a, b, c);
      }
    else
      {
	if (b < c)
	  ShiftL2R (c, b, a);
	else
	  ShiftL2R (a, b, c);
      }
  }
}

namespace mfem
{
  class GridFunction;
  class KnotVector
  {
  protected:
    static const int MaxOrder;
    Vector knot;
    int Order, NumOfControlPoints, NumOfElements;
  public:
      KnotVector ()
    {
    }
    KnotVector (std::istream & input);
      KnotVector (int Order_, int NCP);
      KnotVector (const KnotVector & kv)
    {
      (*this) = kv;
    }
    KnotVector & operator= (const KnotVector & kv);
    int GetNE () const
    {
      return NumOfElements;
    }
    int GetNKS () const
    {
      return NumOfControlPoints - Order;
    }
    int GetNCP () const
    {
      return NumOfControlPoints;
    }
    int GetOrder () const
    {
      return Order;
    }
    int Size () const
    {
      return knot.Size ();
    }
    void GetElements ();
    bool isElement (int i) const
    {
      return (knot (Order + i) != knot (Order + i + 1));
    }
    double getKnotLocation (double xi, int ni) const
    {
      return (xi * knot (ni + 1) + (1. - xi) * knot (ni));
    }
    int findKnotSpan (double u) const;
    void CalcShape (Vector & shape, int i, double xi);
    void CalcDShape (Vector & grad, int i, double xi);
    void Difference (const KnotVector & kv, Vector & diff) const;
    void UniformRefinement (Vector & newknots) const;
    KnotVector *DegreeElevate (int t) const;
    void Flip ();
    void Print (std::ostream & out) const;
    ~KnotVector ()
    {
    }
    double &operator[] (int i)
    {
      return knot (i);
    }
    const double &operator[] (int i) const
    {
      return knot (i);
    }
  };
  class NURBSPatch
  {
  protected:
    int ni, nj, nk, Dim;
    double *data;
      Array < KnotVector * >kv;
    int sd, nd;
    void swap (NURBSPatch * np);
    int SetLoopDirection (int dir);
    inline double &operator () (int i, int j);
    inline const double &operator () (int i, int j) const;
    void init (int dim_);
      NURBSPatch (NURBSPatch * parent, int dir, int Order, int NCP);
  public:
      NURBSPatch (std::istream & input);
      NURBSPatch (KnotVector * kv0, KnotVector * kv1, int dim_);
      NURBSPatch (KnotVector * kv0, KnotVector * kv1, KnotVector * kv2,
		  int dim_);
      NURBSPatch (Array < KnotVector * >&kv, int dim_);
     ~NURBSPatch ();
    void Print (std::ostream & out);
    void DegreeElevate (int dir, int t);
    void KnotInsert (int dir, const KnotVector & knot);
    void KnotInsert (int dir, const Vector & knot);
    void KnotInsert (Array < KnotVector * >&knot);
    void DegreeElevate (int t);
    void UniformRefinement ();
    KnotVector *GetKV (int i)
    {
      return kv[i];
    }
    inline double &operator () (int i, int j, int l);
    inline const double &operator () (int i, int j, int l) const;
    inline double &operator () (int i, int j, int k, int l);
    inline const double &operator () (int i, int j, int k, int l) const;
    static void Get3DRotationMatrix (double n[], double angle, double r,
				     DenseMatrix & T);
    void FlipDirection (int dir);
    void SwapDirections (int dir1, int dir2);
    void Rotate3D (double normal[], double angle);
    int MakeUniformDegree ();
    friend NURBSPatch *Interpolate (NURBSPatch & p1, NURBSPatch & p2);
    friend NURBSPatch *Revolve3D (NURBSPatch & patch, double n[], double ang,
				  int times);
  };
  class NURBSPatchMap;
  class NURBSExtension
  {
    friend class NURBSPatchMap;
  protected:
    int Order;
    int NumOfKnotVectors;
    int NumOfVertices, NumOfElements, NumOfBdrElements, NumOfDofs;
    int NumOfActiveVertices, NumOfActiveElems, NumOfActiveBdrElems;
    int NumOfActiveDofs;
      Array < int >activeVert;
      Array < bool > activeElem;
      Array < bool > activeBdrElem;
      Array < int >activeDof;
    Mesh *patchTopo;
    int own_topo;
      Array < int >edge_to_knot;
      Array < KnotVector * >knotVectors;
    Vector weights;
      Array < int >v_meshOffsets;
      Array < int >e_meshOffsets;
      Array < int >f_meshOffsets;
      Array < int >p_meshOffsets;
      Array < int >v_spaceOffsets;
      Array < int >e_spaceOffsets;
      Array < int >f_spaceOffsets;
      Array < int >p_spaceOffsets;
    Table *el_dof, *bel_dof;
      Array < int >el_to_patch;
      Array < int >bel_to_patch;
      Array2D < int >el_to_IJK;
      Array2D < int >bel_to_IJK;
      Array < NURBSPatch * >patches;
    inline int KnotInd (int edge);
    inline KnotVector *KnotVec (int edge);
    inline KnotVector *KnotVec (int edge, int oedge, int *okv);
    void CheckPatches ();
    void CheckBdrPatches ();
    void GetPatchKnotVectors (int p, Array < KnotVector * >&kv);
    void GetBdrPatchKnotVectors (int p, Array < KnotVector * >&kv);
    void GenerateOffsets ();
    void CountElements ();
    void CountBdrElements ();
    void Get2DElementTopo (Array < Element * >&elements);
    void Get3DElementTopo (Array < Element * >&elements);
    void Get2DBdrElementTopo (Array < Element * >&boundary);
    void Get3DBdrElementTopo (Array < Element * >&boundary);
    void GenerateElementDofTable ();
    void Generate2DElementDofTable ();
    void Generate3DElementDofTable ();
    void GenerateBdrElementDofTable ();
    void Generate2DBdrElementDofTable ();
    void Generate3DBdrElementDofTable ();
    void GetPatchNets (const Vector & Nodes);
    void Get2DPatchNets (const Vector & Nodes);
    void Get3DPatchNets (const Vector & Nodes);
    void SetSolutionVector (Vector & Nodes);
    void Set2DSolutionVector (Vector & Nodes);
    void Set3DSolutionVector (Vector & Nodes);
    void GenerateActiveVertices ();
    void GenerateActiveBdrElems ();
    void MergeWeights (Mesh * mesh_array[], int num_pieces);
      NURBSExtension ()
    {
    }
  public:
      NURBSExtension (std::istream & input);
    NURBSExtension (NURBSExtension * parent, int Order);
    NURBSExtension (Mesh * mesh_array[], int num_pieces);
    void MergeGridFunctions (GridFunction * gf_array[], int num_pieces,
			     GridFunction & merged);
    virtual ~ NURBSExtension ();
    void Print (std::ostream & out) const;
    void PrintCharacteristics (std::ostream & out);
    int Dimension ()
    {
      return patchTopo->Dimension ();
    }
    int GetNP ()
    {
      return patchTopo->GetNE ();
    }
    int GetNBP ()
    {
      return patchTopo->GetNBE ();
    }
    int GetOrder ()
    {
      return Order;
    }
    int GetNKV ()
    {
      return NumOfKnotVectors;
    }
    int GetGNV ()
    {
      return NumOfVertices;
    }
    int GetNV ()
    {
      return NumOfActiveVertices;
    }
    int GetGNE ()
    {
      return NumOfElements;
    }
    int GetNE ()
    {
      return NumOfActiveElems;
    }
    int GetGNBE ()
    {
      return NumOfBdrElements;
    }
    int GetNBE ()
    {
      return NumOfActiveBdrElems;
    }
    int GetNTotalDof ()
    {
      return NumOfDofs;
    }
    int GetNDof ()
    {
      return NumOfActiveDofs;
    }
    const KnotVector *GetKnotVector (int i) const
    {
      return knotVectors[i];
    }
    void GetElementTopo (Array < Element * >&elements);
    void GetBdrElementTopo (Array < Element * >&boundary);
    bool HavePatches ()
    {
      return (patches.Size () != 0);
    }
    Table *GetElementDofTable ()
    {
      return el_dof;
    }
    Table *GetBdrElementDofTable ()
    {
      return bel_dof;
    }
    void GetVertexLocalToGlobal (Array < int >&lvert_vert);
    void GetElementLocalToGlobal (Array < int >&lelem_elem);
    void LoadFE (int i, const FiniteElement * FE);
    void LoadBE (int i, const FiniteElement * BE);
    const Vector & GetWeights () const
    {
      return weights;
    }
    Vector & GetWeights ()
    {
      return weights;
    }
    void ConvertToPatches (const Vector & Nodes);
    void SetKnotsFromPatches ();
    void SetCoordsFromPatches (Vector & Nodes);
    void DegreeElevate (int t);
    void UniformRefinement ();
    void KnotInsert (Array < KnotVector * >&kv);
  };
  class NURBSPatchMap
  {
  private:
    NURBSExtension * Ext;
    int I, J, K, pOffset, opatch;
      Array < int >verts, edges, faces, oedge, oface;
    inline static int F (const int n, const int N)
    {
      return (n < 0) ? 0 : ((n >= N) ? 2 : 1);
    }
    inline static int Or1D (const int n, const int N, const int Or)
    {
      return (Or > 0) ? n : (N - 1 - n);
    }
    inline static int Or2D (const int n1, const int n2,
			    const int N1, const int N2, const int Or);
    void GetPatchKnotVectors (int p, KnotVector * kv[]);
    void GetBdrPatchKnotVectors (int p, KnotVector * kv[], int *okv);
  public:
    NURBSPatchMap (NURBSExtension * ext)
    {
      Ext = ext;
    }
    int nx ()
    {
      return I + 1;
    }
    int ny ()
    {
      return J + 1;
    }
    int nz ()
    {
      return K + 1;
    }
    void SetPatchVertexMap (int p, KnotVector * kv[]);
    void SetPatchDofMap (int p, KnotVector * kv[]);
    void SetBdrPatchVertexMap (int p, KnotVector * kv[], int *okv);
    void SetBdrPatchDofMap (int p, KnotVector * kv[], int *okv);
    inline int operator () (const int i) const;
    inline int operator[] (const int i) const
    {
      return (*this) (i);
    }
    inline int operator () (const int i, const int j) const;
    inline int operator () (const int i, const int j, const int k) const;
  };
  inline double &NURBSPatch::operator () (int i, int j)
  {
    return data[j % sd + sd * (i + (j / sd) * nd)];
  }
  inline const double &NURBSPatch::operator () (int i, int j) const
  {
    return data[j % sd + sd * (i + (j / sd) * nd)];
  }
  inline double &NURBSPatch::operator () (int i, int j, int l)
  {
    return data[(i + j * ni) * Dim + l];
  }
  inline const double &NURBSPatch::operator () (int i, int j, int l) const
  {
    return data[(i + j * ni) * Dim + l];
  }
  inline double &NURBSPatch::operator () (int i, int j, int k, int l)
  {
    return data[(i + (j + k * nj) * ni) * Dim + l];
  }
  inline const double &NURBSPatch::operator () (int i, int j, int k, int l) const
  {
    return data[(i + (j + k * nj) * ni) * Dim + l];
  }
  inline int NURBSExtension::KnotInd (int edge)
  {
    int kv = edge_to_knot[edge];
    return (kv >= 0) ? kv : (-1 - kv);
  }
  inline KnotVector *NURBSExtension::KnotVec (int edge)
  {
    return knotVectors[KnotInd (edge)];
  }
  inline KnotVector *NURBSExtension::KnotVec (int edge, int oedge, int *okv)
  {
    int kv = edge_to_knot[edge];
    if (kv >= 0)
      {
	*okv = oedge;
	return knotVectors[kv];
      }
    else
      {
	*okv = -oedge;
	return knotVectors[-1 - kv];
      }
  }
  inline int NURBSPatchMap::Or2D (const int n1, const int n2,
				  const int N1, const int N2, const int Or)
  {
    switch (Or)
      {
      case 0:
	return n1 + n2 * N1;
      case 1:
	return n2 + n1 * N2;
      case 2:
	return n2 + (N1 - 1 - n1) * N2;
      case 3:
	return (N1 - 1 - n1) + n2 * N1;
      case 4:
	return (N1 - 1 - n1) + (N2 - 1 - n2) * N1;
      case 5:
	return (N2 - 1 - n2) + (N1 - 1 - n1) * N2;
      case 6:
	return (N2 - 1 - n2) + n1 * N2;
      case 7:
	return n1 + (N2 - 1 - n2) * N1;
      }
    return -1;
  }
  inline int NURBSPatchMap::operator () (const int i) const
  {
    int i1 = i - 1;
    switch (F (i1, I))
      {
      case 0:
	return verts[0];
	case 1:return pOffset + Or1D (i1, I, opatch);
	case 2:return verts[1];
      }
    return -1;
  }
  inline int NURBSPatchMap::operator () (const int i, const int j) const
  {
    int i1 = i - 1, j1 = j - 1;
    switch (3 * F (j1, J) + F (i1, I))
      {
      case 0:
	return verts[0];
	case 1:return edges[0] + Or1D (i1, I, oedge[0]);
	case 2:return verts[1];
	case 3:return edges[3] + Or1D (j1, J, -oedge[3]);
	case 4:return pOffset + Or2D (i1, j1, I, J, opatch);
	case 5:return edges[1] + Or1D (j1, J, oedge[1]);
	case 6:return verts[3];
	case 7:return edges[2] + Or1D (i1, I, -oedge[2]);
	case 8:return verts[2];
      }
    return -1;
  }
  inline int NURBSPatchMap::operator () (const int i, const int j,
					 const int k) const
  {
    int i1 = i - 1, j1 = j - 1, k1 = k - 1;
    switch (3 * (3 * F (k1, K) + F (j1, J)) + F (i1, I))
      {
      case 0:
	return verts[0];
	case 1:return edges[0] + Or1D (i1, I, oedge[0]);
	case 2:return verts[1];
	case 3:return edges[3] + Or1D (j1, J, oedge[3]);
	case 4:return faces[0] + Or2D (i1, J - 1 - j1, I, J, oface[0]);
	case 5:return edges[1] + Or1D (j1, J, oedge[1]);
	case 6:return verts[3];
	case 7:return edges[2] + Or1D (i1, I, oedge[2]);
	case 8:return verts[2];
	case 9:return edges[8] + Or1D (k1, K, oedge[8]);
	case 10:return faces[1] + Or2D (i1, k1, I, K, oface[1]);
	case 11:return edges[9] + Or1D (k1, K, oedge[9]);
	case 12:return faces[4] + Or2D (J - 1 - j1, k1, J, K, oface[4]);
	case 13:return pOffset + I * (J * k1 + j1) + i1;
	case 14:return faces[2] + Or2D (j1, k1, J, K, oface[2]);
	case 15:return edges[11] + Or1D (k1, K, oedge[11]);
	case 16:return faces[3] + Or2D (I - 1 - i1, k1, I, K, oface[3]);
	case 17:return edges[10] + Or1D (k1, K, oedge[10]);
	case 18:return verts[4];
	case 19:return edges[4] + Or1D (i1, I, oedge[4]);
	case 20:return verts[5];
	case 21:return edges[7] + Or1D (j1, J, oedge[7]);
	case 22:return faces[5] + Or2D (i1, j1, I, J, oface[5]);
	case 23:return edges[5] + Or1D (j1, J, oedge[5]);
	case 24:return verts[7];
	case 25:return edges[6] + Or1D (i1, I, oedge[6]);
	case 26:return verts[6];
      }
    return -1;
  }
}

namespace mfem
{
  class FiniteElementCollection
  {
  public:
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType) const
      = 0;
    virtual int DofForGeometry (int GeomType) const = 0;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const = 0;
    virtual const char *Name () const
    {
      return "Undefined";
    }
    int HasFaceDofs (int GeomType) const;
    virtual const FiniteElement *TraceFiniteElementForGeometry (int GeomType) const
    {
      return FiniteElementForGeometry (GeomType);
    }
    virtual ~ FiniteElementCollection ()
    {
    }
    static FiniteElementCollection *New (const char *name);
  };
  class H1_FECollection:public FiniteElementCollection
  {
  private:
    char h1_name[32];
    FiniteElement *H1_Elements[Geometry::NumGeom];
    int H1_dof[Geometry::NumGeom];
    int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8];
  public:
      explicit H1_FECollection (const int p, const int dim =
				3, const int type = 0);
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType) const
    {
      return H1_Elements[GeomType];
    }
    virtual int DofForGeometry (int GeomType) const
    {
      return H1_dof[GeomType];
    }
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return h1_name;
    }
    virtual ~ H1_FECollection ();
  };
  class H1Pos_FECollection:public H1_FECollection
  {
  public:
  explicit H1Pos_FECollection (const int p, const int dim = 3):H1_FECollection (p, dim,
		     1)
    {
    }
  };
  class L2_FECollection:public FiniteElementCollection
  {
  private:
    char d_name[32];
    FiniteElement *L2_Elements[Geometry::NumGeom];
    FiniteElement *Tr_Elements[Geometry::NumGeom];
    int *SegDofOrd[2];
    int *TriDofOrd[6];
  public:
      L2_FECollection (const int p, const int dim, const int type = 0);
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType) const
    {
      return L2_Elements[GeomType];
    }
    virtual int DofForGeometry (int GeomType) const
    {
      if (L2_Elements[GeomType])
	return L2_Elements[GeomType]->GetDof ();
      return 0;
    }
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return d_name;
    }
    virtual const FiniteElement *TraceFiniteElementForGeometry (int GeomType) const
    {
      return Tr_Elements[GeomType];
    }
    virtual ~ L2_FECollection ();
  };
  typedef L2_FECollection DG_FECollection;
  class RT_FECollection:public FiniteElementCollection
  {
  protected:
    char rt_name[32];
    FiniteElement *RT_Elements[Geometry::NumGeom];
    int RT_dof[Geometry::NumGeom];
    int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8];
    void InitFaces (const int p, const int dim, const int map_type);
      RT_FECollection (const int p, const int dim, const int map_type)
    {
      InitFaces (p, dim, map_type);
    }
  public:
      RT_FECollection (const int p, const int dim);
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType) const
    {
      return RT_Elements[GeomType];
    }
    virtual int DofForGeometry (int GeomType) const
    {
      return RT_dof[GeomType];
    }
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return rt_name;
    }
    virtual ~ RT_FECollection ();
  };
  class RT_Trace_FECollection:public RT_FECollection
  {
  public:
    RT_Trace_FECollection (const int p, const int dim,
			   const int map_type = FiniteElement::INTEGRAL);
  };
  class ND_FECollection:public FiniteElementCollection
  {
  private:
    char nd_name[32];
    FiniteElement *ND_Elements[Geometry::NumGeom];
    int ND_dof[Geometry::NumGeom];
    int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8];
  public:
      ND_FECollection (const int p, const int dim);
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType) const
    {
      return ND_Elements[GeomType];
    }
    virtual int DofForGeometry (int GeomType) const
    {
      return ND_dof[GeomType];
    }
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return nd_name;
    }
    virtual ~ ND_FECollection ();
  };
  class NURBSFECollection:public FiniteElementCollection
  {
  private:
    NURBS1DFiniteElement * SegmentFE;
    NURBS2DFiniteElement *QuadrilateralFE;
    NURBS3DFiniteElement *ParallelepipedFE;
    char name[16];
    void Allocate (int Order);
    void Deallocate ();
  public:
      explicit NURBSFECollection (int Order)
    {
      Allocate (Order);
    }
    int GetOrder () const
    {
      return SegmentFE->GetOrder ();
    }
    void UpdateOrder (int Order)
    {
      Deallocate ();
      Allocate (Order);
    }
    void Reset () const
    {
      SegmentFE->Reset ();
      QuadrilateralFE->Reset ();
      ParallelepipedFE->Reset ();
    }
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return name;
    }
    virtual ~ NURBSFECollection ()
    {
      Deallocate ();
    }
  };
  class LinearFECollection:public FiniteElementCollection
  {
  private:
    const PointFiniteElement PointFE;
    const Linear1DFiniteElement SegmentFE;
    const Linear2DFiniteElement TriangleFE;
    const BiLinear2DFiniteElement QuadrilateralFE;
    const Linear3DFiniteElement TetrahedronFE;
    const TriLinear3DFiniteElement ParallelepipedFE;
  public:
      LinearFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "Linear";
    };
  };
  class QuadraticFECollection:public FiniteElementCollection
  {
  private:
    const PointFiniteElement PointFE;
    const Quad1DFiniteElement SegmentFE;
    const Quad2DFiniteElement TriangleFE;
    const BiQuad2DFiniteElement QuadrilateralFE;
    const Quadratic3DFiniteElement TetrahedronFE;
    const LagrangeHexFiniteElement ParallelepipedFE;
  public:
      QuadraticFECollection ():ParallelepipedFE (2)
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "Quadratic";
    };
  };
  class QuadraticPosFECollection:public FiniteElementCollection
  {
  private:
    const QuadPos1DFiniteElement SegmentFE;
    const BiQuadPos2DFiniteElement QuadrilateralFE;
  public:
      QuadraticPosFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "QuadraticPos";
    };
  };
  class CubicFECollection:public FiniteElementCollection
  {
  private:
    const PointFiniteElement PointFE;
    const Cubic1DFiniteElement SegmentFE;
    const Cubic2DFiniteElement TriangleFE;
    const BiCubic2DFiniteElement QuadrilateralFE;
    const Cubic3DFiniteElement TetrahedronFE;
    const LagrangeHexFiniteElement ParallelepipedFE;
  public:
      CubicFECollection ():ParallelepipedFE (3)
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "Cubic";
    };
  };
  class CrouzeixRaviartFECollection:public FiniteElementCollection
  {
  private:
    const P0SegmentFiniteElement SegmentFE;
    const CrouzeixRaviartFiniteElement TriangleFE;
    const CrouzeixRaviartQuadFiniteElement QuadrilateralFE;
  public:
      CrouzeixRaviartFECollection ():SegmentFE (1)
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "CrouzeixRaviart";
    };
  };
  class LinearNonConf3DFECollection:public FiniteElementCollection
  {
  private:
    const P0TriangleFiniteElement TriangleFE;
    const P1TetNonConfFiniteElement TetrahedronFE;
    const P0QuadFiniteElement QuadrilateralFE;
    const RotTriLinearHexFiniteElement ParallelepipedFE;
  public:
      LinearNonConf3DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "LinearNonConf3D";
    };
  };
  class RT0_2DFECollection:public FiniteElementCollection
  {
  private:
    const P0SegmentFiniteElement SegmentFE;
    const RT0TriangleFiniteElement TriangleFE;
    const RT0QuadFiniteElement QuadrilateralFE;
  public:
      RT0_2DFECollection ():SegmentFE (0)
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "RT0_2D";
    };
  };
  class RT1_2DFECollection:public FiniteElementCollection
  {
  private:
    const P1SegmentFiniteElement SegmentFE;
    const RT1TriangleFiniteElement TriangleFE;
    const RT1QuadFiniteElement QuadrilateralFE;
  public:
      RT1_2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "RT1_2D";
    };
  };
  class RT2_2DFECollection:public FiniteElementCollection
  {
  private:
    const P2SegmentFiniteElement SegmentFE;
    const RT2TriangleFiniteElement TriangleFE;
    const RT2QuadFiniteElement QuadrilateralFE;
  public:
      RT2_2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "RT2_2D";
    };
  };
  class Const2DFECollection:public FiniteElementCollection
  {
  private:
    const P0TriangleFiniteElement TriangleFE;
    const P0QuadFiniteElement QuadrilateralFE;
  public:
      Const2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "Const2D";
    };
  };
  class LinearDiscont2DFECollection:public FiniteElementCollection
  {
  private:
    const Linear2DFiniteElement TriangleFE;
    const BiLinear2DFiniteElement QuadrilateralFE;
  public:
      LinearDiscont2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "LinearDiscont2D";
    };
  };
  class GaussLinearDiscont2DFECollection:public FiniteElementCollection
  {
  private:
    const GaussLinear2DFiniteElement TriangleFE;
    const GaussBiLinear2DFiniteElement QuadrilateralFE;
  public:
      GaussLinearDiscont2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "GaussLinearDiscont2D";
    };
  };
  class P1OnQuadFECollection:public FiniteElementCollection
  {
  private:
    const P1OnQuadFiniteElement QuadrilateralFE;
  public:
      P1OnQuadFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "P1OnQuad";
    };
  };
  class QuadraticDiscont2DFECollection:public FiniteElementCollection
  {
  private:
    const Quad2DFiniteElement TriangleFE;
    const BiQuad2DFiniteElement QuadrilateralFE;
  public:
      QuadraticDiscont2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "QuadraticDiscont2D";
    };
  };
  class QuadraticPosDiscont2DFECollection:public FiniteElementCollection
  {
  private:
    const BiQuadPos2DFiniteElement QuadrilateralFE;
  public:
      QuadraticPosDiscont2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const
    {
      return __null;
    };
    virtual const char *Name () const
    {
      return "QuadraticPosDiscont2D";
    };
  };
  class GaussQuadraticDiscont2DFECollection:public FiniteElementCollection
  {
  private:
    const GaussQuad2DFiniteElement TriangleFE;
    const GaussBiQuad2DFiniteElement QuadrilateralFE;
  public:
      GaussQuadraticDiscont2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "GaussQuadraticDiscont2D";
    };
  };
  class CubicDiscont2DFECollection:public FiniteElementCollection
  {
  private:
    const Cubic2DFiniteElement TriangleFE;
    const BiCubic2DFiniteElement QuadrilateralFE;
  public:
      CubicDiscont2DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "CubicDiscont2D";
    };
  };
  class Const3DFECollection:public FiniteElementCollection
  {
  private:
    const P0TetFiniteElement TetrahedronFE;
    const P0HexFiniteElement ParallelepipedFE;
  public:
      Const3DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "Const3D";
    };
  };
  class LinearDiscont3DFECollection:public FiniteElementCollection
  {
  private:
    const Linear3DFiniteElement TetrahedronFE;
    const TriLinear3DFiniteElement ParallelepipedFE;
  public:
      LinearDiscont3DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "LinearDiscont3D";
    };
  };
  class QuadraticDiscont3DFECollection:public FiniteElementCollection
  {
  private:
    const Quadratic3DFiniteElement TetrahedronFE;
    const LagrangeHexFiniteElement ParallelepipedFE;
  public:
      QuadraticDiscont3DFECollection ():ParallelepipedFE (2)
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "QuadraticDiscont3D";
    };
  };
  class RefinedLinearFECollection:public FiniteElementCollection
  {
  private:
    const PointFiniteElement PointFE;
    const RefinedLinear1DFiniteElement SegmentFE;
    const RefinedLinear2DFiniteElement TriangleFE;
    const RefinedBiLinear2DFiniteElement QuadrilateralFE;
    const RefinedLinear3DFiniteElement TetrahedronFE;
    const RefinedTriLinear3DFiniteElement ParallelepipedFE;
  public:
      RefinedLinearFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "RefinedLinear";
    };
  };
  class ND1_3DFECollection:public FiniteElementCollection
  {
  private:
    const Nedelec1HexFiniteElement HexahedronFE;
    const Nedelec1TetFiniteElement TetrahedronFE;
  public:
      ND1_3DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "ND1_3D";
    };
  };
  class RT0_3DFECollection:public FiniteElementCollection
  {
  private:
    const P0TriangleFiniteElement TriangleFE;
    const P0QuadFiniteElement QuadrilateralFE;
    const RT0HexFiniteElement HexahedronFE;
    const RT0TetFiniteElement TetrahedronFE;
  public:
      RT0_3DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "RT0_3D";
    };
  };
  class RT1_3DFECollection:public FiniteElementCollection
  {
  private:
    const Linear2DFiniteElement TriangleFE;
    const BiLinear2DFiniteElement QuadrilateralFE;
    const RT1HexFiniteElement HexahedronFE;
  public:
      RT1_3DFECollection ()
    {
    };
    virtual const FiniteElement *FiniteElementForGeometry (int GeomType)
      const;
    virtual int DofForGeometry (int GeomType) const;
    virtual int *DofOrderForOrientation (int GeomType, int Or) const;
    virtual const char *Name () const
    {
      return "RT1_3D";
    };
  };
  class Local_FECollection:public FiniteElementCollection
  {
  private:
    char d_name[32];
    int GeomType;
    FiniteElement *Local_Element;
  public:
      Local_FECollection (const char *fe_name);
    virtual const FiniteElement *FiniteElementForGeometry (int _GeomType) const
    {
      return (GeomType == _GeomType) ? Local_Element : __null;
    }
    virtual int DofForGeometry (int _GeomType) const
    {
      return (GeomType == _GeomType) ? Local_Element->GetDof () : 0;
    }
    virtual int *DofOrderForOrientation (int GeomType, int Or) const
    {
      return __null;
    }
    virtual const char *Name () const
    {
      return d_name;
    }
    virtual ~ Local_FECollection ()
    {
      delete Local_Element;
    }
  };
}

namespace mfem
{
  class LinearFormIntegrator
  {
  protected:
    const IntegrationRule *IntRule;
      LinearFormIntegrator (const IntegrationRule * ir = __null)
    {
      IntRule = ir;
    }
  public:
      virtual void AssembleRHSElementVect (const FiniteElement & el,
					   ElementTransformation & Tr,
					   Vector & elvect) = 0;
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 FaceElementTransformations & Tr,
					 Vector & elvect);
    void SetIntRule (const IntegrationRule * ir)
    {
      IntRule = ir;
    }
    virtual ~ LinearFormIntegrator ()
    {
    };
  };
  class DomainLFIntegrator:public LinearFormIntegrator
  {
    Vector shape;
      Coefficient & Q;
    int oa, ob;
  public:
    DomainLFIntegrator (Coefficient & QF, int a = 2, int b = 0):Q (QF), oa (a),
      ob (b)
    {
    }
    DomainLFIntegrator (Coefficient & QF,
			const IntegrationRule * ir):LinearFormIntegrator (ir),
      Q (QF), oa (1), ob (1)
    {
    }
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class BoundaryLFIntegrator:public LinearFormIntegrator
  {
    Vector shape;
      Coefficient & Q;
    int oa, ob;
  public:
    BoundaryLFIntegrator (Coefficient & QG, int a = 1, int b = 1):Q (QG), oa (a),
      ob
      (b)
    {
    };
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class BoundaryNormalLFIntegrator:public LinearFormIntegrator
  {
    Vector shape;
      VectorCoefficient & Q;
    int oa, ob;
  public:
    BoundaryNormalLFIntegrator (VectorCoefficient & QG, int a = 1, int b = 1):Q (QG), oa (a),
      ob
      (b)
    {
    };
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class BoundaryTangentialLFIntegrator:public LinearFormIntegrator
  {
    Vector shape;
      VectorCoefficient & Q;
    int oa, ob;
  public:
    BoundaryTangentialLFIntegrator (VectorCoefficient & QG, int a = 1, int b = 1):Q (QG), oa (a),
      ob
      (b)
    {
    };
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class VectorDomainLFIntegrator:public LinearFormIntegrator
  {
  private:
    Vector shape, Qvec;
    VectorCoefficient & Q;
  public:
    VectorDomainLFIntegrator (VectorCoefficient & QF):Q (QF)
    {
    };
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class VectorBoundaryLFIntegrator:public LinearFormIntegrator
  {
  private:
    Vector shape, vec;
    VectorCoefficient & Q;
  public:
    VectorBoundaryLFIntegrator (VectorCoefficient & QG):Q (QG)
    {
    };
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class VectorFEDomainLFIntegrator:public LinearFormIntegrator
  {
  private:
    VectorCoefficient & QF;
    DenseMatrix vshape;
    Vector vec;
  public:
      VectorFEDomainLFIntegrator (VectorCoefficient & F):QF (F)
    {
    }
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class VectorBoundaryFluxLFIntegrator:public LinearFormIntegrator
  {
  private:
    double Sign;
    Coefficient *F;
    Vector shape, nor;
  public:
    VectorBoundaryFluxLFIntegrator (Coefficient & f, double s = 1.0, const IntegrationRule * ir = __null):LinearFormIntegrator (ir), Sign (s),
      F
      (&f)
    {
    }
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class VectorFEBoundaryFluxLFIntegrator:public LinearFormIntegrator
  {
  private:
    Coefficient & F;
    Vector shape;
  public:
      VectorFEBoundaryFluxLFIntegrator (Coefficient & f):F (f)
    {
    }
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class VectorFEBoundaryTangentLFIntegrator:public LinearFormIntegrator
  {
  private:
    VectorCoefficient & f;
  public:
    VectorFEBoundaryTangentLFIntegrator (VectorCoefficient & QG):f (QG)
    {
    }
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    using LinearFormIntegrator::AssembleRHSElementVect;
  };
  class BoundaryFlowIntegrator:public LinearFormIntegrator
  {
  private:
    Coefficient * f;
    VectorCoefficient *u;
    double alpha, beta;
    Vector shape;
  public:
      BoundaryFlowIntegrator (Coefficient & _f, VectorCoefficient & _u,
			      double a, double b)
    {
      f = &_f;
      u = &_u;
      alpha = a;
      beta = b;
    }
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 FaceElementTransformations & Tr,
					 Vector & elvect);
  };
  class DGDirichletLFIntegrator:public LinearFormIntegrator
  {
  protected:
    Coefficient * uD, *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;
    Vector shape, dshape_dn, nor, nh, ni;
    DenseMatrix dshape, mq, adjJ;
  public:
      DGDirichletLFIntegrator (Coefficient & u, const double s,
			       const double k):uD (&u), Q (__null),
      MQ (__null), sigma (s), kappa (k)
    {
    }
    DGDirichletLFIntegrator (Coefficient & u, Coefficient & q,
			     const double s, const double k):uD (&u), Q (&q),
      MQ (__null), sigma (s), kappa (k)
    {
    }
    DGDirichletLFIntegrator (Coefficient & u, MatrixCoefficient & q,
			     const double s, const double k):uD (&u),
      Q (__null), MQ (&q), sigma (s), kappa (k)
    {
    }
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 ElementTransformation & Tr,
					 Vector & elvect);
    virtual void AssembleRHSElementVect (const FiniteElement & el,
					 FaceElementTransformations & Tr,
					 Vector & elvect);
  };
}

namespace mfem
{
  class NonlinearFormIntegrator
  {
  public:
    virtual void AssembleElementVector (const FiniteElement & el,
					ElementTransformation & Tr,
					const Vector & elfun,
					Vector & elvect) = 0;
    virtual void AssembleElementGrad (const FiniteElement & el,
				      ElementTransformation & Tr,
				      const Vector & elfun,
				      DenseMatrix & elmat);
    virtual double GetElementEnergy (const FiniteElement & el,
				     ElementTransformation & Tr,
				     const Vector & elfun);
      virtual ~ NonlinearFormIntegrator ()
    {
    }
  };
  class HyperelasticModel
  {
  protected:
    ElementTransformation * T;
  public:
    HyperelasticModel ()
    {
      T = __null;
    }
    void SetTransformation (ElementTransformation & _T)
    {
      T = &_T;
    }
    virtual double EvalW (const DenseMatrix & J) const = 0;
    virtual void EvalP (const DenseMatrix & J, DenseMatrix & P) const = 0;
    virtual void AssembleH (const DenseMatrix & J, const DenseMatrix & DS,
			    const double weight, DenseMatrix & A) const = 0;
    virtual ~ HyperelasticModel ()
    {
    }
  };
  class InverseHarmonicModel:public HyperelasticModel
  {
  protected:
    mutable DenseMatrix Z, S;
    mutable DenseMatrix G, C;
  public:
      virtual double EvalW (const DenseMatrix & J) const;
    virtual void EvalP (const DenseMatrix & J, DenseMatrix & P) const;
    virtual void AssembleH (const DenseMatrix & J, const DenseMatrix & DS,
			    const double weight, DenseMatrix & A) const;
  };
  class NeoHookeanModel:public HyperelasticModel
  {
  protected:
    mutable double mu, K, g;
    Coefficient *c_mu, *c_K, *c_g;
    bool have_coeffs;
    mutable DenseMatrix Z;
    mutable DenseMatrix G, C;
    inline void EvalCoeffs () const;
  public:
    NeoHookeanModel (double _mu, double _K, double _g = 1.0):mu (_mu), K (_K), g (_g),
      have_coeffs
      (false)
    {
      c_mu = c_K = c_g = __null;
    }
  NeoHookeanModel (Coefficient & _mu, Coefficient & _K, Coefficient * _g = __null):mu (0.0), K (0.0), g (1.0), c_mu (&_mu), c_K (&_K), c_g (_g),
      have_coeffs
      (true)
    {
    }
    virtual double EvalW (const DenseMatrix & J) const;
    virtual void EvalP (const DenseMatrix & J, DenseMatrix & P) const;
    virtual void AssembleH (const DenseMatrix & J, const DenseMatrix & DS,
			    const double weight, DenseMatrix & A) const;
  };
  class HyperelasticNLFIntegrator:public NonlinearFormIntegrator
  {
  private:
    HyperelasticModel * model;
    DenseMatrix DSh, DS, J0i, J1, J, P, PMatI, PMatO;
  public:
      HyperelasticNLFIntegrator (HyperelasticModel * m):model (m)
    {
    }
    virtual double GetElementEnergy (const FiniteElement & el,
				     ElementTransformation & Tr,
				     const Vector & elfun);
    virtual void AssembleElementVector (const FiniteElement & el,
					ElementTransformation & Tr,
					const Vector & elfun,
					Vector & elvect);
    virtual void AssembleElementGrad (const FiniteElement & el,
				      ElementTransformation & Tr,
				      const Vector & elfun,
				      DenseMatrix & elmat);
    virtual ~ HyperelasticNLFIntegrator ();
  };
}

namespace mfem
{
  class BilinearFormIntegrator:public NonlinearFormIntegrator
  {
  protected:
    const IntegrationRule *IntRule;
      BilinearFormIntegrator (const IntegrationRule * ir = __null)
    {
      IntRule = ir;
    }
  public:
      virtual void AssembleElementMatrix (const FiniteElement & el,
					  ElementTransformation & Trans,
					  DenseMatrix & elmat);
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
    virtual void AssembleFaceMatrix (const FiniteElement & el1,
				     const FiniteElement & el2,
				     FaceElementTransformations & Trans,
				     DenseMatrix & elmat);
    virtual void AssembleFaceMatrix (const FiniteElement & trial_face_fe,
				     const FiniteElement & test_fe1,
				     const FiniteElement & test_fe2,
				     FaceElementTransformations & Trans,
				     DenseMatrix & elmat);
    virtual void AssembleElementVector (const FiniteElement & el,
					ElementTransformation & Tr,
					const Vector & elfun,
					Vector & elvect);
    virtual void AssembleElementGrad (const FiniteElement & el,
				      ElementTransformation & Tr,
				      const Vector & elfun,
				      DenseMatrix & elmat)
    {
      AssembleElementMatrix (el, Tr, elmat);
    }
    virtual void ComputeElementFlux (const FiniteElement & el,
				     ElementTransformation & Trans,
				     Vector & u,
				     const FiniteElement & fluxelem,
				     Vector & flux, int wcoef = 1)
    {
    }
    virtual double ComputeFluxEnergy (const FiniteElement & fluxelem,
				      ElementTransformation & Trans,
				      Vector & flux)
    {
      return 0.0;
    }
    void SetIntRule (const IntegrationRule * ir)
    {
      IntRule = ir;
    }
    virtual ~ BilinearFormIntegrator ()
    {
    }
  };
  class TransposeIntegrator:public BilinearFormIntegrator
  {
  private:
    int own_bfi;
    BilinearFormIntegrator *bfi;
    DenseMatrix bfi_elmat;
  public:
      TransposeIntegrator (BilinearFormIntegrator * _bfi, int _own_bfi = 1)
    {
      bfi = _bfi;
      own_bfi = _own_bfi;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix (const FiniteElement & el1,
				     const FiniteElement & el2,
				     FaceElementTransformations & Trans,
				     DenseMatrix & elmat);
    virtual ~ TransposeIntegrator ()
    {
      if (own_bfi)
	delete bfi;
    }
  };
  class LumpedIntegrator:public BilinearFormIntegrator
  {
  private:
    int own_bfi;
    BilinearFormIntegrator *bfi;
  public:
      LumpedIntegrator (BilinearFormIntegrator * _bfi, int _own_bfi = 1)
    {
      bfi = _bfi;
      own_bfi = _own_bfi;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual ~ LumpedIntegrator ()
    {
      if (own_bfi)
	delete bfi;
    }
  };
  class InverseIntegrator:public BilinearFormIntegrator
  {
  private:
    int own_integrator;
    BilinearFormIntegrator *integrator;
  public:
      InverseIntegrator (BilinearFormIntegrator * integ, int own_integ = 1)
    {
      integrator = integ;
      own_integrator = own_integ;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual ~ InverseIntegrator ()
    {
      if (own_integrator)
	delete integrator;
    }
  };
  class SumIntegrator:public BilinearFormIntegrator
  {
  private:
    int own_integrators;
    DenseMatrix elem_mat;
      Array < BilinearFormIntegrator * >integrators;
  public:
      SumIntegrator (int own_integs = 1)
    {
      own_integrators = own_integs;
    }
    void AddIntegrator (BilinearFormIntegrator * integ)
    {
      integrators.Append (integ);
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual ~ SumIntegrator ();
  };
  class DiffusionIntegrator:public BilinearFormIntegrator
  {
  private:
    Vector vec, pointflux, shape;
    DenseMatrix dshape, dshapedxt, invdfdx, mq;
    DenseMatrix te_dshape, te_dshapedxt;
    Coefficient *Q;
    MatrixCoefficient *MQ;
  public:
      DiffusionIntegrator ()
    {
      Q = __null;
      MQ = __null;
    }
    DiffusionIntegrator (Coefficient & q):Q (&q)
    {
      MQ = __null;
    }
  DiffusionIntegrator (MatrixCoefficient & q):MQ (&q)
    {
      Q = __null;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
    virtual void AssembleElementVector (const FiniteElement & el,
					ElementTransformation & Tr,
					const Vector & elfun,
					Vector & elvect);
    virtual void ComputeElementFlux (const FiniteElement & el,
				     ElementTransformation & Trans,
				     Vector & u,
				     const FiniteElement & fluxelem,
				     Vector & flux, int wcoef);
    virtual double ComputeFluxEnergy (const FiniteElement & fluxelem,
				      ElementTransformation & Trans,
				      Vector & flux);
  };
  class MassIntegrator:public BilinearFormIntegrator
  {
  private:
    Vector shape, te_shape;
    Coefficient *Q;
  public:
    MassIntegrator (const IntegrationRule * ir = __null):BilinearFormIntegrator
      (ir)
    {
      Q = __null;
    }
  MassIntegrator (Coefficient & q, const IntegrationRule * ir = __null):BilinearFormIntegrator (ir),
      Q
      (&q)
    {
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
  };
  class BoundaryMassIntegrator:public MassIntegrator
  {
  public:
    BoundaryMassIntegrator (Coefficient & q):MassIntegrator (q)
    {
    }
  };
  class ConvectionIntegrator:public BilinearFormIntegrator
  {
  private:
    DenseMatrix dshape, adjJ, Q_ir;
    Vector shape, vec2, BdFidxT;
      VectorCoefficient & Q;
    double alpha;
  public:
    ConvectionIntegrator (VectorCoefficient & q, double a = 1.0):Q (q)
    {
      alpha = a;
    }
    virtual void AssembleElementMatrix (const FiniteElement &,
					ElementTransformation &,
					DenseMatrix &);
  };
  class GroupConvectionIntegrator:public BilinearFormIntegrator
  {
  private:
    DenseMatrix dshape, adjJ, Q_nodal, grad;
    Vector shape;
      VectorCoefficient & Q;
    double alpha;
  public:
    GroupConvectionIntegrator (VectorCoefficient & q, double a = 1.0):Q (q)
    {
      alpha = a;
    }
    virtual void AssembleElementMatrix (const FiniteElement &,
					ElementTransformation &,
					DenseMatrix &);
  };
  class VectorMassIntegrator:public BilinearFormIntegrator
  {
  private:
    Vector shape, te_shape, vec;
    DenseMatrix partelmat;
    DenseMatrix mcoeff;
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    int Q_order;
  public:
      VectorMassIntegrator ()
    {
      Q = __null;
      VQ = __null;
      MQ = __null;
      Q_order = 0;
    }
  VectorMassIntegrator (Coefficient & q, int qo = 0):Q (&q)
    {
      VQ = __null;
      MQ = __null;
      Q_order = qo;
    }
    VectorMassIntegrator (Coefficient & q,
			  const IntegrationRule *
			  ir):BilinearFormIntegrator (ir), Q (&q)
    {
      VQ = __null;
      MQ = __null;
      Q_order = 0;
    }
  VectorMassIntegrator (VectorCoefficient & q, int qo = 0):VQ (&q)
    {
      Q = __null;
      MQ = __null;
      Q_order = qo;
    }
  VectorMassIntegrator (MatrixCoefficient & q, int qo = 0):MQ (&q)
    {
      Q = __null;
      VQ = __null;
      Q_order = qo;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
  };
  class VectorFEDivergenceIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient * Q;
    Vector divshape, shape;
  public:
      VectorFEDivergenceIntegrator ()
    {
      Q = __null;
    }
    VectorFEDivergenceIntegrator (Coefficient & q)
    {
      Q = &q;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat)
    {
    }
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
  };
  class VectorFECurlIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient * Q;
    DenseMatrix curlshapeTrial;
    DenseMatrix vshapeTest;
    DenseMatrix curlshapeTrial_dFT;
  public:
      VectorFECurlIntegrator ()
    {
      Q = __null;
    }
    VectorFECurlIntegrator (Coefficient & q)
    {
      Q = &q;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat)
    {
    }
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
  };
  class DerivativeIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient & Q;
    int xi;
    DenseMatrix dshape, dshapedxt, invdfdx;
    Vector shape, dshapedxi;
  public:
      DerivativeIntegrator (Coefficient & q, int i):Q (q), xi (i)
    {
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat)
    {
      AssembleElementMatrix2 (el, el, Trans, elmat);
    }
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
  };
  class CurlCurlIntegrator:public BilinearFormIntegrator
  {
  private:
    DenseMatrix Curlshape, Curlshape_dFt;
    Coefficient *Q;
  public:
      CurlCurlIntegrator ()
    {
      Q = __null;
    }
    CurlCurlIntegrator (Coefficient & q):Q (&q)
    {
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
  };
  class VectorCurlCurlIntegrator:public BilinearFormIntegrator
  {
  private:
    DenseMatrix dshape_hat, dshape, curlshape, Jadj, grad_hat, grad;
    Coefficient *Q;
  public:
      VectorCurlCurlIntegrator ()
    {
      Q = __null;
    }
    VectorCurlCurlIntegrator (Coefficient & q):Q (&q)
    {
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual double GetElementEnergy (const FiniteElement & el,
				     ElementTransformation & Tr,
				     const Vector & elfun);
  };
  class VectorFEMassIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient * Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init (Coefficient * q, VectorCoefficient * vq,
	       MatrixCoefficient * mq)
    {
      Q = q;
      VQ = vq;
      MQ = mq;
    }
    Vector shape;
    Vector D;
    DenseMatrix K;
    DenseMatrix vshape;
  public:
    VectorFEMassIntegrator ()
    {
      Init (__null, __null, __null);
    }
    VectorFEMassIntegrator (Coefficient * _q)
    {
      Init (_q, __null, __null);
    }
    VectorFEMassIntegrator (Coefficient & q)
    {
      Init (&q, __null, __null);
    }
    VectorFEMassIntegrator (VectorCoefficient * _vq)
    {
      Init (__null, _vq, __null);
    }
    VectorFEMassIntegrator (VectorCoefficient & vq)
    {
      Init (__null, &vq, __null);
    }
    VectorFEMassIntegrator (MatrixCoefficient * _mq)
    {
      Init (__null, __null, _mq);
    }
    VectorFEMassIntegrator (MatrixCoefficient & mq)
    {
      Init (__null, __null, &mq);
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
  };
  class VectorDivergenceIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient * Q;
    Vector shape;
    Vector divshape;
    DenseMatrix dshape;
    DenseMatrix gshape;
    DenseMatrix Jadj;
  public:
      VectorDivergenceIntegrator ()
    {
      Q = __null;
    }
    VectorDivergenceIntegrator (Coefficient * _q)
    {
      Q = _q;
    }
    VectorDivergenceIntegrator (Coefficient & q)
    {
      Q = &q;
    }
    virtual void AssembleElementMatrix2 (const FiniteElement & trial_fe,
					 const FiniteElement & test_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat);
  };
  class DivDivIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient * Q;
    Vector divshape;
  public:
      DivDivIntegrator ()
    {
      Q = __null;
    }
    DivDivIntegrator (Coefficient & q):Q (&q)
    {
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
  };
  class VectorDiffusionIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient * Q;
    DenseMatrix Jinv;
    DenseMatrix dshape;
    DenseMatrix gshape;
    DenseMatrix pelmat;
  public:
      VectorDiffusionIntegrator ()
    {
      Q = __null;
    }
    VectorDiffusionIntegrator (Coefficient & q)
    {
      Q = &q;
    }
    virtual void AssembleElementMatrix (const FiniteElement & el,
					ElementTransformation & Trans,
					DenseMatrix & elmat);
  };
  class ElasticityIntegrator:public BilinearFormIntegrator
  {
  private:
    double q_lambda, q_mu;
    Coefficient *lambda, *mu;
    DenseMatrix dshape, Jinv, gshape, pelmat;
    Vector divshape;
  public:
      ElasticityIntegrator (Coefficient & l, Coefficient & m)
    {
      lambda = &l;
      mu = &m;
    }
    ElasticityIntegrator (Coefficient & m, double q_l, double q_m)
    {
      lambda = __null;
      mu = &m;
      q_lambda = q_l;
      q_mu = q_m;
    }
    virtual void AssembleElementMatrix (const FiniteElement &,
					ElementTransformation &,
					DenseMatrix &);
  };
  class DGTraceIntegrator:public BilinearFormIntegrator
  {
  private:
    Coefficient * rho;
    VectorCoefficient *u;
    double alpha, beta;
    Vector shape1, shape2;
  public:
      DGTraceIntegrator (VectorCoefficient & _u, double a, double b)
    {
      rho = __null;
      u = &_u;
      alpha = a;
      beta = b;
    }
    DGTraceIntegrator (Coefficient & _rho, VectorCoefficient & _u,
		       double a, double b)
    {
      rho = &_rho;
      u = &_u;
      alpha = a;
      beta = b;
    }
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix (const FiniteElement & el1,
				     const FiniteElement & el2,
				     FaceElementTransformations & Trans,
				     DenseMatrix & elmat);
  };
  class DGDiffusionIntegrator:public BilinearFormIntegrator
  {
  protected:
    Coefficient * Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;
    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;
  public:
      DGDiffusionIntegrator (const double s, const double k):Q (__null),
      MQ (__null), sigma (s), kappa (k)
    {
    }
    DGDiffusionIntegrator (Coefficient & q, const double s,
			   const double k):Q (&q), MQ (__null), sigma (s),
      kappa (k)
    {
    }
    DGDiffusionIntegrator (MatrixCoefficient & q, const double s,
			   const double k):Q (__null), MQ (&q), sigma (s),
      kappa (k)
    {
    }
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix (const FiniteElement & el1,
				     const FiniteElement & el2,
				     FaceElementTransformations & Trans,
				     DenseMatrix & elmat);
  };
  class TraceJumpIntegrator:public BilinearFormIntegrator
  {
  private:
    Vector face_shape, shape1, shape2;
  public:
    TraceJumpIntegrator ()
    {
    }
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix (const FiniteElement & trial_face_fe,
				     const FiniteElement & test_fe1,
				     const FiniteElement & test_fe2,
				     FaceElementTransformations & Trans,
				     DenseMatrix & elmat);
  };
  class DiscreteInterpolator:public BilinearFormIntegrator
  {
  };
  class GradientInterpolator:public DiscreteInterpolator
  {
  public:
    virtual void AssembleElementMatrix2 (const FiniteElement & h1_fe,
					 const FiniteElement & nd_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat)
    {
      nd_fe.ProjectGrad (h1_fe, Trans, elmat);
    }
  };
  class IdentityInterpolator:public DiscreteInterpolator
  {
  public:
    virtual void AssembleElementMatrix2 (const FiniteElement & dom_fe,
					 const FiniteElement & ran_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat)
    {
      ran_fe.Project (dom_fe, Trans, elmat);
    }
  };
  class CurlInterpolator:public DiscreteInterpolator
  {
  public:
    virtual void AssembleElementMatrix2 (const FiniteElement & dom_fe,
					 const FiniteElement & ran_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat)
    {
      ran_fe.ProjectCurl (dom_fe, Trans, elmat);
    }
  };
  class DivergenceInterpolator:public DiscreteInterpolator
  {
  public:
    virtual void AssembleElementMatrix2 (const FiniteElement & dom_fe,
					 const FiniteElement & ran_fe,
					 ElementTransformation & Trans,
					 DenseMatrix & elmat)
    {
      ran_fe.ProjectDiv (dom_fe, Trans, elmat);
    }
  };
}

namespace mfem
{
  class Ordering
  {
  public:
    enum Type
    { byNODES, byVDIM };
  };
  typedef int RefinementType;
  class RefinementData
  {
  public:
    RefinementType type;
    int num_fine_elems;
    int num_fine_dofs;
    Table *fl_to_fc;
    DenseMatrix *I;
     ~RefinementData ()
    {
      delete fl_to_fc;
      delete I;
    }
  };
  class NURBSExtension;
  class FiniteElementSpace
  {
  protected:
    Mesh * mesh;
    int vdim;
    int ndofs;
    int ordering;
    const FiniteElementCollection *fec;
    int nvdofs, nedofs, nfdofs, nbdofs;
    int *fdofs, *bdofs;
      Array < RefinementData * >RefData;
    Table *elem_dof;
    Table *bdrElem_dof;
      Array < int >dof_elem_array, dof_ldof_array;
    NURBSExtension *NURBSext;
    int own_ext;
    SparseMatrix *cP;
    SparseMatrix *cR;
    void MarkDependency (const SparseMatrix * D,
			 const Array < int >&row_marker,
			 Array < int >&col_marker);
    void UpdateNURBS ();
    void Constructor ();
    void Destructor ();
      FiniteElementSpace (FiniteElementSpace &);
    void ConstructRefinementData (int k, int cdofs, RefinementType type);
    DenseMatrix *LocalInterpolation (int k, int cdofs,
				     RefinementType type, Array < int >&rows);
    SparseMatrix *NC_GlobalRestrictionMatrix (FiniteElementSpace * cfes,
					      NCMesh * ncmesh);
  public:
      FiniteElementSpace (Mesh * m, const FiniteElementCollection * f,
			  int dim = 1, int order = Ordering::byNODES);
    inline Mesh *GetMesh () const
    {
      return mesh;
    }
    NURBSExtension *GetNURBSext ()
    {
      return NURBSext;
    }
    NURBSExtension *StealNURBSext ();
    SparseMatrix *GetConformingProlongation ()
    {
      return cP;
    }
    const SparseMatrix *GetConformingProlongation () const
    {
      return cP;
    }
    SparseMatrix *GetConformingRestriction ()
    {
      return cR;
    }
    const SparseMatrix *GetConformingRestriction () const
    {
      return cR;
    }
    inline int GetVDim () const
    {
      return vdim;
    }
    int GetOrder (int i) const;
    inline int GetNDofs () const
    {
      return ndofs;
    }
    inline int GetVSize () const
    {
      return vdim * ndofs;
    }
    inline int GetNConformingDofs () const
    {
      return cP ? cP->Width () : ndofs;
    }
    inline int GetConformingVSize () const
    {
      return vdim * GetNConformingDofs ();
    }
    inline int GetOrdering () const
    {
      return ordering;
    }
    const FiniteElementCollection *FEColl () const
    {
      return fec;
    }
    int GetNVDofs () const
    {
      return nvdofs;
    }
    int GetNEDofs () const
    {
      return nedofs;
    }
    int GetNFDofs () const
    {
      return nfdofs;
    }
    inline int GetNE () const
    {
      return mesh->GetNE ();
    }
    inline int GetNV () const
    {
      return mesh->GetNV ();
    }
    inline int GetNBE () const
    {
      return mesh->GetNBE ();
    }
    inline int GetElementType (int i) const
    {
      return mesh->GetElementType (i);
    }
    inline void GetElementVertices (int i, Array < int >&vertices) const
    {
      mesh->GetElementVertices (i, vertices);
    }
    inline int GetBdrElementType (int i) const
    {
      return mesh->GetBdrElementType (i);
    }
    ElementTransformation *GetElementTransformation (int i) const
    {
      return mesh->GetElementTransformation (i);
    }
    void GetElementTransformation (int i, IsoparametricTransformation * ElTr)
    {
      mesh->GetElementTransformation (i, ElTr);
    }
    ElementTransformation *GetBdrElementTransformation (int i) const
    {
      return mesh->GetBdrElementTransformation (i);
    }
    int GetAttribute (int i) const
    {
      return mesh->GetAttribute (i);
    }
    int GetBdrAttribute (int i) const
    {
      return mesh->GetBdrAttribute (i);
    }
    virtual void GetElementDofs (int i, Array < int >&dofs) const;
    virtual void GetBdrElementDofs (int i, Array < int >&dofs) const;
    virtual void GetFaceDofs (int i, Array < int >&dofs) const;
    void GetEdgeDofs (int i, Array < int >&dofs) const;
    void GetVertexDofs (int i, Array < int >&dofs) const;
    void GetElementInteriorDofs (int i, Array < int >&dofs) const;
    void GetEdgeInteriorDofs (int i, Array < int >&dofs) const;
    void DofsToVDofs (Array < int >&dofs) const;
    void DofsToVDofs (int vd, Array < int >&dofs) const;
    int DofToVDof (int dof, int vd) const;
    int VDofToDof (int vdof) const
    {
      return (ordering == Ordering::byNODES) ? (vdof % ndofs) : (vdof / vdim);
    }
    static void AdjustVDofs (Array < int >&vdofs);
    void GetElementVDofs (int i, Array < int >&dofs) const;
    void GetBdrElementVDofs (int i, Array < int >&dofs) const;
    void GetFaceVDofs (int iF, Array < int >&dofs) const;
    void GetEdgeVDofs (int iE, Array < int >&dofs) const;
    void GetElementInteriorVDofs (int i, Array < int >&vdofs) const;
    void GetEdgeInteriorVDofs (int i, Array < int >&vdofs) const;
    void BuildElementToDofTable ();
    void BuildDofToArrays ();
    const Table & GetElementToDofTable () const
    {
      return *elem_dof;
    }
    int GetElementForDof (int i)
    {
      return dof_elem_array[i];
    }
    int GetLocalDofForDof (int i)
    {
      return dof_ldof_array[i];
    }
    const FiniteElement *GetFE (int i) const;
    const FiniteElement *GetBE (int i) const;
    const FiniteElement *GetFaceElement (int i) const;
    const FiniteElement *GetEdgeElement (int i) const;
    const FiniteElement *GetTraceElement (int i, int geom_type) const;
    SparseMatrix *GlobalRestrictionMatrix (FiniteElementSpace * cfes,
					   int one_vdim = -1);
    virtual void GetEssentialVDofs (const Array < int >&bdr_attr_is_ess,
				    Array < int >&ess_dofs) const;
    void ConvertToConformingVDofs (const Array < int >&dofs,
				   Array < int >&cdofs)
    {
      MarkDependency (cP, dofs, cdofs);
    }
    void ConvertFromConformingVDofs (const Array < int >&cdofs,
				     Array < int >&dofs)
    {
      MarkDependency (cR, cdofs, dofs);
    }
    void EliminateEssentialBCFromGRM (FiniteElementSpace * cfes,
				      Array < int >&bdr_attr_is_ess,
				      SparseMatrix * R);
    SparseMatrix *GlobalRestrictionMatrix (FiniteElementSpace * cfes,
					   Array < int >&bdr_attr_is_ess,
					   int one_vdim = -1);
    SparseMatrix *D2C_GlobalRestrictionMatrix (FiniteElementSpace * cfes);
    SparseMatrix *D2Const_GlobalRestrictionMatrix (FiniteElementSpace * cfes);
    SparseMatrix *H2L_GlobalRestrictionMatrix (FiniteElementSpace * lfes);
    virtual void Update ();
    virtual void UpdateAndInterpolate (int num_grid_fns, ...);
    void UpdateAndInterpolate (GridFunction * gf)
    {
      UpdateAndInterpolate (1, gf);
    }
    virtual FiniteElementSpace *SaveUpdate ();
    void Save (std::ostream & out) const;
    virtual ~ FiniteElementSpace ();
  };
}

namespace std __attribute__ ((__visibility__ ("default")))
{

  enum float_round_style
  {
    round_indeterminate = -1,
    round_toward_zero = 0,
    round_to_nearest = 1,
    round_toward_infinity = 2,
    round_toward_neg_infinity = 3
  };
  enum float_denorm_style
  {
    denorm_indeterminate = -1,
    denorm_absent = 0,
    denorm_present = 1
  };
  struct __numeric_limits_base
  {
    static const bool is_specialized = false;
    static const int digits = 0;
    static const int digits10 = 0;
    static const bool is_signed = false;
    static const bool is_integer = false;
    static const bool is_exact = false;
    static const int radix = 0;
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static const bool is_iec559 = false;
    static const bool is_bounded = false;
    static const bool is_modulo = false;
    static const bool traps = false;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template < typename _Tp > struct numeric_limits:public __numeric_limits_base
  {
    static _Tp min () throw ()
    {
      return _Tp ();
    }
    static _Tp max () throw ()
    {
      return _Tp ();
    }
    static _Tp epsilon () throw ()
    {
      return _Tp ();
    }
    static _Tp round_error () throw ()
    {
      return _Tp ();
    }
    static _Tp infinity () throw ()
    {
      return _Tp ();
    }
    static _Tp quiet_NaN () throw ()
    {
      return _Tp ();
    }
    static _Tp signaling_NaN () throw ()
    {
      return _Tp ();
    }
    static _Tp denorm_min () throw ()
    {
      return _Tp ();
    }
  };
  template <> struct numeric_limits <bool >
  {
    static const bool is_specialized = true;
    static bool min () throw ()
    {
      return false;
    }
    static bool max () throw ()
    {
      return true;
    }
    static const int digits = 1;
    static const int digits10 = 0;
    static const bool is_signed = false;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static bool epsilon () throw ()
    {
      return false;
    }
    static bool round_error () throw ()
    {
      return false;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static bool infinity () throw ()
    {
      return false;
    }
    static bool quiet_NaN () throw ()
    {
      return false;
    }
    static bool signaling_NaN () throw ()
    {
      return false;
    }
    static bool denorm_min () throw ()
    {
      return false;
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <char >
  {
    static const bool is_specialized = true;
    static char min () throw ()
    {
      return (((char) (-1) <
	       0) ? -(((char) (-1) <
		       0)
		      ? (((((char) 1 <<
			    ((sizeof (char) * 8 - ((char) (-1) < 0)) - 1)) -
			   1) << 1) + 1) : ~(char) 0) - 1 : (char) 0);
    }
    static char max () throw ()
    {
      return (((char) (-1) <
	       0)
	      ? (((((char) 1 << ((sizeof (char) * 8 - ((char) (-1) < 0)) - 1))
		   - 1) << 1) + 1) : ~(char) 0);
    }
    static const int digits = (sizeof (char) * 8 - ((char) (-1) < 0));
    static const int digits10 =
      ((sizeof (char) * 8 - ((char) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = ((char) (-1) < 0);
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static char epsilon () throw ()
    {
      return 0;
    }
    static char round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static char infinity () throw ()
    {
      return char ();
    }
    static char quiet_NaN () throw ()
    {
      return char ();
    }
    static char signaling_NaN () throw ()
    {
      return char ();
    }
    static char denorm_min () throw ()
    {
      return static_cast < char >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = !is_signed;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <signed char >
  {
    static const bool is_specialized = true;
    static signed char min () throw ()
    {
      return -127 - 1;
    }
    static signed char max () throw ()
    {
      return 127;
    }
    static const int digits =
      (sizeof (signed char) * 8 - ((signed char) (-1) < 0));
    static const int digits10 =
      ((sizeof (signed char) * 8 - ((signed char) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = true;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static signed char epsilon () throw ()
    {
      return 0;
    }
    static signed char round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static signed char infinity () throw ()
    {
      return static_cast < signed char >(0);
    }
    static signed char quiet_NaN () throw ()
    {
      return static_cast < signed char >(0);
    }
    static signed char signaling_NaN () throw ()
    {
      return static_cast < signed char >(0);
    }
    static signed char denorm_min () throw ()
    {
      return static_cast < signed char >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <unsigned char >
  {
    static const bool is_specialized = true;
    static unsigned char min () throw ()
    {
      return 0;
    }
    static unsigned char max () throw ()
    {
      return 127 * 2U + 1;
    }
    static const int digits
      = (sizeof (unsigned char) * 8 - ((unsigned char) (-1) < 0));
    static const int digits10
      =
      ((sizeof (unsigned char) * 8 -
	((unsigned char) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = false;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static unsigned char epsilon () throw ()
    {
      return 0;
    }
    static unsigned char round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static unsigned char infinity () throw ()
    {
      return static_cast < unsigned char >(0);
    }
    static unsigned char quiet_NaN () throw ()
    {
      return static_cast < unsigned char >(0);
    }
    static unsigned char signaling_NaN () throw ()
    {
      return static_cast < unsigned char >(0);
    }
    static unsigned char denorm_min () throw ()
    {
      return static_cast < unsigned char >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = true;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <wchar_t >
  {
    static const bool is_specialized = true;
    static wchar_t min () throw ()
    {
      return (((wchar_t) (-1) <
	       0) ? -(((wchar_t) (-1) <
		       0)
		      ? (((((wchar_t) 1 <<
			    ((sizeof (wchar_t) * 8 - ((wchar_t) (-1) < 0)) -
			     1)) - 1) << 1) + 1) : ~(wchar_t) 0) -
	      1 : (wchar_t) 0);
    }
    static wchar_t max () throw ()
    {
      return (((wchar_t) (-1) <
	       0)
	      ? (((((wchar_t) 1 <<
		    ((sizeof (wchar_t) * 8 - ((wchar_t) (-1) < 0)) - 1)) -
		   1) << 1) + 1) : ~(wchar_t) 0);
    }
    static const int digits = (sizeof (wchar_t) * 8 - ((wchar_t) (-1) < 0));
    static const int digits10
      = ((sizeof (wchar_t) * 8 - ((wchar_t) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = ((wchar_t) (-1) < 0);
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static wchar_t epsilon () throw ()
    {
      return 0;
    }
    static wchar_t round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static wchar_t infinity () throw ()
    {
      return wchar_t ();
    }
    static wchar_t quiet_NaN () throw ()
    {
      return wchar_t ();
    }
    static wchar_t signaling_NaN () throw ()
    {
      return wchar_t ();
    }
    static wchar_t denorm_min () throw ()
    {
      return wchar_t ();
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = !is_signed;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <short >
  {
    static const bool is_specialized = true;
    static short min () throw ()
    {
      return -32767 - 1;
    }
    static short max () throw ()
    {
      return 32767;
    }
    static const int digits = (sizeof (short) * 8 - ((short) (-1) < 0));
    static const int digits10 =
      ((sizeof (short) * 8 - ((short) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = true;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static short epsilon () throw ()
    {
      return 0;
    }
    static short round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static short infinity () throw ()
    {
      return short ();
    }
    static short quiet_NaN () throw ()
    {
      return short ();
    }
    static short signaling_NaN () throw ()
    {
      return short ();
    }
    static short denorm_min () throw ()
    {
      return short ();
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <unsigned short >
  {
    static const bool is_specialized = true;
    static unsigned short min () throw ()
    {
      return 0;
    }
    static unsigned short max () throw ()
    {
      return 32767 * 2U + 1;
    }
    static const int digits
      = (sizeof (unsigned short) * 8 - ((unsigned short) (-1) < 0));
    static const int digits10
      =
      ((sizeof (unsigned short) * 8 -
	((unsigned short) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = false;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static unsigned short epsilon () throw ()
    {
      return 0;
    }
    static unsigned short round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static unsigned short infinity () throw ()
    {
      return static_cast < unsigned short >(0);
    }
    static unsigned short quiet_NaN () throw ()
    {
      return static_cast < unsigned short >(0);
    }
    static unsigned short signaling_NaN () throw ()
    {
      return static_cast < unsigned short >(0);
    }
    static unsigned short denorm_min () throw ()
    {
      return static_cast < unsigned short >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = true;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <int >
  {
    static const bool is_specialized = true;
    static int min () throw ()
    {
      return -2147483647 - 1;
    }
    static int max () throw ()
    {
      return 2147483647;
    }
    static const int digits = (sizeof (int) * 8 - ((int) (-1) < 0));
    static const int digits10 =
      ((sizeof (int) * 8 - ((int) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = true;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static int epsilon () throw ()
    {
      return 0;
    }
    static int round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static int infinity () throw ()
    {
      return static_cast < int >(0);
    }
    static int quiet_NaN () throw ()
    {
      return static_cast < int >(0);
    }
    static int signaling_NaN () throw ()
    {
      return static_cast < int >(0);
    }
    static int denorm_min () throw ()
    {
      return static_cast < int >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <unsigned int >
  {
    static const bool is_specialized = true;
    static unsigned int min () throw ()
    {
      return 0;
    }
    static unsigned int max () throw ()
    {
      return 2147483647 * 2U + 1;
    }
    static const int digits
      = (sizeof (unsigned int) * 8 - ((unsigned int) (-1) < 0));
    static const int digits10
      =
      ((sizeof (unsigned int) * 8 - ((unsigned int) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = false;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static unsigned int epsilon () throw ()
    {
      return 0;
    }
    static unsigned int round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static unsigned int infinity () throw ()
    {
      return static_cast < unsigned int >(0);
    }
    static unsigned int quiet_NaN () throw ()
    {
      return static_cast < unsigned int >(0);
    }
    static unsigned int signaling_NaN () throw ()
    {
      return static_cast < unsigned int >(0);
    }
    static unsigned int denorm_min () throw ()
    {
      return static_cast < unsigned int >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = true;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <long >
  {
    static const bool is_specialized = true;
    static long min () throw ()
    {
      return -9223372036854775807L - 1;
    }
    static long max () throw ()
    {
      return 9223372036854775807L;
    }
    static const int digits = (sizeof (long) * 8 - ((long) (-1) < 0));
    static const int digits10 =
      ((sizeof (long) * 8 - ((long) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = true;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static long epsilon () throw ()
    {
      return 0;
    }
    static long round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static long infinity () throw ()
    {
      return static_cast < long >(0);
    }
    static long quiet_NaN () throw ()
    {
      return static_cast < long >(0);
    }
    static long signaling_NaN () throw ()
    {
      return static_cast < long >(0);
    }
    static long denorm_min () throw ()
    {
      return static_cast < long >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <unsigned long >
  {
    static const bool is_specialized = true;
    static unsigned long min () throw ()
    {
      return 0;
    }
    static unsigned long max () throw ()
    {
      return 9223372036854775807L * 2UL + 1;
    }
    static const int digits
      = (sizeof (unsigned long) * 8 - ((unsigned long) (-1) < 0));
    static const int digits10
      =
      ((sizeof (unsigned long) * 8 -
	((unsigned long) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = false;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static unsigned long epsilon () throw ()
    {
      return 0;
    }
    static unsigned long round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static unsigned long infinity () throw ()
    {
      return static_cast < unsigned long >(0);
    }
    static unsigned long quiet_NaN () throw ()
    {
      return static_cast < unsigned long >(0);
    }
    static unsigned long signaling_NaN () throw ()
    {
      return static_cast < unsigned long >(0);
    }
    static unsigned long denorm_min () throw ()
    {
      return static_cast < unsigned long >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = true;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <long long >
  {
    static const bool is_specialized = true;
    static long long min () throw ()
    {
      return -9223372036854775807LL - 1;
    }
    static long long max () throw ()
    {
      return 9223372036854775807LL;
    }
    static const int digits
      = (sizeof (long long) * 8 - ((long long) (-1) < 0));
    static const int digits10
      = ((sizeof (long long) * 8 - ((long long) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = true;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static long long epsilon () throw ()
    {
      return 0;
    }
    static long long round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static long long infinity () throw ()
    {
      return static_cast < long long >(0);
    }
    static long long quiet_NaN () throw ()
    {
      return static_cast < long long >(0);
    }
    static long long signaling_NaN () throw ()
    {
      return static_cast < long long >(0);
    }
    static long long denorm_min () throw ()
    {
      return static_cast < long long >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <unsigned long long >
  {
    static const bool is_specialized = true;
    static unsigned long long min () throw ()
    {
      return 0;
    }
    static unsigned long long max () throw ()
    {
      return 9223372036854775807LL * 2ULL + 1;
    }
    static const int digits
      = (sizeof (unsigned long long) * 8 - ((unsigned long long) (-1) < 0));
    static const int digits10
      =
      ((sizeof (unsigned long long) * 8 -
	((unsigned long long) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = false;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static unsigned long long epsilon () throw ()
    {
      return 0;
    }
    static unsigned long long round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static unsigned long long infinity () throw ()
    {
      return static_cast < unsigned long long >(0);
    }
    static unsigned long long quiet_NaN () throw ()
    {
      return static_cast < unsigned long long >(0);
    }
    static unsigned long long signaling_NaN () throw ()
    {
      return static_cast < unsigned long long >(0);
    }
    static unsigned long long denorm_min () throw ()
    {
      return static_cast < unsigned long long >(0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = true;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <__int128 >
  {
    static const bool is_specialized = true;
    static __int128 min () throw ()
    {
      return (((__int128) (-1) <
	       0) ? -(((__int128) (-1) <
		       0)
		      ? (((((__int128) 1 <<
			    ((sizeof (__int128) * 8 - ((__int128) (-1) < 0)) -
			     1)) - 1) << 1) + 1) : ~(__int128) 0) -
	      1 : (__int128) 0);
    }
    static __int128 max () throw ()
    {
      return (((__int128) (-1) <
	       0)
	      ? (((((__int128) 1 <<
		    ((sizeof (__int128) * 8 - ((__int128) (-1) < 0)) - 1)) -
		   1) << 1) + 1) : ~(__int128) 0);
    }
    static const int digits = (sizeof (__int128) * 8 - ((__int128) (-1) < 0));
    static const int digits10
      = ((sizeof (__int128) * 8 - ((__int128) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = true;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static __int128 epsilon () throw ()
    {
      return 0;
    }
    static __int128 round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static __int128 infinity () throw ()
    {
      return static_cast < __int128 > (0);
    }
    static __int128 quiet_NaN () throw ()
    {
      return static_cast < __int128 > (0);
    }
    static __int128 signaling_NaN () throw ()
    {
      return static_cast < __int128 > (0);
    }
    static __int128 denorm_min () throw ()
    {
      return static_cast < __int128 > (0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <unsigned __int128 >
  {
    static const bool is_specialized = true;
    static unsigned __int128 min () throw ()
    {
      return 0;
    }
    static unsigned __int128 max () throw ()
    {
      return (((unsigned __int128) (-1) <
	       0)
	      ? (((((unsigned __int128) 1 <<
		    ((sizeof (unsigned __int128) * 8 -
		      ((unsigned __int128) (-1) <
		       0)) - 1)) - 1) << 1) + 1) : ~(unsigned __int128) 0);
    }
    static const int digits
      = (sizeof (unsigned __int128) * 8 - ((unsigned __int128) (-1) < 0));
    static const int digits10
      =
      ((sizeof (unsigned __int128) * 8 -
	((unsigned __int128) (-1) < 0)) * 643L / 2136);
    static const bool is_signed = false;
    static const bool is_integer = true;
    static const bool is_exact = true;
    static const int radix = 2;
    static unsigned __int128 epsilon () throw ()
    {
      return 0;
    }
    static unsigned __int128 round_error () throw ()
    {
      return 0;
    }
    static const int min_exponent = 0;
    static const int min_exponent10 = 0;
    static const int max_exponent = 0;
    static const int max_exponent10 = 0;
    static const bool has_infinity = false;
    static const bool has_quiet_NaN = false;
    static const bool has_signaling_NaN = false;
    static const float_denorm_style has_denorm = denorm_absent;
    static const bool has_denorm_loss = false;
    static unsigned __int128 infinity () throw ()
    {
      return static_cast < unsigned __int128 > (0);
    }
    static unsigned __int128 quiet_NaN () throw ()
    {
      return static_cast < unsigned __int128 > (0);
    }
    static unsigned __int128 signaling_NaN () throw ()
    {
      return static_cast < unsigned __int128 > (0);
    }
    static unsigned __int128 denorm_min () throw ()
    {
      return static_cast < unsigned __int128 > (0);
    }
    static const bool is_iec559 = false;
    static const bool is_bounded = true;
    static const bool is_modulo = true;
    static const bool traps = true;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_toward_zero;
  };
  template <> struct numeric_limits <float >
  {
    static const bool is_specialized = true;
    static float min () throw ()
    {
      return 1.17549435082228750797e-38F;
    }
    static float max () throw ()
    {
      return 3.40282346638528859812e+38F;
    }
    static const int digits = 24;
    static const int digits10 = 6;
    static const bool is_signed = true;
    static const bool is_integer = false;
    static const bool is_exact = false;
    static const int radix = 2;
    static float epsilon () throw ()
    {
      return 1.19209289550781250000e-7F;
    }
    static float round_error () throw ()
    {
      return 0.5F;
    }
    static const int min_exponent = (-125);
    static const int min_exponent10 = (-37);
    static const int max_exponent = 128;
    static const int max_exponent10 = 38;
    static const bool has_infinity = 1;
    static const bool has_quiet_NaN = 1;
    static const bool has_signaling_NaN = has_quiet_NaN;
    static const float_denorm_style has_denorm
      = bool (1) ? denorm_present : denorm_absent;
    static const bool has_denorm_loss = false;
    static float infinity () throw ()
    {
      return __builtin_huge_valf ();
    }
    static float quiet_NaN () throw ()
    {
      return __builtin_nanf ("");
    }
    static float signaling_NaN () throw ()
    {
      return __builtin_nansf ("");
    }
    static float denorm_min () throw ()
    {
      return 1.40129846432481707092e-45F;
    }
    static const bool is_iec559
      = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = false;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_to_nearest;
  };
  template <> struct numeric_limits <double >
  {
    static const bool is_specialized = true;
    static double min () throw ()
    {
      return double (2.22507385850720138309e-308L);
    }
    static double max () throw ()
    {
      return double (1.79769313486231570815e+308L);
    }
    static const int digits = 53;
    static const int digits10 = 15;
    static const bool is_signed = true;
    static const bool is_integer = false;
    static const bool is_exact = false;
    static const int radix = 2;
    static double epsilon () throw ()
    {
      return double (2.22044604925031308085e-16L);
    }
    static double round_error () throw ()
    {
      return 0.5;
    }
    static const int min_exponent = (-1021);
    static const int min_exponent10 = (-307);
    static const int max_exponent = 1024;
    static const int max_exponent10 = 308;
    static const bool has_infinity = 1;
    static const bool has_quiet_NaN = 1;
    static const bool has_signaling_NaN = has_quiet_NaN;
    static const float_denorm_style has_denorm
      = bool (1) ? denorm_present : denorm_absent;
    static const bool has_denorm_loss = false;
    static double infinity () throw ()
    {
      return __builtin_huge_val ();
    }
    static double quiet_NaN () throw ()
    {
      return __builtin_nan ("");
    }
    static double signaling_NaN () throw ()
    {
      return __builtin_nans ("");
    }
    static double denorm_min () throw ()
    {
      return double (4.94065645841246544177e-324L);
    }
    static const bool is_iec559
      = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = false;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_to_nearest;
  };
  template <> struct numeric_limits <long double >
  {
    static const bool is_specialized = true;
    static long double min () throw ()
    {
      return 3.36210314311209350626e-4932L;
    }
    static long double max () throw ()
    {
      return 1.18973149535723176502e+4932L;
    }
    static const int digits = 64;
    static const int digits10 = 18;
    static const bool is_signed = true;
    static const bool is_integer = false;
    static const bool is_exact = false;
    static const int radix = 2;
    static long double epsilon () throw ()
    {
      return 1.08420217248550443401e-19L;
    }
    static long double round_error () throw ()
    {
      return 0.5L;
    }
    static const int min_exponent = (-16381);
    static const int min_exponent10 = (-4931);
    static const int max_exponent = 16384;
    static const int max_exponent10 = 4932;
    static const bool has_infinity = 1;
    static const bool has_quiet_NaN = 1;
    static const bool has_signaling_NaN = has_quiet_NaN;
    static const float_denorm_style has_denorm
      = bool (1) ? denorm_present : denorm_absent;
    static const bool has_denorm_loss = false;
    static long double infinity () throw ()
    {
      return __builtin_huge_vall ();
    }
    static long double quiet_NaN () throw ()
    {
      return __builtin_nanl ("");
    }
    static long double signaling_NaN () throw ()
    {
      return __builtin_nansl ("");
    }
    static long double denorm_min () throw ()
    {
      return 3.64519953188247460253e-4951L;
    }
    static const bool is_iec559
      = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
    static const bool is_bounded = true;
    static const bool is_modulo = false;
    static const bool traps = false;
    static const bool tinyness_before = false;
    static const float_round_style round_style = round_to_nearest;
  };

}

namespace mfem
{
  class GridFunction:public Vector
  {
  protected:
    FiniteElementSpace * fes;
    FiniteElementCollection *fec;
    void SaveSTLTri (std::ostream & out, double p1[], double p2[],
		     double p3[]);
    void GetVectorGradientHat (ElementTransformation & T, DenseMatrix & gh);
    void ProjectDeltaCoefficient (DeltaCoefficient & delta_coeff,
				  double &integral);
  public:
      GridFunction ()
    {
      fes = __null;
      fec = __null;
    }
    GridFunction (FiniteElementSpace * f):Vector (f->GetVSize ())
    {
      fes = f;
      fec = __null;
    }
    GridFunction (Mesh * m, std::istream & input);
    GridFunction (Mesh * m, GridFunction * gf_array[], int num_pieces);
    void MakeOwner (FiniteElementCollection * _fec)
    {
      fec = _fec;
    }
    FiniteElementCollection *OwnFEC ()
    {
      return fec;
    }
    int VectorDim () const;
    void GetNodalValues (int i, Array < double >&nval, int vdim = 1) const;
    virtual double GetValue (int i, const IntegrationPoint & ip,
			     int vdim = 1) const;
    void GetVectorValue (int i, const IntegrationPoint & ip,
			 Vector & val) const;
    void GetValues (int i, const IntegrationRule & ir, Vector & vals,
		    int vdim = 1) const;
    void GetValues (int i, const IntegrationRule & ir, Vector & vals,
		    DenseMatrix & tr, int vdim = 1) const;
    int GetFaceValues (int i, int side, const IntegrationRule & ir,
		       Vector & vals, DenseMatrix & tr, int vdim = 1) const;
    void GetVectorValues (ElementTransformation & T,
			  const IntegrationRule & ir,
			  DenseMatrix & vals) const;
    void GetVectorValues (int i, const IntegrationRule & ir,
			  DenseMatrix & vals, DenseMatrix & tr) const;
    int GetFaceVectorValues (int i, int side, const IntegrationRule & ir,
			     DenseMatrix & vals, DenseMatrix & tr) const;
    void GetValuesFrom (GridFunction &);
    void GetBdrValuesFrom (GridFunction &);
    void GetVectorFieldValues (int i, const IntegrationRule & ir,
			       DenseMatrix & vals,
			       DenseMatrix & tr, int comp = 0) const;
    void ReorderByNodes ();
    void GetNodalValues (Vector & nval, int vdim = 1) const;
    void GetVectorFieldNodalValues (Vector & val, int comp) const;
    void ProjectVectorFieldOn (GridFunction & vec_field, int comp = 0);
    void GetDerivative (int comp, int der_comp, GridFunction & der);
    double GetDivergence (ElementTransformation & tr);
    void GetGradient (ElementTransformation & tr, Vector & grad);
    void GetGradients (const int elem, const IntegrationRule & ir,
		       DenseMatrix & grad);
    void GetVectorGradient (ElementTransformation & tr, DenseMatrix & grad);
    void GetElementAverages (GridFunction & avgs);
    void ImposeBounds (int i, const Vector & weights,
		       const Vector & _lo, const Vector & _hi);
    void ImposeBounds (int i, const Vector & weights,
		       double _min = 0.0, double _max =
		       std::numeric_limits < double >::infinity ());
    void ProjectGridFunction (const GridFunction & src);
    void ProjectCoefficient (Coefficient & coeff);
    void ProjectCoefficient (Coefficient & coeff, Array < int >&dofs, int vd =
			     0);
    void ProjectCoefficient (VectorCoefficient & vcoeff);
    void ProjectCoefficient (Coefficient * coeff[]);
    void ProjectBdrCoefficient (Coefficient & coeff, Array < int >&attr)
    {
      Coefficient *coeff_p = &coeff;
      ProjectBdrCoefficient (&coeff_p, attr);
    }
    void ProjectBdrCoefficient (Coefficient * coeff[], Array < int >&attr);
    void ProjectBdrCoefficientNormal (VectorCoefficient & vcoeff,
				      Array < int >&bdr_attr);
    void ProjectBdrCoefficientTangent (VectorCoefficient & vcoeff,
				       Array < int >&bdr_attr);
    double ComputeL2Error (Coefficient & exsol,
			   const IntegrationRule * irs[] = __null) const
    {
      return ComputeLpError (2.0, exsol, __null, irs);
    }
    double ComputeL2Error (Coefficient * exsol[],
			   const IntegrationRule * irs[] = __null) const;
    double ComputeL2Error (VectorCoefficient & exsol,
			   const IntegrationRule * irs[] = __null,
			   Array < int >*elems = __null) const;
    double ComputeH1Error (Coefficient * exsol, VectorCoefficient * exgrad,
			   Coefficient * ell_coef, double Nu,
			   int norm_type) const;
    double ComputeMaxError (Coefficient & exsol,
			    const IntegrationRule * irs[] = __null) const
    {
      return ComputeLpError (std::numeric_limits < double >::infinity (),
			     exsol, __null, irs);
    }
    double ComputeMaxError (Coefficient * exsol[],
			    const IntegrationRule * irs[] = __null) const;
    double ComputeMaxError (VectorCoefficient & exsol,
			    const IntegrationRule * irs[] = __null) const
    {
      return ComputeLpError (std::numeric_limits < double >::infinity (),
			     exsol, __null, __null, irs);
    }
    double ComputeL1Error (Coefficient & exsol,
			   const IntegrationRule * irs[] = __null) const
    {
      return ComputeLpError (1.0, exsol, __null, irs);
    }
    double ComputeW11Error (Coefficient * exsol, VectorCoefficient * exgrad,
			    int norm_type, Array < int >*elems = __null,
			    const IntegrationRule * irs[] = __null) const;
    double ComputeL1Error (VectorCoefficient & exsol,
			   const IntegrationRule * irs[] = __null) const
    {
      return ComputeLpError (1.0, exsol, __null, __null, irs);
    }
    double ComputeLpError (const double p, Coefficient & exsol,
			   Coefficient * weight = __null,
			   const IntegrationRule * irs[] = __null) const;
    double ComputeLpError (const double p, VectorCoefficient & exsol,
			   Coefficient * weight = __null,
			   VectorCoefficient * v_weight = __null,
			   const IntegrationRule * irs[] = __null) const;
    GridFunction & operator= (double value);
    GridFunction & operator= (const Vector & v);
    GridFunction & operator= (const GridFunction & v);
    void ConformingProlongate (const Vector & x);
    void ConformingProlongate ();
    void ConformingProject (Vector & x) const;
    void ConformingProject ();
    FiniteElementSpace *FESpace ()
    {
      return fes;
    }
    void Update ()
    {
      SetSize (fes->GetVSize ());
    }
    void Update (FiniteElementSpace * f);
    void Update (FiniteElementSpace * f, Vector & v, int v_offset);
    virtual void Save (std::ostream & out) const;
    void SaveVTK (std::ostream & out, const std::string & field_name,
		  int ref);
    void SaveSTL (std::ostream & out, int TimesToRefine = 1);
    virtual ~ GridFunction ();
  };
  std::ostream & operator<< (std::ostream & out, const GridFunction & sol);
  void ComputeFlux (BilinearFormIntegrator & blfi,
		    GridFunction & u,
		    GridFunction & flux, int wcoef = 1, int sd = -1);
  void ZZErrorEstimator (BilinearFormIntegrator & blfi,
			 GridFunction & u,
			 GridFunction & flux, Vector & ErrorEstimates,
			 int wsd = 1);
  class ExtrudeCoefficient:public Coefficient
  {
  private:
    int n;
    Mesh *mesh_in;
      Coefficient & sol_in;
  public:
      ExtrudeCoefficient (Mesh * m, Coefficient & s, int _n):n (_n),
      mesh_in (m), sol_in (s)
    {
    }
    virtual double Eval (ElementTransformation & T,
			 const IntegrationPoint & ip);
    virtual ~ ExtrudeCoefficient ()
    {
    }
  };
  GridFunction *Extrude1DGridFunction (Mesh * mesh, Mesh * mesh2d,
				       GridFunction * sol, const int ny);
}

namespace mfem
{
  class LinearForm:public Vector
  {
  private:
    FiniteElementSpace * fes;
    Array < LinearFormIntegrator * >dlfi;
    Array < LinearFormIntegrator * >blfi;
    Array < LinearFormIntegrator * >flfi;
  public:
    LinearForm (FiniteElementSpace * f):Vector (f->GetVSize ())
    {
      fes = f;
    };
      LinearForm ()
    {
      fes = __null;
    }
    FiniteElementSpace *GetFES ()
    {
      return fes;
    };
    void AddDomainIntegrator (LinearFormIntegrator * lfi);
    void AddBoundaryIntegrator (LinearFormIntegrator * lfi);
    void AddBdrFaceIntegrator (LinearFormIntegrator * lfi);
    void Assemble ();
    void ConformingAssemble (Vector & b) const;
    void ConformingAssemble ();
    void Update ()
    {
      SetSize (fes->GetVSize ());
    }
    void Update (FiniteElementSpace * f)
    {
      fes = f;
      SetSize (f->GetVSize ());
    }
    void Update (FiniteElementSpace * f, Vector & v, int v_offset);
    ~LinearForm ();
  };
}

namespace mfem
{
  class NonlinearForm:public Operator
  {
  protected:
    FiniteElementSpace * fes;
    Array < NonlinearFormIntegrator * >dfi;
    mutable SparseMatrix *Grad;
      Array < int >ess_vdofs;
  public:
      NonlinearForm (FiniteElementSpace * f):Operator (f->GetVSize ())
    {
      fes = f;
      Grad = __null;
    }
    void AddDomainIntegrator (NonlinearFormIntegrator * nlfi)
    {
      dfi.Append (nlfi);
    }
    virtual void SetEssentialBC (const Array < int >&bdr_attr_is_ess,
				 Vector * rhs = __null);
    void SetEssentialVDofs (const Array < int >&ess_vdofs_list)
    {
      ess_vdofs_list.Copy (ess_vdofs);
    }
    virtual double GetEnergy (const Vector & x) const;
    virtual void Mult (const Vector & x, Vector & y) const;
    virtual Operator & GetGradient (const Vector & x) const;
    virtual ~ NonlinearForm ();
  };
}

namespace mfem
{
  class BilinearForm:public Matrix
  {
  protected:
    SparseMatrix * mat;
    SparseMatrix *mat_e;
    FiniteElementSpace *fes;
    int extern_bfs;
      Array < BilinearFormIntegrator * >dbfi;
      Array < BilinearFormIntegrator * >bbfi;
      Array < BilinearFormIntegrator * >fbfi;
      Array < BilinearFormIntegrator * >bfbfi;
    DenseMatrix elemmat;
      Array < int >vdofs;
    DenseTensor *element_matrices;
    int precompute_sparsity;
    void AllocMat ();
      BilinearForm ():Matrix (0)
    {
      fes = __null;
      mat = mat_e = __null;
      extern_bfs = 0;
      element_matrices = __null;
      precompute_sparsity = 0;
    }
  public:
      BilinearForm (FiniteElementSpace * f);
      BilinearForm (FiniteElementSpace * f, BilinearForm * bf, int ps = 0);
    int Size () const
    {
      return height;
    }
    void UsePrecomputedSparsity (int ps = 1)
    {
      precompute_sparsity = ps;
    }
    void AllocateMatrix ()
    {
      if (mat == __null)
	AllocMat ();
    }
    Array < BilinearFormIntegrator * >*GetDBFI ()
    {
      return &dbfi;
    }
    Array < BilinearFormIntegrator * >*GetBBFI ()
    {
      return &bbfi;
    }
    Array < BilinearFormIntegrator * >*GetFBFI ()
    {
      return &fbfi;
    }
    Array < BilinearFormIntegrator * >*GetBFBFI ()
    {
      return &bfbfi;
    }
    const double &operator () (int i, int j)
    {
      return (*mat) (i, j);
    }
    virtual double &Elem (int i, int j);
    virtual const double &Elem (int i, int j) const;
    virtual void Mult (const Vector & x, Vector & y) const;
    void FullMult (const Vector & x, Vector & y) const
    {
      mat->Mult (x, y);
      mat_e->AddMult (x, y);
    }
    virtual void AddMult (const Vector & x, Vector & y, const double a = 1.0) const
    {
      mat->AddMult (x, y, a);
    }
    void FullAddMult (const Vector & x, Vector & y) const
    {
      mat->AddMult (x, y);
      mat_e->AddMult (x, y);
    }
    double InnerProduct (const Vector & x, const Vector & y) const
    {
      return mat->InnerProduct (x, y);
    }
    virtual MatrixInverse *Inverse () const;
    virtual void Finalize (int skip_zeros = 1);
    const SparseMatrix & SpMat () const
    {
      return *mat;
    }
    SparseMatrix & SpMat ()
    {
      return *mat;
    }
    SparseMatrix *LoseMat ()
    {
      SparseMatrix *tmp = mat;
      mat = __null;
      return tmp;
    }
    void AddDomainIntegrator (BilinearFormIntegrator * bfi);
    void AddBoundaryIntegrator (BilinearFormIntegrator * bfi);
    void AddInteriorFaceIntegrator (BilinearFormIntegrator * bfi);
    void AddBdrFaceIntegrator (BilinearFormIntegrator * bfi);
    void operator= (const double a)
    {
      if (mat != __null)
	*mat = a;
      if (mat_e != __null)
	*mat_e = a;
    }
    void Assemble (int skip_zeros = 1);
    void ConformingAssemble ();
    void ConformingAssemble (GridFunction & sol, LinearForm & rhs)
    {
      ConformingAssemble ();
      rhs.ConformingAssemble ();
      sol.ConformingProject ();
    }
    void ComputeElementMatrices ();
    void FreeElementMatrices ()
    {
      delete element_matrices;
      element_matrices = __null;
    }
    void ComputeElementMatrix (int i, DenseMatrix & elmat);
    void AssembleElementMatrix (int i, const DenseMatrix & elmat,
				Array < int >&vdofs, int skip_zeros = 1);
    void EliminateEssentialBC (Array < int >&bdr_attr_is_ess,
			       Vector & sol, Vector & rhs, int d = 0);
    void EliminateVDofs (Array < int >&vdofs, Vector & sol, Vector & rhs,
			 int d = 0);
    void EliminateVDofs (Array < int >&vdofs, int d = 0);
    void EliminateVDofsInRHS (Array < int >&vdofs, const Vector & x,
			      Vector & b);
    double FullInnerProduct (const Vector & x, const Vector & y) const
    {
      return mat->InnerProduct (x, y) + mat_e->InnerProduct (x, y);
    }
    void EliminateEssentialBC (Array < int >&bdr_attr_is_ess, int d = 0);
    void EliminateEssentialBCFromDofs (Array < int >&ess_dofs, Vector & sol,
				       Vector & rhs, int d = 0);
    void EliminateEssentialBCFromDofs (Array < int >&ess_dofs, int d = 0);
    void Update (FiniteElementSpace * nfes = __null);
    FiniteElementSpace *GetFES ()
    {
      return fes;
    }
    virtual ~ BilinearForm ();
  };
  class MixedBilinearForm:public Matrix
  {
  protected:
    SparseMatrix * mat;
    FiniteElementSpace *trial_fes, *test_fes;
      Array < BilinearFormIntegrator * >dom;
      Array < BilinearFormIntegrator * >bdr;
      Array < BilinearFormIntegrator * >skt;
  public:
      MixedBilinearForm (FiniteElementSpace * tr_fes,
			 FiniteElementSpace * te_fes);
    virtual double &Elem (int i, int j);
    virtual const double &Elem (int i, int j) const;
    virtual void Mult (const Vector & x, Vector & y) const;
    virtual void AddMult (const Vector & x, Vector & y,
			  const double a = 1.0) const;
    virtual void AddMultTranspose (const Vector & x, Vector & y,
				   const double a = 1.0) const;
    virtual void MultTranspose (const Vector & x, Vector & y) const
    {
      y = 0.0;
      AddMultTranspose (x, y);
    }
    virtual MatrixInverse *Inverse () const;
    virtual void Finalize (int skip_zeros = 1);
    void GetBlocks (Array2D < SparseMatrix * >&blocks) const;
    const SparseMatrix & SpMat () const
    {
      return *mat;
    }
    SparseMatrix & SpMat ()
    {
      return *mat;
    }
    SparseMatrix *LoseMat ()
    {
      SparseMatrix *tmp = mat;
      mat = __null;
      return tmp;
    }
    void AddDomainIntegrator (BilinearFormIntegrator * bfi);
    void AddBoundaryIntegrator (BilinearFormIntegrator * bfi);
    void AddTraceFaceIntegrator (BilinearFormIntegrator * bfi);
    Array < BilinearFormIntegrator * >*GetDBFI ()
    {
      return &dom;
    }
    Array < BilinearFormIntegrator * >*GetBBFI ()
    {
      return &bdr;
    }
    Array < BilinearFormIntegrator * >*GetTFBFI ()
    {
      return &skt;
    }
    void operator= (const double a)
    {
      *mat = a;
    }
    void Assemble (int skip_zeros = 1);
    void ConformingAssemble ();
    void EliminateTrialDofs (Array < int >&bdr_attr_is_ess,
			     Vector & sol, Vector & rhs);
    void EliminateEssentialBCFromTrialDofs (Array < int >&marked_vdofs,
					    Vector & sol, Vector & rhs);
    virtual void EliminateTestDofs (Array < int >&bdr_attr_is_ess);
    void Update ();
    virtual ~ MixedBilinearForm ();
  };
  class DiscreteLinearOperator:public MixedBilinearForm
  {
  public:
    DiscreteLinearOperator (FiniteElementSpace * domain_fes,
			    FiniteElementSpace *
			    range_fes):MixedBilinearForm (domain_fes,
							  range_fes)
    {
    }
    void AddDomainInterpolator (DiscreteInterpolator * di)
    {
      AddDomainIntegrator (di);
    }
    Array < BilinearFormIntegrator * >*GetDI ()
    {
      return &dom;
    }
    virtual void Assemble (int skip_zeros = 1);
  };
}


namespace __gnu_cxx __attribute__ ((__visibility__ ("default")))
{

  template < typename _Alloc > struct __alloc_traits
  {
    typedef _Alloc allocator_type;
    typedef typename _Alloc::pointer pointer;
    typedef typename _Alloc::const_pointer const_pointer;
    typedef typename _Alloc::value_type value_type;
    typedef typename _Alloc::reference reference;
    typedef typename _Alloc::const_reference const_reference;
    typedef typename _Alloc::size_type size_type;
    typedef typename _Alloc::difference_type difference_type;
    static pointer allocate (_Alloc & __a, size_type __n)
    {
      return __a.allocate (__n);
    }
    static void deallocate (_Alloc & __a, pointer __p, size_type __n)
    {
      __a.deallocate (__p, __n);
    }
    template < typename _Tp >
      static void construct (_Alloc & __a, pointer __p, const _Tp & __arg)
    {
      __a.construct (__p, __arg);
    }
    static void destroy (_Alloc & __a, pointer __p)
    {
      __a.destroy (__p);
    }
    static size_type max_size (const _Alloc & __a)
    {
      return __a.max_size ();
    }
    static const _Alloc & _S_select_on_copy (const _Alloc & __a)
    {
      return __a;
    }
    static void _S_on_swap (_Alloc & __a, _Alloc & __b)
    {
      std::__alloc_swap < _Alloc >::_S_do_it (__a, __b);
    }
    template < typename _Tp > struct rebind
    {
      typedef typename _Alloc::template rebind < _Tp >::other other;
    };
  };

}

namespace std __attribute__ ((__visibility__ ("default")))
{

  enum _Rb_tree_color
  { _S_red = false, _S_black = true };
  struct _Rb_tree_node_base
  {
    typedef _Rb_tree_node_base *_Base_ptr;
    typedef const _Rb_tree_node_base *_Const_Base_ptr;
    _Rb_tree_color _M_color;
    _Base_ptr _M_parent;
    _Base_ptr _M_left;
    _Base_ptr _M_right;
    static _Base_ptr _S_minimum (_Base_ptr __x)
    {
      while (__x->_M_left != 0)
	__x = __x->_M_left;
      return __x;
    }
    static _Const_Base_ptr _S_minimum (_Const_Base_ptr __x)
    {
      while (__x->_M_left != 0)
	__x = __x->_M_left;
      return __x;
    }
    static _Base_ptr _S_maximum (_Base_ptr __x)
    {
      while (__x->_M_right != 0)
	__x = __x->_M_right;
      return __x;
    }
    static _Const_Base_ptr _S_maximum (_Const_Base_ptr __x)
    {
      while (__x->_M_right != 0)
	__x = __x->_M_right;
      return __x;
    }
  };
  template < typename _Val > struct _Rb_tree_node:public _Rb_tree_node_base
  {
    typedef _Rb_tree_node < _Val > *_Link_type;
    _Val _M_value_field;
    _Val *_M_valptr ()
    {
      return std::__addressof (_M_value_field);
    }
    const _Val *_M_valptr () const
    {
      return std::__addressof (_M_value_field);
    }
  };
  __attribute__ ((__pure__)) _Rb_tree_node_base
    *_Rb_tree_increment (_Rb_tree_node_base * __x) throw ();
  __attribute__ ((__pure__)) const _Rb_tree_node_base
    *_Rb_tree_increment (const _Rb_tree_node_base * __x) throw ();
  __attribute__ ((__pure__)) _Rb_tree_node_base
    *_Rb_tree_decrement (_Rb_tree_node_base * __x) throw ();
  __attribute__ ((__pure__)) const _Rb_tree_node_base
    *_Rb_tree_decrement (const _Rb_tree_node_base * __x) throw ();
  template < typename _Tp > struct _Rb_tree_iterator
  {
    typedef _Tp value_type;
    typedef _Tp & reference;
    typedef _Tp *pointer;
    typedef bidirectional_iterator_tag iterator_category;
    typedef ptrdiff_t difference_type;
    typedef _Rb_tree_iterator < _Tp > _Self;
    typedef _Rb_tree_node_base::_Base_ptr _Base_ptr;
    typedef _Rb_tree_node < _Tp > *_Link_type;
      _Rb_tree_iterator ():_M_node ()
    {
    }
    explicit _Rb_tree_iterator (_Link_type __x):_M_node (__x)
    {
    }
    reference operator* ()const
    {
      return *static_cast < _Link_type > (_M_node)->_M_valptr ();
    }
    pointer operator-> () const
    {
      return static_cast < _Link_type > (_M_node)->_M_valptr ();
    }
    _Self & operator++ ()
    {
      _M_node = _Rb_tree_increment (_M_node);
      return *this;
    }
    _Self operator++ (int)
    {
      _Self __tmp = *this;
      _M_node = _Rb_tree_increment (_M_node);
      return __tmp;
    }
    _Self & operator-- ()
    {
      _M_node = _Rb_tree_decrement (_M_node);
      return *this;
    }
    _Self operator-- (int)
    {
      _Self __tmp = *this;
      _M_node = _Rb_tree_decrement (_M_node);
      return __tmp;
    }
    bool operator== (const _Self & __x) const
    {
      return _M_node == __x._M_node;
    }
    bool operator!= (const _Self & __x) const
    {
      return _M_node != __x._M_node;
    }
    _Base_ptr _M_node;
  };
  template < typename _Tp > struct _Rb_tree_const_iterator
  {
    typedef _Tp value_type;
    typedef const _Tp & reference;
    typedef const _Tp *pointer;
    typedef _Rb_tree_iterator < _Tp > iterator;
    typedef bidirectional_iterator_tag iterator_category;
    typedef ptrdiff_t difference_type;
    typedef _Rb_tree_const_iterator < _Tp > _Self;
    typedef _Rb_tree_node_base::_Const_Base_ptr _Base_ptr;
    typedef const _Rb_tree_node < _Tp > *_Link_type;
      _Rb_tree_const_iterator ():_M_node ()
    {
    }
    explicit _Rb_tree_const_iterator (_Link_type __x):_M_node (__x)
    {
    }
    _Rb_tree_const_iterator (const iterator & __it):_M_node (__it._M_node)
    {
    }
    iterator _M_const_cast ()const
    {
      return iterator (static_cast < typename iterator::_Link_type >
		       (const_cast < typename iterator::_Base_ptr >
			(_M_node)));
    }
    reference operator* () const
    {
      return *static_cast < _Link_type > (_M_node)->_M_valptr ();
    }
    pointer operator-> () const
    {
      return static_cast < _Link_type > (_M_node)->_M_valptr ();
    }
    _Self & operator++ ()
    {
      _M_node = _Rb_tree_increment (_M_node);
      return *this;
    }
    _Self operator++ (int)
    {
      _Self __tmp = *this;
      _M_node = _Rb_tree_increment (_M_node);
      return __tmp;
    }
    _Self & operator-- ()
    {
      _M_node = _Rb_tree_decrement (_M_node);
      return *this;
    }
    _Self operator-- (int)
    {
      _Self __tmp = *this;
      _M_node = _Rb_tree_decrement (_M_node);
      return __tmp;
    }
    bool operator== (const _Self & __x) const
    {
      return _M_node == __x._M_node;
    }
    bool operator!= (const _Self & __x) const
    {
      return _M_node != __x._M_node;
    }
    _Base_ptr _M_node;
  };
  template < typename _Val >
    inline bool
    operator== (const _Rb_tree_iterator < _Val > &__x,
		const _Rb_tree_const_iterator < _Val > &__y)
  {
    return __x._M_node == __y._M_node;
  }
  template < typename _Val >
    inline bool
    operator!= (const _Rb_tree_iterator < _Val > &__x,
		const _Rb_tree_const_iterator < _Val > &__y)
  {
    return __x._M_node != __y._M_node;
  }
  void
    _Rb_tree_insert_and_rebalance (const bool __insert_left,
				   _Rb_tree_node_base * __x,
				   _Rb_tree_node_base * __p,
				   _Rb_tree_node_base & __header) throw ();
  _Rb_tree_node_base *_Rb_tree_rebalance_for_erase (_Rb_tree_node_base *
						    const __z,
						    _Rb_tree_node_base &
						    __header) throw ();
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc = allocator < _Val > >class _Rb_tree
  {
    typedef typename __gnu_cxx::__alloc_traits < _Alloc >::template
      rebind < _Rb_tree_node < _Val > >::other _Node_allocator;
    typedef __gnu_cxx::__alloc_traits < _Node_allocator > _Alloc_traits;
  protected:
    typedef _Rb_tree_node_base *_Base_ptr;
    typedef const _Rb_tree_node_base *_Const_Base_ptr;
  public:
    typedef _Key key_type;
    typedef _Val value_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type & reference;
    typedef const value_type & const_reference;
    typedef _Rb_tree_node < _Val > *_Link_type;
    typedef const _Rb_tree_node < _Val > *_Const_Link_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Alloc allocator_type;
    _Node_allocator & _M_get_Node_allocator ()
    {
      return *static_cast < _Node_allocator * >(&this->_M_impl);
    }
    const _Node_allocator & _M_get_Node_allocator () const
    {
      return *static_cast < const _Node_allocator *>(&this->_M_impl);
    }
    allocator_type get_allocator () const
    {
      return allocator_type (_M_get_Node_allocator ());
    }
  protected:
      _Link_type _M_get_node ()
    {
      return _Alloc_traits::allocate (_M_get_Node_allocator (), 1);
    }
    void _M_put_node (_Link_type __p)
    {
      _Alloc_traits::deallocate (_M_get_Node_allocator (), __p, 1);
    }
    _Link_type _M_create_node (const value_type & __x)
    {
      _Link_type __tmp = _M_get_node ();
      try
      {
	get_allocator ().construct (__tmp->_M_valptr (), __x);
      }
      catch ( ...)
      {
	_M_put_node (__tmp);
	throw;
      }
      return __tmp;
    }
    void _M_destroy_node (_Link_type __p)
    {
      get_allocator ().destroy (__p->_M_valptr ());
      _M_put_node (__p);
    }
    _Link_type _M_clone_node (_Const_Link_type __x)
    {
      _Link_type __tmp = _M_create_node (*__x->_M_valptr ());
      __tmp->_M_color = __x->_M_color;
      __tmp->_M_left = 0;
      __tmp->_M_right = 0;
      return __tmp;
    }
  protected:
    template < typename _Key_compare,
      bool _Is_pod_comparator = __is_pod (_Key_compare) >
      struct _Rb_tree_impl:public _Node_allocator
    {
      _Key_compare _M_key_compare;
      _Rb_tree_node_base _M_header;
      size_type _M_node_count;
        _Rb_tree_impl ():_Node_allocator (), _M_key_compare (), _M_header (),
	_M_node_count (0)
      {
	_M_initialize ();
      }
      _Rb_tree_impl (const _Key_compare & __comp,
		     const _Node_allocator & __a):_Node_allocator (__a),
	_M_key_compare (__comp), _M_header (), _M_node_count (0)
      {
	_M_initialize ();
      }
    private:
      void _M_initialize ()
      {
	this->_M_header._M_color = _S_red;
	this->_M_header._M_parent = 0;
	this->_M_header._M_left = &this->_M_header;
	this->_M_header._M_right = &this->_M_header;
      }
    };
    _Rb_tree_impl < _Compare > _M_impl;
  protected:
    _Base_ptr & _M_root ()
    {
      return this->_M_impl._M_header._M_parent;
    }
    _Const_Base_ptr _M_root ()const
    {
      return this->_M_impl._M_header._M_parent;
    }
    _Base_ptr & _M_leftmost ()
    {
      return this->_M_impl._M_header._M_left;
    }
    _Const_Base_ptr _M_leftmost ()const
    {
      return this->_M_impl._M_header._M_left;
    }
    _Base_ptr & _M_rightmost ()
    {
      return this->_M_impl._M_header._M_right;
    }
    _Const_Base_ptr _M_rightmost ()const
    {
      return this->_M_impl._M_header._M_right;
    }
    _Link_type _M_begin ()
    {
      return static_cast < _Link_type > (this->_M_impl._M_header._M_parent);
    }
    _Const_Link_type _M_begin ()const
    {
      return static_cast < _Const_Link_type >
	(this->_M_impl._M_header._M_parent);
    }
    _Link_type _M_end ()
    {
      return reinterpret_cast < _Link_type > (&this->_M_impl._M_header);
    }
    _Const_Link_type _M_end ()const
    {
      return reinterpret_cast < _Const_Link_type > (&this->_M_impl._M_header);
    }
    static const_reference _S_value (_Const_Link_type __x)
    {
      return *__x->_M_valptr ();
    }
    static const _Key & _S_key (_Const_Link_type __x)
    {
      return _KeyOfValue ()(_S_value (__x));
    }
    static _Link_type _S_left (_Base_ptr __x)
    {
      return static_cast < _Link_type > (__x->_M_left);
    }
    static _Const_Link_type _S_left (_Const_Base_ptr __x)
    {
      return static_cast < _Const_Link_type > (__x->_M_left);
    }
    static _Link_type _S_right (_Base_ptr __x)
    {
      return static_cast < _Link_type > (__x->_M_right);
    }
    static _Const_Link_type _S_right (_Const_Base_ptr __x)
    {
      return static_cast < _Const_Link_type > (__x->_M_right);
    }
    static const_reference _S_value (_Const_Base_ptr __x)
    {
      return *static_cast < _Const_Link_type > (__x)->_M_valptr ();
    }
    static const _Key & _S_key (_Const_Base_ptr __x)
    {
      return _KeyOfValue ()(_S_value (__x));
    }
    static _Base_ptr _S_minimum (_Base_ptr __x)
    {
      return _Rb_tree_node_base::_S_minimum (__x);
    }
    static _Const_Base_ptr _S_minimum (_Const_Base_ptr __x)
    {
      return _Rb_tree_node_base::_S_minimum (__x);
    }
    static _Base_ptr _S_maximum (_Base_ptr __x)
    {
      return _Rb_tree_node_base::_S_maximum (__x);
    }
    static _Const_Base_ptr _S_maximum (_Const_Base_ptr __x)
    {
      return _Rb_tree_node_base::_S_maximum (__x);
    }
  public:
    typedef _Rb_tree_iterator < value_type > iterator;
    typedef _Rb_tree_const_iterator < value_type > const_iterator;
    typedef std::reverse_iterator < iterator > reverse_iterator;
    typedef std::reverse_iterator < const_iterator > const_reverse_iterator;
  private:
    pair < _Base_ptr, _Base_ptr >
      _M_get_insert_unique_pos (const key_type & __k);
    pair < _Base_ptr, _Base_ptr >
      _M_get_insert_equal_pos (const key_type & __k);
    pair < _Base_ptr, _Base_ptr >
      _M_get_insert_hint_unique_pos (const_iterator __pos,
				     const key_type & __k);
    pair < _Base_ptr, _Base_ptr >
      _M_get_insert_hint_equal_pos (const_iterator __pos,
				    const key_type & __k);
    iterator
      _M_insert_ (_Base_ptr __x, _Base_ptr __y, const value_type & __v);
    iterator _M_insert_lower (_Base_ptr __y, const value_type & __v);
    iterator _M_insert_equal_lower (const value_type & __x);
    _Link_type _M_copy (_Const_Link_type __x, _Link_type __p);
    void _M_erase (_Link_type __x);
    iterator
      _M_lower_bound (_Link_type __x, _Link_type __y, const _Key & __k);
    const_iterator
      _M_lower_bound (_Const_Link_type __x, _Const_Link_type __y,
		      const _Key & __k) const;
    iterator
      _M_upper_bound (_Link_type __x, _Link_type __y, const _Key & __k);
    const_iterator
      _M_upper_bound (_Const_Link_type __x, _Const_Link_type __y,
		      const _Key & __k) const;
  public:
    _Rb_tree ()
    {
    }
  _Rb_tree (const _Compare & __comp, const allocator_type & __a = allocator_type ()):_M_impl (__comp,
	     _Node_allocator
	     (__a))
    {
    }
    _Rb_tree (const _Rb_tree & __x):_M_impl (__x._M_impl._M_key_compare,
					     _Alloc_traits::
					     _S_select_on_copy (__x.
								_M_get_Node_allocator
								()))
    {
      if (__x._M_root () != 0)
	{
	  _M_root () = _M_copy (__x._M_begin (), _M_end ());
	  _M_leftmost () = _S_minimum (_M_root ());
	  _M_rightmost () = _S_maximum (_M_root ());
	  _M_impl._M_node_count = __x._M_impl._M_node_count;
	}
    }
    ~_Rb_tree ()
    {
      _M_erase (_M_begin ());
    }
    _Rb_tree & operator= (const _Rb_tree & __x);
    _Compare key_comp ()const
    {
      return _M_impl._M_key_compare;
    }
    iterator begin ()
    {
      return iterator (static_cast < _Link_type >
		       (this->_M_impl._M_header._M_left));
    }
    const_iterator begin ()const
    {
      return const_iterator (static_cast < _Const_Link_type >
			     (this->_M_impl._M_header._M_left));
    }
    iterator end ()
    {
      return iterator (static_cast < _Link_type > (&this->_M_impl._M_header));
    }
    const_iterator end ()const
    {
      return const_iterator (static_cast < _Const_Link_type >
			     (&this->_M_impl._M_header));
    }
    reverse_iterator rbegin ()
    {
      return reverse_iterator (end ());
    }
    const_reverse_iterator rbegin ()const
    {
      return const_reverse_iterator (end ());
    }
    reverse_iterator rend ()
    {
      return reverse_iterator (begin ());
    }
    const_reverse_iterator rend ()const
    {
      return const_reverse_iterator (begin ());
    }
    bool empty () const
    {
      return _M_impl._M_node_count == 0;
    }
    size_type size () const
    {
      return _M_impl._M_node_count;
    }
    size_type max_size () const
    {
      return _Alloc_traits::max_size (_M_get_Node_allocator ());
    }
    void swap (_Rb_tree & __t);
    pair < iterator, bool > _M_insert_unique (const value_type & __x);
    iterator _M_insert_equal (const value_type & __x);
    iterator
      _M_insert_unique_ (const_iterator __position, const value_type & __x);
    iterator
      _M_insert_equal_ (const_iterator __position, const value_type & __x);
    template < typename _InputIterator >
      void _M_insert_unique (_InputIterator __first, _InputIterator __last);
    template < typename _InputIterator >
      void _M_insert_equal (_InputIterator __first, _InputIterator __last);
  private:
    void _M_erase_aux (const_iterator __position);
    void _M_erase_aux (const_iterator __first, const_iterator __last);
  public:
    void erase (iterator __position)
    {
      _M_erase_aux (__position);
    }
    void erase (const_iterator __position)
    {
      _M_erase_aux (__position);
    }
    size_type erase (const key_type & __x);
    void erase (iterator __first, iterator __last)
    {
      _M_erase_aux (__first, __last);
    }
    void erase (const_iterator __first, const_iterator __last)
    {
      _M_erase_aux (__first, __last);
    }
    void erase (const key_type * __first, const key_type * __last);
    void clear ()
    {
      _M_erase (_M_begin ());
      _M_leftmost () = _M_end ();
      _M_root () = 0;
      _M_rightmost () = _M_end ();
      _M_impl._M_node_count = 0;
    }
    iterator find (const key_type & __k);
    const_iterator find (const key_type & __k) const;
    size_type count (const key_type & __k) const;
    iterator lower_bound (const key_type & __k)
    {
      return _M_lower_bound (_M_begin (), _M_end (), __k);
    }
    const_iterator lower_bound (const key_type & __k) const
    {
      return _M_lower_bound (_M_begin (), _M_end (), __k);
    }
    iterator upper_bound (const key_type & __k)
    {
      return _M_upper_bound (_M_begin (), _M_end (), __k);
    }
    const_iterator upper_bound (const key_type & __k) const
    {
      return _M_upper_bound (_M_begin (), _M_end (), __k);
    }
    pair < iterator, iterator > equal_range (const key_type & __k);
    pair < const_iterator, const_iterator >
      equal_range (const key_type & __k) const;
    bool __rb_verify ()const;
  };
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    inline bool
    operator== (const _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
		_Alloc > &__x, const _Rb_tree < _Key, _Val, _KeyOfValue,
		_Compare, _Alloc > &__y)
  {
    return __x.size () == __y.size ()
      && std::equal (__x.begin (), __x.end (), __y.begin ());
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    inline bool
    operator< (const _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
	       _Alloc > &__x, const _Rb_tree < _Key, _Val, _KeyOfValue,
	       _Compare, _Alloc > &__y)
  {
    return std::lexicographical_compare (__x.begin (), __x.end (),
					 __y.begin (), __y.end ());
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    inline bool
    operator!= (const _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
		_Alloc > &__x, const _Rb_tree < _Key, _Val, _KeyOfValue,
		_Compare, _Alloc > &__y)
  {
    return !(__x == __y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    inline bool
    operator> (const _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
	       _Alloc > &__x, const _Rb_tree < _Key, _Val, _KeyOfValue,
	       _Compare, _Alloc > &__y)
  {
    return __y < __x;
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    inline bool
    operator<= (const _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
		_Alloc > &__x, const _Rb_tree < _Key, _Val, _KeyOfValue,
		_Compare, _Alloc > &__y)
  {
    return !(__y < __x);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    inline bool
    operator>= (const _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
		_Alloc > &__x, const _Rb_tree < _Key, _Val, _KeyOfValue,
		_Compare, _Alloc > &__y)
  {
    return !(__x < __y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    inline void
    swap (_Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc > &__x,
	  _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc > &__y)
  {
    __x.swap (__y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc > &_Rb_tree < _Key,
    _Val, _KeyOfValue, _Compare, _Alloc >::operator= (const _Rb_tree & __x)
  {
    if (this != &__x)
      {
	clear ();
	_M_impl._M_key_compare = __x._M_impl._M_key_compare;
	if (__x._M_root () != 0)
	  {
	    _M_root () = _M_copy (__x._M_begin (), _M_end ());
	    _M_leftmost () = _S_minimum (_M_root ());
	    _M_rightmost () = _S_maximum (_M_root ());
	    _M_impl._M_node_count = __x._M_impl._M_node_count;
	  }
      }
    return *this;
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_insert_ (_Base_ptr __x, _Base_ptr __p, const _Val & __v)
  {
    bool __insert_left = (__x != 0 || __p == _M_end ()
			  || _M_impl._M_key_compare (_KeyOfValue ()(__v),
						     _S_key (__p)));
    _Link_type __z = _M_create_node ((__v));
    _Rb_tree_insert_and_rebalance (__insert_left, __z, __p,
				   this->_M_impl._M_header);
    ++_M_impl._M_node_count;
    return iterator (__z);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_insert_lower (_Base_ptr __p, const _Val & __v)
  {
    bool __insert_left = (__p == _M_end ()
			  || !_M_impl._M_key_compare (_S_key (__p),
						      _KeyOfValue ()(__v)));
    _Link_type __z = _M_create_node ((__v));
    _Rb_tree_insert_and_rebalance (__insert_left, __z, __p,
				   this->_M_impl._M_header);
    ++_M_impl._M_node_count;
    return iterator (__z);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_insert_equal_lower (const _Val & __v)
  {
    _Link_type __x = _M_begin ();
    _Link_type __y = _M_end ();
    while (__x != 0)
      {
	__y = __x;
	__x = !_M_impl._M_key_compare (_S_key (__x), _KeyOfValue ()(__v)) ?
	  _S_left (__x) : _S_right (__x);
      }
    return _M_insert_lower (__y, (__v));
  }
  template < typename _Key, typename _Val, typename _KoV,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KoV, _Compare, _Alloc >::_Link_type
    _Rb_tree < _Key, _Val, _KoV, _Compare,
    _Alloc >::_M_copy (_Const_Link_type __x, _Link_type __p)
  {
    _Link_type __top = _M_clone_node (__x);
    __top->_M_parent = __p;
    try
    {
      if (__x->_M_right)
	__top->_M_right = _M_copy (_S_right (__x), __top);
      __p = __top;
      __x = _S_left (__x);
      while (__x != 0)
	{
	  _Link_type __y = _M_clone_node (__x);
	  __p->_M_left = __y;
	  __y->_M_parent = __p;
	  if (__x->_M_right)
	    __y->_M_right = _M_copy (_S_right (__x), __y);
	  __p = __y;
	  __x = _S_left (__x);
	}
    }
    catch ( ...)
    {
      _M_erase (__top);
      throw;
    }
    return __top;
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    void
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_erase (_Link_type __x)
  {
    while (__x != 0)
      {
	_M_erase (_S_right (__x));
	_Link_type __y = _S_left (__x);
	_M_destroy_node (__x);
	__x = __y;
      }
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_lower_bound (_Link_type __x, _Link_type __y,
			      const _Key & __k)
  {
    while (__x != 0)
      if (!_M_impl._M_key_compare (_S_key (__x), __k))
	__y = __x, __x = _S_left (__x);
      else
	__x = _S_right (__x);
    return iterator (__y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::const_iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_lower_bound (_Const_Link_type __x, _Const_Link_type __y,
			      const _Key & __k) const
  {
    while (__x != 0)
      if (!_M_impl._M_key_compare (_S_key (__x), __k))
	__y = __x, __x = _S_left (__x);
      else
	__x = _S_right (__x);
    return const_iterator (__y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_upper_bound (_Link_type __x, _Link_type __y,
			      const _Key & __k)
  {
    while (__x != 0)
      if (_M_impl._M_key_compare (__k, _S_key (__x)))
	__y = __x, __x = _S_left (__x);
      else
	__x = _S_right (__x);
    return iterator (__y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::const_iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_upper_bound (_Const_Link_type __x, _Const_Link_type __y,
			      const _Key & __k) const
  {
    while (__x != 0)
      if (_M_impl._M_key_compare (__k, _S_key (__x)))
	__y = __x, __x = _S_left (__x);
      else
	__x = _S_right (__x);
    return const_iterator (__y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    pair < typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::iterator,
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::iterator >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::equal_range (const _Key & __k)
  {
    _Link_type __x = _M_begin ();
    _Link_type __y = _M_end ();
    while (__x != 0)
      {
	if (_M_impl._M_key_compare (_S_key (__x), __k))
	  __x = _S_right (__x);
	else if (_M_impl._M_key_compare (__k, _S_key (__x)))
	  __y = __x, __x = _S_left (__x);
	else
	  {
	    _Link_type __xu (__x), __yu (__y);
	    __y = __x, __x = _S_left (__x);
	    __xu = _S_right (__xu);
	    return pair < iterator,
	      iterator > (_M_lower_bound (__x, __y, __k),
			  _M_upper_bound (__xu, __yu, __k));
	  }
      }
    return pair < iterator, iterator > (iterator (__y), iterator (__y));
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    pair < typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::const_iterator,
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::const_iterator >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::equal_range (const _Key & __k) const
  {
    _Const_Link_type __x = _M_begin ();
    _Const_Link_type __y = _M_end ();
    while (__x != 0)
      {
	if (_M_impl._M_key_compare (_S_key (__x), __k))
	  __x = _S_right (__x);
	else if (_M_impl._M_key_compare (__k, _S_key (__x)))
	  __y = __x, __x = _S_left (__x);
	else
	  {
	    _Const_Link_type __xu (__x), __yu (__y);
	      __y = __x, __x = _S_left (__x);
	      __xu = _S_right (__xu);
	      return pair < const_iterator,
	      const_iterator > (_M_lower_bound (__x, __y, __k),
				_M_upper_bound (__xu, __yu, __k));
	  }
      }
    return pair < const_iterator, const_iterator > (const_iterator (__y),
						    const_iterator (__y));
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    void
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::swap (_Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
		    _Alloc > &__t)
  {
    if (_M_root () == 0)
      {
	if (__t._M_root () != 0)
	  {
	    _M_root () = __t._M_root ();
	    _M_leftmost () = __t._M_leftmost ();
	    _M_rightmost () = __t._M_rightmost ();
	    _M_root ()->_M_parent = _M_end ();
	    __t._M_root () = 0;
	    __t._M_leftmost () = __t._M_end ();
	    __t._M_rightmost () = __t._M_end ();
	  }
      }
    else if (__t._M_root () == 0)
      {
	__t._M_root () = _M_root ();
	__t._M_leftmost () = _M_leftmost ();
	__t._M_rightmost () = _M_rightmost ();
	__t._M_root ()->_M_parent = __t._M_end ();
	_M_root () = 0;
	_M_leftmost () = _M_end ();
	_M_rightmost () = _M_end ();
      }
    else
      {
	std::swap (_M_root (), __t._M_root ());
	std::swap (_M_leftmost (), __t._M_leftmost ());
	std::swap (_M_rightmost (), __t._M_rightmost ());
	_M_root ()->_M_parent = _M_end ();
	__t._M_root ()->_M_parent = __t._M_end ();
      }
    std::swap (this->_M_impl._M_node_count, __t._M_impl._M_node_count);
    std::swap (this->_M_impl._M_key_compare, __t._M_impl._M_key_compare);
    _Alloc_traits::_S_on_swap (_M_get_Node_allocator (),
			       __t._M_get_Node_allocator ());
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    pair < typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr,
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_get_insert_unique_pos (const key_type & __k)
  {
    typedef pair < _Base_ptr, _Base_ptr > _Res;
    _Link_type __x = _M_begin ();
    _Link_type __y = _M_end ();
    bool __comp = true;
    while (__x != 0)
      {
	__y = __x;
	__comp = _M_impl._M_key_compare (__k, _S_key (__x));
	__x = __comp ? _S_left (__x) : _S_right (__x);
      }
    iterator __j = iterator (__y);
    if (__comp)
      {
	if (__j == begin ())
	  return _Res (__x, __y);
	else
	  --__j;
      }
    if (_M_impl._M_key_compare (_S_key (__j._M_node), __k))
      return _Res (__x, __y);
    return _Res (__j._M_node, 0);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    pair < typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr,
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_get_insert_equal_pos (const key_type & __k)
  {
    typedef pair < _Base_ptr, _Base_ptr > _Res;
    _Link_type __x = _M_begin ();
    _Link_type __y = _M_end ();
    while (__x != 0)
      {
	__y = __x;
	__x = _M_impl._M_key_compare (__k, _S_key (__x)) ?
	  _S_left (__x) : _S_right (__x);
      }
    return _Res (__x, __y);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    pair < typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::iterator, bool >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_insert_unique (const _Val & __v)
  {
    typedef pair < iterator, bool > _Res;
    pair < _Base_ptr, _Base_ptr > __res
      = _M_get_insert_unique_pos (_KeyOfValue ()(__v));
    if (__res.second)
      return _Res (_M_insert_ (__res.first, __res.second, (__v)), true);
    return _Res (iterator (static_cast < _Link_type > (__res.first)), false);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_insert_equal (const _Val & __v)
  {
    pair < _Base_ptr, _Base_ptr > __res
      = _M_get_insert_equal_pos (_KeyOfValue ()(__v));
    return _M_insert_ (__res.first, __res.second, (__v));
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    pair < typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr,
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_get_insert_hint_unique_pos (const_iterator __position,
					     const key_type & __k)
  {
    iterator __pos = __position._M_const_cast ();
    typedef pair < _Base_ptr, _Base_ptr > _Res;
    if (__pos._M_node == _M_end ())
      {
	if (size () > 0
	    && _M_impl._M_key_compare (_S_key (_M_rightmost ()), __k))
	  return _Res (0, _M_rightmost ());
	else
	  return _M_get_insert_unique_pos (__k);
      }
    else if (_M_impl._M_key_compare (__k, _S_key (__pos._M_node)))
      {
	iterator __before = __pos;
	if (__pos._M_node == _M_leftmost ())
	  return _Res (_M_leftmost (), _M_leftmost ());
	else if (_M_impl._M_key_compare (_S_key ((--__before)._M_node), __k))
	  {
	    if (_S_right (__before._M_node) == 0)
	      return _Res (0, __before._M_node);
	    else
	      return _Res (__pos._M_node, __pos._M_node);
	  }
	else
	  return _M_get_insert_unique_pos (__k);
      }
    else if (_M_impl._M_key_compare (_S_key (__pos._M_node), __k))
      {
	iterator __after = __pos;
	if (__pos._M_node == _M_rightmost ())
	  return _Res (0, _M_rightmost ());
	else if (_M_impl._M_key_compare (__k, _S_key ((++__after)._M_node)))
	  {
	    if (_S_right (__pos._M_node) == 0)
	      return _Res (0, __pos._M_node);
	    else
	      return _Res (__after._M_node, __after._M_node);
	  }
	else
	  return _M_get_insert_unique_pos (__k);
      }
    else
      return _Res (__pos._M_node, 0);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_insert_unique_ (const_iterator __position, const _Val & __v)
  {
    pair < _Base_ptr, _Base_ptr > __res
      = _M_get_insert_hint_unique_pos (__position, _KeyOfValue ()(__v));
    if (__res.second)
      return _M_insert_ (__res.first, __res.second, (__v));
    return iterator (static_cast < _Link_type > (__res.first));
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    pair < typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr,
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::_Base_ptr >
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_get_insert_hint_equal_pos (const_iterator __position,
					    const key_type & __k)
  {
    iterator __pos = __position._M_const_cast ();
    typedef pair < _Base_ptr, _Base_ptr > _Res;
    if (__pos._M_node == _M_end ())
      {
	if (size () > 0
	    && !_M_impl._M_key_compare (__k, _S_key (_M_rightmost ())))
	  return _Res (0, _M_rightmost ());
	else
	  return _M_get_insert_equal_pos (__k);
      }
    else if (!_M_impl._M_key_compare (_S_key (__pos._M_node), __k))
      {
	iterator __before = __pos;
	if (__pos._M_node == _M_leftmost ())
	  return _Res (_M_leftmost (), _M_leftmost ());
	else if (!_M_impl._M_key_compare (__k, _S_key ((--__before)._M_node)))
	  {
	    if (_S_right (__before._M_node) == 0)
	      return _Res (0, __before._M_node);
	    else
	      return _Res (__pos._M_node, __pos._M_node);
	  }
	else
	  return _M_get_insert_equal_pos (__k);
      }
    else
      {
	iterator __after = __pos;
	if (__pos._M_node == _M_rightmost ())
	  return _Res (0, _M_rightmost ());
	else if (!_M_impl._M_key_compare (_S_key ((++__after)._M_node), __k))
	  {
	    if (_S_right (__pos._M_node) == 0)
	      return _Res (0, __pos._M_node);
	    else
	      return _Res (__after._M_node, __after._M_node);
	  }
	else
	  return _Res (0, 0);
      }
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_insert_equal_ (const_iterator __position, const _Val & __v)
  {
    pair < _Base_ptr, _Base_ptr > __res
      = _M_get_insert_hint_equal_pos (__position, _KeyOfValue ()(__v));
    if (__res.second)
      return _M_insert_ (__res.first, __res.second, (__v));
    return _M_insert_equal_lower ((__v));
  }
  template < typename _Key, typename _Val, typename _KoV,
    typename _Cmp, typename _Alloc >
    template < class _II >
    void
    _Rb_tree < _Key, _Val, _KoV, _Cmp,
    _Alloc >::_M_insert_unique (_II __first, _II __last)
  {
    for (; __first != __last; ++__first)
      _M_insert_unique_ (end (), *__first);
  }
  template < typename _Key, typename _Val, typename _KoV,
    typename _Cmp, typename _Alloc >
    template < class _II >
    void
    _Rb_tree < _Key, _Val, _KoV, _Cmp, _Alloc >::_M_insert_equal (_II __first,
								  _II __last)
  {
    for (; __first != __last; ++__first)
      _M_insert_equal_ (end (), *__first);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    void
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_erase_aux (const_iterator __position)
  {
    _Link_type __y =
      static_cast < _Link_type > (_Rb_tree_rebalance_for_erase
				  (const_cast < _Base_ptr >
				   (__position._M_node),
				   this->_M_impl._M_header));
    _M_destroy_node (__y);
    --_M_impl._M_node_count;
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    void
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::_M_erase_aux (const_iterator __first, const_iterator __last)
  {
    if (__first == begin () && __last == end ())
      clear ();
    else
      while (__first != __last)
	erase (__first++);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::size_type
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::erase (const _Key & __x)
  {
    pair < iterator, iterator > __p = equal_range (__x);
    const size_type __old_size = size ();
    erase (__p.first, __p.second);
    return __old_size - size ();
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    void
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::erase (const _Key * __first, const _Key * __last)
  {
    while (__first != __last)
      erase (*__first++);
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::find (const _Key & __k)
  {
    iterator __j = _M_lower_bound (_M_begin (), _M_end (), __k);
    return (__j == end ()
	    || _M_impl._M_key_compare (__k,
				       _S_key (__j._M_node))) ? end () : __j;
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue,
    _Compare, _Alloc >::const_iterator
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::find (const _Key & __k) const
  {
    const_iterator __j = _M_lower_bound (_M_begin (), _M_end (), __k);
      return (__j == end ()
	      || _M_impl._M_key_compare (__k,
					 _S_key (__j.
						 _M_node))) ? end () : __j;
  }
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    typename _Rb_tree < _Key, _Val, _KeyOfValue, _Compare, _Alloc >::size_type
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::count (const _Key & __k) const
  {
    pair < const_iterator, const_iterator > __p = equal_range (__k);
    const size_type __n = std::distance (__p.first, __p.second);
      return __n;
  }
  __attribute__ ((__pure__)) unsigned int
    _Rb_tree_black_count (const _Rb_tree_node_base * __node,
			  const _Rb_tree_node_base * __root) throw ();
  template < typename _Key, typename _Val, typename _KeyOfValue,
    typename _Compare, typename _Alloc >
    bool
    _Rb_tree < _Key, _Val, _KeyOfValue, _Compare,
    _Alloc >::__rb_verify ()const
  {
    if (_M_impl._M_node_count == 0 || begin () == end ())
      return _M_impl._M_node_count == 0 && begin () == end ()
	&& this->_M_impl._M_header._M_left == _M_end ()
	&& this->_M_impl._M_header._M_right == _M_end ();
    unsigned int __len = _Rb_tree_black_count (_M_leftmost (), _M_root ());
    for (const_iterator __it = begin (); __it != end (); ++__it)
      {
	_Const_Link_type __x =
	  static_cast < _Const_Link_type > (__it._M_node);
	_Const_Link_type __L = _S_left (__x);
	_Const_Link_type __R = _S_right (__x);
	if (__x->_M_color == _S_red)
	  if ((__L && __L->_M_color == _S_red)
	      || (__R && __R->_M_color == _S_red))
	    return false;
	if (__L && _M_impl._M_key_compare (_S_key (__x), _S_key (__L)))
	    return false;
	if (__R && _M_impl._M_key_compare (_S_key (__R), _S_key (__x)))
	    return false;
	if (!__L && !__R && _Rb_tree_black_count (__x, _M_root ()) != __len)
	    return false;
      }
    if (_M_leftmost () != _Rb_tree_node_base::_S_minimum (_M_root ()))
        return false;
    if (_M_rightmost () != _Rb_tree_node_base::_S_maximum (_M_root ()))
      return false;
    return true;
  }

}

namespace std __attribute__ ((__visibility__ ("default")))
{

  template < typename _Key, typename _Tp, typename _Compare =
    std::less < _Key >, typename _Alloc =
    std::allocator < std::pair < const _Key, _Tp > >>class map
  {
  public:
    typedef _Key key_type;
    typedef _Tp mapped_type;
    typedef std::pair < const _Key, _Tp > value_type;
    typedef _Compare key_compare;
    typedef _Alloc allocator_type;
  private:
    typedef typename _Alloc::value_type _Alloc_value_type;



  public:
      class value_compare:public std::binary_function < value_type,
      value_type, bool >
    {
      friend class map < _Key, _Tp, _Compare, _Alloc >;
    protected:
        _Compare comp;
        value_compare (_Compare __c):comp (__c)
      {
      }
    public:
        bool operator () (const value_type & __x, const value_type & __y) const
      {
	return comp (__x.first, __y.first);
      }
    };
  private:
    typedef typename __gnu_cxx::__alloc_traits < _Alloc >::template
      rebind < value_type >::other _Pair_alloc_type;
    typedef _Rb_tree < key_type, value_type, _Select1st < value_type >,
      key_compare, _Pair_alloc_type > _Rep_type;
    _Rep_type _M_t;
    typedef __gnu_cxx::__alloc_traits < _Pair_alloc_type > _Alloc_traits;
  public:
    typedef typename _Alloc_traits::pointer pointer;
    typedef typename _Alloc_traits::const_pointer const_pointer;
    typedef typename _Alloc_traits::reference reference;
    typedef typename _Alloc_traits::const_reference const_reference;
    typedef typename _Rep_type::iterator iterator;
    typedef typename _Rep_type::const_iterator const_iterator;
    typedef typename _Rep_type::size_type size_type;
    typedef typename _Rep_type::difference_type difference_type;
    typedef typename _Rep_type::reverse_iterator reverse_iterator;
    typedef typename _Rep_type::const_reverse_iterator const_reverse_iterator;
  map ():_M_t ()
    {
    }
  explicit map (const _Compare & __comp, const allocator_type & __a = allocator_type ()):_M_t (__comp,
	  _Pair_alloc_type
	  (__a))
    {
    }
    map (const map & __x):_M_t (__x._M_t)
    {
    }
  template < typename _InputIterator > map (_InputIterator __first, _InputIterator __last):_M_t
      ()
    {
      _M_t._M_insert_unique (__first, __last);
    }
  template < typename _InputIterator > map (_InputIterator __first, _InputIterator __last, const _Compare & __comp, const allocator_type & __a = allocator_type ()):_M_t (__comp,
	  _Pair_alloc_type
	  (__a))
    {
      _M_t._M_insert_unique (__first, __last);
    }
    map & operator= (const map & __x)
    {
      _M_t = __x._M_t;
      return *this;
    }
    allocator_type get_allocator ()const
    {
      return allocator_type (_M_t.get_allocator ());
    }
    iterator begin ()
    {
      return _M_t.begin ();
    }
    const_iterator begin ()const
    {
      return _M_t.begin ();
    }
    iterator end ()
    {
      return _M_t.end ();
    }
    const_iterator end ()const
    {
      return _M_t.end ();
    }
    reverse_iterator rbegin ()
    {
      return _M_t.rbegin ();
    }
    const_reverse_iterator rbegin ()const
    {
      return _M_t.rbegin ();
    }
    reverse_iterator rend ()
    {
      return _M_t.rend ();
    }
    const_reverse_iterator rend ()const
    {
      return _M_t.rend ();
    }
    bool empty () const
    {
      return _M_t.empty ();
    }
    size_type size () const
    {
      return _M_t.size ();
    }
    size_type max_size () const
    {
      return _M_t.max_size ();
    }
    mapped_type & operator[] (const key_type & __k)
    {

      iterator __i = lower_bound (__k);
      if (__i == end () || key_comp ()(__k, (*__i).first))
	__i = insert (__i, value_type (__k, mapped_type ()));
      return (*__i).second;
    }
    mapped_type & at (const key_type & __k)
    {
      iterator __i = lower_bound (__k);
      if (__i == end () || key_comp ()(__k, (*__i).first))
	__throw_out_of_range (("map::at"));
      return (*__i).second;
    }
    const mapped_type & at (const key_type & __k) const
    {
      const_iterator __i = lower_bound (__k);
      if (__i == end () || key_comp ()(__k, (*__i).first))
	__throw_out_of_range (("map::at"));
        return (*__i).second;
    }
    std::pair < iterator, bool > insert (const value_type & __x)
    {
      return _M_t._M_insert_unique (__x);
    }
    iterator insert (iterator __position, const value_type & __x)
    {
      return _M_t._M_insert_unique_ (__position, __x);
    }
    template < typename _InputIterator >
      void insert (_InputIterator __first, _InputIterator __last)
    {
      _M_t._M_insert_unique (__first, __last);
    }
    void erase (iterator __position)
    {
      _M_t.erase (__position);
    }
    size_type erase (const key_type & __x)
    {
      return _M_t.erase (__x);
    }
    void erase (iterator __first, iterator __last)
    {
      _M_t.erase (__first, __last);
    }
    void swap (map & __x)
    {
      _M_t.swap (__x._M_t);
    }
    void clear ()
    {
      _M_t.clear ();
    }
    key_compare key_comp ()const
    {
      return _M_t.key_comp ();
    }
    value_compare value_comp () const
    {
      return value_compare (_M_t.key_comp ());
    }
    iterator find (const key_type & __x)
    {
      return _M_t.find (__x);
    }
    const_iterator find (const key_type & __x) const
    {
      return _M_t.find (__x);
    }
    size_type count (const key_type & __x) const
    {
      return _M_t.find (__x) == _M_t.end ()? 0 : 1;
    }
    iterator lower_bound (const key_type & __x)
    {
      return _M_t.lower_bound (__x);
    }
    const_iterator lower_bound (const key_type & __x) const
    {
      return _M_t.lower_bound (__x);
    }
    iterator upper_bound (const key_type & __x)
    {
      return _M_t.upper_bound (__x);
    }
    const_iterator upper_bound (const key_type & __x) const
    {
      return _M_t.upper_bound (__x);
    }
    std::pair < iterator, iterator > equal_range (const key_type & __x)
    {
      return _M_t.equal_range (__x);
    }
    std::pair < const_iterator, const_iterator >
      equal_range (const key_type & __x) const
    {
      return _M_t.equal_range (__x);
    }
    template < typename _K1, typename _T1, typename _C1, typename _A1 >
      friend bool
      operator== (const map < _K1, _T1, _C1, _A1 > &,
		  const map < _K1, _T1, _C1, _A1 > &);
    template < typename _K1, typename _T1, typename _C1, typename _A1 >
      friend bool
      operator< (const map < _K1, _T1, _C1, _A1 > &,
		 const map < _K1, _T1, _C1, _A1 > &);
  };
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator== (const map < _Key, _Tp, _Compare, _Alloc > &__x,
		const map < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return __x._M_t == __y._M_t;
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator< (const map < _Key, _Tp, _Compare, _Alloc > &__x,
	       const map < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return __x._M_t < __y._M_t;
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator!= (const map < _Key, _Tp, _Compare, _Alloc > &__x,
		const map < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return !(__x == __y);
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator> (const map < _Key, _Tp, _Compare, _Alloc > &__x,
	       const map < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return __y < __x;
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator<= (const map < _Key, _Tp, _Compare, _Alloc > &__x,
		const map < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return !(__y < __x);
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator>= (const map < _Key, _Tp, _Compare, _Alloc > &__x,
		const map < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return !(__x < __y);
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline void
    swap (map < _Key, _Tp, _Compare, _Alloc > &__x,
	  map < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    __x.swap (__y);
  }

}

namespace std __attribute__ ((__visibility__ ("default")))
{

  template < typename _Key, typename _Tp,
    typename _Compare = std::less < _Key >,
    typename _Alloc =
    std::allocator < std::pair < const _Key, _Tp > >>class multimap
  {
  public:
    typedef _Key key_type;
    typedef _Tp mapped_type;
    typedef std::pair < const _Key, _Tp > value_type;
    typedef _Compare key_compare;
    typedef _Alloc allocator_type;
  private:
    typedef typename _Alloc::value_type _Alloc_value_type;



  public:
      class value_compare:public std::binary_function < value_type,
      value_type, bool >
    {
      friend class multimap < _Key, _Tp, _Compare, _Alloc >;
    protected:
        _Compare comp;
        value_compare (_Compare __c):comp (__c)
      {
      }
    public:
        bool operator () (const value_type & __x, const value_type & __y) const
      {
	return comp (__x.first, __y.first);
      }
    };
  private:
    typedef typename __gnu_cxx::__alloc_traits < _Alloc >::template
      rebind < value_type >::other _Pair_alloc_type;
    typedef _Rb_tree < key_type, value_type, _Select1st < value_type >,
      key_compare, _Pair_alloc_type > _Rep_type;
    _Rep_type _M_t;
    typedef __gnu_cxx::__alloc_traits < _Pair_alloc_type > _Alloc_traits;
  public:
    typedef typename _Alloc_traits::pointer pointer;
    typedef typename _Alloc_traits::const_pointer const_pointer;
    typedef typename _Alloc_traits::reference reference;
    typedef typename _Alloc_traits::const_reference const_reference;
    typedef typename _Rep_type::iterator iterator;
    typedef typename _Rep_type::const_iterator const_iterator;
    typedef typename _Rep_type::size_type size_type;
    typedef typename _Rep_type::difference_type difference_type;
    typedef typename _Rep_type::reverse_iterator reverse_iterator;
    typedef typename _Rep_type::const_reverse_iterator const_reverse_iterator;
  multimap ():_M_t ()
    {
    }
  explicit multimap (const _Compare & __comp, const allocator_type & __a = allocator_type ()):_M_t (__comp,
	  _Pair_alloc_type
	  (__a))
    {
    }
    multimap (const multimap & __x):_M_t (__x._M_t)
    {
    }
  template < typename _InputIterator > multimap (_InputIterator __first, _InputIterator __last):_M_t
      ()
    {
      _M_t._M_insert_equal (__first, __last);
    }
  template < typename _InputIterator > multimap (_InputIterator __first, _InputIterator __last, const _Compare & __comp, const allocator_type & __a = allocator_type ()):_M_t (__comp,
	  _Pair_alloc_type
	  (__a))
    {
      _M_t._M_insert_equal (__first, __last);
    }
    multimap & operator= (const multimap & __x)
    {
      _M_t = __x._M_t;
      return *this;
    }
    allocator_type get_allocator ()const
    {
      return allocator_type (_M_t.get_allocator ());
    }
    iterator begin ()
    {
      return _M_t.begin ();
    }
    const_iterator begin ()const
    {
      return _M_t.begin ();
    }
    iterator end ()
    {
      return _M_t.end ();
    }
    const_iterator end ()const
    {
      return _M_t.end ();
    }
    reverse_iterator rbegin ()
    {
      return _M_t.rbegin ();
    }
    const_reverse_iterator rbegin ()const
    {
      return _M_t.rbegin ();
    }
    reverse_iterator rend ()
    {
      return _M_t.rend ();
    }
    const_reverse_iterator rend ()const
    {
      return _M_t.rend ();
    }
    bool empty () const
    {
      return _M_t.empty ();
    }
    size_type size () const
    {
      return _M_t.size ();
    }
    size_type max_size () const
    {
      return _M_t.max_size ();
    }
    iterator insert (const value_type & __x)
    {
      return _M_t._M_insert_equal (__x);
    }
    iterator insert (iterator __position, const value_type & __x)
    {
      return _M_t._M_insert_equal_ (__position, __x);
    }
    template < typename _InputIterator >
      void insert (_InputIterator __first, _InputIterator __last)
    {
      _M_t._M_insert_equal (__first, __last);
    }
    void erase (iterator __position)
    {
      _M_t.erase (__position);
    }
    size_type erase (const key_type & __x)
    {
      return _M_t.erase (__x);
    }
    void erase (iterator __first, iterator __last)
    {
      _M_t.erase (__first, __last);
    }
    void swap (multimap & __x)
    {
      _M_t.swap (__x._M_t);
    }
    void clear ()
    {
      _M_t.clear ();
    }
    key_compare key_comp ()const
    {
      return _M_t.key_comp ();
    }
    value_compare value_comp () const
    {
      return value_compare (_M_t.key_comp ());
    }
    iterator find (const key_type & __x)
    {
      return _M_t.find (__x);
    }
    const_iterator find (const key_type & __x) const
    {
      return _M_t.find (__x);
    }
    size_type count (const key_type & __x) const
    {
      return _M_t.count (__x);
    }
    iterator lower_bound (const key_type & __x)
    {
      return _M_t.lower_bound (__x);
    }
    const_iterator lower_bound (const key_type & __x) const
    {
      return _M_t.lower_bound (__x);
    }
    iterator upper_bound (const key_type & __x)
    {
      return _M_t.upper_bound (__x);
    }
    const_iterator upper_bound (const key_type & __x) const
    {
      return _M_t.upper_bound (__x);
    }
    std::pair < iterator, iterator > equal_range (const key_type & __x)
    {
      return _M_t.equal_range (__x);
    }
    std::pair < const_iterator, const_iterator >
      equal_range (const key_type & __x) const
    {
      return _M_t.equal_range (__x);
    }
    template < typename _K1, typename _T1, typename _C1, typename _A1 >
      friend bool
      operator== (const multimap < _K1, _T1, _C1, _A1 > &,
		  const multimap < _K1, _T1, _C1, _A1 > &);
    template < typename _K1, typename _T1, typename _C1, typename _A1 >
      friend bool
      operator< (const multimap < _K1, _T1, _C1, _A1 > &,
		 const multimap < _K1, _T1, _C1, _A1 > &);
  };
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator== (const multimap < _Key, _Tp, _Compare, _Alloc > &__x,
		const multimap < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return __x._M_t == __y._M_t;
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator< (const multimap < _Key, _Tp, _Compare, _Alloc > &__x,
	       const multimap < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return __x._M_t < __y._M_t;
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator!= (const multimap < _Key, _Tp, _Compare, _Alloc > &__x,
		const multimap < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return !(__x == __y);
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator> (const multimap < _Key, _Tp, _Compare, _Alloc > &__x,
	       const multimap < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return __y < __x;
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator<= (const multimap < _Key, _Tp, _Compare, _Alloc > &__x,
		const multimap < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return !(__y < __x);
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline bool
    operator>= (const multimap < _Key, _Tp, _Compare, _Alloc > &__x,
		const multimap < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    return !(__x < __y);
  }
  template < typename _Key, typename _Tp, typename _Compare, typename _Alloc >
    inline void
    swap (multimap < _Key, _Tp, _Compare, _Alloc > &__x,
	  multimap < _Key, _Tp, _Compare, _Alloc > &__y)
  {
    __x.swap (__y);
  }

}

namespace mfem
{
  class DataCollection
  {
  protected:
    std::string name;
    std::map < std::string, GridFunction * >field_map;
    Mesh *mesh;
    int cycle;
    double time;
    bool serial;
    int myid;
    int num_procs;
    int pad_digits;
    static const int pad_digits_default = 6;
    bool own_data;
    int error;
      DataCollection (const char *collection_name);
    void DeleteData ();
    void DeleteAll ();
  public:
      DataCollection (const char *collection_name, Mesh * _mesh);
    virtual void RegisterField (const char *field_name, GridFunction * gf);
    GridFunction *GetField (const char *field_name);
    bool HasField (const char *name)
    {
      return field_map.count (name) == 1;
    }
    Mesh *GetMesh ()
    {
      return mesh;
    }
    void SetCycle (int c)
    {
      cycle = c;
    }
    void SetTime (double t)
    {
      time = t;
    }
    int GetCycle ()
    {
      return cycle;
    }
    double GetTime ()
    {
      return time;
    }
    const char *GetCollectionName ()
    {
      return name.c_str ();
    }
    void SetOwnData (bool o)
    {
      own_data = o;
    }
    void SetPadDigits (int digits)
    {
      pad_digits = digits;
    }
    virtual void Save ();
    virtual ~ DataCollection ();
    enum
    { NO_ERROR = 0, READ_ERROR = 1, WRITE_ERROR = 2 };
    int Error () const
    {
      return error;
    }
    void ResetError (int err = NO_ERROR)
    {
      error = err;
    }
  };
  class VisItFieldInfo
  {
  public:
    std::string association;
    int num_components;
      VisItFieldInfo ()
    {
      association = "";
      num_components = 0;
    }
    VisItFieldInfo (std::string _association, int _num_components)
    {
      association = _association;
      num_components = _num_components;
    }
  };
  class VisItDataCollection:public DataCollection
  {
  protected:
    int spatial_dim, topo_dim;
    int visit_max_levels_of_detail;
      std::map < std::string, VisItFieldInfo > field_info_map;
      std::string GetVisItRootString ();
    void ParseVisItRootString (std::string json);
    void LoadVisItRootFile (std::string root_name);
    void LoadMesh ();
    void LoadFields ();
  public:
      VisItDataCollection (const char *collection_name);
      VisItDataCollection (const char *collection_name, Mesh * _mesh);
    virtual void RegisterField (const char *field_name, GridFunction * gf);
    void SetMaxLevelsOfDetail (int max_levels_of_detail);
    void DeleteAll ();
    virtual void Save ();
    void Load (int _cycle = 0);
      virtual ~ VisItDataCollection ()
    {
    }
  };
}

namespace mfem
{
  namespace internal
  {
    class StopWatch;
  }
  class StopWatch
  {
  private:
    internal::StopWatch * M;
  public:
    StopWatch ();
    void Clear ();
    void Start ();
    void Stop ();
    double Resolution ();
    double RealTime ();
    double UserTime ();
    double SystTime ();
     ~StopWatch ();
  };
  extern StopWatch tic_toc;
  extern void tic ();
  extern double toc ();
}
namespace mfem
{
  class isockstream
  {
  private:
    int portnum, portID, socketID, error;
    char *Buf;
    int establish ();
    int read_data (int socketid, char *buf, int size);
  public:
      explicit isockstream (int port);
    bool good ()
    {
      return (!error);
    }
    void receive (std::istringstream ** in);
     ~isockstream ();
  };
}

namespace mfem
{
  class socketbuf:public std::streambuf
  {
  private:
    int socket_descriptor;
    static const int buflen = 1024;
    char ibuf[buflen], obuf[buflen];
  public:
      socketbuf ()
    {
      socket_descriptor = -1;
    }
    explicit socketbuf (int sd)
    {
      socket_descriptor = sd;
      setp (obuf, obuf + buflen);
    }
    socketbuf (const char hostname[], int port)
    {
      socket_descriptor = -1;
      open (hostname, port);
    }
    int attach (int sd);
    int detach ()
    {
      return attach (-1);
    }
    int open (const char hostname[], int port);
    int close ();
    int getsocketdescriptor ()
    {
      return socket_descriptor;
    }
    bool is_open ()
    {
      return (socket_descriptor >= 0);
    }
    ~socketbuf ()
    {
      close ();
    }
  protected:
    virtual int sync ();
    virtual int_type underflow ();
    virtual int_type overflow (int_type c = traits_type::eof ());
    virtual std::streamsize xsgetn (char_type * __s, std::streamsize __n);
    virtual std::streamsize xsputn (const char_type * __s,
				    std::streamsize __n);
  };
  class socketstream:public std::iostream
  {
  private:
    socketbuf __buf;
  public:
    socketstream ():std::iostream (&__buf)
    {
    }
    explicit socketstream (int s):std::iostream (&__buf), __buf (s)
    {
    }
    socketstream (const char hostname[], int port):std::iostream (&__buf)
    {
      open (hostname, port);
    }
    socketbuf *rdbuf ()
    {
      return &__buf;
    }
    int open (const char hostname[], int port)
    {
      int err = __buf.open (hostname, port);
      if (err)
	setstate (std::ios::failbit);
      return err;
    }
    int close ()
    {
      return __buf.close ();
    }
    bool is_open ()
    {
      return __buf.is_open ();
    }
    virtual ~ socketstream ()
    {
    }
  };
  class socketserver
  {
  private:
    int listen_socket;
  public:
      explicit socketserver (int port);
    bool good ()
    {
      return (listen_socket >= 0);
    }
    int close ();
    int accept (socketstream & sockstr);
    ~socketserver ()
    {
      close ();
    }
  };
}

namespace mfem
{
  class osockstream:public socketstream
  {
  public:
    osockstream (int port, const char *hostname);
    int send ()
    {
      (*this) << std::flush;
      return 0;
    }
    virtual ~ osockstream ()
    {
    }
  };
}

namespace mfem
{
  class Vector;
  class OptionsParser
  {
  public:
    enum OptionType
    { INT, DOUBLE, STRING, ENABLE, DISABLE, ARRAY, VECTOR };
  private:
    struct Option
    {
      OptionType type;
      void *var_ptr;
      const char *short_name;
      const char *long_name;
      const char *description;
      bool required;
        Option (OptionType _type, void *_var_ptr, const char *_short_name,
		const char *_long_name, const char *_description,
		bool req):type (_type), var_ptr (_var_ptr),
	short_name (_short_name), long_name (_long_name),
	description (_description), required (req)
      {
      }
    };
    int argc;
    char **argv;
      Array < Option > options;
      Array < int >option_check;
    int error_type, error_idx;
    static void WriteValue (const Option & opt, std::ostream & out);
  public:
      OptionsParser (int _argc, char *_argv[]):argc (_argc), argv (_argv)
    {
      error_type = error_idx = 0;
    }
    void AddOption (bool * var, const char *enable_short_name,
		    const char *enable_long_name,
		    const char *disable_short_name,
		    const char *disable_long_name, const char *description,
		    bool required = false)
    {
      options.
	Append (Option
		(ENABLE, var, enable_short_name, enable_long_name,
		 description, required));
      options.
	Append (Option
		(DISABLE, var, disable_short_name, disable_long_name,
		 description, required));
    }
    void AddOption (int *var, const char *short_name, const char *long_name,
		    const char *description, bool required = false)
    {
      options.Append (Option (INT, var, short_name, long_name, description,
			      required));
    }
    void AddOption (double *var, const char *short_name,
		    const char *long_name, const char *description,
		    bool required = false)
    {
      options.Append (Option (DOUBLE, var, short_name, long_name, description,
			      required));
    }
    void AddOption (const char **var, const char *short_name,
		    const char *long_name, const char *description,
		    bool required = false)
    {
      options.Append (Option (STRING, var, short_name, long_name, description,
			      required));
    }
    void AddOption (Array < int >*var, const char *short_name,
		    const char *long_name, const char *description,
		    bool required = false)
    {
      options.Append (Option (ARRAY, var, short_name, long_name, description,
			      required));
    }
    void AddOption (Vector * var, const char *short_name,
		    const char *long_name, const char *description,
		    bool required = false)
    {
      options.Append (Option (VECTOR, var, short_name, long_name, description,
			      required));
    }
    void Parse ();
    bool Good () const
    {
      return (error_type == 0);
    }
    bool Help () const
    {
      return (error_type == 1);
    }
    void PrintOptions (std::ostream & out) const;
    void PrintError (std::ostream & out) const;
    void PrintHelp (std::ostream & out) const;
    void PrintUsage (std::ostream & out) const;
  };
}


namespace std __attribute__ ((__visibility__ ("default")))
{

  class codecvt_base
  {
  public:
    enum result
    {
      ok,
      partial,
      error,
      noconv
    };
  };
template < typename _InternT, typename _ExternT, typename _StateT > class __codecvt_abstract_base:public locale::facet,
    public
    codecvt_base
  {
  public:
    typedef codecvt_base::result result;
    typedef _InternT intern_type;
    typedef _ExternT extern_type;
    typedef _StateT state_type;
    result
      out (state_type & __state, const intern_type * __from,
	   const intern_type * __from_end, const intern_type * &__from_next,
	   extern_type * __to, extern_type * __to_end,
	   extern_type * &__to_next) const
    {
      return this->do_out (__state, __from, __from_end, __from_next,
			   __to, __to_end, __to_next);
    }
    result
      unshift (state_type & __state, extern_type * __to,
	       extern_type * __to_end, extern_type * &__to_next) const
    {
      return this->do_unshift (__state, __to, __to_end, __to_next);
    }
    result
      in (state_type & __state, const extern_type * __from,
	  const extern_type * __from_end, const extern_type * &__from_next,
	  intern_type * __to, intern_type * __to_end,
	  intern_type * &__to_next) const
    {
      return this->do_in (__state, __from, __from_end, __from_next,
			  __to, __to_end, __to_next);
    }
    int encoding () const throw ()
    {
      return this->do_encoding ();
    }
    bool always_noconv ()const throw ()
    {
      return this->do_always_noconv ();
    }
    int
      length (state_type & __state, const extern_type * __from,
	      const extern_type * __end, size_t __max) const
    {
      return this->do_length (__state, __from, __end, __max);
    }
    int max_length () const throw ()
    {
      return this->do_max_length ();
    }
  protected:
  explicit __codecvt_abstract_base (size_t __refs = 0):locale::
      facet (__refs)
    {
    }
    virtual ~ __codecvt_abstract_base ()
    {
    }
    virtual result
      do_out (state_type & __state, const intern_type * __from,
	      const intern_type * __from_end,
	      const intern_type * &__from_next, extern_type * __to,
	      extern_type * __to_end, extern_type * &__to_next) const = 0;
    virtual result do_unshift (state_type & __state, extern_type * __to,
			       extern_type * __to_end,
			       extern_type * &__to_next) const = 0;
    virtual result do_in (state_type & __state, const extern_type * __from,
			  const extern_type * __from_end,
			  const extern_type * &__from_next,
			  intern_type * __to, intern_type * __to_end,
			  intern_type * &__to_next) const = 0;
    virtual int do_encoding () const throw () = 0;
    virtual bool do_always_noconv () const throw () = 0;
    virtual int
      do_length (state_type &, const extern_type * __from,
		 const extern_type * __end, size_t __max) const = 0;
    virtual int do_max_length () const throw () = 0;
  };
template < typename _InternT, typename _ExternT, typename _StateT > class codecvt:public __codecvt_abstract_base < _InternT, _ExternT,
    _StateT
    >
  {
  public:
    typedef codecvt_base::result result;
    typedef _InternT intern_type;
    typedef _ExternT extern_type;
    typedef _StateT state_type;
  protected:
    __c_locale _M_c_locale_codecvt;
  public:
    static locale::id id;
  explicit codecvt (size_t __refs = 0):__codecvt_abstract_base < _InternT, _ExternT, _StateT > (__refs),
      _M_c_locale_codecvt (0)
    {
    }
    explicit codecvt (__c_locale __cloc, size_t __refs = 0);
  protected:
    virtual ~ codecvt ()
    {
    }
    virtual result
      do_out (state_type & __state, const intern_type * __from,
	      const intern_type * __from_end,
	      const intern_type * &__from_next, extern_type * __to,
	      extern_type * __to_end, extern_type * &__to_next) const;
    virtual result do_unshift (state_type & __state, extern_type * __to,
			       extern_type * __to_end,
			       extern_type * &__to_next) const;
    virtual result do_in (state_type & __state, const extern_type * __from,
			  const extern_type * __from_end,
			  const extern_type * &__from_next,
			  intern_type * __to, intern_type * __to_end,
			  intern_type * &__to_next) const;
    virtual int do_encoding () const throw ();
    virtual bool do_always_noconv () const throw ();
    virtual int
      do_length (state_type &, const extern_type * __from,
		 const extern_type * __end, size_t __max) const;
    virtual int do_max_length () const throw ();
  };
  template < typename _InternT, typename _ExternT, typename _StateT >
    locale::id codecvt < _InternT, _ExternT, _StateT >::id;
  template <>
    class codecvt < char, char,
    mbstate_t >:public __codecvt_abstract_base < char, char, mbstate_t >
  {
  public:
    typedef char intern_type;
    typedef char extern_type;
    typedef mbstate_t state_type;
  protected:
      __c_locale _M_c_locale_codecvt;
  public:
    static locale::id id;
      explicit codecvt (size_t __refs = 0);
      explicit codecvt (__c_locale __cloc, size_t __refs = 0);
  protected:
      virtual ~ codecvt ();
    virtual result
      do_out (state_type & __state, const intern_type * __from,
	      const intern_type * __from_end,
	      const intern_type * &__from_next, extern_type * __to,
	      extern_type * __to_end, extern_type * &__to_next) const;
    virtual result do_unshift (state_type & __state, extern_type * __to,
			       extern_type * __to_end,
			       extern_type * &__to_next) const;
    virtual result do_in (state_type & __state, const extern_type * __from,
			  const extern_type * __from_end,
			  const extern_type * &__from_next,
			  intern_type * __to, intern_type * __to_end,
			  intern_type * &__to_next) const;
    virtual int do_encoding () const throw ();
    virtual bool do_always_noconv () const throw ();
    virtual int
      do_length (state_type &, const extern_type * __from,
		 const extern_type * __end, size_t __max) const;
    virtual int do_max_length () const throw ();
  };
  template <>
    class codecvt < wchar_t, char,
    mbstate_t >:public __codecvt_abstract_base < wchar_t, char, mbstate_t >
  {
  public:
    typedef wchar_t intern_type;
    typedef char extern_type;
    typedef mbstate_t state_type;
  protected:
      __c_locale _M_c_locale_codecvt;
  public:
    static locale::id id;
      explicit codecvt (size_t __refs = 0);
      explicit codecvt (__c_locale __cloc, size_t __refs = 0);
  protected:
      virtual ~ codecvt ();
    virtual result
      do_out (state_type & __state, const intern_type * __from,
	      const intern_type * __from_end,
	      const intern_type * &__from_next, extern_type * __to,
	      extern_type * __to_end, extern_type * &__to_next) const;
    virtual result do_unshift (state_type & __state, extern_type * __to,
			       extern_type * __to_end,
			       extern_type * &__to_next) const;
    virtual result do_in (state_type & __state, const extern_type * __from,
			  const extern_type * __from_end,
			  const extern_type * &__from_next,
			  intern_type * __to, intern_type * __to_end,
			  intern_type * &__to_next) const;
      virtual int do_encoding () const throw ();
      virtual bool do_always_noconv () const throw ();
      virtual
      int do_length (state_type &, const extern_type * __from,
		     const extern_type * __end, size_t __max) const;
    virtual int do_max_length () const throw ();
  };
template < typename _InternT, typename _ExternT, typename _StateT > class codecvt_byname:public codecvt < _InternT, _ExternT,
    _StateT
    >
  {
  public:
  explicit codecvt_byname (const char *__s, size_t __refs = 0):codecvt < _InternT, _ExternT,
      _StateT >
      (__refs)
    {
      if (__builtin_strcmp (__s, "C") != 0
	  && __builtin_strcmp (__s, "POSIX") != 0)
	{
	  this->_S_destroy_c_locale (this->_M_c_locale_codecvt);
	  this->_S_create_c_locale (this->_M_c_locale_codecvt, __s);
	}
    }
  protected:
    virtual ~ codecvt_byname ()
    {
    }
  };
  extern template class codecvt_byname < char, char, mbstate_t >;
  extern template
    const codecvt < char, char, mbstate_t > &use_facet < codecvt < char, char,
    mbstate_t > >(const locale &);
  extern template bool has_facet < codecvt < char, char,
    mbstate_t > >(const locale &);
  extern template class codecvt_byname < wchar_t, char, mbstate_t >;
  extern template
    const codecvt < wchar_t, char, mbstate_t > &use_facet < codecvt < wchar_t,
    char, mbstate_t > >(const locale &);
  extern template bool has_facet < codecvt < wchar_t, char,
    mbstate_t > >(const locale &);

}

namespace std
{
  using::FILE;
  using::fpos_t;
  using::clearerr;
  using::fclose;
  using::feof;
  using::ferror;
  using::fflush;
  using::fgetc;
  using::fgetpos;
  using::fgets;
  using::fopen;
  using::fprintf;
  using::fputc;
  using::fputs;
  using::fread;
  using::freopen;
  using::fscanf;
  using::fseek;
  using::fsetpos;
  using::ftell;
  using::fwrite;
  using::getc;
  using::getchar;
  using::gets;
  using::perror;
  using::printf;
  using::putc;
  using::putchar;
  using::puts;
  using::remove;
  using::rename;
  using::rewind;
  using::scanf;
  using::setbuf;
  using::setvbuf;
  using::sprintf;
  using::sscanf;
  using::tmpfile;
  using::tmpnam;
  using::ungetc;
  using::vfprintf;
  using::vprintf;
  using::vsprintf;
}
namespace __gnu_cxx
{
  using::snprintf;
  using::vfscanf;
  using::vscanf;
  using::vsnprintf;
  using::vsscanf;
}
namespace std
{
  using::__gnu_cxx::snprintf;
  using::__gnu_cxx::vfscanf;
  using::__gnu_cxx::vscanf;
  using::__gnu_cxx::vsnprintf;
  using::__gnu_cxx::vsscanf;
}


namespace std __attribute__ ((__visibility__ ("default")))
{

  typedef __gthread_mutex_t __c_lock;
  typedef FILE __c_file;

}

namespace std __attribute__ ((__visibility__ ("default")))
{

  template < typename _CharT > class __basic_file;
  template <> class __basic_file < char >
  {
    __c_file *_M_cfile;
    bool _M_cfile_created;
  public:
      __basic_file (__c_lock * __lock = 0) throw ();
    __basic_file *open (const char *__name, ios_base::openmode __mode,
			int __prot = 0664);
    __basic_file *sys_open (__c_file * __file, ios_base::openmode);
    __basic_file *sys_open (int __fd, ios_base::openmode __mode) throw ();
    __basic_file *close ();
    __attribute__ ((__pure__)) bool is_open ()const throw ();
    __attribute__ ((__pure__)) int fd () throw ();
    __attribute__ ((__pure__)) __c_file *file () throw ();
     ~__basic_file ();
      streamsize xsputn (const char *__s, streamsize __n);
      streamsize
      xsputn_2 (const char *__s1, streamsize __n1,
		const char *__s2, streamsize __n2);
      streamsize xsgetn (char *__s, streamsize __n);
      streamoff seekoff (streamoff __off, ios_base::seekdir __way) throw ();
    int sync ();
      streamsize showmanyc ();
  };

}

namespace std __attribute__ ((__visibility__ ("default")))
{

template < typename _CharT, typename _Traits > class basic_filebuf:public basic_streambuf < _CharT,
    _Traits
    >
  {
  public:
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef typename traits_type::int_type int_type;
    typedef typename traits_type::pos_type pos_type;
    typedef typename traits_type::off_type off_type;
    typedef basic_streambuf < char_type, traits_type > __streambuf_type;
    typedef basic_filebuf < char_type, traits_type > __filebuf_type;
    typedef __basic_file < char >__file_type;
    typedef typename traits_type::state_type __state_type;
    typedef codecvt < char_type, char, __state_type > __codecvt_type;
    friend class ios_base;
  protected:
    __c_lock _M_lock;
    __file_type _M_file;
    ios_base::openmode _M_mode;
    __state_type _M_state_beg;
    __state_type _M_state_cur;
    __state_type _M_state_last;
    char_type *_M_buf;
    size_t _M_buf_size;
    bool _M_buf_allocated;
    bool _M_reading;
    bool _M_writing;
    char_type _M_pback;
    char_type *_M_pback_cur_save;
    char_type *_M_pback_end_save;
    bool _M_pback_init;
    const __codecvt_type *_M_codecvt;
    char *_M_ext_buf;
    streamsize _M_ext_buf_size;
    const char *_M_ext_next;
    char *_M_ext_end;
    void _M_create_pback ()
    {
      if (!_M_pback_init)
	{
	  _M_pback_cur_save = this->gptr ();
	  _M_pback_end_save = this->egptr ();
	  this->setg (&_M_pback, &_M_pback, &_M_pback + 1);
	  _M_pback_init = true;
	}
    }
    void _M_destroy_pback () throw ()
    {
      if (_M_pback_init)
	{
	  _M_pback_cur_save += this->gptr () != this->eback ();
	  this->setg (_M_buf, _M_pback_cur_save, _M_pback_end_save);
	  _M_pback_init = false;
	}
    }
  public:
    basic_filebuf ();
    virtual ~ basic_filebuf ()
    {
      this->close ();
    }
    bool is_open ()const throw ()
    {
      return _M_file.is_open ();
    }
    __filebuf_type *open (const char *__s, ios_base::openmode __mode);
    __filebuf_type *close ();
  protected:
    void _M_allocate_internal_buffer ();
    void _M_destroy_internal_buffer () throw ();
    virtual streamsize showmanyc ();
    virtual int_type underflow ();
    virtual int_type pbackfail (int_type __c = _Traits::eof ());
    virtual int_type overflow (int_type __c = _Traits::eof ());
    bool _M_convert_to_external (char_type *, streamsize);
    virtual __streambuf_type *setbuf (char_type * __s, streamsize __n);
    virtual pos_type
      seekoff (off_type __off, ios_base::seekdir __way,
	       ios_base::openmode __mode = ios_base::in | ios_base::out);
    virtual pos_type
      seekpos (pos_type __pos,
	       ios_base::openmode __mode = ios_base::in | ios_base::out);
    pos_type
      _M_seek (off_type __off, ios_base::seekdir __way, __state_type __state);
    int _M_get_ext_pos (__state_type & __state);
    virtual int sync ();
    virtual void imbue (const locale & __loc);
    virtual streamsize xsgetn (char_type * __s, streamsize __n);
    virtual streamsize xsputn (const char_type * __s, streamsize __n);
    bool _M_terminate_output ();
    void _M_set_buffer (streamsize __off)
    {
      const bool __testin = _M_mode & ios_base::in;
      const bool __testout = (_M_mode & ios_base::out
			      || _M_mode & ios_base::app);
      if (__testin && __off > 0)
	this->setg (_M_buf, _M_buf, _M_buf + __off);
      else
	this->setg (_M_buf, _M_buf, _M_buf);
      if (__testout && __off == 0 && _M_buf_size > 1)
	this->setp (_M_buf, _M_buf + _M_buf_size - 1);
      else
	this->setp (0, 0);
    }
  };
template < typename _CharT, typename _Traits > class basic_ifstream:public basic_istream < _CharT,
    _Traits
    >
  {
  public:
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef typename traits_type::int_type int_type;
    typedef typename traits_type::pos_type pos_type;
    typedef typename traits_type::off_type off_type;
    typedef basic_filebuf < char_type, traits_type > __filebuf_type;
    typedef basic_istream < char_type, traits_type > __istream_type;
  private:
    __filebuf_type _M_filebuf;
  public:
  basic_ifstream ():__istream_type (), _M_filebuf ()
    {
      this->init (&_M_filebuf);
    }
  explicit basic_ifstream (const char *__s, ios_base::openmode __mode = ios_base::in):__istream_type (),
      _M_filebuf
      ()
    {
      this->init (&_M_filebuf);
      this->open (__s, __mode);
    }
    ~basic_ifstream ()
    {
    }
    __filebuf_type *rdbuf () const
    {
      return const_cast < __filebuf_type * >(&_M_filebuf);
    }
    bool is_open ()
    {
      return _M_filebuf.is_open ();
    }
    bool is_open ()const
    {
      return _M_filebuf.is_open ();
    }
    void open (const char *__s, ios_base::openmode __mode = ios_base::in)
    {
      if (!_M_filebuf.open (__s, __mode | ios_base::in))
	this->setstate (ios_base::failbit);
      else
	this->clear ();
    }
    void close ()
    {
      if (!_M_filebuf.close ())
	this->setstate (ios_base::failbit);
    }
  };
template < typename _CharT, typename _Traits > class basic_ofstream:public basic_ostream < _CharT,
    _Traits
    >
  {
  public:
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef typename traits_type::int_type int_type;
    typedef typename traits_type::pos_type pos_type;
    typedef typename traits_type::off_type off_type;
    typedef basic_filebuf < char_type, traits_type > __filebuf_type;
    typedef basic_ostream < char_type, traits_type > __ostream_type;
  private:
    __filebuf_type _M_filebuf;
  public:
  basic_ofstream ():__ostream_type (), _M_filebuf ()
    {
      this->init (&_M_filebuf);
    }
  explicit basic_ofstream (const char *__s, ios_base::openmode __mode = ios_base::out | ios_base::trunc):__ostream_type (),
      _M_filebuf
      ()
    {
      this->init (&_M_filebuf);
      this->open (__s, __mode);
    }
    ~basic_ofstream ()
    {
    }
    __filebuf_type *rdbuf () const
    {
      return const_cast < __filebuf_type * >(&_M_filebuf);
    }
    bool is_open ()
    {
      return _M_filebuf.is_open ();
    }
    bool is_open ()const
    {
      return _M_filebuf.is_open ();
    }
    void
      open (const char *__s,
	    ios_base::openmode __mode = ios_base::out | ios_base::trunc)
    {
      if (!_M_filebuf.open (__s, __mode | ios_base::out))
	this->setstate (ios_base::failbit);
      else
	this->clear ();
    }
    void close ()
    {
      if (!_M_filebuf.close ())
	this->setstate (ios_base::failbit);
    }
  };
template < typename _CharT, typename _Traits > class basic_fstream:public basic_iostream < _CharT,
    _Traits
    >
  {
  public:
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef typename traits_type::int_type int_type;
    typedef typename traits_type::pos_type pos_type;
    typedef typename traits_type::off_type off_type;
    typedef basic_filebuf < char_type, traits_type > __filebuf_type;
    typedef basic_ios < char_type, traits_type > __ios_type;
    typedef basic_iostream < char_type, traits_type > __iostream_type;
  private:
    __filebuf_type _M_filebuf;
  public:
  basic_fstream ():__iostream_type (), _M_filebuf ()
    {
      this->init (&_M_filebuf);
    }
  explicit basic_fstream (const char *__s, ios_base::openmode __mode = ios_base::in | ios_base::out):__iostream_type (0),
      _M_filebuf
      ()
    {
      this->init (&_M_filebuf);
      this->open (__s, __mode);
    }
    ~basic_fstream ()
    {
    }
    __filebuf_type *rdbuf () const
    {
      return const_cast < __filebuf_type * >(&_M_filebuf);
    }
    bool is_open ()
    {
      return _M_filebuf.is_open ();
    }
    bool is_open ()const
    {
      return _M_filebuf.is_open ();
    }
    void
      open (const char *__s,
	    ios_base::openmode __mode = ios_base::in | ios_base::out)
    {
      if (!_M_filebuf.open (__s, __mode))
	this->setstate (ios_base::failbit);
      else
	this->clear ();
    }
    void close ()
    {
      if (!_M_filebuf.close ())
	this->setstate (ios_base::failbit);
    }
  };

}

namespace std __attribute__ ((__visibility__ ("default")))
{

  template < typename _CharT, typename _Traits >
    void basic_filebuf < _CharT, _Traits >::_M_allocate_internal_buffer ()
  {
    if (!_M_buf_allocated && !_M_buf)
      {
	_M_buf = new char_type[_M_buf_size];
	_M_buf_allocated = true;
      }
  }
  template < typename _CharT, typename _Traits >
    void
    basic_filebuf < _CharT, _Traits >::_M_destroy_internal_buffer () throw ()
  {
    if (_M_buf_allocated)
      {
	delete[]_M_buf;
	_M_buf = 0;
	_M_buf_allocated = false;
      }
    delete[]_M_ext_buf;
    _M_ext_buf = 0;
    _M_ext_buf_size = 0;
    _M_ext_next = 0;
    _M_ext_end = 0;
  }
template < typename _CharT, typename _Traits > basic_filebuf < _CharT, _Traits >::basic_filebuf ():__streambuf_type (), _M_lock (), _M_file (&_M_lock),
    _M_mode (ios_base::openmode (0)), _M_state_beg (), _M_state_cur (),
    _M_state_last (), _M_buf (0), _M_buf_size (8192),
    _M_buf_allocated (false), _M_reading (false), _M_writing (false),
    _M_pback (), _M_pback_cur_save (0), _M_pback_end_save (0),
    _M_pback_init (false), _M_codecvt (0), _M_ext_buf (0),
    _M_ext_buf_size (0), _M_ext_next (0), _M_ext_end (0)
  {
    if (has_facet < __codecvt_type > (this->_M_buf_locale))
      _M_codecvt = &use_facet < __codecvt_type > (this->_M_buf_locale);
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::__filebuf_type *
    basic_filebuf < _CharT, _Traits >::open (const char *__s,
					     ios_base::openmode __mode)
  {
    __filebuf_type *__ret = 0;
    if (!this->is_open ())
      {
	_M_file.open (__s, __mode);
	if (this->is_open ())
	  {
	    _M_allocate_internal_buffer ();
	    _M_mode = __mode;
	    _M_reading = false;
	    _M_writing = false;
	    _M_set_buffer (-1);
	    _M_state_last = _M_state_cur = _M_state_beg;
	    if ((__mode & ios_base::ate)
		&& this->seekoff (0, ios_base::end, __mode)
		== pos_type (off_type (-1)))
	      this->close ();
	    else
	      __ret = this;
	  }
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::__filebuf_type *
    basic_filebuf < _CharT, _Traits >::close ()
  {
    if (!this->is_open ())
      return 0;
    bool __testfail = false;
    {
      struct __close_sentry
      {
	basic_filebuf *__fb;
	  __close_sentry (basic_filebuf * __fbi):__fb (__fbi)
	{
	}
	 ~__close_sentry ()
	{
	  __fb->_M_mode = ios_base::openmode (0);
	  __fb->_M_pback_init = false;
	  __fb->_M_destroy_internal_buffer ();
	  __fb->_M_reading = false;
	  __fb->_M_writing = false;
	  __fb->_M_set_buffer (-1);
	  __fb->_M_state_last = __fb->_M_state_cur = __fb->_M_state_beg;
	}
      } __cs (this);
      try
      {
	if (!_M_terminate_output ())
	  __testfail = true;
      }
      catch (__cxxabiv1::__forced_unwind &)
      {
	_M_file.close ();
	throw;
      }
      catch ( ...)
      {
	__testfail = true;
      }
    }
    if (!_M_file.close ())
      __testfail = true;
    if (__testfail)
      return 0;
    else
      return this;
  }
  template < typename _CharT, typename _Traits >
    streamsize basic_filebuf < _CharT, _Traits >::showmanyc ()
  {
    streamsize __ret = -1;
    const bool __testin = _M_mode & ios_base::in;
    if (__testin && this->is_open ())
      {
	__ret = this->egptr () - this->gptr ();
	if (__check_facet (_M_codecvt).encoding () >= 0)
	  __ret += _M_file.showmanyc () / _M_codecvt->max_length ();
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::int_type
    basic_filebuf < _CharT, _Traits >::underflow ()
  {
    int_type __ret = traits_type::eof ();
    const bool __testin = _M_mode & ios_base::in;
    if (__testin)
      {
	if (_M_writing)
	  {
	    if (overflow () == traits_type::eof ())
	      return __ret;
	    _M_set_buffer (-1);
	    _M_writing = false;
	  }
	_M_destroy_pback ();
	if (this->gptr () < this->egptr ())
	  return traits_type::to_int_type (*this->gptr ());
	const size_t __buflen = _M_buf_size > 1 ? _M_buf_size - 1 : 1;
	bool __got_eof = false;
	streamsize __ilen = 0;
	codecvt_base::result __r = codecvt_base::ok;
	if (__check_facet (_M_codecvt).always_noconv ())
	  {
	    __ilen =
	      _M_file.xsgetn (reinterpret_cast < char *>(this->eback ()),
			      __buflen);
	    if (__ilen == 0)
	      __got_eof = true;
	  }
	else
	  {
	    const int __enc = _M_codecvt->encoding ();
	    streamsize __blen;
	    streamsize __rlen;
	    if (__enc > 0)
	      __blen = __rlen = __buflen * __enc;
	    else
	      {
		__blen = __buflen + _M_codecvt->max_length () - 1;
		__rlen = __buflen;
	      }
	    const streamsize __remainder = _M_ext_end - _M_ext_next;
	    __rlen = __rlen > __remainder ? __rlen - __remainder : 0;
	    if (_M_reading && this->egptr () == this->eback () && __remainder)
	      __rlen = 0;
	    if (_M_ext_buf_size < __blen)
	      {
		char *__buf = new char[__blen];
		if (__remainder)
		  __builtin_memcpy (__buf, _M_ext_next, __remainder);
		delete[]_M_ext_buf;
		_M_ext_buf = __buf;
		_M_ext_buf_size = __blen;
	      }
	    else if (__remainder)
	      __builtin_memmove (_M_ext_buf, _M_ext_next, __remainder);
	    _M_ext_next = _M_ext_buf;
	    _M_ext_end = _M_ext_buf + __remainder;
	    _M_state_last = _M_state_cur;
	    do
	      {
		if (__rlen > 0)
		  {
		    if (_M_ext_end - _M_ext_buf + __rlen > _M_ext_buf_size)
		      {
			__throw_ios_failure (("basic_filebuf::underflow "
					      "codecvt::max_length() "
					      "is not valid"));
		      }
		    streamsize __elen = _M_file.xsgetn (_M_ext_end, __rlen);
		    if (__elen == 0)
		      __got_eof = true;
		    else if (__elen == -1)
		      break;
		    _M_ext_end += __elen;
		  }
		char_type *__iend = this->eback ();
		if (_M_ext_next < _M_ext_end)
		  __r = _M_codecvt->in (_M_state_cur, _M_ext_next,
					_M_ext_end, _M_ext_next,
					this->eback (),
					this->eback () + __buflen, __iend);
		if (__r == codecvt_base::noconv)
		  {
		    size_t __avail = _M_ext_end - _M_ext_buf;
		    __ilen = std::min (__avail, __buflen);
		    traits_type::copy (this->eback (),
				       reinterpret_cast < char_type * >
				       (_M_ext_buf), __ilen);
		    _M_ext_next = _M_ext_buf + __ilen;
		  }
		else
		  __ilen = __iend - this->eback ();
		if (__r == codecvt_base::error)
		  break;
		__rlen = 1;
	      }
	    while (__ilen == 0 && !__got_eof);
	  }
	if (__ilen > 0)
	  {
	    _M_set_buffer (__ilen);
	    _M_reading = true;
	    __ret = traits_type::to_int_type (*this->gptr ());
	  }
	else if (__got_eof)
	  {
	    _M_set_buffer (-1);
	    _M_reading = false;
	    if (__r == codecvt_base::partial)
	      __throw_ios_failure (("basic_filebuf::underflow "
				    "incomplete character in file"));
	  }
	else if (__r == codecvt_base::error)
	  __throw_ios_failure (("basic_filebuf::underflow "
				"invalid byte sequence in file"));
	else
	  __throw_ios_failure (("basic_filebuf::underflow "
				"error reading the file"));
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::int_type
    basic_filebuf < _CharT, _Traits >::pbackfail (int_type __i)
  {
    int_type __ret = traits_type::eof ();
    const bool __testin = _M_mode & ios_base::in;
    if (__testin)
      {
	if (_M_writing)
	  {
	    if (overflow () == traits_type::eof ())
	      return __ret;
	    _M_set_buffer (-1);
	    _M_writing = false;
	  }
	const bool __testpb = _M_pback_init;
	const bool __testeof = traits_type::eq_int_type (__i, __ret);
	int_type __tmp;
	if (this->eback () < this->gptr ())
	  {
	    this->gbump (-1);
	    __tmp = traits_type::to_int_type (*this->gptr ());
	  }
	else if (this->seekoff (-1, ios_base::cur) !=
		 pos_type (off_type (-1)))
	  {
	    __tmp = this->underflow ();
	    if (traits_type::eq_int_type (__tmp, __ret))
	      return __ret;
	  }
	else
	  {
	    return __ret;
	  }
	if (!__testeof && traits_type::eq_int_type (__i, __tmp))
	  __ret = __i;
	else if (__testeof)
	  __ret = traits_type::not_eof (__i);
	else if (!__testpb)
	  {
	    _M_create_pback ();
	    _M_reading = true;
	    *this->gptr () = traits_type::to_char_type (__i);
	    __ret = __i;
	  }
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::int_type
    basic_filebuf < _CharT, _Traits >::overflow (int_type __c)
  {
    int_type __ret = traits_type::eof ();
    const bool __testeof = traits_type::eq_int_type (__c, __ret);
    const bool __testout = (_M_mode & ios_base::out
			    || _M_mode & ios_base::app);
    if (__testout)
      {
	if (_M_reading)
	  {
	    _M_destroy_pback ();
	    const int __gptr_off = _M_get_ext_pos (_M_state_last);
	    if (_M_seek (__gptr_off, ios_base::cur, _M_state_last)
		== pos_type (off_type (-1)))
	      return __ret;
	  }
	if (this->pbase () < this->pptr ())
	  {
	    if (!__testeof)
	      {
		*this->pptr () = traits_type::to_char_type (__c);
		this->pbump (1);
	      }
	    if (_M_convert_to_external (this->pbase (),
					this->pptr () - this->pbase ()))
	      {
		_M_set_buffer (0);
		__ret = traits_type::not_eof (__c);
	      }
	  }
	else if (_M_buf_size > 1)
	  {
	    _M_set_buffer (0);
	    _M_writing = true;
	    if (!__testeof)
	      {
		*this->pptr () = traits_type::to_char_type (__c);
		this->pbump (1);
	      }
	    __ret = traits_type::not_eof (__c);
	  }
	else
	  {
	    char_type __conv = traits_type::to_char_type (__c);
	    if (__testeof || _M_convert_to_external (&__conv, 1))
	      {
		_M_writing = true;
		__ret = traits_type::not_eof (__c);
	      }
	  }
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    bool
    basic_filebuf < _CharT,
    _Traits >::_M_convert_to_external (_CharT * __ibuf, streamsize __ilen)
  {
    streamsize __elen;
    streamsize __plen;
    if (__check_facet (_M_codecvt).always_noconv ())
      {
	__elen = _M_file.xsputn (reinterpret_cast < char *>(__ibuf), __ilen);
	__plen = __ilen;
      }
    else
      {
	streamsize __blen = __ilen * _M_codecvt->max_length ();
	char *__buf = static_cast < char *>(__builtin_alloca (__blen));
	char *__bend;
	const char_type *__iend;
	codecvt_base::result __r;
	__r = _M_codecvt->out (_M_state_cur, __ibuf, __ibuf + __ilen,
			       __iend, __buf, __buf + __blen, __bend);
	if (__r == codecvt_base::ok || __r == codecvt_base::partial)
	  __blen = __bend - __buf;
	else if (__r == codecvt_base::noconv)
	  {
	    __buf = reinterpret_cast < char *>(__ibuf);
	    __blen = __ilen;
	  }
	else
	  __throw_ios_failure (("basic_filebuf::_M_convert_to_external "
				"conversion error"));
	__elen = _M_file.xsputn (__buf, __blen);
	__plen = __blen;
	if (__r == codecvt_base::partial && __elen == __plen)
	  {
	    const char_type *__iresume = __iend;
	    streamsize __rlen = this->pptr () - __iend;
	    __r = _M_codecvt->out (_M_state_cur, __iresume,
				   __iresume + __rlen, __iend, __buf,
				   __buf + __blen, __bend);
	    if (__r != codecvt_base::error)
	      {
		__rlen = __bend - __buf;
		__elen = _M_file.xsputn (__buf, __rlen);
		__plen = __rlen;
	      }
	    else
	      __throw_ios_failure (("basic_filebuf::_M_convert_to_external "
				    "conversion error"));
	  }
      }
    return __elen == __plen;
  }
  template < typename _CharT, typename _Traits >
    streamsize
    basic_filebuf < _CharT, _Traits >::xsgetn (_CharT * __s, streamsize __n)
  {
    streamsize __ret = 0;
    if (_M_pback_init)
      {
	if (__n > 0 && this->gptr () == this->eback ())
	  {
	    *__s++ = *this->gptr ();
	    this->gbump (1);
	    __ret = 1;
	    --__n;
	  }
	_M_destroy_pback ();
      }
    else if (_M_writing)
      {
	if (overflow () == traits_type::eof ())
	  return __ret;
	_M_set_buffer (-1);
	_M_writing = false;
      }
    const bool __testin = _M_mode & ios_base::in;
    const streamsize __buflen = _M_buf_size > 1 ? _M_buf_size - 1 : 1;
    if (__n > __buflen && __check_facet (_M_codecvt).always_noconv ()
	&& __testin)
      {
	const streamsize __avail = this->egptr () - this->gptr ();
	if (__avail != 0)
	  {
	    traits_type::copy (__s, this->gptr (), __avail);
	    __s += __avail;
	    this->setg (this->eback (), this->gptr () + __avail,
			this->egptr ());
	    __ret += __avail;
	    __n -= __avail;
	  }
	streamsize __len;
	for (;;)
	  {
	    __len = _M_file.xsgetn (reinterpret_cast < char *>(__s), __n);
	    if (__len == -1)
	      __throw_ios_failure (("basic_filebuf::xsgetn "
				    "error reading the file"));
	    if (__len == 0)
	      break;
	    __n -= __len;
	    __ret += __len;
	    if (__n == 0)
	      break;
	    __s += __len;
	  }
	if (__n == 0)
	  {
	    _M_set_buffer (0);
	    _M_reading = true;
	  }
	else if (__len == 0)
	  {
	    _M_set_buffer (-1);
	    _M_reading = false;
	  }
      }
    else
      __ret += __streambuf_type::xsgetn (__s, __n);
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    streamsize
    basic_filebuf < _CharT, _Traits >::xsputn (const _CharT * __s,
					       streamsize __n)
  {
    streamsize __ret = 0;
    const bool __testout = (_M_mode & ios_base::out
			    || _M_mode & ios_base::app);
    if (__check_facet (_M_codecvt).always_noconv ()
	&& __testout && !_M_reading)
      {
	const streamsize __chunk = 1ul << 10;
	streamsize __bufavail = this->epptr () - this->pptr ();
	if (!_M_writing && _M_buf_size > 1)
	  __bufavail = _M_buf_size - 1;
	const streamsize __limit = std::min (__chunk, __bufavail);
	if (__n >= __limit)
	  {
	    const streamsize __buffill = this->pptr () - this->pbase ();
	    const char *__buf =
	      reinterpret_cast < const char *>(this->pbase ());
	    __ret =
	      _M_file.xsputn_2 (__buf, __buffill,
				reinterpret_cast < const char *>(__s), __n);
	    if (__ret == __buffill + __n)
	      {
		_M_set_buffer (0);
		_M_writing = true;
	      }
	    if (__ret > __buffill)
	      __ret -= __buffill;
	    else
	      __ret = 0;
	  }
	else
	  __ret = __streambuf_type::xsputn (__s, __n);
      }
    else
      __ret = __streambuf_type::xsputn (__s, __n);
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::__streambuf_type *
    basic_filebuf < _CharT, _Traits >::setbuf (char_type * __s,
					       streamsize __n)
  {
    if (!this->is_open ())
      {
	if (__s == 0 && __n == 0)
	  _M_buf_size = 1;
	else if (__s && __n > 0)
	  {
	    _M_buf = __s;
	    _M_buf_size = __n;
	  }
      }
    return this;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::pos_type
    basic_filebuf < _CharT, _Traits >::seekoff (off_type __off,
						ios_base::seekdir __way,
						ios_base::openmode)
  {
    int __width = 0;
    if (_M_codecvt)
      __width = _M_codecvt->encoding ();
    if (__width < 0)
      __width = 0;
    pos_type __ret = pos_type (off_type (-1));
    const bool __testfail = __off != 0 && __width <= 0;
    if (this->is_open () && !__testfail)
      {
	bool __no_movement = __way == ios_base::cur && __off == 0
	  && (!_M_writing || _M_codecvt->always_noconv ());
	if (!__no_movement)
	  _M_destroy_pback ();
	__state_type __state = _M_state_beg;
	off_type __computed_off = __off * __width;
	if (_M_reading && __way == ios_base::cur)
	  {
	    __state = _M_state_last;
	    __computed_off += _M_get_ext_pos (__state);
	  }
	if (!__no_movement)
	  __ret = _M_seek (__computed_off, __way, __state);
	else
	  {
	    if (_M_writing)
	      __computed_off = this->pptr () - this->pbase ();
	    off_type __file_off = _M_file.seekoff (0, ios_base::cur);
	    if (__file_off != off_type (-1))
	      {
		__ret = __file_off + __computed_off;
		__ret.state (__state);
	      }
	  }
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::pos_type
    basic_filebuf < _CharT, _Traits >::seekpos (pos_type __pos,
						ios_base::openmode)
  {
    pos_type __ret = pos_type (off_type (-1));
    if (this->is_open ())
      {
	_M_destroy_pback ();
	__ret = _M_seek (off_type (__pos), ios_base::beg, __pos.state ());
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    typename basic_filebuf < _CharT, _Traits >::pos_type
    basic_filebuf < _CharT, _Traits >::_M_seek (off_type __off,
						ios_base::seekdir __way,
						__state_type __state)
  {
    pos_type __ret = pos_type (off_type (-1));
    if (_M_terminate_output ())
      {
	off_type __file_off = _M_file.seekoff (__off, __way);
	if (__file_off != off_type (-1))
	  {
	    _M_reading = false;
	    _M_writing = false;
	    _M_ext_next = _M_ext_end = _M_ext_buf;
	    _M_set_buffer (-1);
	    _M_state_cur = __state;
	    __ret = __file_off;
	    __ret.state (_M_state_cur);
	  }
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    int basic_filebuf < _CharT,
    _Traits >::_M_get_ext_pos (__state_type & __state)
  {
    if (_M_codecvt->always_noconv ())
      return this->gptr () - this->egptr ();
    else
      {
	const int __gptr_off =
	  _M_codecvt->length (__state, _M_ext_buf, _M_ext_next,
			      this->gptr () - this->eback ());
	return _M_ext_buf + __gptr_off - _M_ext_end;
      }
  }
  template < typename _CharT, typename _Traits >
    bool basic_filebuf < _CharT, _Traits >::_M_terminate_output ()
  {
    bool __testvalid = true;
    if (this->pbase () < this->pptr ())
      {
	const int_type __tmp = this->overflow ();
	if (traits_type::eq_int_type (__tmp, traits_type::eof ()))
	  __testvalid = false;
      }
    if (_M_writing && !__check_facet (_M_codecvt).always_noconv ()
	&& __testvalid)
      {
	const size_t __blen = 128;
	char __buf[__blen];
	codecvt_base::result __r;
	streamsize __ilen = 0;
	do
	  {
	    char *__next;
	    __r = _M_codecvt->unshift (_M_state_cur, __buf,
				       __buf + __blen, __next);
	    if (__r == codecvt_base::error)
	      __testvalid = false;
	    else if (__r == codecvt_base::ok || __r == codecvt_base::partial)
	      {
		__ilen = __next - __buf;
		if (__ilen > 0)
		  {
		    const streamsize __elen = _M_file.xsputn (__buf, __ilen);
		    if (__elen != __ilen)
		      __testvalid = false;
		  }
	      }
	  }
	while (__r == codecvt_base::partial && __ilen > 0 && __testvalid);
	if (__testvalid)
	  {
	    const int_type __tmp = this->overflow ();
	    if (traits_type::eq_int_type (__tmp, traits_type::eof ()))
	      __testvalid = false;
	  }
      }
    return __testvalid;
  }
  template < typename _CharT, typename _Traits >
    int basic_filebuf < _CharT, _Traits >::sync ()
  {
    int __ret = 0;
    if (this->pbase () < this->pptr ())
      {
	const int_type __tmp = this->overflow ();
	if (traits_type::eq_int_type (__tmp, traits_type::eof ()))
	  __ret = -1;
      }
    return __ret;
  }
  template < typename _CharT, typename _Traits >
    void basic_filebuf < _CharT, _Traits >::imbue (const locale & __loc)
  {
    bool __testvalid = true;
    const __codecvt_type *_M_codecvt_tmp = 0;
    if (__builtin_expect (has_facet < __codecvt_type > (__loc), true))
      _M_codecvt_tmp = &use_facet < __codecvt_type > (__loc);
    if (this->is_open ())
      {
	if ((_M_reading || _M_writing)
	    && __check_facet (_M_codecvt).encoding () == -1)
	  __testvalid = false;
	else
	  {
	    if (_M_reading)
	      {
		if (__check_facet (_M_codecvt).always_noconv ())
		  {
		    if (_M_codecvt_tmp
			&& !__check_facet (_M_codecvt_tmp).always_noconv ())
		      __testvalid = this->seekoff (0, ios_base::cur, _M_mode)
			!= pos_type (off_type (-1));
		  }
		else
		  {
		    _M_ext_next = _M_ext_buf
		      + _M_codecvt->length (_M_state_last, _M_ext_buf,
					    _M_ext_next,
					    this->gptr () - this->eback ());
		    const streamsize __remainder = _M_ext_end - _M_ext_next;
		    if (__remainder)
		      __builtin_memmove (_M_ext_buf, _M_ext_next,
					 __remainder);
		    _M_ext_next = _M_ext_buf;
		    _M_ext_end = _M_ext_buf + __remainder;
		    _M_set_buffer (-1);
		    _M_state_last = _M_state_cur = _M_state_beg;
		  }
	      }
	    else if (_M_writing && (__testvalid = _M_terminate_output ()))
	      _M_set_buffer (-1);
	  }
      }
    if (__testvalid)
      _M_codecvt = _M_codecvt_tmp;
    else
      _M_codecvt = 0;
  }
  extern template class basic_filebuf < char >;
  extern template class basic_ifstream < char >;
  extern template class basic_ofstream < char >;
  extern template class basic_fstream < char >;
  extern template class basic_filebuf < wchar_t >;
  extern template class basic_ifstream < wchar_t >;
  extern template class basic_ofstream < wchar_t >;
  extern template class basic_fstream < wchar_t >;

}

using namespace std;
using namespace mfem;
int
main (int argc, char *argv[])
{
  const char *mesh_file = "../data/star.mesh";
  int order = 1;
  bool visualization = 1;
  OptionsParser args (argc, argv);
  args.AddOption (&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption (&order, "-o", "--order",
		  "Finite element order (polynomial degree) or -1 for"
		  " isoparametric space.");
  args.AddOption (&visualization, "-vis", "--visualization", "-no-vis",
		  "--no-visualization",
		  "Enable or disable GLVis visualization.");
  args.Parse ();
  if (!args.Good ())
    {
      args.PrintUsage (cout);
      return 1;
    }
  args.PrintOptions (cout);
  Mesh *mesh;
  ifstream imesh (mesh_file);
  if (!imesh)
    {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
    }
  mesh = new Mesh (imesh, 1, 1);
  imesh.close ();
  int dim = mesh->Dimension ();
  {
    int ref_levels =
      (int) floor (log (50000. / mesh->GetNE ()) / log (2.) / dim);
    for (int l = 0; l < ref_levels; l++)
      mesh->UniformRefinement ();
  }
  
  FiniteElementCollection *fec;
  if (order > 0)
    fec = new H1_FECollection (order, dim);
  else if (mesh->GetNodes ())
    fec = mesh->GetNodes ()->OwnFEC ();
  else
    fec = new H1_FECollection (order = 1, dim);
  cout << "FiniteElementCollection" << endl;
  
  FiniteElementSpace *fespace = new FiniteElementSpace (mesh, fec);
  cout << "Number of unknowns: " << fespace->GetVSize () << endl;
  cout << "FiniteElementSpace" << endl;
  
  LinearForm *b = new LinearForm (fespace);
  cout << "LinearForm" << endl;
  
 ConstantCoefficient one (1.0);
  b->AddDomainIntegrator (new DomainLFIntegrator (one));
  b->Assemble ();
  cout << "b->Assembled" << endl;
  
  GridFunction x (fespace);
  x = 0.0;
  cout << "GridFunction X" << endl;
  
  BilinearForm *a = new BilinearForm (fespace);
  cout << "BilinearForm a" << endl;
  
//#warning return BilinearForm
  //return 0;

  a->AddDomainIntegrator (new MassIntegrator (one));
  cout << "MassIntegrator" << endl;
  
//#warning return MassIntegrator
    //return 0;
    
  a->Assemble();
  cout << "Assemble" << endl;
  
//#warning return Assemble
  //return 0;

  Array < int >ess_bdr (mesh->bdr_attributes.Max ());
  ess_bdr = 1;
  a->EliminateEssentialBC (ess_bdr, x, *b);
  a->Finalize ();
  cout << "Finalize" << endl;
  
  //#warning return Finalize
  //return 0;

  const SparseMatrix & A = a->SpMat ();
  cout << "SparseMatrix A" << endl;
  
  GSSmoother M (A);
  PCG (A, M, *b, x, 1, 200, 1e-12, 0.0);
  cout << "PCG" << endl;
  
  ofstream mesh_ofs ("refined.mesh");
  mesh_ofs.precision (8);
  mesh->Print (mesh_ofs);
  ofstream sol_ofs ("sol.gf");
  sol_ofs.precision (8);
  x.Save (sol_ofs);
  if (visualization)
    {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock (vishost, visport);
      sol_sock.precision (8);
      sol_sock << "solution\n" << *mesh << x << flush;
    }
  delete a;
  delete b;
  delete fespace;
  if (order > 0)
    delete fec;
  delete mesh;
  return 0;
}
