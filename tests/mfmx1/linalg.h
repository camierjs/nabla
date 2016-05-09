
namespace mfem
{
inline int CheckFinite(const double *v, const int n);
class Vector
{
protected:
   int size, allocsize;
   double * data;
public:
   Vector () { allocsize = size = 0; data = 0; }
   Vector(const Vector &);
   explicit Vector (int s);
   Vector (double *_data, int _size)
   { data = _data; size = _size; allocsize = -size; }
   void Load (std::istream ** in, int np, int * dim);
   void Load(std::istream &in, int Size);
   void Load(std::istream &in) { int s; in >> s; Load (in, s); }
   void SetSize(int s);
   void SetData(double *d) { data = d; }
   void SetDataAndSize(double *d, int s)
   { data = d; size = s; allocsize = -s; }
   void NewDataAndSize(double *d, int s)
   { if (allocsize > 0) delete [] data; SetDataAndSize(d, s); }
   void MakeDataOwner() { allocsize = abs(allocsize); }
   void Destroy();
   inline int Size() const { return size; }
   inline double *GetData() const { return data; }
   inline operator double *() { return data; }
   inline operator const double *() const { return data; }
   inline bool OwnsData() const { return (allocsize > 0); }
   inline void StealData(double **p)
   { *p = data; data = 0; size = allocsize = 0; }
   inline double *StealData() { double *p; StealData(&p); return p; }
   double & Elem (int i);
   const double & Elem (int i) const;
   inline double & operator() (int i);
   inline const double & operator() (int i) const;
   double operator*(const double *) const;
   double operator*(const Vector &v) const;
   Vector & operator=(const double *v);
   Vector & operator=(const Vector &v);
   Vector & operator=(double value);
   Vector & operator*=(double c);
   Vector & operator/=(double c);
   Vector & operator-=(double c);
   Vector & operator-=(const Vector &v);
   Vector & operator+=(const Vector &v);
   Vector & Add(const double a, const Vector &Va);
   Vector & Set(const double a, const Vector &x);
   void SetVector (const Vector &v, int offset);
   void Neg();
   friend void swap(Vector *v1, Vector *v2);
   friend void add(const Vector &v1, const Vector &v2, Vector &v);
   friend void add(const Vector &v1, double alpha, const Vector &v2, Vector &v);
   friend void add(const double a, const Vector &x, const Vector &y, Vector &z);
   friend void add (const double a, const Vector &x,
                    const double b, const Vector &y, Vector &z);
   friend void subtract(const Vector &v1, const Vector &v2, Vector &v);
   friend void subtract(const double a, const Vector &x,
                        const Vector &y, Vector &z);
   void median(const Vector &lo, const Vector &hi);
   void GetSubVector(const Array<int> &dofs, Vector &elemvect) const;
   void GetSubVector(const Array<int> &dofs, double *elem_data) const;
   void SetSubVector(const Array<int> &dofs, const Vector &elemvect);
   void SetSubVector(const Array<int> &dofs, double *elem_data);
   void AddElementVector(const Array<int> & dofs, const Vector & elemvect);
   void AddElementVector(const Array<int> & dofs, double *elem_data);
   void AddElementVector(const Array<int> & dofs, const double a,
                         const Vector & elemvect);
   void Print(std::ostream & out = std::cout, int width = 8) const;
   void Print_HYPRE(std::ostream &out) const;
   void Randomize(int seed = 0);
   double Norml2() const;
   double Normlinf() const;
   double Norml1() const;
   double Normlp(double p) const;
   double Max() const;
   double Min() const;
   double Sum() const;
   double DistanceTo (const double *p) const;
   int CheckFinite() const { return mfem::CheckFinite(data, size); }
   ~Vector ();
};
inline int CheckFinite(const double *v, const int n)
{
   int bad = 0;
   for (int i = 0; i < n; i++)
   {
         if (!std::isfinite(v[i]))
         {
            bad++;
         }
   }
   return bad;
}
inline Vector::Vector (int s)
{
   if (s > 0)
   {
      allocsize = size = s;
      data = new double[s];
   }
   else
   {
      allocsize = size = 0;
      data = __null;
   }
}
inline void Vector::SetSize(int s)
{
   if (s == size)
      return;
   if (s <= abs(allocsize))
   {
      size = s;
      return;
   }
   if (allocsize > 0)
      delete [] data;
   allocsize = size = s;
   data = new double[s];
}
inline void Vector::Destroy()
{
   if (allocsize > 0)
      delete [] data;
   allocsize = size = 0;
   data = __null;
}
inline double & Vector::operator() (int i)
{
   return data[i];
}
inline const double & Vector::operator() (int i) const
{
   return data[i];
}
inline Vector::~Vector()
{
   if (allocsize > 0)
      delete [] data;
}
inline double Distance(const double *x, const double *y, const int n)
{
   using namespace std;
   double d = 0.0;
   for (int i = 0; i < n; i++)
      d += (x[i]-y[i])*(x[i]-y[i]);
   return sqrt(d);
}
}
namespace mfem
{
class Operator
{
protected:
   int height, width;
public:
   explicit Operator(int s = 0) { height = width = s; }
   Operator(int h, int w) { height = h; width = w; }
   inline int Height() const { return height; }
   inline int NumRows() const { return height; }
   inline int Width() const { return width; }
   inline int NumCols() const { return width; }
   virtual void Mult(const Vector &x, Vector &y) const = 0;
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { mfem_error ("Operator::MultTranspose() is not overloaded!"); }
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return *((Operator *)this);
   }
   void PrintMatlab (std::ostream & out, int n = 0, int m = 0);
   virtual ~Operator() { }
};
class TimeDependentOperator : public Operator
{
protected:
   double t;
public:
   explicit TimeDependentOperator(int n = 0, double _t = 0.0)
      : Operator(n) { t = _t; }
   TimeDependentOperator(int h, int w, double _t = 0.0)
      : Operator(h, w) { t = _t; }
   virtual double GetTime() const { return t; }
   virtual void SetTime(const double _t) { t = _t; }
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      mfem_error("TimeDependentOperator::ImplicitSolve() is not overloaded!");
   }
   virtual ~TimeDependentOperator() { }
};
class Solver : public Operator
{
public:
   bool iterative_mode;
   explicit Solver(int s = 0, bool iter_mode = false)
      : Operator(s) { iterative_mode = iter_mode; }
   Solver(int h, int w, bool iter_mode = false)
      : Operator(h, w) { iterative_mode = iter_mode; }
   virtual void SetOperator(const Operator &op) = 0;
};
class IdentityOperator : public Operator
{
public:
   explicit IdentityOperator(int n) : Operator(n) { }
   virtual void Mult(const Vector &x, Vector &y) const { y = x; }
   ~IdentityOperator() { }
};
class TransposeOperator : public Operator
{
private:
   const Operator &A;
public:
   TransposeOperator(const Operator *a)
      : Operator(a->Width(), a->Height()), A(*a) { }
   TransposeOperator(const Operator &a)
      : Operator(a.Width(), a.Height()), A(a) { }
   virtual void Mult(const Vector &x, Vector &y) const
   { A.MultTranspose(x, y); }
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A.Mult(x, y); }
   ~TransposeOperator() { }
};
class RAPOperator : public Operator
{
private:
   Operator & Rt;
   Operator & A;
   Operator & P;
   mutable Vector Px;
   mutable Vector APx;
public:
   RAPOperator(Operator &Rt_, Operator &A_, Operator &P_)
      : Operator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }
   void Mult(const Vector & x, Vector & y) const
   { P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y); }
   void MultTranspose(const Vector & x, Vector & y) const
   { Rt.Mult(x, APx); A.MultTranspose(APx, Px); P.MultTranspose(Px, y); }
   ~RAPOperator() { }
};
}
namespace mfem
{
class MatrixInverse;
class Matrix : public Operator
{
   friend class MatrixInverse;
public:
   explicit Matrix(int s) : Operator(s) { }
   explicit Matrix(int h, int w) : Operator(h, w) { }
   virtual double &Elem(int i, int j) = 0;
   virtual const double &Elem(int i, int j) const = 0;
   virtual MatrixInverse *Inverse() const = 0;
   virtual void Finalize(int) { }
   virtual void Print (std::ostream & out = std::cout, int width_ = 4) const;
   virtual ~Matrix() { }
};
class MatrixInverse : public Solver
{
public:
   MatrixInverse() { }
   MatrixInverse(const Matrix &mat)
      : Solver(mat.height, mat.width) { }
};
class AbstractSparseMatrix : public Matrix
{
public:
   explicit AbstractSparseMatrix(int s = 0) : Matrix(s) { }
   explicit AbstractSparseMatrix(int h, int w) : Matrix(h, w) { }
   virtual int NumNonZeroElems() const = 0;
   virtual int GetRow(const int row, Array<int> &cols, Vector &srow) const = 0;
   virtual void EliminateZeroRows() = 0;
   virtual void Mult(const Vector &x, Vector &y) const = 0;
   virtual void AddMult(const Vector &x, Vector &y,
                        const double val = 1.) const = 0;
   virtual void MultTranspose(const Vector &x, Vector &y) const = 0;
   virtual void AddMultTranspose(const Vector &x, Vector &y,
                                 const double val = 1.) const = 0;
   virtual ~AbstractSparseMatrix() { }
};
}
namespace mfem
{
template <class Elem, int Num>
class StackPart
{
public:
   StackPart<Elem, Num> *Prev;
   Elem Elements[Num];
};
template <class Elem, int Num>
class Stack
{
private:
   StackPart <Elem, Num> *TopPart, *TopFreePart;
   int UsedInTop, SSize;
public:
   Stack() { TopPart = TopFreePart = __null; UsedInTop = Num; SSize = 0; };
   int Size() { return SSize; };
   void Push (Elem E);
   Elem Pop();
   void Clear();
   ~Stack() { Clear(); };
};
template <class Elem, int Num>
void Stack <Elem, Num>::Push (Elem E)
{
   StackPart <Elem, Num> *aux;
   if (UsedInTop == Num)
   {
      if (TopFreePart == __null)
         aux = new StackPart <Elem, Num>;
      else
         TopFreePart = (aux = TopFreePart)->Prev;
      aux->Prev = TopPart;
      TopPart = aux;
      UsedInTop = 0;
   }
   TopPart->Elements[UsedInTop++] = E;
   SSize++;
}
template <class Elem, int Num>
Elem Stack <Elem, Num>::Pop()
{
   StackPart <Elem, Num> *aux;
   if (UsedInTop == 0)
   {
      TopPart = (aux = TopPart)->Prev;
      aux->Prev = TopFreePart;
      TopFreePart = aux;
      UsedInTop = Num;
   }
   SSize--;
   return TopPart->Elements[--UsedInTop];
}
template <class Elem, int Num>
void Stack <Elem, Num>::Clear()
{
   StackPart <Elem, Num> *aux;
   while (TopPart != __null)
   {
      TopPart = (aux = TopPart)->Prev;
      delete aux;
   }
   while (TopFreePart != __null)
   {
      TopFreePart = (aux = TopFreePart)->Prev;
      delete aux;
   }
   UsedInTop = Num;
   SSize = 0;
}
template <class Elem, int Num>
class MemAllocNode
{
public:
   MemAllocNode <Elem, Num> *Prev;
   Elem Elements[Num];
};
template <class Elem, int Num>
class MemAlloc
{
private:
   MemAllocNode <Elem, Num> *Last;
   int AllocatedInLast;
   Stack <Elem *, Num> UsedMem;
public:
   MemAlloc() { Last = __null; AllocatedInLast = Num; };
   Elem *Alloc();
   void Free (Elem *);
   void Clear();
   ~MemAlloc() { Clear(); };
};
template <class Elem, int Num>
Elem *MemAlloc <Elem, Num>::Alloc()
{
   MemAllocNode <Elem, Num> *aux;
   if (UsedMem.Size() > 0)
      return UsedMem.Pop();
   if (AllocatedInLast == Num)
   {
      aux = Last;
      Last = new MemAllocNode <Elem, Num>;
      Last->Prev = aux;
      AllocatedInLast = 0;
   }
   return &(Last->Elements[AllocatedInLast++]);
}
template <class Elem, int Num>
void MemAlloc <Elem, Num>::Free (Elem *E)
{
   UsedMem.Push (E);
}
template <class Elem, int Num>
void MemAlloc <Elem, Num>::Clear()
{
   MemAllocNode <Elem, Num> *aux;
   while (Last != __null)
   {
      aux = Last->Prev;
      delete Last;
      Last = aux;
   }
   AllocatedInLast = Num;
   UsedMem.Clear();
}
}
namespace mfem
{
class Table
{
protected:
   int size;
   int *I, *J;
public:
   Table() { size = -1; I = J = __null; }
   explicit Table (int dim, int connections_per_row = 3);
   Table (int nrows, int *partitioning);
   void MakeI (int nrows);
   void AddAColumnInRow (int r) { I[r]++; }
   void AddColumnsInRow (int r, int ncol) { I[r] += ncol; }
   void MakeJ();
   void AddConnection (int r, int c) { J[I[r]++] = c; }
   void AddConnections (int r, const int *c, int nc);
   void ShiftUpI();
   void SetSize(int dim, int connections_per_row);
   void SetDims(int rows, int nnz);
   inline int Size() const { return size; }
   inline int Size_of_connections() const { return I[size]; }
   int operator() (int i, int j) const;
   void GetRow(int i, Array<int> &row) const;
   int RowSize(int i) const { return I[i+1]-I[i]; }
   const int *GetRow(int i) const { return J+I[i]; }
   int *GetRow(int i) { return J+I[i]; }
   int *GetI() { return I; };
   int *GetJ() { return J; };
   const int *GetI() const { return I; };
   const int *GetJ() const { return J; };
   void SetIJ(int *newI, int *newJ, int newsize = -1);
   int Push( int i, int j );
   void Finalize();
   int Width() const;
   void LoseData() { size = -1; I = J = __null; }
   void Print(std::ostream & out = std::cout, int width = 4) const;
   void PrintMatlab(std::ostream & out) const;
   void Save(std::ostream & out) const;
   void Copy(Table & copy) const;
   void Swap(Table & other);
   void Clear();
   ~Table();
};
void Transpose (const Table &A, Table &At, int _ncols_A = -1);
Table * Transpose (const Table &A);
void Transpose(const Array<int> &A, Table &At, int _ncols_A = -1);
void Mult (const Table &A, const Table &B, Table &C);
Table * Mult (const Table &A, const Table &B);
class STable : public Table
{
public:
   STable (int dim, int connections_per_row = 3);
   int operator() (int i, int j) const;
   int Push( int i, int j );
   ~STable() {}
};
class DSTable
{
private:
   class Node
   {
   public:
      Node *Prev;
      int Column, Index;
   };
   int NumRows, NumEntries;
   Node **Rows;
   MemAlloc <Node, 1024> NodesMem;
   int Push_(int r, int c);
   int Index(int r, int c) const;
public:
   DSTable(int nrows);
   int NumberOfRows() const { return(NumRows); }
   int NumberOfEntries() const { return(NumEntries); }
   int Push(int a, int b)
   { return((a <= b) ? Push_(a, b) : Push_(b, a)); }
   int operator()(int a, int b) const
   { return((a <= b) ? Index(a, b) : Index(b, a)); }
   ~DSTable();
   class RowIterator
   {
   private:
      Node *n;
   public:
      RowIterator (const DSTable &t, int r) { n = t.Rows[r]; }
      int operator!() { return(n != __null); }
      void operator++() { n = n->Prev; }
      int Column() { return(n->Column); }
      int Index() { return(n->Index); }
   };
};
}
namespace mfem
{
class DenseMatrix : public Matrix
{
   friend class DenseTensor;
private:
   double *data;
   friend class DenseMatrixInverse;
   friend void Mult(const DenseMatrix &b,
                    const DenseMatrix &c,
                    DenseMatrix &a);
   void Eigensystem(Vector &ev, DenseMatrix *evect = __null);
public:
   DenseMatrix();
   DenseMatrix(const DenseMatrix &);
   explicit DenseMatrix(int s);
   DenseMatrix(int m, int n);
   DenseMatrix(const DenseMatrix &mat, char ch);
   DenseMatrix(double *d, int h, int w) : Matrix(h, w) { data = d; }
   void UseExternalData(double *d, int h, int w)
   { data = d; height = h; width = w; }
   void ClearExternalData() { data = __null; height = width = 0; }
   int Size() const { return Width(); }
   void SetSize(int s);
   void SetSize(int h, int w);
   inline double *Data() const { return data; }
   inline double &operator()(int i, int j);
   inline const double &operator()(int i, int j) const;
   double operator*(const DenseMatrix &m) const;
   double Trace() const;
   virtual double &Elem(int i, int j);
   virtual const double &Elem(int i, int j) const;
   void Mult(const double *x, double *y) const;
   virtual void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const double *x, double *y) const;
   virtual void MultTranspose(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y) const;
   double InnerProduct(const double *x, const double *y) const;
   void LeftScaling(const Vector & s);
   void InvLeftScaling(const Vector & s);
   void RightScaling(const Vector & s);
   void InvRightScaling(const Vector & s);
   void SymmetricScaling(const Vector & s);
   void InvSymmetricScaling(const Vector & s);
   double InnerProduct(const Vector &x, const Vector &y) const
   { return InnerProduct((const double *)x, (const double *)y); }
   virtual MatrixInverse *Inverse() const;
   void Invert();
   double Det() const;
   double Weight() const;
   void Add(const double c, const DenseMatrix &A);
   DenseMatrix &operator=(double c);
   DenseMatrix &operator=(const double *d);
   DenseMatrix &operator=(const DenseMatrix &m);
   DenseMatrix &operator+=(DenseMatrix &m);
   DenseMatrix &operator-=(DenseMatrix &m);
   DenseMatrix &operator*=(double c);
   void Neg();
   void Norm2(double *v) const;
   double MaxMaxNorm() const;
   double FNorm() const;
   void Eigenvalues(Vector &ev)
   { Eigensystem(ev); }
   void Eigenvalues(Vector &ev, DenseMatrix &evect)
   { Eigensystem(ev, &evect); }
   void Eigensystem(Vector &ev, DenseMatrix &evect)
   { Eigensystem(ev, &evect); }
   void SingularValues(Vector &sv) const;
   int Rank(double tol) const;
   double CalcSingularvalue(const int i) const;
   void CalcEigenvalues(double *lambda, double *vec) const;
   void GetColumn(int c, Vector &col);
   void GetColumnReference(int c, Vector &col)
   { col.SetDataAndSize(data + c * height, height); }
   void GetDiag(Vector &d);
   void Getl1Diag(Vector &l);
   void Diag(double c, int n);
   void Diag(double *diag, int n);
   void Transpose();
   void Transpose(DenseMatrix &A);
   void Symmetrize();
   void Lump();
   void GradToCurl(DenseMatrix &curl);
   void GradToDiv(Vector &div);
   void CopyRows(DenseMatrix &A, int row1, int row2);
   void CopyCols(DenseMatrix &A, int col1, int col2);
   void CopyMN(DenseMatrix &A, int m, int n, int Aro, int Aco);
   void CopyMN(DenseMatrix &A, int row_offset, int col_offset);
   void CopyMNt(DenseMatrix &A, int row_offset, int col_offset);
   void CopyMNDiag(double c, int n, int row_offset, int col_offset);
   void CopyMNDiag(double *diag, int n, int row_offset, int col_offset);
   void AddMatrix(DenseMatrix &A, int ro, int co);
   void AddMatrix(double a, DenseMatrix &A, int ro, int co);
   void AddToVector(int offset, Vector &v) const;
   void GetFromVector(int offset, const Vector &v);
   void AdjustDofDirection(Array<int> &dofs);
   void SetRow(int row, double value);
   void SetCol(int col, double value);
   int CheckFinite() const { return mfem::CheckFinite(data, height*width); }
   virtual void Print(std::ostream &out = std::cout, int width_ = 4) const;
   virtual void PrintMatlab(std::ostream &out = std::cout) const;
   virtual void PrintT(std::ostream &out = std::cout, int width_ = 4) const;
   void TestInversion();
   virtual ~DenseMatrix();
};
void Add(const DenseMatrix &A, const DenseMatrix &B,
         double alpha, DenseMatrix &C);
void Mult(const DenseMatrix &b, const DenseMatrix &c, DenseMatrix &a);
void CalcAdjugate(const DenseMatrix &a, DenseMatrix &adja);
void CalcAdjugateTranspose(const DenseMatrix &a, DenseMatrix &adjat);
void CalcInverse(const DenseMatrix &a, DenseMatrix &inva);
void CalcInverseTranspose(const DenseMatrix &a, DenseMatrix &inva);
void CalcOrtho(const DenseMatrix &J, Vector &n);
void MultAAt(const DenseMatrix &a, DenseMatrix &aat);
void MultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt);
void AddMultADAt(const DenseMatrix &A, const Vector &D, DenseMatrix &ADAt);
void MultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt);
void AddMultABt(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &ABt);
void MultAtB(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &AtB);
void AddMult_a_AAt(double a, const DenseMatrix &A, DenseMatrix &AAt);
void Mult_a_AAt(double a, const DenseMatrix &A, DenseMatrix &AAt);
void MultVVt(const Vector &v, DenseMatrix &vvt);
void MultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt);
void AddMultVWt(const Vector &v, const Vector &w, DenseMatrix &VWt);
void AddMult_a_VWt(const double a, const Vector &v, const Vector &w, DenseMatrix &VWt);
void AddMult_a_VVt(const double a, const Vector &v, DenseMatrix &VVt);
class DenseMatrixInverse : public MatrixInverse
{
private:
   const DenseMatrix *a;
   double *data;
public:
   DenseMatrixInverse(const DenseMatrix &mat);
   DenseMatrixInverse(const DenseMatrix *mat);
   int Size() const { return Width(); }
   void Factor();
   void Factor(const DenseMatrix &mat);
   virtual void SetOperator(const Operator &op);
   virtual void Mult(const Vector &x, Vector &y) const;
   virtual ~DenseMatrixInverse();
};
class DenseMatrixEigensystem
{
   DenseMatrix &mat;
   Vector EVal;
   DenseMatrix EVect;
   Vector ev;
   int n;
public:
   DenseMatrixEigensystem(DenseMatrix &m);
   void Eval();
   Vector &Eigenvalues() { return EVal; }
   DenseMatrix &Eigenvectors() { return EVect; }
   double Eigenvalue(int i) { return EVal(i); }
   const Vector &Eigenvector(int i)
   {
      ev.SetData(EVect.Data() + i * EVect.Height());
      return ev;
   }
   ~DenseMatrixEigensystem();
};
class DenseMatrixSVD
{
   Vector sv;
   int m, n;
   void Init();
public:
   DenseMatrixSVD(DenseMatrix &M);
   DenseMatrixSVD(int h, int w);
   void Eval(DenseMatrix &M);
   Vector &Singularvalues() { return sv; }
   double Singularvalue(int i) { return sv(i); }
   ~DenseMatrixSVD();
};
class Table;
class DenseTensor
{
private:
   DenseMatrix Mk;
   double *tdata;
   int nk;
public:
   DenseTensor() { nk = 0; tdata = __null; }
   DenseTensor(int i, int j, int k)
      : Mk(__null, i, j)
   { nk = k; tdata = new double[i*j*k]; }
   int SizeI() const { return Mk.Height(); }
   int SizeJ() const { return Mk.Width(); }
   int SizeK() const { return nk; }
   void SetSize(int i, int j, int k)
   {
      delete [] tdata;
      Mk.UseExternalData(__null, i, j);
      nk = k;
      tdata = new double[i*j*k];
   }
   DenseMatrix &operator()(int k) { Mk.data = GetData(k); return Mk; }
   double &operator()(int i, int j, int k)
   { return tdata[i+SizeI()*(j+SizeJ()*k)]; }
   const double &operator()(int i, int j, int k) const
   { return tdata[i+SizeI()*(j+SizeJ()*k)]; }
   double *GetData(int k) { return tdata+k*Mk.Height()*Mk.Width(); }
   double *Data() { return tdata; }
   void AddMult(const Table &elem_dof, const Vector &x, Vector &y) const;
   ~DenseTensor() { delete [] tdata; Mk.ClearExternalData(); }
};
inline double &DenseMatrix::operator()(int i, int j)
{
   return data[i+j*height];
}
inline const double &DenseMatrix::operator()(int i, int j) const
{
   return data[i+j*height];
}
}
namespace mfem
{
class
   RowNode
{
public:
   double Value;
   RowNode *Prev;
   int Column;
};
class SparseMatrix : public AbstractSparseMatrix
{
private:
   int *I, *J;
   double *A;
   RowNode **Rows;
   mutable int current_row;
   mutable int* ColPtrJ;
   mutable RowNode ** ColPtrNode;
   typedef MemAlloc <RowNode, 1024> RowNodeAlloc;
   RowNodeAlloc * NodesMem;
   bool ownGraph;
   bool ownData;
   bool isSorted;
   inline void SetColPtr(const int row) const;
   inline void ClearColPtr() const;
   inline double &SearchRow(const int col);
   inline void _Add_(const int col, const double a)
   { SearchRow(col) += a; }
   inline void _Set_(const int col, const double a)
   { SearchRow(col) = a; }
   inline double _Get_(const int col) const;
   inline double &SearchRow(const int row, const int col);
   inline void _Add_(const int row, const int col, const double a)
   { SearchRow(row, col) += a; }
   inline void _Set_(const int row, const int col, const double a)
   { SearchRow(row, col) = a; }
public:
   explicit SparseMatrix(int nrows, int ncols = 0);
   SparseMatrix(int *i, int *j, double *data, int m, int n);
   SparseMatrix(int *i, int *j, double *data, int m, int n, bool ownij, bool owna,
                bool issorted);
   int Size() const { return Height(); }
   inline int *GetI() const { return I; }
   inline int *GetJ() const { return J; }
   inline double *GetData() const { return A; }
   int RowSize(const int i) const;
   int MaxRowSize() const;
   int *GetRowColumns(const int row);
   const int *GetRowColumns(const int row) const;
   double *GetRowEntries(const int row);
   const double *GetRowEntries(const int row) const;
   void SetWidth(int width_ = -1);
   int ActualWidth();
   void SortColumnIndices();
   virtual double &Elem(int i, int j);
   virtual const double &Elem(int i, int j) const;
   double &operator()(int i, int j);
   const double &operator()(int i, int j) const;
   void GetDiag(Vector & d) const;
   virtual void Mult(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y, const double a = 1.0) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void AddMultTranspose(const Vector &x, Vector &y,
                         const double a = 1.0) const;
   void PartMult(const Array<int> &rows, const Vector &x, Vector &y) const;
   void PartAddMult(const Array<int> &rows, const Vector &x, Vector &y,
                    const double a=1.0) const;
   double InnerProduct(const Vector &x, const Vector &y) const;
   void GetRowSums(Vector &x) const;
   double GetRowNorml1(int irow) const;
   virtual MatrixInverse *Inverse() const;
   void EliminateRow(int row, const double sol, Vector &rhs);
   void EliminateRow(int row, int setOneDiagonal = 0);
   void EliminateCol(int col);
   void EliminateCols(Array<int> &cols, Vector *x = __null, Vector *b = __null);
   void EliminateRowCol(int rc, const double sol, Vector &rhs, int d = 0);
   void EliminateRowColMultipleRHS(int rc, const Vector &sol,
                                   DenseMatrix &rhs, int d = 0);
   void EliminateRowCol(int rc, int d = 0);
   void EliminateRowCol(int rc, SparseMatrix &Ae, int d = 0);
   void SetDiagIdentity();
   void EliminateZeroRows();
   void Gauss_Seidel_forw(const Vector &x, Vector &y) const;
   void Gauss_Seidel_back(const Vector &x, Vector &y) const;
   double GetJacobiScaling() const;
   void Jacobi(const Vector &b, const Vector &x0, Vector &x1, double sc) const;
   void DiagScale(const Vector &b, Vector &x, double sc = 1.0) const;
   void Jacobi2(const Vector &b, const Vector &x0, Vector &x1,
                double sc = 1.0) const;
   void Jacobi3(const Vector &b, const Vector &x0, Vector &x1,
                double sc = 1.0) const;
   virtual void Finalize(int skip_zeros = 1);
   bool Finalized() const { return (A != __null); }
   bool areColumnsSorted() const { return isSorted; }
   void GetBlocks(Array2D<SparseMatrix *> &blocks) const;
   void GetSubMatrix(const Array<int> &rows, const Array<int> &cols,
                     DenseMatrix &subm);
   void Set(const int i, const int j, const double a);
   void Add(const int i, const int j, const double a);
   void SetSubMatrix(const Array<int> &rows, const Array<int> &cols,
                     const DenseMatrix &subm, int skip_zeros = 1);
   void SetSubMatrixTranspose(const Array<int> &rows, const Array<int> &cols,
                              const DenseMatrix &subm, int skip_zeros = 1);
   void AddSubMatrix(const Array<int> &rows, const Array<int> &cols,
                     const DenseMatrix &subm, int skip_zeros = 1);
   bool RowIsEmpty(const int row) const;
   virtual int GetRow(const int row, Array<int> &cols, Vector &srow) const;
   void SetRow(const int row, const Array<int> &cols, const Vector &srow);
   void AddRow(const int row, const Array<int> &cols, const Vector &srow);
   void ScaleRow(const int row, const double scale);
   void ScaleRows(const Vector & sl);
   void ScaleColumns(const Vector & sr);
   SparseMatrix &operator+=(SparseMatrix &B);
   void Add(const double a, const SparseMatrix &B);
   SparseMatrix &operator=(double a);
   SparseMatrix &operator*=(double a);
   void Print(std::ostream &out = std::cout, int width_ = 4) const;
   void PrintMatlab(std::ostream &out = std::cout) const;
   void PrintMM(std::ostream &out = std::cout) const;
   void PrintCSR(std::ostream &out) const;
   void PrintCSR2(std::ostream &out) const;
   int Walk(int &i, int &j, double &a);
   double IsSymmetric() const;
   void Symmetrize();
   virtual int NumNonZeroElems() const;
   double MaxNorm() const;
   int CountSmallElems(double tol) const;
   void LoseData() { I=0; J=0; A=0; }
   friend void Swap(SparseMatrix & A, SparseMatrix & B);
   virtual ~SparseMatrix();
};
void SparseMatrixFunction(SparseMatrix &S, double (*f)(double));
SparseMatrix *Transpose(const SparseMatrix &A);
SparseMatrix *TransposeAbstractSparseMatrix (const AbstractSparseMatrix &A,
                                             int useActualWidth);
SparseMatrix *Mult(const SparseMatrix &A, const SparseMatrix &B,
                   SparseMatrix *OAB = __null);
SparseMatrix *MultAbstractSparseMatrix (const AbstractSparseMatrix &A,
                                        const AbstractSparseMatrix &B);
SparseMatrix *RAP(const SparseMatrix &A, const SparseMatrix &R,
                  SparseMatrix *ORAP = __null);
SparseMatrix *RAP(const SparseMatrix &Rt, const SparseMatrix &A,
                  const SparseMatrix &P);
SparseMatrix *Mult_AtDA(const SparseMatrix &A, const Vector &D,
                        SparseMatrix *OAtDA = __null);
SparseMatrix * Add(const SparseMatrix & A, const SparseMatrix & B);
SparseMatrix * Add(double a, const SparseMatrix & A, double b,
                   const SparseMatrix & B);
SparseMatrix * Add(Array<SparseMatrix *> & Ai);
inline void SparseMatrix::SetColPtr(const int row) const
{
   if (Rows)
   {
      if (ColPtrNode == __null)
      {
         ColPtrNode = new RowNode *[width];
         for (int i = 0; i < width; i++)
         {
            ColPtrNode[i] = __null;
         }
      }
      for (RowNode *node_p = Rows[row]; node_p != __null; node_p = node_p->Prev)
      {
         ColPtrNode[node_p->Column] = node_p;
      }
   }
   else
   {
      if (ColPtrJ == __null)
      {
         ColPtrJ = new int[width];
         for (int i = 0; i < width; i++)
         {
            ColPtrJ[i] = -1;
         }
      }
      for (int j = I[row], end = I[row+1]; j < end; j++)
      {
         ColPtrJ[J[j]] = j;
      }
   }
   current_row = row;
}
inline void SparseMatrix::ClearColPtr() const
{
   if (Rows)
      for (RowNode *node_p = Rows[current_row]; node_p != __null;
           node_p = node_p->Prev)
      {
         ColPtrNode[node_p->Column] = __null;
      }
   else
      for (int j = I[current_row], end = I[current_row+1]; j < end; j++)
      {
         ColPtrJ[J[j]] = -1;
      }
}
inline double &SparseMatrix::SearchRow(const int col)
{
   if (Rows)
   {
      RowNode *node_p = ColPtrNode[col];
      if (node_p == __null)
      {
         node_p = NodesMem->Alloc();
         node_p->Prev = Rows[current_row];
         node_p->Column = col;
         node_p->Value = 0.0;
         Rows[current_row] = ColPtrNode[col] = node_p;
      }
      return node_p->Value;
   }
   else
   {
      const int j = ColPtrJ[col];
      if (!(j != -1)) { { std::ostringstream s; s << std::setprecision(16); s << std::setiosflags(std::ios_base::scientific); s << "Verification failed: (" << "j != -1" << ") is false: " << "Entry for column " << col << " is not allocated." << '\n'; s << " ... at line " << 452; s << " in " << __PRETTY_FUNCTION__ << " of file " << "../fem/../linalg/sparsemat.hpp" << "."; s << std::ends; if (!(0)) mfem::mfem_error(s.str().c_str()); else mfem::mfem_warning(s.str().c_str()); }; };
      return A[j];
   }
}
inline double SparseMatrix::_Get_(const int col) const
{
   if (Rows)
   {
      RowNode *node_p = ColPtrNode[col];
      return (node_p == __null) ? 0.0 : node_p->Value;
   }
   else
   {
      const int j = ColPtrJ[col];
      return (j == -1) ? 0.0 : A[j];
   }
}
inline double &SparseMatrix::SearchRow(const int row, const int col)
{
   if (Rows)
   {
      RowNode *node_p;
      for (node_p = Rows[row]; 1; node_p = node_p->Prev)
      {
         if (node_p == __null)
         {
            node_p = NodesMem->Alloc();
            node_p->Prev = Rows[row];
            node_p->Column = col;
            node_p->Value = 0.0;
            Rows[row] = node_p;
            break;
         }
         else if (node_p->Column == col)
         {
            break;
         }
      }
      return node_p->Value;
   }
   else
   {
      int *Ip = I+row, *Jp = J;
      for (int k = Ip[0], end = Ip[1]; k < end; k++)
      {
         if (Jp[k] == col)
         {
            return A[k];
         }
      }
      { std::ostringstream s; s << std::setprecision(16); s << std::setiosflags(std::ios_base::scientific); s << "MFEM abort: " << "Could not find entry for row = " << row << ", col = " << col << '\n'; s << " ... at line " << 509; s << " in " << __PRETTY_FUNCTION__ << " of file " << "../fem/../linalg/sparsemat.hpp" << "."; s << std::ends; if (!(0)) mfem::mfem_error(s.str().c_str()); else mfem::mfem_warning(s.str().c_str()); };
   }
   return A[0];
}
}
namespace mfem
{
class BlockVector: public Vector
{
protected:
   int numBlocks;
   const int *blockOffsets;
   Array<Vector *> tmp_block;
public:
   BlockVector();
   BlockVector(const Array<int> & bOffsets);
   BlockVector(const BlockVector & block);
   BlockVector(double *data, const Array<int> & bOffsets);
   BlockVector & operator=(const BlockVector & original);
   BlockVector & operator=(double val);
   ~BlockVector();
   Vector & GetBlock(int i);
   const Vector & GetBlock(int i) const;
   void GetBlockView(int i, Vector & blockView);
   int BlockSize(int i){ return blockOffsets[i+1] - blockOffsets[i];}
   void Update(double *data, const Array<int> & bOffsets);
};
}
namespace mfem
{
class BlockMatrix : public AbstractSparseMatrix
{
public:
   BlockMatrix(const Array<int> & offsets);
   BlockMatrix(const Array<int> & row_offsets, const Array<int> & col_offsets);
   void SetBlock(int i, int j, SparseMatrix * mat);
   int NumRowBlocks() const {return nRowBlocks; }
   int NumColBlocks() const {return nColBlocks; }
   SparseMatrix & GetBlock(int i, int j);
   const SparseMatrix & GetBlock(int i, int j) const;
   int IsZeroBlock(int i, int j) const {return (Aij(i,j)==__null) ? 1 : 0; }
   Array<int> & RowOffsets() { return row_offsets; }
   Array<int> & ColOffsets() { return col_offsets; }
   const Array<int> & RowOffsets() const { return row_offsets; }
   const Array<int> & ColOffsets() const { return col_offsets; }
   int RowSize(const int i) const;
   void EliminateRowCol(Array<int> & ess_bc_dofs, Vector & sol, Vector & rhs);
   SparseMatrix * CreateMonolithic() const;
   void PrintMatlab(std::ostream & os = std::cout) const;
   virtual double& Elem (int i, int j);
   virtual const double& Elem (int i, int j) const;
   virtual MatrixInverse * Inverse() const
   {
      mfem_error("BlockMatrix::Inverse not implemented \n");
      return static_cast<MatrixInverse*>(__null);
   }
   virtual int NumNonZeroElems() const;
   virtual int GetRow(const int row, Array<int> &cols, Vector &srow) const;
   virtual void EliminateZeroRows();
   virtual void Mult(const Vector & x, Vector & y) const;
   virtual void AddMult(const Vector & x, Vector & y, const double val = 1.) const;
   virtual void MultTranspose(const Vector & x, Vector & y) const;
   virtual void AddMultTranspose(const Vector & x, Vector & y, const double val = 1.) const;
   virtual ~BlockMatrix();
   int owns_blocks;
private:
   inline void findGlobalRow(int iglobal, int & iblock, int & iloc) const;
   inline void findGlobalCol(int jglobal, int & jblock, int & jloc) const;
   int nRowBlocks;
   int nColBlocks;
   Array<int> row_offsets;
   Array<int> col_offsets;
   Array2D<SparseMatrix *> Aij;
};
BlockMatrix * Transpose(const BlockMatrix & A);
BlockMatrix * Mult(const BlockMatrix & A, const BlockMatrix & B);
inline void BlockMatrix::findGlobalRow(int iglobal, int & iblock, int & iloc) const
{
   if(iglobal > row_offsets[nRowBlocks])
      mfem_error("BlockMatrix::findGlobalRow");
   for(iblock = 0; iblock < nRowBlocks; ++iblock)
      if(row_offsets[iblock+1] > iglobal)
         break;
   iloc = iglobal - row_offsets[iblock];
}
inline void BlockMatrix::findGlobalCol(int jglobal, int & jblock, int & jloc) const
{
   if(jglobal > col_offsets[nColBlocks])
      mfem_error("BlockMatrix::findGlobalCol");
   for(jblock = 0; jblock < nColBlocks; ++jblock)
      if(col_offsets[jblock+1] > jglobal)
         break;
   jloc = jglobal - col_offsets[jblock];
}
}
namespace mfem
{
class BlockOperator : public Operator
{
public:
   BlockOperator(const Array<int> & offsets);
   BlockOperator(const Array<int> & row_offsets, const Array<int> & col_offsets);
   void SetDiagonalBlock(int iblock, Operator *op);
   void SetBlock(int iRow, int iCol, Operator *op);
   int NumRowBlocks() const { return nRowBlocks; }
   int NumColBlocks() const { return nColBlocks; }
   int IsZeroBlock(int i, int j) const { return (op(i,j)==__null) ? 1 : 0; }
   Operator & GetBlock(int i, int j)
   { if (!(op(i,j))) { { std::ostringstream s; s << std::setprecision(16); s << std::setiosflags(std::ios_base::scientific); s << "Verification failed: (" << "op(i,j)" << ") is false: " << "" << '\n'; s << " ... at line " << 76; s << " in " << __PRETTY_FUNCTION__ << " of file " << "../fem/../linalg/blockoperator.hpp" << "."; s << std::ends; if (!(0)) mfem::mfem_error(s.str().c_str()); else mfem::mfem_warning(s.str().c_str()); }; }; return *op(i,j); }
   Array<int> & RowOffsets() { return row_offsets; }
   Array<int> & ColOffsets() { return col_offsets; }
   virtual void Mult (const Vector & x, Vector & y) const;
   virtual void MultTranspose (const Vector & x, Vector & y) const;
   ~BlockOperator();
   int owns_blocks;
private:
   int nRowBlocks;
   int nColBlocks;
   Array<int> row_offsets;
   Array<int> col_offsets;
   Array2D<Operator *> op;
   mutable BlockVector xblock;
   mutable BlockVector yblock;
   mutable Vector tmp;
};
class BlockDiagonalPreconditioner : public Solver
{
public:
   BlockDiagonalPreconditioner(const Array<int> & offsets);
   void SetDiagonalBlock(int iblock, Operator *op);
   virtual void SetOperator(const Operator &op){ }
   int NumBlocks() const { return nBlocks; }
   Operator & GetDiagonalBlock(int iblock)
   { if (!(op[iblock])) { { std::ostringstream s; s << std::setprecision(16); s << std::setiosflags(std::ios_base::scientific); s << "Verification failed: (" << "op[iblock]" << ") is false: " << "" << '\n'; s << " ... at line " << 144; s << " in " << __PRETTY_FUNCTION__ << " of file " << "../fem/../linalg/blockoperator.hpp" << "."; s << std::ends; if (!(0)) mfem::mfem_error(s.str().c_str()); else mfem::mfem_warning(s.str().c_str()); }; }; return *op[iblock]; }
   Array<int> & Offsets() { return offsets; }
   virtual void Mult (const Vector & x, Vector & y) const;
   virtual void MultTranspose (const Vector & x, Vector & y) const;
   ~BlockDiagonalPreconditioner();
   int owns_blocks;
private:
   int nBlocks;
   Array<int> offsets;
   Array<Operator *> op;
   mutable BlockVector xblock;
   mutable BlockVector yblock;
};
}
namespace mfem
{
class SparseSmoother : public MatrixInverse
{
protected:
   const SparseMatrix *oper;
public:
   SparseSmoother() { oper = __null; }
   SparseSmoother(const SparseMatrix &a)
      : MatrixInverse(a) { oper = &a; }
   virtual void SetOperator(const Operator &a);
};
class GSSmoother : public SparseSmoother
{
protected:
   int type;
   int iterations;
public:
   GSSmoother(int t = 0, int it = 1) { type = t; iterations = it; }
   GSSmoother(const SparseMatrix &a, int t = 0, int it = 1)
      : SparseSmoother(a) { type = t; iterations = it; }
   virtual void Mult(const Vector &x, Vector &y) const;
};
class DSmoother : public SparseSmoother
{
protected:
   int type;
   double scale;
   int iterations;
   mutable Vector z;
public:
   DSmoother(int t = 0, double s = 1., int it = 1)
   { type = t; scale = s; iterations = it; }
   DSmoother(const SparseMatrix &a, int t = 0, double s = 1., int it = 1);
   virtual void Mult(const Vector &x, Vector &y) const;
};
}
namespace mfem
{
class ODESolver
{
protected:
   TimeDependentOperator *f;
public:
   ODESolver() : f(__null) { }
   virtual void Init(TimeDependentOperator &_f) { f = &_f; }
   virtual void Step(Vector &x, double &t, double &dt) = 0;
   virtual ~ODESolver() { }
};
class ForwardEulerSolver : public ODESolver
{
private:
   Vector dxdt;
public:
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class RK2Solver : public ODESolver
{
private:
   double a;
   Vector dxdt, x1;
public:
   RK2Solver(const double _a = 2./3.) : a(_a) { }
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class RK3SSPSolver : public ODESolver
{
private:
   Vector y, k;
public:
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class RK4Solver : public ODESolver
{
private:
   Vector y, k, z;
public:
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class ExplicitRKSolver : public ODESolver
{
private:
   int s;
   const double *a, *b, *c;
   Vector y, *k;
public:
   ExplicitRKSolver(int _s, const double *_a, const double *_b,
                    const double *_c);
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
   virtual ~ExplicitRKSolver();
};
class RK6Solver : public ExplicitRKSolver
{
private:
   static const double a[28], b[8], c[7];
public:
   RK6Solver() : ExplicitRKSolver(8, a, b, c) { }
};
class RK8Solver : public ExplicitRKSolver
{
private:
   static const double a[66], b[12], c[11];
public:
   RK8Solver() : ExplicitRKSolver(12, a, b, c) { }
};
class BackwardEulerSolver : public ODESolver
{
protected:
   Vector k;
public:
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class ImplicitMidpointSolver : public ODESolver
{
protected:
   Vector k;
public:
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class SDIRK23Solver : public ODESolver
{
protected:
   double gamma;
   Vector k, y;
public:
   SDIRK23Solver(int gamma_opt = 1);
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class SDIRK34Solver : public ODESolver
{
protected:
   Vector k, y, z;
public:
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
class SDIRK33Solver : public ODESolver
{
protected:
   Vector k, y;
public:
   virtual void Init(TimeDependentOperator &_f);
   virtual void Step(Vector &x, double &t, double &dt);
};
}
namespace mfem
{
class IterativeSolver : public Solver
{
protected:
   const Operator *oper;
   Solver *prec;
   int max_iter, print_level;
   double rel_tol, abs_tol;
   mutable int final_iter, converged;
   mutable double final_norm;
   double Dot(const Vector &x, const Vector &y) const;
   double Norm(const Vector &x) const { return sqrt(Dot(x, x)); }
public:
   IterativeSolver();
   void SetRelTol(double rtol) { rel_tol = rtol; }
   void SetAbsTol(double atol) { abs_tol = atol; }
   void SetMaxIter(int max_it) { max_iter = max_it; }
   void SetPrintLevel(int print_lvl);
   int GetNumIterations() { return final_iter; }
   int GetConverged() { return converged; }
   double GetFinalNorm() { return final_norm; }
   virtual void SetPreconditioner(Solver &pr);
   virtual void SetOperator(const Operator &op);
};
class SLISolver : public IterativeSolver
{
protected:
   mutable Vector r, z;
   void UpdateVectors();
public:
   SLISolver() { }
   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }
   virtual void Mult(const Vector &x, Vector &y) const;
};
void SLI(const Operator &A, const Vector &b, Vector &x,
         int print_iter = 0, int max_num_iter = 1000,
         double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);
void SLI(const Operator &A, Solver &B, const Vector &b, Vector &x,
         int print_iter = 0, int max_num_iter = 1000,
         double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);
class CGSolver : public IterativeSolver
{
protected:
   mutable Vector r, d, z;
   void UpdateVectors();
public:
   CGSolver() { }
   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }
   virtual void Mult(const Vector &x, Vector &y) const;
};
void CG(const Operator &A, const Vector &b, Vector &x,
        int print_iter = 0, int max_num_iter = 1000,
        double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);
void PCG(const Operator &A, Solver &B, const Vector &b, Vector &x,
         int print_iter = 0, int max_num_iter = 1000,
         double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24);
class GMRESSolver : public IterativeSolver
{
protected:
   int m;
public:
   GMRESSolver() { m = 50; }
   void SetKDim(int dim) { m = dim; }
   virtual void Mult(const Vector &x, Vector &y) const;
};
class FGMRESSolver : public IterativeSolver
{
protected:
   int m;
public:
   FGMRESSolver() { m = 50; }
   void SetKDim(int dim) { m = dim; }
   virtual void Mult(const Vector &x, Vector &y) const;
};
int GMRES(const Operator &A, Vector &x, const Vector &b, Solver &M,
          int &max_iter, int m, double &tol, double atol, int printit);
void GMRES(const Operator &A, Solver &B, const Vector &b, Vector &x,
           int print_iter = 0, int max_num_iter = 1000, int m = 50,
           double rtol = 1e-12, double atol = 1e-24);
class BiCGSTABSolver : public IterativeSolver
{
protected:
   mutable Vector p, phat, s, shat, t, v, r, rtilde;
   void UpdateVectors();
public:
   BiCGSTABSolver() { }
   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }
   virtual void Mult(const Vector &x, Vector &y) const;
};
int BiCGSTAB(const Operator &A, Vector &x, const Vector &b, Solver &M,
             int &max_iter, double &tol, double atol, int printit);
void BiCGSTAB(const Operator &A, Solver &B, const Vector &b, Vector &x,
              int print_iter = 0, int max_num_iter = 1000,
              double rtol = 1e-12, double atol = 1e-24);
class MINRESSolver : public IterativeSolver
{
protected:
   mutable Vector v0, v1, w0, w1, q;
   mutable Vector u1;
public:
   MINRESSolver() { }
   virtual void SetPreconditioner(Solver &pr)
   { IterativeSolver::SetPreconditioner(pr); if (oper) u1.SetSize(width); }
   virtual void SetOperator(const Operator &op);
   virtual void Mult(const Vector &b, Vector &x) const;
};
void MINRES(const Operator &A, const Vector &b, Vector &x, int print_it = 0,
            int max_it = 1000, double rtol = 1e-12, double atol = 1e-24);
void MINRES(const Operator &A, Solver &B, const Vector &b, Vector &x,
            int print_it = 0, int max_it = 1000,
            double rtol = 1e-12, double atol = 1e-24);
class NewtonSolver : public IterativeSolver
{
protected:
   mutable Vector r, c;
public:
   NewtonSolver() { }
   virtual void SetOperator(const Operator &op);
   void SetSolver(Solver &solver) { prec = &solver; }
   virtual void Mult(const Vector &b, Vector &x) const;
};
int aGMRES(const Operator &A, Vector &x, const Vector &b,
           const Operator &M, int &max_iter,
           int m_max, int m_min, int m_step, double cf,
           double &tol, double &atol, int printit);
class SLBQPOptimizer : public IterativeSolver
{
protected:
   Vector lo, hi, w;
   double a;
   inline double solve(double l, const Vector &xt, Vector &x, int &nclip) const
   {
      add(xt, l, w, x);
      x.median(lo,hi);
      nclip++;
      return Dot(w,x)-a;
   }
   inline void print_iteration(int it, double r, double l) const;
public:
   SLBQPOptimizer() {}
   void SetBounds(const Vector &_lo, const Vector &_hi);
   void SetLinearConstraint(const Vector &_w, double _a);
   virtual void Mult(const Vector &xt, Vector &x) const;
   virtual void SetPreconditioner(Solver &pr);
   virtual void SetOperator(const Operator &op);
};
}
