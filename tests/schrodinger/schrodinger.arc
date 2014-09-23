<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="schrodinger" xml:lang="en">
  <arcane>
	 <title>Exemple Schrodinger</title>
	 <timeloop>schrodingerLoop</timeloop>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
	  <output-period>0</output-period>
		<output>
		  <variable>node_node_density</variable>
		  <variable>cell_cell_density</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>

   <mesh>
     <file internal-partition="true">../nabla.unf</file>
   </mesh>

   <schrodinger>
     <option_deltat>0.0</option_deltat>
     <option_epsilon>0.01</option_epsilon>
     <option_ini_borders>1.0</option_ini_borders>
     <option_ini_iterations>1</option_ini_iterations>
     <option_max_iterations>2</option_max_iterations>

     <alephEpsilon>1.e-8</alephEpsilon>
     <alephMaxIterations>16384</alephMaxIterations>
     <!-- DIAGONAL=0, AINV=1, AMG=2, IC=3, POLY=4, ILU=5, ILUp=6 -->
     <alephPreconditionerMethod>5</alephPreconditionerMethod>
     <!-- 0:Auto(=Sloop), 1:Sloop, 2:Hypre, 3:Trilinos, 4:Cuda, 5:PETSc -->
     <alephUnderlyingSolver>0</alephUnderlyingSolver>
     <!--PCG=0, BiCGStab=1 , BiCGStab2=2, GMRES=3, SAMG=4, QMR=5, SuperLU=6 -->
     <alephSolverMethod>3</alephSolverMethod>
     <alephNumberOfCores>0</alephNumberOfCores>
   </schrodinger>
</case>
