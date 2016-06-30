<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="heat" xml:lang="en">
  <arcane>
	 <title>Exemple Heat</title>
	 <timeloop>heatLoop</timeloop>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
	  <output-period>0</output-period>
		<output>
		  <variable>cell_greek_theta</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>
   
   <arcane-checkpoint>
     <period>0</period>
     <do-dump-at-end>false</do-dump-at-end>
   </arcane-checkpoint>

   <mesh>
     <meshgenerator>
 		 <cartesian>
			<nsd>2 2</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="4" prx="1.0">1.0</lx>
			<ly ny="4" pry="1.0">1.0</ly>
		 </cartesian> 
     </meshgenerator>
   </mesh>

   <heat>
     <option_deltat>0.001</option_deltat>
     <option_ini_temperature>300.0</option_ini_temperature>
     <option_hot_temperature>700.0</option_hot_temperature>
     <option_max_iterations>1</option_max_iterations>

     <alephEpsilon>1.e-10</alephEpsilon>
     <alephMaxIterations>16384</alephMaxIterations>
     <!-- DIAGONAL=0, AINV=1, AMG=2, IC=3, POLY=4, ILU=5, ILUp=6 -->
     <alephPreconditionerMethod>0</alephPreconditionerMethod>
     <!-- 0:Auto(=Sloop), 1:Sloop, 2:Hypre, 3:Trilinos, 4:Cuda, 5:PETSc -->
     <alephUnderlyingSolver>2</alephUnderlyingSolver>
     <!--PCG=0, BiCGStab=1 , BiCGStab2=2, GMRES=3, SAMG=4, QMR=5, SuperLU=6 -->
     <alephSolverMethod>3</alephSolverMethod>
     <alephNumberOfCores>0</alephNumberOfCores>
   </heat>
</case>
