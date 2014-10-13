<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="positive_ddfv" xml:lang="en">
  <arcane>
	 <title>Exemple Poison Positive DDFV</title>
	 <timeloop>positive_ddfvLoop</timeloop>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
	  <output-period>0</output-period>
		<output>
		  <variable>cell_cell_th</variable>
		  <variable>node_node_th</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>

   <mesh>
     <!-- qtd qud sqr -->
     <file internal-partition="true">qtd.unf</file>
     <!--meshgenerator>
 		 <cartesian>
			<nsd>4 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="128" prx="1.0">1.0</lx>
			<ly ny="128" pry="1.0">1.0</ly>
		 </cartesian> 
     </meshgenerator-->
   </mesh>

   <positive_ddfv>
     <option_quads>0</option_quads>
     <option_triangles>1</option_triangles>
     <option_deltat>0.001</option_deltat>
     <option_deltat_factor>10</option_deltat_factor>
     <option_ini_temperature>300</option_ini_temperature>
     <option_hot_temperature>700</option_hot_temperature>
     <option_max_iterations>8192</option_max_iterations>
     <alephEpsilon>1e-08</alephEpsilon>
     <alephUnderlyingSolver>0</alephUnderlyingSolver>
     <alephMaxIterations>128</alephMaxIterations>
     <alephPreconditionerMethod>0</alephPreconditionerMethod>
     <alephSolverMethod>6</alephSolverMethod>
     <alephNumberOfCores>0</alephNumberOfCores>
   </positive_ddfv>
</case>
