<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="poisson_ddfv" xml:lang="en">
  <arcane>
	 <title>Exemple Poisson</title>
	 <timeloop>poisson_ddfvLoop</timeloop>
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
     <!--file internal-partition="true">nabla.unf</file-->
     <meshgenerator>
 		 <cartesian>
			<nsd>4 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="16" prx="1.0">1.0</lx>
			<ly ny="16" pry="1.0">1.0</ly>
		 </cartesian> 
     </meshgenerator>
   </mesh>

   <poisson_ddfv>
     <option_quads>true</option_quads>
     <option_triangles>false</option_triangles>

     <option_deltat>0.0001</option_deltat>
    <option_max_iterations>2</option_max_iterations>
   </poisson_ddfv>
</case>
