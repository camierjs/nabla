<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="diffusion_ddfv" xml:lang="en">
  <arcane>
	 <title>Exemple Poison</title>
	 <timeloop>diffusion_ddfvLoop</timeloop>
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

   <diffusion_ddfv>
     <option_quads>false</option_quads>
     <option_triangles>true</option_triangles>

     <option_deltat>0.001</option_deltat>
    <option_max_iterations>1</option_max_iterations>
   </diffusion_ddfv>
</case>
