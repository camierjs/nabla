<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="poisson_diff" xml:lang="en">
  <arcane>
	 <title>Exemple Poison</title>
	 <timeloop>poisson_diffLoop</timeloop>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
	  <output-period>0</output-period>
		<output>
		  <variable>cell_th</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>

   <mesh>
     <meshgenerator>
 		 <cartesian>
			<nsd>2 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="64" prx="1.0">1.0</lx>
			<ly ny="64" pry="1.0">1.0</ly>
		 </cartesian> 
     </meshgenerator>
   </mesh>

   <poisson_diff>
     <option_deltat>0.001</option_deltat>
     <option_ini_temperature>300.0</option_ini_temperature>
     <option_hot_temperature>700.0</option_hot_temperature>
     <option_max_iterations>1024</option_max_iterations>
   </poisson_diff>
</case>
