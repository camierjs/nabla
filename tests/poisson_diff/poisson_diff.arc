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
		  <variable>cell_cell_temperature</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>

  <mesh>
     <meshgenerator>
       <sod>
         <x set='true' delta='0.01'>4</x>
         <y set='true' delta='0.01'>4</y>
         <z set='true' delta='0.01' total='false'>2</z>
       </sod>
     </meshgenerator>
   </mesh>

   <poisson_diff>
     <option_deltat>0.001</option_deltat>
     <option_ini_temperature>300.0</option_ini_temperature>
     <option_hot_temperature>2700.0</option_hot_temperature>
     <option_max_iterations>8</option_max_iterations>
   </poisson_diff>
</case>
