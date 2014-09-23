<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="ddfv" xml:lang="en">
  <arcane>
	 <title>Exemple ddfv</title>
	 <timeloop>ddfvLoop</timeloop>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
	  <output-period>0</output-period>
		<output>
		  <variable>cell_cell_density</variable>
 		  <variable>cell_cell_density_re_zero</variable>
 		  <variable>cell_cell_density_im_zero</variable>
 		  <variable>cell_cell_density_or_reim_zero</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>

   <mesh>
     <file internal-partition="true">../nabla.unf</file>
   </mesh>

   <ddfv>
     <option_deltat>0.00001</option_deltat>
     <option_epsilon>0.001</option_epsilon>
     <option_ini_borders>1.0</option_ini_borders>
     <option_ini_iterations>1</option_ini_iterations>
     <option_max_iterations>4</option_max_iterations>
   </ddfv>
</case>
