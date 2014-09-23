<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="glace" xml:lang="en">
  <arcane>
	 <title>GLACE module</title>
	 <timeloop>glaceLoop</timeloop>
  </arcane>

  <main>
    <do-time-history>0</do-time-history>
  </main>

  <arcane-post-processing>
    <save-init>0</save-init>
	 <output-period>0</output-period>
    <output-history-period>0</output-history-period>
    <end-execution-output>0</end-execution-output>
    <output>
      <variable>cell_rh</variable>
      <variable>cell_p</variable>
      <!--variable>cell_rhEp</variable>
      <variable>cell_itEp</variable>
      <variable>cell_glace_dtt</variable-->
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

 	<mesh>
     <file internal-partition="true">../sod_triangles.unf</file>
	</mesh>

   <glace>
     <option_test_sod>true</option_test_sod>

     <option_quads>false</option_quads>
     <option_triangles>true</option_triangles>

     <option_dtt_ini>0.001</option_dtt_ini>
     <option_dtt_min>1e-12</option_dtt_min>
     <option_dtt_max>0.01</option_dtt_max>
     <option_dtt_control>0.9</option_dtt_control>
     <option_dtt_end>0.01</option_dtt_end>

     <option_x_max>1.0</option_x_max>
     <option_y_max>0.1</option_y_max>

   </glace>
</case>
