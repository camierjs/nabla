<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="glace2D" xml:lang="en">
  <arcane>
	 <title>GLACE module</title>
	 <timeloop>glace2DLoop</timeloop>
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
      <!--variable>cell_m</variable>
      <variable>cell_V</variable>
      <variable>cell_u</variable>
      <variable>cell_c</variable>
      <variable>node_node_u</variable>
      <variable>node_node_u_second_member</variable-->
      <!--variable>cell_rh</variable-->
      <variable>cell_p</variable>
      <variable>cell_ZG</variable>
      <!--variable>cell_ZD</variable-->
      <!--variable>cell_rhEp</variable>
      <variable>cell_itEp</variable>
      <variable>cell_glace_dtt</variable-->
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

 	<mesh>
     <!--file internal-partition="true">block_mesh.mli</file-->
     <!--file internal-partition="true">../thex_mesh.unf</file-->
     <!--file internal-partition="true">cyl200.mli</file-->
	  <meshgenerator>
       <!--sod zyx='true'>
         <x set='false' delta='0.25'>4</x>
         <y set='false' delta='1.0'>4</y>
     </sod-->
     <cartesian>
			<nsd>8 1 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="32" prx="1.0">1.0</lx>
			<ly ny="4" pry="1.0">4.0</ly>
	  </cartesian> 
	  </meshgenerator>
	</mesh>
   <glace2-d>
     <DEBUG>false</DEBUG>
     
     <option_chic>false</option_chic>
     <option_glace>true</option_glace>

     <option_test_sod>true</option_test_sod>

     <option_hexa>false</option_hexa>
     <option_cylinder>false</option_cylinder>

     <option_quads>true</option_quads>
     <option_triangles>false</option_triangles>

     <option_dtt_initial>0.00001</option_dtt_initial>
     <option_dtt_min>1e-12</option_dtt_min>
     <option_dtt_max>0.01</option_dtt_max>
     <option_dtt_control>0.15</option_dtt_control> <!-- CFL -->
     <option_stoptime>0.2</option_stoptime>

     <option_x_min>0.0</option_x_min>
     <option_x_interface>0.5</option_x_interface>
     <option_x_max>1.0</option_x_max>

     <option_y_min>0.0</option_y_min>
     <option_y_max>4.0</option_y_max>

     <option_z_min>0.0</option_z_min>
     <option_z_max>0.0</option_z_max>

     <option_ini_zg_rh>1.0</option_ini_zg_rh>
     <option_ini_zg_p>1.0</option_ini_zg_p>

     <option_ini_zd_rh>0.125</option_ini_zd_rh>
     <option_ini_zd_p>0.1</option_ini_zd_p>

     <option_max_iterations>8</option_max_iterations>
   </glace2-d>
</case>
