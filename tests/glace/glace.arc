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
    <save-init>1</save-init>
	 <output-period>10</output-period>
    <output-history-period>0</output-history-period>
    <end-execution-output>1</end-execution-output>
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
 		 <cartesian>
			<nsd>4 1 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="128" prx="1.0">10.0</lx>
			<ly ny="16" pry="1.0">1.0</ly>
			<lz nz="16" prz="1.0">1.0</lz>
		 </cartesian> 
	  </meshgenerator>
	</mesh>
   <glace>
     <option_chic>false</option_chic>
     <option_glace>true</option_glace>

     <option_test_sod>true</option_test_sod>

     <option_hexa>false</option_hexa>
     <option_cylinder>true</option_cylinder>

     <option_quads>false</option_quads>
     <option_triangles>false</option_triangles>

     <option_greek_deltat_ini>0.001</option_greek_deltat_ini>
     <option_greek_deltat_min>1e-12</option_greek_deltat_min>
     <option_greek_deltat_max>0.01</option_greek_deltat_max>
     <option_greek_deltat_control>0.9</option_greek_deltat_control>
     <option_greek_deltat_end>1.0</option_greek_deltat_end>

     <option_x_min>0.0</option_x_min>
     <option_x_interface>0.5</option_x_interface>
     <option_x_max>1.0</option_x_max>

     <option_y_min>0.0</option_y_min>
     <option_y_max>0.15</option_y_max>

     <option_z_min>-0.15</option_z_min>
     <option_z_max>+0.15</option_z_max>

     <option_ini_zg_greek_rho>2.0</option_ini_zg_greek_rho>
     <option_ini_zg_p>2.0</option_ini_zg_p>

     <option_ini_zd_greek_rho>1.0</option_ini_zd_greek_rho>
     <option_ini_zd_p>1.0</option_ini_zd_p>

     <option_max_iterations>8</option_max_iterations>
   </glace>
</case>
