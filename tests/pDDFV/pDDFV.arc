<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="pDDFV" xml:lang="en">
  <arcane>
	 <title>Exemple Poisson Positive DDFV</title>
	 <timeloop>pDDFVLoop</timeloop>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
     <!--format name="Ensight7PostProcessor"/-->
	  <output-period>0</output-period>
	  <output>
		  <variable>cell_cell_greek_theta</variable>
		  <variable>cell_cell_greek_theta_diff</variable>
		  <!--variable>cell_cell_exact_solution</variable-->
		  <!--variable>cell_cell_sd_id</variable-->
		  <!--variable>cell_cell_th_kp1mk</variable-->
		  <variable>node_node_greek_theta</variable>
		  <variable>node_node_greek_theta_diff</variable>
		  <!--variable>node_node_unique_id</variable-->
		  <!--variable>node_node_exact_solution</variable-->
		  <!--variable>node_node_sd_id</variable-->
		  <!--variable>node_node_th_kp1mk</variable-->
		  <!--variable>face_face_th</variable-->
        <group>AllCells</group>
 		</output> 
	</arcane-post-processing>

   <arcane-checkpoint>
     <period>0</period>
     <do-dump-at-end>false</do-dump-at-end>
   </arcane-checkpoint>

   <mesh>
     <!--sqrh sqro sqrhq -->
     <!-- qtd qud sqr -->
     <!-- sqr10 sqr20 sqr40 sqr80 sqr160 sqr320 sqr640  -->
     <!-- zzz10.unf zzz20.unf zzz40.unf zzz80.unf zzz160.unf zzz320.unf zzz640.unf -->
     <!-- zzzSoft10.unf zzzSoft20.unf zzzSoft40.unf zzzSoft80.unf zzzSoft160.unf zzzSoft320.unf zzzSoft640.unf -->
     <!-- zzzStep zzzStep10 -->
     <!--file internal-partition="true">../../unf/sqrhq.unf</file-->
     <meshgenerator>
 		 <cartesian>
			<nsd>4 1 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="10" prx="1.0">1.0</lx>
			<ly ny="10" pry="1.0">1.0</ly>
	    </cartesian> 
     </meshgenerator>
   </mesh>

   <p-d-d-f-v>
     <!--option_dtt_ini>0.01</option_dtt_ini--> <!-- 10: 9.285e-01 -->
     <option_greek_deltat_ini>1.0</option_greek_deltat_ini> <!-- 20: -->
     <!--option_dtt_ini>0.000625</option_dtt_ini--> <!-- 40: -->
     
      <!-- Maillage Triangles ou Quads -->
     <option_quads>true</option_quads>
     <option_triangles>false</option_triangles>

     <!-- Utilisation d'un maillage indirect (dual barycentrique) -->
     <option_indirect>true</option_indirect>

     <!-- Utilisation des valeures aux faces ou de la moyenne -->
     <option_trial>false</option_trial>
     <option_trial_average>true</option_trial_average>

     <!-- Options de deformations de maillages-->
     <option_rdq>false</option_rdq>
     <option_rdq_al>0.3</option_rdq_al>

     <option_sncq>false</option_sncq>
     <option_sncq_th>0.25</option_sncq_th>
     
     <option_kershaw>false</option_kershaw>

     <!-- Options des solutions Analytiques *ou pas* -->
     <option_hole>false</option_hole>
     <option_atan>false</option_atan>

     <!-- Test du 'Parabolic problem with Discontinuous Coefficients -->
     <option_gao_wu>false</option_gao_wu>
     <option_gao_wu_k>1.0</option_gao_wu_k>
     <option_gao_wu_area>true</option_gao_wu_area>
     
     <!-- Options de l'anisotropisme -->
     <option_isotropic>true</option_isotropic>

     <option_k>1.0</option_k>
     <option_greek_theta>0.0</option_greek_theta><!-- 0.52359877559829887308 -->

     <option_spin_greek_theta>false</option_spin_greek_theta>
     <option_spin_greek_theta_x>3.0</option_spin_greek_theta_x>
     <option_spin_greek_theta_y>3.0</option_spin_greek_theta_y>

     <option_ini_temperature>0.0</option_ini_temperature>
     <option_PartMg_temperature>0.0</option_PartMg_temperature>
     <option_max_iterations>8192</option_max_iterations>
     <option_picard_ep>1e-6</option_picard_ep>

     <alephEpsilon>1e-10</alephEpsilon>
     <alephMaxIterations>8192</alephMaxIterations>

     <alephUnderlyingSolver>2</alephUnderlyingSolver>
     <alephPreconditionerMethod>2</alephPreconditionerMethod>
     <alephSolverMethod>3</alephSolverMethod>
     <alephNumberOfCores>0</alephNumberOfCores>
     <!--  5-3-0 ou x-6-1 -->
     
     <option_debug_errors>false</option_debug_errors>
     <option_debug_geometry>false</option_debug_geometry>

     <option_debug_primal>false</option_debug_primal>

     <option_debug_dual>false</option_debug_dual>
     <option_debug_dual_aleph>false</option_debug_dual_aleph>

     <option_debug_trial>false</option_debug_trial>
     <option_debug_trial_aleph>false</option_debug_trial_aleph>

     <option_aleph_dump_matrix>false</option_aleph_dump_matrix>

     <option_dag>false</option_dag>

     <option_quit_when_finish>true</option_quit_when_finish>
     <option_only_one_iteration>true</option_only_one_iteration>
   </p-d-d-f-v>
</case>
