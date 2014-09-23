<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="mctb" xml:lang="en">
  <arcane>
	 <title>mctb</title>
	 <timeloop>mctbLoop</timeloop>
    <modules>
      <module name="ArcaneCheckpoint" actif="false" />
    </modules>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
	  <output-period>0</output-period>
     <output-history-period>0</output-history-period>
		<output>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>
   
   <mesh nb-ghostlayer="1" ghostlayer-builder-version="3">
     <meshgenerator>
       <cartesian>
         <nsd>2 2</nsd> 
         <origine>0.0 0.0 0.0</origine>
         <lx nx='280'>1.0</lx>
         <ly ny='80'>0.12</ly>
       </cartesian>
     </meshgenerator>
   </mesh> 

   <mctb>
     <dim>2</dim>                             <!--Problemdimension-->
     <NX>280</NX>                             <!--NumberofcellsinXdirection-->
     <NY>80</NY>                              <!--NumberofcellsinYdirection-->
     <min_coord_x>0.0</min_coord_x>           <!--Leftcornercoordinates-->
     <min_coord_y>0.0</min_coord_y>           <!--Downcornercoordinates-->
     <max_coord_x>1.0</max_coord_x>           <!--Rightcornercoordinates-->
     <max_coord_y>0.12</max_coord_y>          <!--Upcornercoordinates-->

     <absorption_rate>0.1</absorption_rate>   <!--sigma_aabsorption_rate*sigma_t,sigma_ssigma_t-sigma_a-->
     <vsrc_sig_ratio>1.8e-18</vsrc_sig_ratio> <!--volumicsourcetermvol_src/(sigma_t*dt)-->
     <bnd_src>2.1e-15</bnd_src>               <!--Surfacicsourcetermfortheleftboundary([Weight]/[S]/[T])-->
     <velocity>3e2</velocity>                 <!--particlevelocity(samevelocityforallparticles)-->
     <Nmc_ini>100000</Nmc_ini>                <!--NumberofMCparticles(asagoal)-->
     <dt_cycle>1.e-2</dt_cycle>               <!--Timestep-->
     <nb_of_iter>100</nb_of_iter>             <!--Numberoftimestepiterations-->

     <bc_x_L>3</bc_x_L>                       <!--bc_incoming_current-->
     <bc_x_R>1</bc_x_R>                       <!--bc_vaccum-->
     <bc_y_L>2</bc_y_L>                       <!--bc_specular-->
     <bc_y_R>1</bc_y_R>                       <!--bc_vaccum-->
     <zc>0.5</zc>                             <!--cm-->
     <rc>0.0</rc>                             <!--cm-->
     <option_z0>0.1</option_z0>               <!--cm-->
     <option_r0>0.06</option_r0>              <!--cm-->
     <cst_sigma_t0>1.e-1</cst_sigma_t0>
     <cst_sigma_t1>1.e+5</cst_sigma_t1>
   </mctb>
</case>
