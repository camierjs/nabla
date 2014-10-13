<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="nahea" xml:lang="en">
  <arcane>
	 <title>Nahea</title>
	 <timeloop>naheaLoop</timeloop>
    <modules>
      <module name="ArcanePostProcessing" active="true" />
      <module name="ArcaneCheckpoint" actif="true" />
    </modules>
  </arcane>

  <arcane-checkpoint>
    <period>0</period> <!--Nombre d'itérations entre deux protections.-->
    <do-dump-at-end>true</do-dump-at-end>
    <!-- Service possible: ArcaneHdf5MultiCheckpoint ArcaneHdf5Checkpoint2-->
    <checkpoint-service name="ArcaneHdf5Checkpoint2">
      <fileset-size>32</fileset-size>
      <index-modulo>4</index-modulo>
    </checkpoint-service>
  </arcane-checkpoint>

	<arcane-post-processing>
     <save-init>0</save-init>
	  <output-period>0</output-period>
     <output-history-period>0</output-history-period>
     <output-history-shrink>1</output-history-shrink>
	</arcane-post-processing>
 
   <mesh nb-ghostlayer="0" ghostlayer-builder-version="3">
     <meshgenerator>
       <cartesian>
         <nsd>1 1 1</nsd> 
         <origine>0.0 0.0 0.0</origine>
         <lx nx='1'>1.0</lx>
         <ly ny='1'>1.0</ly>
         <lz nz='1'>1.0</lz>
       </cartesian>
     </meshgenerator>
   </mesh> 

   <nahea>
     <!--    583 ~    4253 ROUNDOFF ERROR
           10489 ~  110503 dft =   6 : 4 threads
           60745 ~  756839
           68301 ~  859433
          106991 ~ 1398269 dft =  72 : 1400s@2-threads
          215208 ~ 2976221 dft = 160 : 3991s@4-threads
       -->
     <option_nth_prime>218239</option_nth_prime>

     <!-- Log console de l'avancement -->
     <option_log_modulo>8</option_log_modulo>

     <!-- A 5mn de la fin, on sort pour écrire le checkpoint -->
     <option_tst_tremain>300</option_tst_tremain>

     <!-- Nombre max d'itération -->
     <option_max_iterations>1024</option_max_iterations>
   </nahea>
</case>
