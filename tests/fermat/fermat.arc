<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="fermat" xml:lang="en">
  <arcane>
	 <title>Fermat</title>
	 <timeloop>fermatLoop</timeloop>
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
    </checkpoint-service>
  </arcane-checkpoint>

	<arcane-post-processing>
     <save-init>0</save-init>
	  <output-period>0</output-period>
     <output-history-period>0</output-history-period>
		<output>
		  <variable>cell_nth</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>
 
   <mesh nb-ghostlayer="0" ghostlayer-builder-version="3">
     <meshgenerator>
       <cartesian>
         <nsd>2 2 1</nsd> 
         <origine>0.0 0.0 0.0</origine>
         <lx nx='2'>1.0</lx>
         <ly ny='2'>1.0</ly>
         <lz nz='2'>1.0</lz>
       </cartesian>
     </meshgenerator>
   </mesh> 

   <fermat-module>
     <option_ini_nth>1</option_ini_nth>
     <!-- Log console de l'avancement -->
     <option_log_tremain>8</option_log_tremain>
     <!-- A 20mn de la fin, on sort pour écrire le checkpoint -->
     <option_tst_tremain>3000</option_tst_tremain>
     <!-- Nombre max d'itération, à mettre en vis-à-vis du prime (exposant) recherché
          1024 4096 32768 1048576 4194304 16777216 67108864 134217728 -->
     <option_max_iterations>14</option_max_iterations>
   </fermat-module>
</case>
