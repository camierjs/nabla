<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="darcy" xml:lang="en">
  
  <arcane>
    <title>Experimentation Arcane</title>
    <timeloop>darcyLoop</timeloop>
  </arcane>

   <mesh>
     <meshgenerator>
 		 <cartesian>
			<nsd>4 1 0</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="4" prx="1.0">1.0</lx>
			<ly ny="4" pry="1.0">1.0</ly>
			<lz nz="4" prz="1.0">1.0</lz>
		 </cartesian> 
     </meshgenerator>
   </mesh>
   
   <arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
	  <output-history-period>0</output-history-period>
     <output>
       <group>AllCells</group>
       <variable>cell_pressure</variable>
       <!--variable>permeability</variable-->
       <variable>cell_porosity</variable>
       <variable>cell_total_mobility</variable>
       <variable>cell_water_saturation</variable>
     </output>
   </arcane-post-processing>
   
   <arcane-checkpoint>
     <do-dump-at-end>false</do-dump-at-end>
   </arcane-checkpoint>

  <darcy>
    <option_stoptime>1.</option_stoptime>
    <option_ini_porosity>1.</option_ini_porosity>
    <option_ini_permeability>1.</option_ini_permeability>
    <option_ini_oil_density>1.</option_ini_oil_density>
    <option_ini_water_density>1.</option_ini_water_density>
    <option_ini_oil_viscosity>1.</option_ini_oil_viscosity>
    <option_ini_water_viscosity>1.</option_ini_water_viscosity>
   </darcy>

</case>
