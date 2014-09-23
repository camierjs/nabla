<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="comd" xml:lang="en">
  <arcane>
	 <title>comd</title>
	 <timeloop>comdLoop</timeloop>
    <modules>
      <module name="ArcaneCheckpoint" actif="false" />
    </modules>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
	  <output-period>0</output-period>
     <output-history-period>0</output-history-period>
		<output>
		  <variable>cell_natom</variable>
        <group>AllCells</group>
 		</output>
	</arcane-post-processing>
 
   <mesh nb-ghostlayer="1" ghostlayer-builder-version="3">
     <meshgenerator>
       <cartesian>
         <nsd>1 1 1</nsd> 
         <origine>0.0 0.0 0.0</origine>
         <lx nx='4'>103.042016006400004358</lx>
         <ly ny='4'>103.042016006400004358</ly>
         <lz nz='4'>103.042016006400004358</lz>
       </cartesian>
     </meshgenerator>
   </mesh> 

   <comd>
     <option_nx>4</option_nx>
     <option_ny>4</option_ny>
     <option_nz>4</option_nz>
     <option_lat>5.152101</option_lat>
     <option_defgrad>1</option_defgrad>
     <option_cutoff>4.59</option_cutoff>
     <option_boxfactor>1</option_boxfactor>
     <option_max_iteration>4</option_max_iteration>
   </comd>
</case>
