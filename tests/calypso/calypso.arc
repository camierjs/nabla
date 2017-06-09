<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="calypso" xml:lang="en">
  
  <arcane>
	 <title>MonaiValeyTest</title>
	 <timeloop>calypsoLoop</timeloop>
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
		<variable>cell_h</variable>
		<variable>cell_hn</variable>
		<!--variable>cell_d</variable-->
 		<variable>cell_un</variable>
 		<variable>cell_vn</variable>
     <group>AllCells</group>
 	 </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <period>0</period>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh>
    <meshgenerator>
      <cartesian>
		  <nsd>2 1</nsd>
		  <origine>0.0 0.0 0.0</origine>
		  <lx nx="66" prx="1.0">0.1</lx>
		  <ly ny="50" pry="1.0">0.1</ly>
	   </cartesian>
    </meshgenerator>
  </mesh>

  <calypso>
    <NX>64</NX>
    <NY>48</NY>
    <X_EDGE_ELEMS>66</X_EDGE_ELEMS>
    <Y_EDGE_ELEMS>50</Y_EDGE_ELEMS>
    <option_eps_fp>1.0e-12</option_eps_fp>
    <option_fill>false</option_fill>
    <option_arcane>true</option_arcane>
    <option_max_iterations>8</option_max_iterations>
  </calypso>
</case>
