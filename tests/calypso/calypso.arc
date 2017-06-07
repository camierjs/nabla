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
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh>
    <meshgenerator>
      <cartesian>
		  <nsd>4 1</nsd>
		  <origine>0.0 0.0 0.0</origine>
		  <lx nx="66" prx="1.0">1.0</lx>
		  <ly ny="50" pry="1.0">1.0</ly>
	   </cartesian>
    </meshgenerator>
  </mesh>

  <calypso>
    <NX>64</NX>
    <NY>48</NY>
    <X_EDGE_ELEMS>66</X_EDGE_ELEMS>
    <Y_EDGE_ELEMS>50</Y_EDGE_ELEMS>
    <option_eps_fp>1.0e-12</option_eps_fp>
  </calypso>
</case>
