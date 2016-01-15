<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="anyItem" xml:lang="en">
  <arcane>
	 <title>AnyItem Test</title>
	 <timeloop>anyItemLoop</timeloop>
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
			<nsd>4 1 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="1000" prx="1.0">1.0</lx>
			<ly ny="10" pry="1.0">1.0</ly>
			<lz nz="10" prz="1.0">1.0</lz>
		 </cartesian> 
	  </meshgenerator>
	</mesh>
</case>
