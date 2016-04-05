<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="fmly" xml:lang="en">
  <arcane>
	 <title>Exemple Family</title>
	 <timeloop>fmlyLoop</timeloop>
  </arcane>

	<arcane-post-processing>
     <save-init>0</save-init>
     <end-execution-output>0</end-execution-output>
	  <output-period>0</output-period>
	</arcane-post-processing>

   <arcane-checkpoint>
     <period>0</period>
     <do-dump-at-end>false</do-dump-at-end>
   </arcane-checkpoint>
   
   <mesh>
     <meshgenerator>
       <sod zyx='true'>
         <x set='true' delta='0.1'>16</x>
         <y set='true' delta='0.1'>16</y>
       </sod>
     </meshgenerator>
   </mesh>
</case>
