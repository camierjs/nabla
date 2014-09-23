<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="materials" xml:lang="en">
  <arcane>
	 <title>Exemple material</title>
	 <timeloop>materialsLoop</timeloop>
    <configuration>
      <parametre name="NotParallel" value="true" />
    </configuration>
  </arcane>

  <arcane-post-processing>
    <save-init>0</save-init>
    <end-execution-output>0</end-execution-output>
	 <output-period>0</output-period>
	 <output>
		<variable>cell_pressure</variable>
      <group>AllCells</group>
 	 </output>
  </arcane-post-processing>
  
  <mesh>
    <meshgenerator>
      <sod>
        <x delta='0.05'>4</x>
        <y delta='0.25'>4</y>
      </sod>
    </meshgenerator>
  </mesh>
  <materials>
    <compatibility_mode>true</compatibility_mode>
  </materials>
</case>
