<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="cartesian" xml:lang="en">
  <arcane>
	 <title>Exemple cartesian</title>
	 <timeloop>cartesianLoop</timeloop>
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
        <x delta='0.05'>50</x>
        <y delta='0.25'>20</y>
      </sod>
    </meshgenerator>
    <initialisation>
      <variable nom="cell_density" valeur="1." groupe="ZG" />
      <variable nom="cell_pressure" valeur="1." groupe="ZG" />
      <variable nom="cell_density" valeur="0.125" groupe="ZD" />
      <variable nom="cell_pressure" valeur="0.1" groupe="ZD" />
    </initialisation>
  </mesh>
</case>
