<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="mhydro_splited" xml:lang="en">
  <arcane>
	 <title>Exemple Nabla MicroHydro</title>
	 <timeloop>mhydro_splitedLoop</timeloop>
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
      <variable>cell_rh</variable>
      <variable>cell_pressure</variable>
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh>
    <!--1049600 noeuds, 984064 mailles-->
    <meshgenerator><sod>
        <x set='false' delta='0.02'>64</x>
        <y set='true' delta='0.02'>7</y>
        <z set='true' delta='0.02' total='true'>7</z>
      </sod></meshgenerator>
    <!--meshgenerator><sod>
        <x set='true' delta='0.25'>4</x>
        <y set='true' delta='0.02'>3</y>
        <z set='true' delta='0.02' total='true'>3</z>
      </sod></meshgenerator-->
	 <initialisation>
		<variable nom="cell_rh" valeur="1." groupe="ZG" />
		<variable nom="cell_pressure" valeur="1." groupe="ZG" />
		<variable nom="cell_adiabatic_cst" valeur="1.4" groupe="ZG" />
		<variable nom="cell_rh" valeur="0.125" groupe="ZD" />
		<variable nom="cell_pressure" valeur="0.1" groupe="ZD" />
		<variable nom="cell_adiabatic_cst" valeur="1.4" groupe="ZD" />
	 </initialisation>
  </mesh>

  <module-main></module-main>

</case>
