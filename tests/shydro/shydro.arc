<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="shydro" xml:lang="en">
  <arcane>
	 <title>Exemple Nabla SimpleHydro</title>
	 <timeloop>shydroLoop</timeloop>
  </arcane>

  <main>
    <do-time-history>0</do-time-history>
  </main>

  <arcane-post-processing>
    <save-init>1</save-init>
	 <output-period>0</output-period>
    <output-history-period>0</output-history-period>
    <end-execution-output>0</end-execution-output>
    <output>
      <variable>cell_density</variable>
      <variable>cell_pressure</variable>
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh>
    <meshgenerator><sod><x>64</x><y>8</y><z>8</z></sod></meshgenerator>
 	 <initialisation>
		<variable nom="cell_density" valeur="1." groupe="ZG" />
		<variable nom="cell_pressure" valeur="1." groupe="ZG" />
		<variable nom="cell_adiabatic_cst" valeur="1.4" groupe="ZG" />
		<variable nom="cell_density" valeur="0.125" groupe="ZD" />
		<variable nom="cell_pressure" valeur="0.1" groupe="ZD" />
		<variable nom="cell_adiabatic_cst" valeur="1.4" groupe="ZD" />
	 </initialisation>
  </mesh>

  <module-main></module-main>

</case>
