<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="upwind" xml:lang="en">
  <arcane>
	 <title>Upwind Test Scheme</title>
	 <timeloop>upwindLoop</timeloop>
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
      <!--variable>cell_p</variable-->
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh>
    <meshgenerator><sod zyx='true'>
        <x set='true' delta='0.1125'>4</x>
        <y set='true' delta='0.1125'>4</y>
        <z set='true' delta='0.1125' total='true'>4</z>
      </sod></meshgenerator>
  </mesh>

  <module-main></module-main>
</case>
