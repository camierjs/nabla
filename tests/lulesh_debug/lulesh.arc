<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="lulesh" xml:lang="en">
  <arcane>
	 <title>Livermore Unstructured Lagrange Explicit Shock Hydrodynamics</title>
	 <timeloop>luleshLoop</timeloop>
  </arcane>

  <main>
    <do-time-history>0</do-time-history>
  </main>

  <!-- ARCANE_HYODA_PPM_RENDER=1 -->
  <arcane-post-processing>
    <save-init>0</save-init>
	 <output-period>0</output-period>
    <output-history-period>0</output-history-period>
    <end-execution-output>0</end-execution-output>
    <output>
      <variable>cell_p</variable>
      <variable>cell_e</variable>
      <variable>cell_q</variable>
      <variable>cell_v</variable>
      <variable>cell_vdov</variable>
      <variable>cell_delv</variable>
      <variable>cell_ql</variable>
      <variable>cell_qq</variable>
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh>
    <meshgenerator>
      <sod zyx='false'> <!-- 'false' pour mimer nabla+avx||mic-->
        <x set='true' delta='0.28125'>4</x> <!-- 1.125/(nx+1) -->
        <y set='true' delta='0.375'>3</y>
        <z set='true' delta='0.375' total='true'>3</z>
      </sod>
    </meshgenerator>
  </mesh>
  
  <lulesh>
    <option_initial_energy>3.948746e+4</option_initial_energy>
    <option_stoptime>1.0e-2</option_stoptime>
    <option_max_iterations>1024</option_max_iterations>
  </lulesh>
</case>
