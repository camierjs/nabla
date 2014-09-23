<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="sethi" xml:lang="en">
  <arcane>
	 <title>Sethi is Nabla's HYDRO (extracted from the code RAMSES)</title>
	 <timeloop>sethiLoop</timeloop>
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
      <variable>cell_u</variable>
      <variable>cell_v</variable>
      <variable>cell_E</variable>
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh nb-ghostlayer="1" ghostlayer-builder-version="3">
    <meshgenerator>
      <cartesian>
        <nsd>2 2</nsd> 
        <origine>0.0 0.0</origine>
        <lx nx='16'>1.0</lx>
        <ly ny='16'>1.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh> 

  <sethi> 
    <global_nx>16</global_nx>
    <global_ny>16</global_ny>
    <nstepmax>8</nstepmax>
  </sethi>
</case>
