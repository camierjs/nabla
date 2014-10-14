<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="sethi" xml:lang="en">
  <arcane>
	 <title>Sethi is Nabla's HYDRO (extracted from a real code RAMSES)</title>
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
        <nsd>4 4</nsd> 
        <origine>0.0 0.0</origine>
        <lx nx='256'>1.0</lx>
        <ly ny='256'>1.0</ly>
      </cartesian>
    </meshgenerator>
  </mesh> 

  <sethi> 
    <testcase>1</testcase>
    <uid_bubble_one>32640</uid_bubble_one>
    <uid_bubble_two>26480</uid_bubble_two>
    <nstepmax>4096</nstepmax>
  </sethi>
</case>
