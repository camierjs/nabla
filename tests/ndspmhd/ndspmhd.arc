<?xml version='1.0' encoding='ISO-8859-1'?>
<case codeversion="1.0" codename="ndspmhd" xml:lang="en">
  <arcane>
	 <timeloop>ndspmhdLoop</timeloop>
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
        <group>AllCells</group>
    </output>
  </arcane-post-processing>

  <arcane-checkpoint>
    <do-dump-at-end>false</do-dump-at-end>
  </arcane-checkpoint>

  <mesh nb-ghostlayer="1" ghostlayer-builder-version="3">
    <meshgenerator>
 		 <cartesian>
			<nsd>1 1 1</nsd>
			<origine>0.0 0.0 0.0</origine>
			<lx nx="32" prx="1.0">1.0</lx>
			<ly ny="32" pry="1.0">1.0</ly>
	    </cartesian> 
    </meshgenerator>
  </mesh>
  
  <ndspmhd>   
    <option_nx>32</option_nx>
    <option_ny>32</option_ny>
    <option_nz>1</option_nz>

    <option_max_iterations>2</option_max_iterations>

    <!--############################
        nsplash ndspmhd_*.dat
        nsplash -render 12 -dev /xw ndspmhd_*.dat
        nsplash -render 12 -dev /png ndspmhd_*.dat
        /usr/local/opendev1/gcc/ffmpeg/1.2/bin/ffmpeg -i splash_%04d.png -f avi -vcodec mpeg4 -b:v 2M output.avi
        /usr/local/opendev1/gcc/ffmpeg/1.2/bin/ffmpeg -i splash_%04d.png -f avi -vcodec mjpeg -qscale 1 output.avi
      -->
    <option_psep>0.005</option_psep>

    <option_tmax>100.0</option_tmax>
    <option_tout>0.5</option_tout>
    <option_nmax>1000000</option_nmax>
    <option_nout>-1</option_nout>

    <option_gamma>1.666666666667</option_gamma>

    <option_iener>2</option_iener>
    <option_polyk>1.0</option_polyk>

    <option_icty>0</option_icty>
    <option_ndirect>1000000</option_ndirect>
    <option_maxdensits>250</option_maxdensits>

    <option_iprterm>0</option_iprterm>

    <option_iav>3</option_iav>
    <option_alphamin>1.0</option_alphamin>
    <option_alphaumin>0.0</option_alphaumin>
    <option_alphaBmin>0.0</option_alphaBmin>
    <option_beta>0.0</option_beta>

    <option_iavlimx>0</option_iavlimx>
    <option_iavlimy>0</option_iavlimy>
    <option_iavlimz>0</option_iavlimz>
    <option_avdecayconst>0.1</option_avdecayconst>

    <option_ikernav>3</option_ikernav>

    <option_ihvar>3</option_ihvar>
    <option_hfact>1.2</option_hfact>
    <option_tolh>1.0e-3</option_tolh>

    <option_idumpghost>1</option_idumpghost>

    <option_imhd>0</option_imhd>
    <option_imagforce>2</option_imagforce>

    <option_idivBzero>0</option_idivBzero>
    <option_psidecayfact>0.100</option_psidecayfact>

    <option_iresist>0</option_iresist>
    <option_etamhd>0.0</option_etamhd>

    <option_ixsph>0</option_ixsph>
    <option_xsphfac>0.0</option_xsphfac>

    <option_igravity>0</option_igravity>
    <option_hsoft>0.0</option_hsoft>

    <option_damp>0.0</option_damp>
    <option_dampr>0.0</option_dampr>
    <option_dampz>0.0</option_dampz>

    <option_ikernel>0</option_ikernel>
    <option_iexternal_force>0</option_iexternal_force>

    <option_C_cour>0.3</option_C_cour>
    <option_C_force>0.25</option_C_force>

    <option_usenumdens>false</option_usenumdens>

    <option_idust>0</option_idust>
    <option_idrag_nature>0</option_idrag_nature>
    <option_idrag_structure>0</option_idrag_structure>
    <option_Kdrag>0.0</option_Kdrag>
    <option_ismooth>0</option_ismooth>
    
    <!--physical viscosity-->
    <option_ivisc>0</option_ivisc>
    <option_shearvisc>0.0</option_shearvisc>
    <option_bulkvisc>0.0</option_bulkvisc>
  </ndspmhd>
</case>
