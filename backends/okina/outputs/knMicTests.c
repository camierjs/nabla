#ifdef __MIC__

// *****************************************************************************
// * Flush des données pour les gathers/scatters
// *****************************************************************************
__attribute__((unused)) static void micTestFlush(Real *data){
  data[0]=set(1.+1./9., 1.+2./9., 1.+3./9., 1.+4./9., 1.+5./9., 1.+6./9., 1.+7./9., 1.+8./9.);
  data[1]=set(2.+1./9., 2.+2./9., 2.+3./9., 2.+4./9., 2.+5./9., 2.+6./9., 2.+7./9., 2.+8./9.);
  data[2]=set(3.+1./9., 3.+2./9., 3.+3./9., 3.+4./9., 3.+5./9., 3.+6./9., 3.+7./9., 3.+8./9.);
  data[3]=set(4.+1./9., 4.+2./9., 4.+3./9., 4.+4./9., 4.+5./9., 4.+6./9., 4.+7./9., 4.+8./9.);
  data[4]=set(5.+1./9., 5.+2./9., 5.+3./9., 5.+4./9., 5.+5./9., 5.+6./9., 5.+7./9., 5.+8./9.);
  data[5]=set(6.+1./9., 6.+2./9., 6.+3./9., 6.+4./9., 6.+5./9., 6.+6./9., 6.+7./9., 6.+8./9.);
  data[6]=set(7.+1./9., 7.+2./9., 7.+3./9., 7.+4./9., 7.+5./9., 7.+6./9., 7.+7./9., 7.+8./9.);
  data[7]=set(8.+1./9., 8.+2./9., 8.+3./9., 8.+4./9., 8.+5./9., 8.+6./9., 8.+7./9., 8.+8./9.);
}


// *****************************************************************************
// * Dump des données pour les gathers/scatters
// *****************************************************************************
__attribute__((unused)) static void micTestDump(Real *data){
  dbgReal(DBG_ALL,data[0]);printf("\n");
  dbgReal(DBG_ALL,data[1]);printf("\n");
  dbgReal(DBG_ALL,data[2]);printf("\n");
  dbgReal(DBG_ALL,data[3]);printf("\n");
  dbgReal(DBG_ALL,data[4]);printf("\n");
  dbgReal(DBG_ALL,data[5]);printf("\n");
  dbgReal(DBG_ALL,data[6]);printf("\n");
  dbgReal(DBG_ALL,data[7]);printf("\n");
}


// *****************************************************************************
// * Test MIC des gathers/scatters sur des Reals
// *****************************************************************************
__attribute__((unused)) static void micTestReal(void){
  Real data[8];
  Real gthr;
  Real scttr=set(0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008);
  
  printf("WARP_SIZE=%%d\nData for Real test are:\n",WARP_SIZE);
  micTestFlush(data);
  micTestDump(data);
  printf("data[0~7]=\n");
  printf("On gatherk pour avoir la diagonale:\n\t\t\t");
  gatherk(0*WARP_SIZE+0,
          1*WARP_SIZE+1,
          2*WARP_SIZE+2,
          3*WARP_SIZE+3,
          4*WARP_SIZE+4,
          5*WARP_SIZE+5,
          6*WARP_SIZE+6,
          7*WARP_SIZE+7,data,gthr);dbgReal(DBG_ALL,gthr);printf("\n");
  
  printf("Now scattering:");
  scatter(0*WARP_SIZE+0,
          1*WARP_SIZE+1,
          2*WARP_SIZE+2,
          3*WARP_SIZE+3,
          4*WARP_SIZE+4,
          5*WARP_SIZE+5,
          6*WARP_SIZE+6,
          7*WARP_SIZE+7,data,scttr);
  printf("\n");dbgReal(DBG_ALL,data[0]);
  printf("\n");dbgReal(DBG_ALL,data[1]);
  printf("\n");dbgReal(DBG_ALL,data[2]);
  printf("\n");dbgReal(DBG_ALL,data[3]);
  printf("\n");dbgReal(DBG_ALL,data[4]);
  printf("\n");dbgReal(DBG_ALL,data[5]);
  printf("\n");dbgReal(DBG_ALL,data[6]);
  printf("\n");dbgReal(DBG_ALL,data[7]);
}


// *****************************************************************************
// * Test MIC des gathers/scatters sur des Reals
// *****************************************************************************
__attribute__((unused)) static void micTestReal3(void){
  Real3 data3[8];
  Real3 gthr3;
  Real3 scttr3=Real3(set(1.00001,1.00002,1.00003,1.00004,1.00005,1.00006,1.00007,1.00008),
                     set(2.00001,2.00002,2.00003,2.00004,1.00005,2.00006,2.00007,2.00008),
                     set(3.00001,3.00002,3.00003,3.00004,3.00005,3.00006,3.00007,3.00008));
  printf("\n\nmicTestReal3\nX:\n");
  data3[0].x=set(10.+1./9., 10.+2./9., 10.+3./9., 10.+4./9., 10.+5./9., 10.+6./9., 10.+7./9., 10.+8./9.);dbgReal(DBG_ALL,data3[0].x);printf("\n");
  data3[1].x=set(20.+1./9., 20.+2./9., 20.+3./9., 20.+4./9., 20.+5./9., 20.+6./9., 20.+7./9., 20.+8./9.);dbgReal(DBG_ALL,data3[1].x);printf("\n");
  data3[2].x=set(30.+1./9., 30.+2./9., 30.+3./9., 30.+4./9., 30.+5./9., 30.+6./9., 30.+7./9., 30.+8./9.);dbgReal(DBG_ALL,data3[2].x);printf("\n");
  data3[3].x=set(40.+1./9., 40.+2./9., 40.+3./9., 40.+4./9., 40.+5./9., 30.+6./9., 40.+7./9., 40.+8./9.);dbgReal(DBG_ALL,data3[3].x);printf("\n");
  data3[4].x=set(50.+1./9., 50.+2./9., 50.+3./9., 50.+4./9., 50.+5./9., 40.+6./9., 50.+7./9., 50.+8./9.);dbgReal(DBG_ALL,data3[4].x);printf("\n");
  data3[5].x=set(60.+1./9., 60.+2./9., 60.+3./9., 60.+4./9., 60.+5./9., 50.+6./9., 60.+7./9., 60.+8./9.);dbgReal(DBG_ALL,data3[5].x);printf("\n");
  data3[6].x=set(70.+1./9., 70.+2./9., 70.+3./9., 70.+4./9., 70.+5./9., 60.+6./9., 70.+7./9., 70.+8./9.);dbgReal(DBG_ALL,data3[6].x);printf("\n");
  data3[7].x=set(80.+1./9., 80.+2./9., 80.+3./9., 80.+4./9., 80.+5./9., 70.+6./9., 80.+7./9., 80.+8./9.);dbgReal(DBG_ALL,data3[7].x);printf("\n");
  printf("Y:\n");
  data3[0].y=set(11.+1./9., 11.+2./9., 11.+3./9., 11.+4./9., 11.+5./9., 11.+6./9., 11.+7./9., 11.+8./9.);dbgReal(DBG_ALL,data3[0].y);printf("\n");
  data3[1].y=set(21.+1./9., 21.+2./9., 21.+3./9., 21.+4./9., 21.+5./9., 21.+6./9., 21.+7./9., 21.+8./9.);dbgReal(DBG_ALL,data3[1].y);printf("\n");
  data3[2].y=set(31.+1./9., 31.+2./9., 31.+3./9., 31.+4./9., 31.+5./9., 31.+6./9., 31.+7./9., 31.+8./9.);dbgReal(DBG_ALL,data3[2].y);printf("\n");
  data3[3].y=set(41.+1./9., 41.+2./9., 41.+3./9., 41.+4./9., 41.+5./9., 41.+6./9., 41.+7./9., 41.+8./9.);dbgReal(DBG_ALL,data3[3].y);printf("\n");
  data3[4].y=set(51.+1./9., 51.+2./9., 51.+3./9., 51.+4./9., 51.+5./9., 51.+6./9., 51.+7./9., 51.+8./9.);dbgReal(DBG_ALL,data3[4].y);printf("\n");
  data3[5].y=set(61.+1./9., 61.+2./9., 61.+3./9., 61.+4./9., 61.+5./9., 61.+6./9., 61.+7./9., 61.+8./9.);dbgReal(DBG_ALL,data3[5].y);printf("\n");
  data3[6].y=set(71.+1./9., 71.+2./9., 71.+3./9., 71.+4./9., 71.+5./9., 71.+6./9., 71.+7./9., 71.+8./9.);dbgReal(DBG_ALL,data3[6].y);printf("\n");
  data3[7].y=set(81.+1./9., 81.+2./9., 81.+3./9., 81.+4./9., 81.+5./9., 81.+6./9., 81.+7./9., 81.+8./9.);dbgReal(DBG_ALL,data3[7].y);printf("\n");
  printf("Z:\n");
  data3[0].z=set(12.+1./9., 12.+2./9., 12.+3./9., 12.+4./9., 12.+5./9., 12.+6./9., 12.+7./9., 12.+8./9.);dbgReal(DBG_ALL,data3[0].z);printf("\n");
  data3[1].z=set(22.+1./9., 22.+2./9., 22.+3./9., 22.+4./9., 22.+5./9., 22.+6./9., 22.+7./9., 22.+8./9.);dbgReal(DBG_ALL,data3[1].z);printf("\n");
  data3[2].z=set(32.+1./9., 32.+2./9., 32.+3./9., 32.+4./9., 32.+5./9., 32.+6./9., 32.+7./9., 32.+8./9.);dbgReal(DBG_ALL,data3[2].z);printf("\n");
  data3[3].z=set(42.+1./9., 42.+2./9., 42.+3./9., 42.+4./9., 42.+5./9., 42.+6./9., 42.+7./9., 42.+8./9.);dbgReal(DBG_ALL,data3[3].z);printf("\n");
  data3[4].z=set(52.+1./9., 52.+2./9., 52.+3./9., 52.+4./9., 52.+5./9., 52.+6./9., 52.+7./9., 52.+8./9.);dbgReal(DBG_ALL,data3[4].z);printf("\n");
  data3[5].z=set(62.+1./9., 62.+2./9., 62.+3./9., 62.+4./9., 62.+5./9., 62.+6./9., 62.+7./9., 62.+8./9.);dbgReal(DBG_ALL,data3[5].z);printf("\n");
  data3[6].z=set(72.+1./9., 72.+2./9., 72.+3./9., 72.+4./9., 72.+5./9., 72.+6./9., 72.+7./9., 72.+8./9.);dbgReal(DBG_ALL,data3[6].z);printf("\n");
  data3[7].z=set(82.+1./9., 82.+2./9., 82.+3./9., 82.+4./9., 82.+5./9., 82.+6./9., 82.+7./9., 82.+8./9.);dbgReal(DBG_ALL,data3[7].z);printf("\n");
  printf("En memoire:\n");
  dbgReal(DBG_ALL,data3[0].x);printf("\n");
  dbgReal(DBG_ALL,data3[0].y);printf("\n");
  dbgReal(DBG_ALL,data3[0].z);printf("\n");
  dbgReal(DBG_ALL,data3[1].x);printf("\n");
  dbgReal(DBG_ALL,data3[1].y);printf("\n");
  dbgReal(DBG_ALL,data3[1].z);printf("\n");
  dbgReal(DBG_ALL,data3[2].x);printf("\n");
  dbgReal(DBG_ALL,data3[2].y);printf("\n");
  dbgReal(DBG_ALL,data3[2].z);printf("\n");
  dbgReal(DBG_ALL,data3[3].x);printf("\n");
  dbgReal(DBG_ALL,data3[3].y);printf("\n");
  dbgReal(DBG_ALL,data3[3].z);printf("\n");
  dbgReal(DBG_ALL,data3[4].x);printf("\n");
  dbgReal(DBG_ALL,data3[4].y);printf("\n");
  dbgReal(DBG_ALL,data3[4].z);printf("\n");
  dbgReal(DBG_ALL,data3[5].x);printf("\n");
  dbgReal(DBG_ALL,data3[5].y);printf("\n");
  dbgReal(DBG_ALL,data3[5].z);printf("\n");
  dbgReal(DBG_ALL,data3[6].x);printf("\n");
  dbgReal(DBG_ALL,data3[6].y);printf("\n");
  dbgReal(DBG_ALL,data3[6].z);printf("\n");
  dbgReal(DBG_ALL,data3[7].x);printf("\n");
  dbgReal(DBG_ALL,data3[7].y);printf("\n");
  dbgReal(DBG_ALL,data3[7].z);printf("\n");
  printf("\nOn veut la diagonale:\n");
  gather3k(0*3*WARP_SIZE+0,
           1*3*WARP_SIZE+1,
           2*3*WARP_SIZE+2,
           3*3*WARP_SIZE+3,
           4*3*WARP_SIZE+4,
           5*3*WARP_SIZE+5,
           6*3*WARP_SIZE+6,
           7*3*WARP_SIZE+7,data3,gthr3);dbgReal3(DBG_ALL,gthr3);printf("\n");
  printf("\n");

  printf("Now scattering:\n");
  scatter3(0*3*WARP_SIZE+0,
           1*3*WARP_SIZE+1,
           2*3*WARP_SIZE+2,
           3*3*WARP_SIZE+3,
           4*3*WARP_SIZE+4,
           5*3*WARP_SIZE+5,
           6*3*WARP_SIZE+6,
           7*3*WARP_SIZE+7,data3,scttr3);
  dbgReal(DBG_ALL,data3[0].x);printf("\n");
  dbgReal(DBG_ALL,data3[0].y);printf("\n");
  dbgReal(DBG_ALL,data3[0].z);printf("\n");
  dbgReal(DBG_ALL,data3[1].x);printf("\n");
  dbgReal(DBG_ALL,data3[1].y);printf("\n");
  dbgReal(DBG_ALL,data3[1].z);printf("\n");
  dbgReal(DBG_ALL,data3[2].x);printf("\n");
  dbgReal(DBG_ALL,data3[2].y);printf("\n");
  dbgReal(DBG_ALL,data3[2].z);printf("\n");
  dbgReal(DBG_ALL,data3[3].x);printf("\n");
  dbgReal(DBG_ALL,data3[3].y);printf("\n");
  dbgReal(DBG_ALL,data3[3].z);printf("\n");
  dbgReal(DBG_ALL,data3[4].x);printf("\n");
  dbgReal(DBG_ALL,data3[4].y);printf("\n");
  dbgReal(DBG_ALL,data3[4].z);printf("\n");
  dbgReal(DBG_ALL,data3[5].x);printf("\n");
  dbgReal(DBG_ALL,data3[5].y);printf("\n");
  dbgReal(DBG_ALL,data3[5].z);printf("\n");
  dbgReal(DBG_ALL,data3[6].x);printf("\n");
  dbgReal(DBG_ALL,data3[6].y);printf("\n");
  dbgReal(DBG_ALL,data3[6].z);printf("\n");
  dbgReal(DBG_ALL,data3[7].x);printf("\n");
  dbgReal(DBG_ALL,data3[7].y);printf("\n");
  dbgReal(DBG_ALL,data3[7].z);printf("\n");
  
  //Flush
  data3[0].x=set(10.+1./9., 10.+2./9., 10.+3./9., 10.+4./9., 10.+5./9., 10.+6./9., 10.+7./9., 10.+8./9.);
  data3[1].x=set(20.+1./9., 20.+2./9., 20.+3./9., 20.+4./9., 20.+5./9., 20.+6./9., 20.+7./9., 20.+8./9.);
  data3[2].x=set(30.+1./9., 30.+2./9., 30.+3./9., 30.+4./9., 30.+5./9., 30.+6./9., 30.+7./9., 30.+8./9.);
  data3[3].x=set(40.+1./9., 40.+2./9., 40.+3./9., 40.+4./9., 40.+5./9., 40.+6./9., 40.+7./9., 40.+8./9.);
 
  data3[0].y=set(11.+1./9., 11.+2./9., 11.+3./9., 11.+4./9., 11.+5./9., 11.+6./9., 11.+7./9., 11.+8./9.);
  data3[1].y=set(21.+1./9., 21.+2./9., 21.+3./9., 21.+4./9., 21.+5./9., 21.+6./9., 21.+7./9., 21.+8./9.);
  data3[2].y=set(31.+1./9., 31.+2./9., 31.+3./9., 31.+4./9., 31.+5./9., 31.+6./9., 31.+7./9., 31.+8./9.);
  data3[3].y=set(41.+1./9., 41.+2./9., 41.+3./9., 41.+4./9., 41.+5./9., 41.+6./9., 41.+7./9., 41.+8./9.);
  
  data3[0].z=set(12.+1./9., 12.+2./9., 12.+3./9., 12.+4./9., 12.+5./9., 12.+6./9., 12.+7./9., 12.+8./9.);
  data3[1].z=set(22.+1./9., 22.+2./9., 22.+3./9., 22.+4./9., 22.+5./9., 22.+6./9., 22.+7./9., 22.+8./9.);
  data3[2].z=set(32.+1./9., 32.+2./9., 32.+3./9., 32.+4./9., 32.+5./9., 32.+6./9., 32.+7./9., 32.+8./9.);
  data3[3].z=set(42.+1./9., 42.+2./9., 42.+3./9., 42.+4./9., 42.+5./9., 42.+6./9., 42.+7./9., 42.+8./9.);

  printf("\nNow K-scattering:\n");
  scatter3(0+0,4+1,8+2,12+3,16+4,20+5,24+6,28+7,data3,scttr3);
  dbgReal(DBG_ALL,data3[0].x);printf("\n");
  dbgReal(DBG_ALL,data3[0].y);printf("\n");
  dbgReal(DBG_ALL,data3[0].z);printf("\n");
  
  dbgReal(DBG_ALL,data3[1].x);printf("\n");
  dbgReal(DBG_ALL,data3[1].y);printf("\n");
  dbgReal(DBG_ALL,data3[1].z);printf("\n");
  
  dbgReal(DBG_ALL,data3[2].x);printf("\n");
  dbgReal(DBG_ALL,data3[2].y);printf("\n");
  dbgReal(DBG_ALL,data3[2].z);printf("\n");
  
  dbgReal(DBG_ALL,data3[3].x);printf("\n");
  dbgReal(DBG_ALL,data3[3].y);printf("\n");
  dbgReal(DBG_ALL,data3[3].z);printf("\n");
  
  dbgReal(DBG_ALL,data3[4].x);printf("\n");
  dbgReal(DBG_ALL,data3[4].y);printf("\n");
  dbgReal(DBG_ALL,data3[4].z);printf("\n");
  
  dbgReal(DBG_ALL,data3[5].x);printf("\n");
  dbgReal(DBG_ALL,data3[5].y);printf("\n");
  dbgReal(DBG_ALL,data3[5].z);printf("\n");
  
  dbgReal(DBG_ALL,data3[6].x);printf("\n");
  dbgReal(DBG_ALL,data3[6].y);printf("\n");
  dbgReal(DBG_ALL,data3[6].z);printf("\n");
  
  dbgReal(DBG_ALL,data3[7].x);printf("\n");
  dbgReal(DBG_ALL,data3[7].y);printf("\n");
  dbgReal(DBG_ALL,data3[7].z);printf("\n");
  

  printf("\n");
  //exit(0);
}

#endif
