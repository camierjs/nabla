///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2016 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
// This NABLA port is an implementation of the NDSPMHD software
// Implementation of the exact Riemann solver given in Toro (1992)
//
// Solves for the post-shock pressure (pr) and velocity (vstar)
// given the initial left and right states
//
// Does not matter if high P / high rho is on left or right
//
// Daniel Price, Institute of Astronomy, Cambridge, UK, 2004
// dprice@ast.cam.ac.uk
void riemannsolver(const Real gamma,
                   const Real p_L, const Real p_R,
                   const Real v_L, const Real v_R,
                   const Real rho_L, const Real rho_R,
                   Real* pr, Real* vstar){
  const Real tol = 1.5e-2;
  const Integer maxits = 30;
  Integer its;
  Real c_L, c_R;
  Real prnew, f_L, f_R, dfdp_L, dfdp_R, f, df, dp;
  Real power, denom, cs2;

//--use isothermal solver if appropriate
  c_L = sqrt(gamma*p_L/rho_L);
  c_R = sqrt(gamma*p_R/rho_R);
  if (gamma < 1.0001){
    cs2 = p_L/rho_L;
    get_pstar_isothermal(cs2,v_L,v_R,rho_L,rho_R,pr,vstar);
    return;
  }
  
//--get an initial starting estimate of intermediate pressure
//  this one is from Toro(1992) - gives basically the right answer
//  for pressure jumps below about 4
  power = (gamma-1.)/(2.*gamma);
  denom = c_L/p_L**power + c_R/p_R**power;
  prnew = ((c_L + c_R + (v_L - v_R)*0.5*(gamma-1.))/denom)**(1./power);
  pr = p_L;
  its = 0;

  ////print*,'initial guess = ',prnew
  while (abs(prnew-pr)>tol && its<maxits){
    its = its + 1;
    pr = prnew;
    // evaluate the function and its derivatives
    f_and_df(pr,p_L,c_L,gamma,f_L,dfdp_L);
    f_and_df(pr,p_R,c_R,gamma,f_R,dfdp_R);
    // then get new estimate of pr
    f = f_L + f_R + (v_R - v_L);
    df = dfdp_L + dfdp_R;
    // Newton-Raphson iterations
    dp = -f/df;
    prnew = pr + dp;
  }

  if (its==maxits) info()<<"WARNING: its not converged in riemann solver";
  if (prnew<=0.0){
    info()<<"ERROR: pr < 0 in riemann solver";
    info()<<"its = "<<its<<"p_L<< p_R = "<<p_L<<p_R<<" v_L<< v_R = "<<v_L<<v_R<<" p* = "<<prnew<<"v = "<<vstar<<v_R + f_R;
  }
  *pr = prnew;
  *vstar = v_L - f_L;
//  if (its.gt.0) then
//     print*,'its = ',its,'p_L, p_R = ',p_L,p_R,' v_L, v_R = ',v_L,v_R,' p* = ',prnew,'v = ',vstar,v_R + f_R
//  endif
}



//--pressure function
//  H is pstar/p_L or pstar/p_R
void f_and_df(const Real prstar,const Real pr,const Real cs,const Real gam,
              Real* fp, Real* dfdp){
  Real H,term, power, gamm1, denom;

  H = prstar/pr;
  gamm1 = gam - 1.0;
  
  if (H>1.0) {  // shock
    denom = gam*((gam+1.)*H + gamm1);
    term = sqrt(2./denom);
    fp = (H - 1.)*cs*term;
    dfdp = cs*term/pr + (H - 1.)*cs/term*(-1./denom**2)*gam*(gam+1.)/pr;
  }else{               // rarefaction
    power = gamm1/(2.*gam);
    fp = (H**power - 1.)*(2.*cs/gamm1);
    dfdp = 2.*cs/gamm1*power*H**(power-1.)/pr;
  }
}


//-------------------------------------------------------------
// Non-iterative isothermal Riemann solver 
// from Balsara (1994), ApJ 420, 197-212
//
// See also Cha & Whitworth (2003), MNRAS 340, 73-90
//-------------------------------------------------------------
void get_pstar_isothermal(const Real cs2,
                          const Real v_L,const Real v_R,
                          const Real rho_L,const Real rho_R,
                          Real* pstar, Real* vstar){
  Real sqrtrho_L, sqrtrho_R, X, vdiff, determinant, vstar2;

  sqrtrho_L = sqrt(rho_L);
  sqrtrho_R = sqrt(rho_R);
  
  X = sqrtrho_L*sqrtrho_R/(sqrtrho_L + sqrtrho_R);
  vdiff = v_L - v_R;
  determinant = (X*vdiff)**2 + 4.*cs2*X*(sqrtrho_L + sqrtrho_R);
  
  pstar = 0.25*(X*vdiff + sqrt(determinant))**2  ;
  vstar = v_L - (pstar - cs2*rho_L)/(sqrt(pstar*rho_L));
  vstar2 = v_R + (pstar - cs2*rho_R)/(sqrt(pstar*rho_R));
  if (abs(vstar2-vstar)>1.e-5)
    info()<<"error: vstar = "<<vstar<<", "<<vstar2;
  //print*,' pstar = ',pstar,' v_R,v_L = ',v_L,v_R,cs2,' vstar = ',vstar
}
