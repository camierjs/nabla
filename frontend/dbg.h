///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
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
#ifndef _NABLA_DBG_H_
#define _NABLA_DBG_H_


// ****************************************************************************
// * DEBUG level Definitions
// ****************************************************************************
#define	DBG_OFF		0x0000ul // Debug off
#define	DBG_EMG		0x0001ul // system is unusable
#define	DBG_ALR		0x0002ul // action must be taken immediately
#define	DBG_CTL		0x0004ul // critical conditions
#define	DBG_ERR		0x0008ul // error conditions
#define	DBG_SYS		0x000Ful
			
#define	DBG_WNG		0x0010ul // warning conditions
#define	DBG_NTC		0x0020ul // normal but significant condition
#define	DBG_LOG		0x0040ul // debug-level messages
#define	DBG_LOW		0x0080ul
#define	DBG_STD		0x00F0ul
	
#define	DBG_HASH		0x0100ul
#define	DBG_MODULO	0x0200ul
#define	DBG_CYC		0x0400ul
#define	DBG_DBG		0x0800ul
#define	DBG_DEBUG	0x0F00ul
	
#define	DBG_ALL		0xFFFFul

#define NCC_DBG_STD_LVL DBG_OFF


// ****************************************************************************
// * Forward declaration of debug functions
// ****************************************************************************
void dbgt(const bool flag, const int fd, const char *msg, ...);
NABLA_STATUS dbg(const char *str, ...);
void printfdbg(const char *str, ...);

void dbgOpenTraceFile(const char *file);
void dbgCloseTraceFile(void);

unsigned long dbgGet(void);
unsigned long dbgSet(unsigned long flg);

void busy(unsigned long flg);

void ncc_printff(const char *str, ...);

bool ncc_key_q(void);


#endif // _NABLA_DBG_H_