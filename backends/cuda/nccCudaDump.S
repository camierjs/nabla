// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
        .section ".rodata"
        .global real3_h
        .global debug_h
        .global error_h
        .global extra_h
        .global cuMsh_h
real3_h:
	     .incbin "real3.h"
        .byte 0
debug_h:
	     .incbin "debug.h"
        .byte 0
error_h: 
	     .incbin "error.h"
        .byte 0
extra_h: 
	     .incbin "extra.h"
        .byte 0   
cuMsh_h: 
	     .incbin "cuMsh.h"
        .byte 0   
         
