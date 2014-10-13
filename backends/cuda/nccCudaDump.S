        .section ".rodata"
        .global real3_h
        .global debug_h
        .global error_h
        .global extra_h
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
         
