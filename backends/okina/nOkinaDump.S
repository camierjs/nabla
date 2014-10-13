        .section ".rodata"
        .global knDbg_h
        .global knMsh_c
        .global knMth_h
        
        .global knStdReal_h
        .global knStdReal3_h
        .global knStdInteger_h
        .global knStdTernary_h
        .global knStdGather_h
        .global knStdScatter_h
        .global knStdOStream_h

        .global knSseReal_h
        .global knSseReal3_h
        .global knSseInteger_h
        .global knSseTernary_h
        .global knSseGather_h
        .global knSseScatter_h
        .global knSseOStream_h

        .global knAvxReal_h
        .global knAvxReal3_h
        .global knAvxInteger_h
        .global knAvxTernary_h
        .global knAvxGather_h
        .global knAvxScatter_h
        .global knAvxOStream_h
        
        .global knMicReal_h
        .global knMicReal3_h
        .global knMicInteger_h
        .global knMicTernary_h
        .global knMicGather_h
        .global knMicScatter_h
        .global knMicOStream_h
        

        
knDbg_h:
	     .incbin "knDbg.h"
        .byte 0
knMsh_c: 
	     .incbin "knMsh.c"
        .byte 0
knMth_h: 
	     .incbin "knMth.h"
        .byte 0


        
knStdReal_h:
	     .incbin "knStdReal.h"
        .byte 0
knStdReal3_h:
	     .incbin "knStdReal3.h"
        .byte 0
knStdInteger_h:
	     .incbin "knStdInteger.h"
        .byte 0
knStdTernary_h: 
	     .incbin "knStdTernary.h"
        .byte 0
knStdGather_h: 
	     .incbin "knStdGather.h"
        .byte 0
knStdScatter_h: 
	     .incbin "knStdScatter.h"
        .byte 0
knStdOStream_h: 
	     .incbin "knStdOStream.h"
        .byte 0


        
knSseReal_h:
	     .incbin "knSseReal.h"
        .byte 0
knSseReal3_h:
	     .incbin "knSseReal3.h"
        .byte 0
knSseInteger_h:
	     .incbin "knSseInteger.h"
        .byte 0
knSseTernary_h: 
	     .incbin "knSseTernary.h"
        .byte 0
knSseGather_h: 
	     .incbin "knSseGather.h"
        .byte 0
knSseScatter_h: 
	     .incbin "knSseScatter.h"
        .byte 0
knSseOStream_h: 
	     .incbin "knSseOStream.h"
        .byte 0


        
knAvxReal_h:
	     .incbin "knAvxReal.h"
        .byte 0
knAvxReal3_h:
	     .incbin "knAvxReal3.h"
        .byte 0
knAvxInteger_h:
	     .incbin "knAvxInteger.h"
        .byte 0
knAvxTernary_h: 
	     .incbin "knAvxTernary.h"
        .byte 0
knAvxGather_h: 
	     .incbin "knAvxGather.h"
        .byte 0
knAvxScatter_h: 
	     .incbin "knAvxScatter.h"
        .byte 0
knAvxOStream_h: 
	     .incbin "knAvxOStream.h"
        .byte 0


        
knMicReal_h:
	     .incbin "knMicReal.h"
        .byte 0
knMicReal3_h:
	     .incbin "knMicReal3.h"
        .byte 0
knMicInteger_h:
	     .incbin "knMicInteger.h"
        .byte 0
knMicTernary_h: 
	     .incbin "knMicTernary.h"
        .byte 0
knMicGather_h: 
	     .incbin "knMicGather.h"
        .byte 0
knMicScatter_h: 
	     .incbin "knMicScatter.h"
        .byte 0
knMicOStream_h: 
	     .incbin "knMicOStream.h"
        .byte 0

