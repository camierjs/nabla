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
        .global knAvx2Gather_h
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
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knDbg.h"
        .byte 0
knMsh_c: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMsh.c"
        .byte 0
knMth_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMth.h"
        .byte 0


        
knStdReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knStdReal.h"
        .byte 0
knStdReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knStdReal3.h"
        .byte 0
knStdInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knStdInteger.h"
        .byte 0
knStdTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knStdTernary.h"
        .byte 0
knStdGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knStdGather.h"
        .byte 0
knStdScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knStdScatter.h"
        .byte 0
knStdOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knStdOStream.h"
        .byte 0


        
knSseReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knSseReal.h"
        .byte 0
knSseReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knSseReal3.h"
        .byte 0
knSseInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knSseInteger.h"
        .byte 0
knSseTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knSseTernary.h"
        .byte 0
knSseGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knSseGather.h"
        .byte 0
knSseScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knSseScatter.h"
        .byte 0
knSseOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knSseOStream.h"
        .byte 0


        
knAvxReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvxReal.h"
        .byte 0
knAvxReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvxReal3.h"
        .byte 0
knAvxInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvxInteger.h"
        .byte 0
knAvxTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvxTernary.h"
        .byte 0
knAvxGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvxGather.h"
        .byte 0
knAvx2Gather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvx2Gather.h"
        .byte 0
knAvxScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvxScatter.h"
        .byte 0
knAvxOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knAvxOStream.h"
        .byte 0


        
knMicReal_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMicReal.h"
        .byte 0
knMicReal3_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMicReal3.h"
        .byte 0
knMicInteger_h:
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMicInteger.h"
        .byte 0
knMicTernary_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMicTernary.h"
        .byte 0
knMicGather_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMicGather.h"
        .byte 0
knMicScatter_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMicScatter.h"
        .byte 0
knMicOStream_h: 
	     .incbin "${CMAKE_CURRENT_SOURCE_DIR}/outputs/knMicOStream.h"
        .byte 0

