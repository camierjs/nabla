TGT=gram
LOG=-t -v $(TGT).log

SIMD = std
COLORS = 
PARALLEL = omp

NABLA = /tmp/nabla/nabla


all:$(TGT).cc

$(TGT).cc: $(TGT_FILE) $(NABLA)
	$(NABLA) $(LOG) --arcane --alone gram -i $(TGT).n
#	$(NABLA) $(LOG) --okina $(TGT) --std --seq -i $(TGT).n
#	$(NABLA) $(LOG) --lambda $(TGT) -i $(TGT).n

png:
	dot -Tpng gram.dot -o gram.png
