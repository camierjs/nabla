TGT=gram
LOG=-t -v $(TGT).log

SIMD = std
COLORS = aos
PARALLEL = omp

NABLA = /tmp/nabla/nabla


all:$(TGT).cc

$(TGT).cc: $(TGT_FILE) $(NABLA)
	$(NABLA) $(LOG) --okina $(TGT) --$(SIMD) --$(COLORS) --$(PARALLEL) -i $(TGT).n

png:
	dot -Tpng gram.dot -o gram.png
