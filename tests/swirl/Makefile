export TGT = swirl
export MESH = 1

export LOG = --tnl #-v $(TGT).log
export ADDITIONAL_NABLA_FILES = 
export ADDITIONAL_DEFINES = -DNB_PARTICLES=200000

include ./Makefile.nabla.lambda

gSwirl: Makefile gSwirl.c
	gcc -O2 -Wall -std=gnu99 gSwirl.c -o gSwirl

merge:
	/tmp/nabla/nabla --tnl --lambda gSwirlOne --std --seq -i gSwirl.1.n
	/usr/bin/dot -Tsvg  gSwirlOne.time.dot -o gSwirlOne.svg
	/tmp/nabla/nabla --tnl --lambda gSwirlTwo --std --seq -i gSwirl.2.n
	/usr/bin/dot -Tsvg  gSwirlTwo.time.dot -o gSwirlTwo.svg
	/tmp/nabla/nabla --tnl --lambda gSwirlMerged --std --seq -i gSwirl.1.n gSwirl.2.n
	/usr/bin/dot -Tsvg  gSwirlMerged.time.dot -o gSwirlMerged.svg
	cp *.svg /tmp/
	chmod 755 /tmp/*.svg


compose:
	/tmp/nabla/nabla --tnl --lambda top --std --seq -i top.n
	/usr/bin/dot -Tsvg top.time.dot -o top.svg
	/tmp/nabla/nabla --tnl --lambda deep --std --seq -i deep.n
	/usr/bin/dot -Tsvg deep.time.dot -o deep.svg
#	/tmp/nabla/nabla --tnl --lambda composed --std --seq -i composed.n
# hack composed.time.dot to add inner loop
# node_deepExit_4d40 -> node_deepLoop_4d31;
	/usr/bin/dot -Tsvg composed.time.dot -o composed.svg
	cp *.svg /tmp/ && chmod 755 /tmp/*.svg
