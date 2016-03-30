TGT = glcNxtStp2en
#ahosla
#glcNxtStp2
#glcNxtStp2en (43,-35) (78,-0)

ZOOM = 16 # 5 is good
#QUALITY = 100

TAIL = 78 # 43 pour enlever header CeCILL
HEAD = -0 # -0 pour tout

EMACS_FLAGS = --no-site-lisp --no-init-file --no-site-file
HTM2IMG_FLAGS = --encoding utf-8 --zoom $(ZOOM) #--quality $(QUALITY)

all:
	/bin/echo -n -e > /tmp/$(TGT).n
	tail -n +$(TAIL) $(TGT).n | head -n $(HEAD) |cat -s >> /tmp/$(TGT).n
	/bin/echo -e \\n >> /tmp/$(TGT).n
	emacs $(EMACS_FLAGS) --batch -u $(shell id -un) -l htmlz.el /tmp/$(TGT).n > /tmp/$(TGT).html
# Now wkhtmltoimage html to /tmp/png file
	wkhtmltoimage $(HTM2IMG_FLAGS) /tmp/$(TGT).html /tmp/$(TGT).png
# convert trim & repage
	convert -verbose -trim +repage /tmp/$(TGT).png /tmp/$(TGT)_trimed.png
#	convert -verbose -trim +repage /tmp/$(TGT).png /tmp/$(TGT)_trimed.pdf
# convert strip & synchronize /tmp/pdf to here
#	convert -verbose -strip -synchronize /tmp/$(TGT)_trimed.pdf $(TGT).pdf
# Cleanup
# 	\rm -rf /tmp/$(TGT).png /tmp/$(TGT)*.pdf
cln:
	\rm -rf *.html


#####################
# Old method: org+sed
#####################
file=$(TGT).n
title= ∇-Glace sans @
org_header = \\\#+title: $(title)\\n\\\#+author: Camier Jean-Sylvain\\n\\\#+email: Jean-Sylvain.Camier@cea.fr\\n\\\#+date:\\n\\\#+LANGUAGE: en\\\n\\\#+OPTIONS: H:1 num:nil toc:nil todo:nil pri:nil tags:nil date:t\\n\\\#+OPTIONS: author:t timestamp:nil\\n\\\#+OPTIONS:\\\\n:nil ^:t \<:t \*:t \|:t\\n\\\#+STARTUP: latex indent nolatexpreview\\n\\\#+LaTeX_CLASS: article\\n\\\#+LaTeX_HEADER: \\\usepackage{hyperref}\\n

org:
	@/bin/echo -e $(org_header) > $(file)
	for tgt in $(tgts);do\
		/bin/echo -e \* $$tgt >> $(file);\
		/bin/echo -e \#+begin_src nabla >> $(file);\
		tail -n +43 $$tgt.n |cat -s >> $(file);\
		/bin/echo -e \#+end_src >> $(file);\
		/bin/echo -e \\n >> $(file);\
	done

sed:
# Remove h1 title
	sed -i 's/<h1 class="title">.*<\/h1>//g' $(TGT).html
# Remove preamble
	sed -i '/<div id="preamble"/,/<\/div>/d' $(TGT).html
	sed -i '/<h2 id="orgheadline1">.*<\/h2>/d' $(TGT).html
	sed -i 's/Validate//g;s/border/#border/g;s/box-shadow/#box-shadow/g' $(TGT).html
	sed -i '/<p class="author">Author.*<\/p>/d' $(TGT).html
html:
	emacs -u $(shell id -un) $(file) --batch -f org-html-export-to-html
