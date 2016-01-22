TGT = gloci
file=$(TGT).org
tgts = glociStep2 #glociStep1
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
