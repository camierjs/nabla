#########################################
# xpdf ~debian/open/gnuplot/gnuplot.pdf #
#########################################
reset
set encoding iso_8859_15
set terminal png transparent truecolor size 1024,1024 enhanced
set output "manu.png"

######################
# SIZE, SCALE & TICS #
######################
set size 1,1
set autoscale
set origin 0,0
unset xtics
unset ytics
#show border
#show margin

###########
# Palette #
###########
set palette model RGB maxcolors 512
set palette defined (1 "blue", 2 "cyan", 3 "green", 4 "yellow", 5 "red")

#############
# Color Box #
#############
set style line 1024 linetype -1
#set colorbox vertical border 1024 front #user #size .025,.5 #origin 0.9,.4
#set cbtics nomirror norotate offset -0.5
#set colorbox

################
# Bloc & File  #
# !! 0..n-1 !! #
################
bloc = 0
file = "/tmp/nabla/tests/xgnplt/lambda/1/run/seq/output.plot"

############
# Awk & Bc #
############
nbEmptyLines = system(sprintf("grep -e ^$ %s | wc --lines",file))
print nbEmptyLines." empty lines have been found"
nbBlocs = system(sprintf('echo %s/2|bc',nbEmptyLines))
print nbBlocs." blocs have been found"
allLines = system(sprintf('wc --lines %s | sed "s/ .*//" ',file))
print allLines." lines have been found"
lines = system(sprintf('echo %s/%s|bc',allLines,nbBlocs))
print "Targetting ".lines." lines/bloc"
l0 = system(sprintf("echo '%d*%s+1'|bc",bloc,lines))
lN = system(sprintf("echo %s+%s-2-1|bc",l0,lines))
print "Focussing bloc #".bloc.", lines: [".l0.",".lN."]"

#####################
# Colors & Polygone #
#####################
color(i) = system(sprintf('awk ''{if (NR==%d){print $1}}'' %s',i,file))
#print "colors: ".color(1).", ".color(2)."...".color(9)
polygone(i) = sprintf('awk ''{if (NR==%d){print $2,$3,"\n",$4,$5,"\n",$6,$7,"\n",$8,$9,"\n",$10,$11,"\n\n"}}'' %s',i,file)
#print "'".system(polygone(2))."'"

###########
# CBRANGE #
###########
stats file index bloc using 1 nooutput
set cbrange [STATS_min:STATS_max]

########
# PLOT #
########
plot for [i=l0:lN] '< '.polygone(i) using 1:2 with filledcurves palette cb color(i) notitle,\
     for [i=l0:lN] '< '.polygone(i) with lines lt -1 lw 2 notitle
