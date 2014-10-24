Set(ALEPH_SOURCES AlephKernel.cc   AlephMatrix.cc
                  AlephVector.cc   AlephParams.cc
                  AlephTopology.cc AlephOrdering.cc
                  AlephIndexing.cc AlephFactory.cc)

info("\t${VT100_FG_YELLOW}Aleph${VT100_RESET}")

add_library(NablaOkinaAleph ${ALEPH_SOURCES})
