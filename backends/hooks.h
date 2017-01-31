///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2017 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
#ifndef _NABLA_MIDDLEND_HOOKS_H_
#define _NABLA_MIDDLEND_HOOKS_H_


typedef struct hookForAllStruct{
  // Prefix à l'ENUMERATE_*
  char* (*prefix)(nablaJob*); // jobs, functions
  // Dump l'ENUMERATE_*
  char* (*dump)(nablaJob*); // jobs
  // Dump la référence à un item au sein d'un ENUMERATE_*
  char* (*item)(nablaJob*,const char, const char, char); // jobs
  // Dump l'ENUMERATE_*
  char* (*postfix)(nablaJob*); // jobs, functions
} hookForAll;


typedef struct hookTokenStruct{
  char* (*prefix)(nablaMain*); // variables
  // Gestion des différentes actions pour un job
  void (*svvitch)(astNode*, nablaJob*); // jobs
  // Transformation de tokens en variables selon l'ENUMERATE_*
  nablaVariable* (*variable)(astNode*, nablaMain*, nablaJob*); // jobs, functions
  // Hook pour mettre en forme les options
  void (*option)(nablaMain*,nablaOption*); // options
  void (*system)(astNode*, nablaMain*, const char, char); // variables
  void (*iteration)(nablaMain*); // functions
  void (*exit)(nablaMain*,nablaJob*);
  void (*error)(nablaMain*,nablaJob*);
  void (*time)(nablaMain*); // functions
  void (*fatal)(nablaMain*); // functions
  void (*turnBracketsToParentheses)(nablaMain*, nablaJob*, nablaVariable*, char); // jobs
  void (*isTest)(nablaMain*,nablaJob*,astNode*,int); // jobs
  char* (*postfix)(nablaMain*); // variables
  char *comment; // prefix d'un commentaire
} hookToken;


typedef struct hookGrammarStruct{
  // Hook de génération d'un kernel associé à une fonction
  void (*function)(nablaMain*, astNode*); // grammar
  // Génération d'un kernel associé à un support
  void (*job)(nablaMain*, astNode*); // grammar
  // Génération d'un kernel associé à une reduction
  void (*reduction)(nablaMain*, astNode*); // grammar
  // Should be removed: Hook pour transformer les variables à returner
  bool (*primary_expression_to_return)(nablaMain*, nablaJob*, astNode*); // jobs
  // Hook returnFromArgument for OKINA and OMP
  void (*returnFromArgument)(nablaMain*, nablaJob*); // jobs
  bool (*dfsVariable)(nablaMain*); // jobs, functions
  bool (*dfsExtra)(nablaMain*,nablaJob*,bool); // args
  char* (*dfsArgType)(nablaMain*,nablaVariable*); // args
  char* (*eoe)(nablaMain*); // jobs
  bool (*hit)(nablaMain*,bool); // jobs
  bool (*dump)(nablaMain*);
} hookGrammar;


typedef struct hookCallStruct{
  // Hooks pour rajouter au fur et à mesure qu'on les découvre
  // les fonctions appelées et les arguments
  void (*addCallNames)(nablaMain*,nablaJob*,astNode*); // functions
  void (*addArguments)(nablaMain*,nablaJob*); // functions
  // Hook pour préfixer les points d'entrée (à-la inline, par exemple)
  char* (*entryPointPrefix)(nablaMain*,nablaJob*); // arguments, jobs, functions
  // Hook pour associer aux fonctions appelées les arguments à rajouter
  void (*dfsForCalls)(nablaMain*,nablaJob*,astNode*,const char *,astNode *); // functions
  void (*addExtraParameters)(nablaMain*, nablaJob*, int*); // jobs, functions
  void (*dumpNablaParameterList)(nablaMain*, nablaJob*, astNode*,int*); // jobs
  char* (*prefixType)(nablaMain*,const char*);
  char* (*iTask)(nablaMain*,nablaJob*);
  char* (*oTask)(nablaMain*,nablaJob*);
} hookCall;


// Structure des hooks utilisé pour la gestion des directions
typedef struct hookXyzStruct{
  char* (*prefix)(void); // variables
  char* (*prevCell)(int); // variables
  char* (*nextCell)(int); // variables
  char* (*postfix)(void); // variables
} hookXyz;


// Structure des hooks que l'on va utiliser afin de générer les pragmas
typedef struct hookPragmaStruct{
  char* (*align)(void); // functions
} hookPragma;


typedef struct hookHeaderStruct{
  void (*dump)(nablaMain*); // animate
  void (*dumpWithLibs)(nablaMain*); // animate
  void (*open)(nablaMain*); // animate
  void (*enums)(nablaMain*); // animate
  void (*prefix)(nablaMain*); // animate
  void (*include)(nablaMain*); // animate
  void (*alloc)(nablaMain*); // animate
  void (*postfix)(nablaMain*); // animate
} hookHeader;


// Hooks for Sources
typedef struct hookSourceStruct{
  void (*open)(nablaMain *); // animate
  void (*include)(nablaMain *); // animate
  char* (*name)(nablaMain *); // gram
} hookSource;


// Mesh Hooks
typedef struct hookMeshStruct{
  void (*prefix)(nablaMain *); // animate
  void (*core)(nablaMain *); // animate
  void (*postfix)(nablaMain *); // animate
} hookMesh;


// Variables Hooks
typedef struct hookVarsStruct{
  void (*init)(nablaMain *); // animate
  void (*prefix)(nablaMain *); // animate
  void (*malloc)(nablaMain *); // animate
  void (*free)(nablaMain *); // animate
  char* (*idecl)(nablaMain *); // args
  char* (*odecl)(nablaMain *); // args
} hookVars;


// Hooks for Main
typedef struct hookMainStruct{
  NABLA_STATUS (*prefix)(nablaMain *); // animate
  NABLA_STATUS (*preInit)(nablaMain *); // animate
  NABLA_STATUS (*varInitKernel)(nablaMain *); // animate
  NABLA_STATUS (*varInitCall)(nablaMain *); // animate
  NABLA_STATUS (*main)(nablaMain *); // animate
  NABLA_STATUS (*postInit)(nablaMain *); // animate
  NABLA_STATUS (*postfix)(nablaMain *); // animate
} hookMain;


// ****************************************************************************
// * Backend HOOKS Structure
// ****************************************************************************
typedef struct hookStruct{
  const hookForAll *forall;
  const hookToken *token;
  const hookGrammar *grammar;
  const hookCall *call;
  const hookXyz *xyz;
  const hookPragma *pragma;
  const hookHeader *header;
  const hookSource *source;
  const hookMesh *mesh;
  const hookVars *vars;
  const hookMain *main;
} hooks;

#endif // _NABLA_MIDDLEND_HOOKS_H_
