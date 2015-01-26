;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; NABLA - a Numerical Analysis Based LAnguage                               ;;
;;                                                                           ;;
;; Copyright (C) 2014~2015 CEA/DAM/DIF                                       ;;
;; IDDN.FR.001.520002.000.S.P.2014.000.10500                                 ;;
;;                                                                           ;;
;; Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          ;;
;;                                                                           ;;
;; This software is a computer program whose purpose is to translate         ;;
;; numerical-analysis specific sources and to generate optimized code        ;;
;; for different targets and architectures.                                  ;;
;;                                                                           ;;
;; This software is governed by the CeCILL license under French law and      ;;
;; abiding by the rules of distribution of free software. You can  use,      ;;
;; modify and/or redistribute the software under the terms of the CeCILL     ;;
;; license as circulated by CEA, CNRS and INRIA at the following URL:        ;;
;; "http:;;www.cecill.info".                                                 ;;
;;                                                                           ;;
;; The CeCILL is a free software license, explicitly compatible with         ;;
;; the GNU GPL.                                                              ;;
;;                                                                           ;;
;; As a counterpart to the access to the source code and rights to copy,     ;;
;; modify and redistribute granted by the license, users are provided only   ;;
;; with a limited warranty and the software's author, the holder of the      ;;
;; economic rights, and the successive licensors have only limited liability.;;
;;                                                                           ;;
;; In this respect, the user's attention is drawn to the risks associated    ;;
;; with loading, using, modifying and/or developing or reproducing the       ;;
;; software by the user in light of its specific status of free software,    ;;
;; that may mean that it is complicated to manipulate, and that also         ;;
;; therefore means that it is reserved for developers and experienced        ;;
;; professionals having in-depth computer knowledge. Users are therefore     ;;
;; encouraged to load and test the software's suitability as regards their   ;;
;; requirements in conditions enabling the security of their systems and/or  ;;
;; data to be ensured and, more generally, to use and operate it in the      ;;
;; same conditions as regards security.                                      ;;
;;                                                                           ;;
;; The fact that you are presently reading this means that you have had      ;;
;; knowledge of the CeCILL license and that you accept its terms.            ;;
;;                                                                           ;;
;; See the LICENSE file for details.                                         ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; nabla-mode.el --- major mode for editing Nabla files.
(defconst nabla-version "130604" "Nabla Mode version number")
(defconst nabla-time-stamp "2013-06-04" "Nabla Mode time stamp")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defconst nabla-keywords
  '("inline" "restrict" "aligned" "const" "return" "register" "volatile"
    "if" "else" "do" "while" "continue" "break" "for"
    "foreach"
    "all" "own" "inner" "outer"
    "in" "out" "inout" "with")
  "List of Nabla keywords regexps")

(defconst nabla-maths
  '("sqrt" "norm" "dot" "cross" "min" "max" 
    "mathlink" "Prime"
    ) "List of Nabla mathematical tools")

(defconst nabla-builtins
  '("exit" "checkpoint" "coord" 
    "uid" "lid" "sid" "this" "nbNode" "iNode" "fatal"
    "backCell" "backCellUid" "frontCell" "frontCellUid" 
    "nextCell" "prevCell"
    "nextNextCell" "prevPrevCell"
    "deltat" "nbCell"
    "iteration" "time" "exit"
    "precise" "remain"
    ) "List of Nabla builtins")

(defconst nabla-libraries
  '("mpi" 
    "mail" 
    "cartesian" "xyz" 
    "mat" "env" "matenv" 
    "gmp"
    "dft" 
    "slurm" 
    "mathematica"
    "aleph" "rhs" "lhs" "matrix" "solve" "setValue" "getValue" "addValue" "newValue" "reset"
    ) "List of Nabla libraries")

(defconst nabla-items
  '("global"
    "cell" "cells" 
    "node" "nodes"
    "edge" "edges"
    "face" "faces" 
    "particle" "particles"
    "material" "materials" 
    "environment" "environments" 
    "options" "option" "enum"
    ) "List of Nabla items")

(defconst nabla-types
  '("bool" "Bool" "Uid"
    "void" "int" "char" "Integer" "Int32" "Int64"
    "float" "double" "Real" "Real3" "Real3x3" 
    "cell" "cells" "Cell"
    "node" "nodes" "Node"
    "Particle"
    "edge" "edges"
    "face" "faces" "Face"
    ) "List of Nabla types")

(defconst nabla-preprocessors
  '() "List of Nabla preprocessors")

(defconst nabla-warnings
  '("warning" "error")
  "List of Nabla warnings")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defconst nabla-keywords-regexp
  (concat "\\<\\(" (regexp-opt nabla-keywords) "\\)\\>")
  "Regexp for Nabla keywords.")
(defconst nabla-maths-regexp
  (concat "\\<\\(" (regexp-opt nabla-maths) "\\)\\>")
  "Regexp for Nabla maths.")
(defconst nabla-builtins-regexp
  (concat "\\<\\(" (regexp-opt nabla-builtins) "\\)\\>")
  "Regexp for Nabla builtins.")
(defconst nabla-libraries-regexp
  (concat "\\<\\(" (regexp-opt nabla-libraries) "\\)\\>")
  "Regexp for Nabla builtins.")
(defconst nabla-items-regexp
  (concat "\\<\\(" (regexp-opt nabla-items) "\\)\\>")
  "Regexp for Nabla items.")
(defconst nabla-types-regexp
  (concat "\\<\\(" (regexp-opt nabla-types) "\\)\\>")
  "Regexp for Nabla types.")
(defconst nabla-preprocessors-regexp
  (concat "\\<\\(" (regexp-opt nabla-preprocessors) "\\)\\>")
  "Regexp for Nabla preprocessors.")
(defconst nabla-warnings-regexp
  (concat "\\<\\(" (regexp-opt nabla-warnings) "\\)\\>")
  "Regexp for Nabla warnings.")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defvar nabla-font-inout-face 'nabla-font-inout-face)

(defvar nabla-font-at-face 'nabla-font-at-face)

(defvar nabla-font-comment-face 'nabla-font-comment-face)

(defface nabla-font-inout-face
  '((((class color) (background light)) (:foreground "Blue" :bold t))
    (((class color) (background dark)) (:foreground "RoyalBlue" :bold t))
    (t (:italic t :bold t)))
  "Font lock mode face used to highlight @ definitions."
  :group 'font-lock-highlighting-faces)

(defface nabla-font-at-face
  '((((class color) (background light)) (:foreground "Blue" :bold nil))
    (((class color) (background dark)) (:foreground "RoyalBlue" :bold nil))
    (t (:italic t :bold t)))
  "Font lock mode face used to highlight @ definitions."
  :group 'font-lock-highlighting-faces)

(defface nabla-font-comment-face
  '((((class color) (background light)) (:foreground "Salmon"))
    (((class color) (background dark)) (:foreground "Salmon"))
    (t (:italic t :bold t)))
  "Font lock mode face used for comments."
  :group 'font-lock-highlighting-faces)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Keywords
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; font-lock-comment-face
;;; font-lock-comment-delimiter-face
;;; font-lock-doc-face
;;; font-lock-string-face
;;; font-lock-keyword-face
;;; font-lock-builtin-face
;;; font-lock-function-name-face
;;; font-lock-variable-name-face
;;; font-lock-type-face
;;; font-lock-constant-face
;;; font-lock-preprocessor-face
;;; font-lock-negation-char-face
;;; font-lock-warning-face
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defconst nabla-font-lock-keywords
  (list
   (list nabla-preprocessors-regexp 1 'font-lock-preprocessor-face)
   (list nabla-keywords-regexp 1 'font-lock-keyword-face)
   (list nabla-maths-regexp 1 'font-lock-builtin-face)
   (list nabla-builtins-regexp 1 'font-lock-builtin-face)
   (list nabla-libraries-regexp 1 'font-lock-builtin-face)
   (list nabla-items-regexp 1 'font-lock-type-face)
   (list nabla-types-regexp 1 'font-lock-type-face)
   (list nabla-warnings-regexp 1 'font-lock-warning-face)
   ;; highlight special keywords
   '("\\(∀\\)" . font-lock-keyword-face)
   ;; highlight numbers[-+]?
   '("\\W\\([0-9._]+\\)\\>" 1 font-lock-constant-face)
   ;; highlight true, false
   '("\\(true\\|false\\)" . font-lock-constant-face)
   ;; highlight functions
   '("\\<\\(\\w+\\)\\s-*(" 1 font-lock-function-name-face)
   ;; highlight @,#
   '("\\(@\\)" . font-lock-function-name-face)
   '("\\(#\\)" . font-lock-constant-face)
   ;; highlight (/*-+)=
   '("\\<\\(/=\\|*=\\|-=\\|+=\\)\\>" . font-lock-builtin-face)
   ;; highlight /*-+
   '("\\(/\\|*\\|-\\|+\\|=\\)" 1 font-lock-builtin-face)
   ;; highlight and, or, not, xor
   '("\\<\\(&&\\|||\\|!\\|^\\)\\>" 1 font-lock-builtin-face)
   '("\\<\\(and\\|or\\|not\\|xor\\)\\>" . font-lock-builtin-face)
    ;; highlight directives and directive names
   '("^\\s-*\\(#\\)\\(\\w+\\)\\s-*\\(\\w+\\|\"\\).*" (1 font-lock-builtin-face) (2 font-lock-preprocessor-face) (3 font-lock-variable-name-face))
   ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Mode definitions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define-derived-mode nabla-mode c-mode "Nabla"
  "Minor mode for editing NABLA code." 
  (setq font-lock-defaults '(nabla-font-lock-keywords nil nil ((?\_ . "w"))))
  (message "Loading Nabla minor mode %s@%s" nabla-version nabla-time-stamp)
  (run-hooks 'nabla-mode-hook))
(provide 'nabla-mode)
