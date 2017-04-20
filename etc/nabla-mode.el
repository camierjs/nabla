;;; nabla-mode.el --- major mode for editing Nabla files.
(defconst nabla-version "nabla" "Nabla Mode version number")
(defconst nabla-time-stamp "2017-04-18"
  "Nabla Mode time stamp for last update.")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defconst nabla-keywords
  '("inline" "restrict" "aligned" "const" "return" "register" "volatile"
    "if" "is" "else" "do" "while" "continue" "break" "for"
    "foreach"
    "all" "own" "inner" "outer"
    "in" "out" "inout" "with")
  "List of Nabla keywords regexps")

(defconst nabla-maths
  '("sqrt" "norm" "dot" "cross" "min" "max" 
    "mathlink" "Prime" "operator"
    ) "List of Nabla mathematical tools")

(defconst nabla-builtins
  '("exit" "checkpoint" "coord" 
    "uid" "lid" "sid" "this" "nbNode" "iNode" "fatal"
    "backCell" "backCellUid" "frontCell" "frontCellUid" 
    "next" "prev"
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
  '("global" "set"
    "cell" "cells" 
    "node" "nodes"
    "edge" "edges"
    "face" "faces" 
    "particle" "particles"
    "material" "materials" 
    "environment" "environments" 
    "options" "option" "enum"
    "variables"
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
    ;"ℕ" "ℝ" "ℝᵈ" "ℝᵈ⨯ℝᵈ" 
    "real" "natural"
    "reals" "naturals"
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
   ;'("\\(⁰\\)" . font-lock-variable-name-face)
   '("\\(ⁿ⁺¹\\|ⁿ⁼⁰\\|ⁿ\\)" . font-lock-constant-face)
   '("\\(²ˣ²\\|³ˣ³\\)" . font-lock-type-face)
   '("\\(ˣ\\|⁺\\|⁼\\|¹\\|²\\|³\\|ˣ\\)" . font-lock-type-face)
   ;; highlight special keywords
   '("\\(∀\\|ℵ\\)" . font-lock-keyword-face)
   ;; highlight special characters
   '("\\(ℝ\\|ℤ\\|ℕ\\|ℾ\\|ℂ\\|ℿ\\)" . font-lock-type-face)
   ;; highlight numbers[-+]?
   '("\\W\\([0-9._]+\\)\\>" 1 font-lock-constant-face)
   ;; highlight true, false
   '("\\(true\\|false\\)" . font-lock-constant-face)
   ;; highlight functions
   '("\\<\\(\\w+\\)\\s-*(" 1 font-lock-function-name-face)
   ;; highlight @,#
   '("\\(@\\)" . font-lock-type-face)
   ;'("\\(#nodes\\)" . font-lock-builtin-face)
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
   ;; highlight job and function names
   '("\\<\\(\\w+\\)\\s-*\\(@\\)" (1 font-lock-function-name-face) (2 font-lock-type-face))
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

;;; nabla-mode.el ends here
