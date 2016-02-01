(custom-set-faces 
 '(default                      ((t (:foreground "black"))))
 '(font-lock-builtin-face       ((t (:foreground "#1e90ff"))))
 '(font-lock-comment-face       ((t (:foreground "#006400")))) ; DarkGreen
 '(font-lock-constant-face      ((t (:foreground "magenta"))))
 '(font-lock-function-name-face ((t (:foreground "blue" :bold t))))
 '(font-lock-keyword-face       ((t (:foreground "black" :bold t))))
 '(font-lock-string-face        ((t (:foreground "#ff00ff")))) ; magenta1
 '(font-lock-type-face          ((t (:foreground "#cd853f" :bold t))))
 '(font-lock-variable-name-face ((t (:foreground "cyan"))))
 '(font-lock-warning-face       ((t (:foreground "red" :weight bold))))
 )

;(setq nabla-mode)
(setq htmlize-use-rgb-map 'force)
;(setq tab-width 3)

(require 'htmlize)
;(require 'nabla-mode)

;(autoload 'nabla-mode "nabla-mode" "Nabla Mode" t)
;(setq auto-mode-alist (cons '("\\.n\\'" . nabla-mode) auto-mode-alist))

;(setq locale-coding-system 'utf-8)
;(setq face-ignored-fonts '("Latin Modern Math" "TeX Gyre Schola Math"))

(find-file (pop command-line-args-left))

(font-lock-fontify-buffer)

(with-current-buffer (htmlize-buffer)  (princ (buffer-string)))
