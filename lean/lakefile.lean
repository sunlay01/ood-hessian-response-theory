import Lake
open Lake DSL

package ood_response where
  moreLeanArgs := #["-DwarningAsError=false"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @
  "0df444a360eaa60ab8c11dca51a86af692955474"

@[default_target]
lean_lib OODResponse

lean_lib SpectralResponse

lean_lib UnifiedBound

lean_lib TransferSufficiency

lean_lib AlgorithmCorollaries
