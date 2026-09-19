import TransferSufficiency
import OODResponse

namespace OODResponse

open scoped BigOperators

/-! Exact finite-dimensional kernel witnesses used in theory/13.
The analytic risk derivations stay in Markdown; this file checks the
observation-kernel algebra without introducing an analytic axiom. -/

def irmObservation : Mat 2 :=
  fun _ _ => 1

/- The zero-padded square representation of the row map [1 0]. -/
def vrexObservation : Mat 2 :=
  fun i j => if i = 0 ∧ j = 0 then 1 else 0

def mixedWitness : Vec 2 := ![1, -1]

def spuriousWitness : Vec 2 := ![0, 1]

theorem irm_mixed_is_blind :
    matVec irmObservation mixedWitness = 0 := by
  funext i
  fin_cases i <;> simp [matVec, irmObservation, mixedWitness]

theorem vrex_spurious_is_blind :
    matVec vrexObservation spuriousWitness = 0 := by
  funext i
  fin_cases i <;> simp [matVec, vrexObservation, spuriousWitness]

def identityTransfer : Mat 2 :=
  fun i j => if i = j then 1 else 0

theorem mixed_transfer_is_nonzero :
    matVec identityTransfer mixedWitness ≠ 0 := by
  intro h
  have h0 := congrFun h (0 : Fin 2)
  simp [matVec, identityTransfer, mixedWitness] at h0

theorem spurious_transfer_is_nonzero :
    matVec identityTransfer spuriousWitness ≠ 0 := by
  intro h
  have h1 := congrFun h (1 : Fin 2)
  simp [matVec, identityTransfer, spuriousWitness] at h1

theorem irm_target_gap_positive {ε : ℝ} (hε : 0 < ε) :
    (1 + ε)^2 - 1 > 0 := by
  nlinarith [sq_nonneg ε]

theorem vrex_target_gap_positive {ε : ℝ} (hε : 0 < ε) :
    ((1 + ε)^2 / 2 - 1 / 2) > 0 := by
  nlinarith [sq_nonneg ε]

end OODResponse
