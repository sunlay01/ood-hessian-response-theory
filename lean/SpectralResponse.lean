import OODResponse

/-!
  Algebraic companion for the spectral-response part of the OOD framework.

  The calculus statements (existence of third derivatives, eigengaps, and
  differentiability of eigenvalue maps) stay in the accompanying Markdown.
  This file records the finite-dimensional identities used by those
  statements: matrix products and traces, the cyclic trace identity, the
  tensor adjoint which turns a spectral matrix sensitivity into a parameter
  force, and an algebraic rotation-blindness interface.
-/

namespace OODResponse

open scoped BigOperators

/-! ### Finite matrices and the trace -/

def matMul {n : Nat} (A B : Mat n) : Mat n :=
  fun i k => ∑ j, A i j * B j k

def matTrace {n : Nat} (A : Mat n) : ℝ :=
  ∑ i, A i i

theorem matMul_assoc {n : Nat} (A B C : Mat n) :
    matMul (matMul A B) C = matMul A (matMul B C) := by
  classical
  funext i k
  simp only [matMul]
  calc
    (∑ j, (∑ l, A i l * B l j) * C j k) =
        ∑ j, ∑ l, (A i l * B l j) * C j k := by
      refine Finset.sum_congr rfl ?_
      intro j _
      rw [Finset.sum_mul]
    _ = ∑ l, ∑ j, (A i l * B l j) * C j k := by
      rw [Finset.sum_comm]
    _ = ∑ l, A i l * (∑ j, B l j * C j k) := by
      refine Finset.sum_congr rfl ?_
      intro l _
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro j _
      ring
    _ = ∑ l, A i l * ∑ j, B l j * C j k := rfl

theorem matTrace_mul_comm {n : Nat} (A B : Mat n) :
    matTrace (matMul A B) = matTrace (matMul B A) := by
  classical
  calc
    matTrace (matMul A B) = ∑ i, ∑ j, A i j * B j i := rfl
    _ = ∑ j, ∑ i, A i j * B j i := by rw [Finset.sum_comm]
    _ = ∑ j, ∑ i, B j i * A i j := by
      refine Finset.sum_congr rfl ?_
      intro j _
      refine Finset.sum_congr rfl ?_
      intro i _
      rw [mul_comm]
    _ = matTrace (matMul B A) := rfl

theorem matTrace_sub {n : Nat} (A B : Mat n) :
    matTrace (A - B) = matTrace A - matTrace B := by
  classical
  simp [matTrace, sub_eq_add_neg, Finset.sum_add_distrib]

def commutator {n : Nat} (K H : Mat n) : Mat n :=
  matMul K H - matMul H K

theorem matTrace_commutator_zero {n : Nat} (K H : Mat n) :
    matTrace (commutator K H) = 0 := by
  rw [commutator, matTrace_sub, matTrace_mul_comm]
  ring

/-! ### Tensor adjoints and spectral parameter forces -/

def tensorAdjoint {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (B : Mat n) : Vec n :=
  fun k => ∑ i, ∑ j, T i j k * B i j

def spectralForce {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (B : Mat n) : Vec n :=
  tensorAdjoint T B

theorem tensorAdjoint_add {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (A B : Mat n) :
    tensorAdjoint T (A + B) = tensorAdjoint T A + tensorAdjoint T B := by
  classical
  funext k
  simp only [tensorAdjoint, Pi.add_apply, mul_add, Finset.sum_add_distrib]

theorem tensorAdjoint_neg {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (A : Mat n) :
    tensorAdjoint T (-A) = -(tensorAdjoint T A) := by
  classical
  funext k
  simp [tensorAdjoint]

theorem spectralForce_linear {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (A B : Mat n) :
    spectralForce T (A + B) = spectralForce T A + spectralForce T B := by
  exact tensorAdjoint_add T A B

theorem spectralForce_zero_of_zero_sensitivity {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) :
    spectralForce T (0 : Mat n) = 0 := by
  funext k
  simp [spectralForce, tensorAdjoint]

/-! ### Rotation-blind directions -/

theorem trace_direction_rotation_blind {n : Nat} (K H : Mat n) :
    matTrace (commutator K H) = 0 :=
  matTrace_commutator_zero K H

/-
  `rotationBlind` is the algebraic interface used for a general
  eigenvalue-only spectral functional.  In the analytic theory its field is
  discharged from orthogonal invariance and the chain rule.  The concrete
  trace functional above discharges the same condition without an axiom.
-/
structure RotationBlind {n : Nat} where
  differential : Mat n → Mat n → ℝ
  vanishes_on_commutator : ∀ H K, differential H (commutator K H) = 0

theorem rotationBlind_zero {n : Nat} (ρ : RotationBlind (n := n))
    (H K : Mat n) :
    ρ.differential H (commutator K H) = 0 :=
  ρ.vanishes_on_commutator H K

end OODResponse
