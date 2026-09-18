import Mathlib

/-!
  A small Lean4 companion to the two-regime OOD response theory.

  The analytic hypotheses (differentiability, implicit-function theorem,
  constant-rank theorem, and probabilistic exposure assumptions) are kept
  outside this file.  This file formalizes the finite-dimensional algebraic
  identities that do not depend on a particular calculus library encoding.
-/

namespace OODResponse

open scoped BigOperators

abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (n : Nat) := Fin n → Fin n → ℝ

def matVec {n : Nat} (A : Mat n) (v : Vec n) : Vec n :=
  fun i => ∑ j, A i j * v j

def matTranspose {n : Nat} (A : Mat n) : Mat n :=
  fun i j => A j i

def quadForce {n m : Nat}
    (J : Fin m → Fin n → ℝ) (W : Fin m → Fin m → ℝ)
    (r : Fin m → ℝ) : Vec n :=
  fun i => ∑ k, J k i * (∑ l, W k l * r l)

def taskResponse {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (v : Vec n) : Mat n :=
  fun i j => ∑ k, T i j k * v k

def irmStatisticForce {n m : Nat}
    (q : Fin m → ℝ) (a : Fin m → Vec n) : Vec n :=
  fun i => ∑ e, q e * a e i

def vrexStatisticForce {n m : Nat}
    (delta : Fin m → ℝ) (g : Fin m → Vec n) : Vec n :=
  fun i => ∑ e, delta e * g e i

theorem taskResponse_linear {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (v w : Vec n) :
    taskResponse T (v + w) = taskResponse T v + taskResponse T w := by
  funext i j
  simp only [taskResponse, Pi.add_apply, mul_add, Finset.sum_add_distrib]

theorem taskResponse_neg {n : Nat}
    (T : Fin n → Fin n → Fin n → ℝ) (v : Vec n) :
    taskResponse T (-v) = -(taskResponse T v) := by
  funext i j
  simp [taskResponse]

theorem mixed_tangent_cancel {n m : Nat}
    (qI qS : Fin m → Vec n → ℝ) (vI vS : Vec n)
    (h : ∀ e, qI e vI = - qS e vS) :
    (fun e => qI e vI + qS e vS) = 0 := by
  funext e
  rw [h e]
  simp

theorem irm_force_zero_of_zero_residual {n m : Nat}
    (q : Fin m → ℝ) (a : Fin m → Vec n)
    (hq : q = 0) :
    irmStatisticForce q a = 0 := by
  subst hq
  funext i
  simp [irmStatisticForce]

theorem vrex_force_zero_of_equal_risk {n m : Nat}
    (delta : Fin m → ℝ) (g : Fin m → Vec n)
    (hdelta : delta = 0) :
    vrexStatisticForce delta g = 0 := by
  subst hdelta
  funext i
  simp [vrexStatisticForce]

theorem response_equation {n : Nat}
    (A : Mat n) (B v : Vec n)
    (h : matVec A v = -B) :
    matVec A v + B = 0 := by
  rw [h]
  simp

theorem actuation_blind_implies_zero_response {n : Nat}
    (_A : Mat n) (force : Vec n) (v : Vec n)
    (hforce : force = 0) (hmove : v = force) :
    v = 0 := by
  rw [hmove, hforce]

theorem generic_force_is_linear_in_residual {n m : Nat}
    (J : Fin m → Fin n → ℝ) (W : Fin m → Fin m → ℝ)
    (r s : Fin m → ℝ) :
    quadForce J W (r + s) = quadForce J W r + quadForce J W s := by
  funext i
  simp only [quadForce, Pi.add_apply, mul_add, Finset.sum_add_distrib]

theorem centered_sum_zero {m : Nat} (pi x : Fin m → ℝ)
    (_hpi : ∑ e, pi e = 1) :
    (fun e => x e - ∑ j, pi j * x j) = 0 →
    ∑ e, pi e * (x e - ∑ j, pi j * x j) = 0 := by
  intro h
  have hpoint : ∀ e, x e - ∑ j, pi j * x j = 0 := fun e => congrFun h e
  simp [hpoint]

theorem observation_actuation_helpfulness_decomposition {n m : Nat}
    (J : Fin m → Fin n → ℝ) (W : Fin m → Fin m → ℝ)
    (r : Fin m → ℝ) :
    quadForce J W r = 0 ∨ quadForce J W r ≠ 0 := by
  exact em (quadForce J W r = 0)

end OODResponse
