import OODResponse

/-!
  Algebraic core of the mechanism-to-bound bridge.

  The analytic target-risk Taylor estimate, Moore--Penrose identities, and
  source-coverage assumptions are stated in
  11_mechanism_to_generalization_bound.md.  The theorem below formalizes the
  part that is independent of those analytic assumptions: once a mechanism
  shift is decomposed into a visible component and a blind component, a linear
  transfer map obeys the corresponding visible-plus-blind norm bound.
-/

namespace OODResponse

theorem visible_blind_bound
    {n m p : Nat}
    (G : Vec n →ₗ[ℝ] Vec p)
    (A : Vec n →ₗ[ℝ] Vec m)
    (xi xiVisible xiBlind : Vec n)
    (hdecomp : xi = xiVisible + xiBlind)
    (κ b : ℝ)
    (hVisible : ‖G xiVisible‖ ≤ κ * ‖A xi‖)
    (hBlind : ‖G xiBlind‖ ≤ b) :
    ‖G xi‖ ≤ κ * ‖A xi‖ + b := by
  calc
    ‖G xi‖ = ‖G (xiVisible + xiBlind)‖ := by rw [hdecomp]
    _ = ‖G xiVisible + G xiBlind‖ := by rw [map_add]
    _ ≤ ‖G xiVisible‖ + ‖G xiBlind‖ := norm_add_le _ _
    _ ≤ κ * ‖A xi‖ + b := add_le_add hVisible hBlind

theorem visible_blind_zero_remainder
    {n m p : Nat}
    (G : Vec n →ₗ[ℝ] Vec p)
    (A : Vec n →ₗ[ℝ] Vec m)
    (xi xiVisible : Vec n)
    (hdecomp : xi = xiVisible)
    (κ : ℝ)
    (hVisible : ‖G xiVisible‖ ≤ κ * ‖A xi‖) :
    ‖G xi‖ ≤ κ * ‖A xi‖ := by
  simpa [hdecomp] using hVisible

theorem blind_kernel_zero_transfer
    {n m p : Nat}
    (G : Vec n →ₗ[ℝ] Vec p)
    (A : Vec n →ₗ[ℝ] Vec m)
    (hKernel : ∀ x, A x = 0 → G x = 0)
    (xi : Vec n)
    (hxi : A xi = 0) :
    G xi = 0 :=
  hKernel xi hxi

theorem source_span_bound
    {n p E : Nat}
    (G : Vec n →ₗ[ℝ] Vec p)
    (alpha : Fin E → ℝ)
    (xi : Fin E → Vec n)
    (xiPerp xiTarget : Vec n)
    (hdecomp : xiTarget = (∑ e, alpha e • xi e) + xiPerp)
    (c : Fin E → ℝ)
    (hEach : ∀ e, ‖G (xi e)‖ ≤ c e)
    (unseen : ℝ)
    (hUnseen : ‖G xiPerp‖ ≤ unseen) :
    ‖G xiTarget‖ ≤ (∑ e, ‖alpha e‖ * c e) + unseen := by
  calc
    ‖G xiTarget‖ = ‖G ((∑ e, alpha e • xi e) + xiPerp)‖ := by
      rw [hdecomp]
    _ = ‖(∑ e, alpha e • G (xi e)) + G xiPerp‖ := by
      simp only [map_add, map_sum, map_smul]
    _ ≤ ‖∑ e, alpha e • G (xi e)‖ + ‖G xiPerp‖ := norm_add_le _ _
    _ ≤ (∑ e, ‖alpha e • G (xi e)‖) + ‖G xiPerp‖ := by
      exact add_le_add_left
        (by simpa using
          (norm_sum_le (Finset.univ : Finset (Fin E))
            (fun e => alpha e • G (xi e)))) _
    _ = (∑ e, ‖alpha e‖ * ‖G (xi e)‖) + ‖G xiPerp‖ := by
      simp only [norm_smul]
    _ ≤ (∑ e, ‖alpha e‖ * c e) + unseen := by
      apply add_le_add
      · exact Finset.sum_le_sum (fun e _ => mul_le_mul_of_nonneg_left
          (hEach e) (norm_nonneg _))
      · exact hUnseen

end OODResponse
