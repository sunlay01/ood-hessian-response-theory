import Mathlib

namespace OODResponse

/-! A formal factorization companion.  We state the extension theorem with a
surjective observation map; this is enough for the scalar IRMv1 and V-REx
maps used in the concrete corollaries. -/

def locallyTransferSufficient
    {V Y Z : Type*}
    [AddCommGroup V] [AddCommGroup Y] [AddCommGroup Z]
    [Module ℝ V] [Module ℝ Y] [Module ℝ Z]
    (O : V →ₗ[ℝ] Y) (G : V →ₗ[ℝ] Z) : Prop :=
  ∀ x, O x = 0 → G x = 0

theorem factorization_iff_of_surjective
    {V Y Z : Type*}
    [AddCommGroup V] [AddCommGroup Y] [AddCommGroup Z]
    [Module ℝ V] [Module ℝ Y] [Module ℝ Z]
    (O : V →ₗ[ℝ] Y) (G : V →ₗ[ℝ] Z)
    (hO : Function.Surjective O) :
    locallyTransferSufficient O G ↔
      ∃ L : Y →ₗ[ℝ] Z, G = L.comp O := by
  constructor
  · intro hker
    let qG : (V ⧸ LinearMap.ker O) →ₗ[ℝ] Z :=
      (LinearMap.ker O).liftQ G (by
        intro x hx
        exact hker x hx)
    let e : (V ⧸ LinearMap.ker O) ≃ₗ[ℝ] Y :=
      O.quotKerEquivOfSurjective hO
    let L : Y →ₗ[ℝ] Z := qG.comp e.symm.toLinearMap
    refine ⟨L, ?_⟩
    ext x
    change G x = qG (e.symm (O x))
    rw [LinearMap.quotKerEquivOfSurjective_symm_apply O hO x]
    rfl
  · rintro ⟨L, rfl⟩ x hx
    simp [hx]

theorem transfer_sufficient_iff_factorization
    {V Y Z : Type*}
    [AddCommGroup V] [AddCommGroup Y] [AddCommGroup Z]
    [Module ℝ V] [Module ℝ Y] [Module ℝ Z]
    (O : V →ₗ[ℝ] Y) (G : V →ₗ[ℝ] Z)
    (hO : Function.Surjective O) :
    locallyTransferSufficient O G ↔
      ∃ L : Y →ₗ[ℝ] Z, G = L.comp O :=
  factorization_iff_of_surjective O G hO

theorem factorization_iff
    {V Y Z : Type*}
    [AddCommGroup V] [AddCommGroup Y] [AddCommGroup Z]
    [Module ℝ V] [Module ℝ Y] [Module ℝ Z]
    [FiniteDimensional ℝ V] [FiniteDimensional ℝ Y]
    (O : V →ₗ[ℝ] Y) (G : V →ₗ[ℝ] Z) :
    locallyTransferSufficient O G ↔
      ∃ L : Y →ₗ[ℝ] Z, G = L.comp O := by
  constructor
  · intro hker
    let qG : (V ⧸ LinearMap.ker O) →ₗ[ℝ] Z :=
      (LinearMap.ker O).liftQ G (by
        intro x hx
        exact hker x hx)
    let e : (V ⧸ LinearMap.ker O) ≃ₗ[ℝ] LinearMap.range O :=
      O.quotKerEquivRange
    let Lrange : LinearMap.range O →ₗ[ℝ] Z :=
      qG.comp e.symm.toLinearMap
    have hrange :
        G = Lrange.comp O.rangeRestrict := by
      ext x
      change G x = qG (e.symm (O.rangeRestrict x))
      have he :
          e.symm (O.rangeRestrict x) = (LinearMap.ker O).mkQ x := by
        exact LinearMap.quotKerEquivRange_symm_apply_image O x
          (LinearMap.mem_range_self O x)
      rw [he]
      rfl
    obtain ⟨L, hL⟩ := LinearMap.exists_extend Lrange
    refine ⟨L, ?_⟩
    ext x
    have hx := congrArg (fun f => f (O.rangeRestrict x)) hL
    have hrx := congrArg (fun f => f x) hrange
    exact hrx.trans hx.symm
  · rintro ⟨L, rfl⟩ x hx
    simp [hx]

end OODResponse
