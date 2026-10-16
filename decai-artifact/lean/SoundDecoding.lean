namespace SoundDecoding

/-! ## Counting lemma -/

theorem filter_length_mono {β : Type} (l : List β) (p q : β → Bool)
    (h : ∀ a, p a = true → q a = true) :
    (l.filter p).length ≤ (l.filter q).length := by
  induction l with
  | nil => simp
  | cons a t ih =>
    cases hp : p a with
    | true =>
      have hq : q a = true := h a hp
      simp [List.filter_cons, hp, hq]
      exact ih
    | false =>
      cases hq : q a with
      | true =>
        simp [List.filter_cons, hp, hq]
        exact Nat.le_succ_of_le ih
      | false =>
        simp [List.filter_cons, hp, hq]
        exact ih

/-! ## Scores -/

/-- The one law of ordered additive groups that the soundness proof uses. -/
def SubLaw (α : Type) [LT α] [LE α] [Sub α] [OfNat α 0] : Prop :=
  ∀ a b c : α, 0 < a → a ≤ b - c → c < b

/-- The law holds for the integers. -/
theorem int_subLaw : SubLaw Int := by
  intro a b c h1 h2
  omega

/-! ## Lemma 1: abstract top-k -/

section TopK
variable {V : Type} (vocab : List V)
variable {α : Type} [LT α] [LE α] [Sub α] [OfNat α 0] [∀ a b : α, Decidable (a < b)]

/-- The admissible top-k set: fewer than `k` tokens have a strictly larger score. -/
def Admissible (z : V → α) (k : Nat) (t : V) : Prop :=
  (vocab.filter (fun j => decide (z t < z j))).length < k

/-- The abstract candidate set computed from a margin matrix `m j t` (lower bounds on z_j - z_t). -/
def Cand (m : V → V → α) (k : Nat) (t : V) : Prop :=
  (vocab.filter (fun j => decide (0 < m j t))).length < k

/-- Lemma 1. If the margins are sound for the scores `z`, every admissible token is a candidate. -/
theorem topk_sound (hlaw : SubLaw α) (z : V → α) (m : V → V → α) (k : Nat) (t : V)
    (hm : ∀ j, m j t ≤ z j - z t) (ht : Admissible vocab z k t) : Cand vocab m k t := by
  unfold Admissible at ht
  unfold Cand
  have hmono := filter_length_mono vocab (fun j => decide (0 < m j t)) (fun j => decide (z t < z j)) (by
    intro j hj
    have h1 : 0 < m j t := of_decide_eq_true hj
    exact decide_eq_true (hlaw (m j t) (z j) (z t) h1 (hm j)))
  exact Nat.lt_of_le_of_lt hmono ht

end TopK

/-! ## Theorem 1: soundness of abstract decoding with the partitioned join -/

/-- An abstract output string: the set of tokens allowed at each position. -/
abbrev AStr (V : Type) := Nat → V → Prop

/-- Concretisation of an abstract output string of length `j`. -/
def gamma {V : Type} (S : AStr V) (j : Nat) (y : Nat → V) : Prop := ∀ i, i < j → S i (y i)

/-- The monitor run on the first `j` output tokens. -/
def run {V Q : Type} (q0 : Q) (δ : Q → V → Q) (y : Nat → V) : Nat → Q
  | 0 => q0
  | j + 1 => δ (run q0 δ y j) (y j)

section Decoding
variable {V Q X : Type}
variable (q0 qbad : Q) (δ : Q → V → Q)
-- `adm x j y t`: token `t` may be decoded at step `j` on input `x` after the output prefix of `y`.
variable (adm : X → Nat → (Nat → V) → V → Prop)
-- The inputs described by the abstract input.
variable (Xs : X → Prop)
-- `cand j S t`: the abstract top-k operator at step `j` for the abstract output string `S`.
variable (cand : Nat → AStr V → V → Prop)

/-- One iteration of Algorithm 2: the reached monitor states and, for each, the joined abstract string. -/
def step (D : Q → Prop) (R : Q → AStr V) (j : Nat) : (Q → Prop) × (Q → AStr V) :=
  (fun q' => ∃ q t, D q ∧ cand j (R q) t ∧ δ q t = q',
   fun q' i v => ∃ q t, D q ∧ cand j (R q) t ∧ δ q t = q' ∧ ((i < j ∧ R q i v) ∨ (i = j ∧ v = t)))

/-- The abstract state after `j` iterations: `R_j` of Algorithm 2 as (domain, entries). -/
def absState : Nat → (Q → Prop) × (Q → AStr V)
  | 0 => (fun q => q = q0, fun _ _ _ => False)
  | j + 1 => step δ cand (absState j).1 (absState j).2 j

/-- Algorithm 2 raises an alarm within `L` steps. -/
def Alarm (L : Nat) : Prop :=
  ∃ j, j < L ∧ ∃ q t, (absState q0 δ cand j).1 q ∧ cand j ((absState q0 δ cand j).2 q) t ∧ δ q t = qbad

/-- Soundness of the abstract top-k operator (cf. Lemma 1 and Definition 3). -/
def OracleSound : Prop :=
  ∀ x, Xs x → ∀ j (S : AStr V) y t, gamma S j y → adm x j y t → cand j S t

/-- The invariant of Theorem 1: every admissible output prefix is covered by the entry of its monitor state. -/
theorem invariant (hs : OracleSound adm Xs cand) (x : X) (hx : Xs x) (y : Nat → V) :
    ∀ j, (∀ i, i < j → adm x i y (y i)) →
      (absState q0 δ cand j).1 (run q0 δ y j) ∧ gamma ((absState q0 δ cand j).2 (run q0 δ y j)) j y := by
  intro j
  induction j with
  | zero =>
    intro _
    exact ⟨rfl, fun i hi => absurd hi (Nat.not_lt_zero i)⟩
  | succ j ih =>
    intro hadm
    obtain ⟨hD, hγ⟩ := ih (fun i hi => hadm i (Nat.lt_succ_of_lt hi))
    have hc : cand j ((absState q0 δ cand j).2 (run q0 δ y j)) (y j) :=
      hs x hx j _ y (y j) hγ (hadm j (Nat.lt_succ_self j))
    refine ⟨⟨run q0 δ y j, y j, hD, hc, rfl⟩, ?_⟩
    intro i hi
    refine ⟨run q0 δ y j, y j, hD, hc, rfl, ?_⟩
    cases Nat.lt_or_ge i j with
    | inl hlt => exact Or.inl ⟨hlt, hγ i hlt⟩
    | inr hge =>
      have heq : i = j := Nat.le_antisymm (Nat.le_of_lt_succ hi) hge
      subst heq
      exact Or.inr ⟨rfl, rfl⟩

/-- Theorem 1. If Algorithm 2 raises no alarm within `L` steps, then no admissible output of any input
    in `Xs` reaches the forbidden state `qbad` within `L` steps. -/
theorem soundness (hs : OracleSound adm Xs cand) (hq0 : q0 ≠ qbad) (L : Nat)
    (hv : ¬ Alarm q0 qbad δ cand L) :
    ∀ x, Xs x → ∀ y : Nat → V, (∀ i, i < L → adm x i y (y i)) →
      ∀ j, j ≤ L → run q0 δ y j ≠ qbad := by
  intro x hx y hadm j hj
  cases j with
  | zero => exact hq0
  | succ j =>
    intro hbad
    have hinv := invariant q0 δ adm Xs cand hs x hx y j (fun i hi => hadm i (by omega))
    have hc : cand j ((absState q0 δ cand j).2 (run q0 δ y j)) (y j) :=
      hs x hx j _ y (y j) hinv.2 (hadm j (by omega))
    exact hv ⟨j, by omega, run q0 δ y j, y j, hinv.1, hc, hbad⟩

end Decoding

/-! ## Corollary: Lemma 1 and Theorem 1 combined -/

section Combined
variable {V Q X : Type} (vocab : List V) (k : Nat)
variable {α : Type} [LT α] [LE α] [Sub α] [OfNat α 0] [∀ a b : α, Decidable (a < b)]
variable (q0 qbad : Q) (δ : Q → V → Q) (Xs : X → Prop)
-- `score x j y`: the logits of the model on input `x` after the output prefix of `y` of length `j`.
variable (score : X → Nat → (Nat → V) → V → α)
-- `margin j S`: the margin oracle on the abstract context with abstract output string `S`.
variable (margin : Nat → AStr V → V → V → α)

/-- Corollary. With a sound margin oracle, if the abstract decoding (with candidates computed from the
    margins) raises no alarm, the model is F-free on `Xs` under top-k decoding. -/
theorem freedom (hlaw : SubLaw α)
    (hm : ∀ x, Xs x → ∀ j (S : AStr V) y, gamma S j y → ∀ i t, margin j S i t ≤ score x j y i - score x j y t)
    (hq0 : q0 ≠ qbad) (L : Nat)
    (hv : ¬ Alarm q0 qbad δ (fun j S t => Cand vocab (margin j S) k t) L) :
    ∀ x, Xs x → ∀ y : Nat → V, (∀ i, i < L → Admissible vocab (score x i y) k (y i)) →
      ∀ j, j ≤ L → run q0 δ y j ≠ qbad := by
  have hs : OracleSound (fun x j y t => Admissible vocab (score x j y) k t) Xs
      (fun j S t => Cand vocab (margin j S) k t) := by
    intro x hx j S y t hγ ht
    exact topk_sound vocab hlaw (score x j y) (margin j S) k t (fun i => hm x hx j S y hγ i t) ht
  exact soundness q0 qbad δ (fun x j y t => Admissible vocab (score x j y) k t) Xs
    (fun j S t => Cand vocab (margin j S) k t) hs hq0 L hv

end Combined

/-- The corollary for integer scores, where the algebraic law is discharged by `int_subLaw`. -/
theorem freedom_int {V Q X : Type} (vocab : List V) (k : Nat) (q0 qbad : Q) (δ : Q → V → Q)
    (Xs : X → Prop) (score : X → Nat → (Nat → V) → V → Int) (margin : Nat → AStr V → V → V → Int)
    (hm : ∀ x, Xs x → ∀ j (S : AStr V) y, gamma S j y → ∀ i t, margin j S i t ≤ score x j y i - score x j y t)
    (hq0 : q0 ≠ qbad) (L : Nat)
    (hv : ¬ Alarm q0 qbad δ (fun j S t => Cand vocab (margin j S) k t) L) :
    ∀ x, Xs x → ∀ y : Nat → V, (∀ i, i < L → Admissible vocab (score x i y) k (y i)) →
      ∀ j, j ≤ L → run q0 δ y j ≠ qbad :=
  freedom vocab k q0 qbad δ Xs score margin int_subLaw hm hq0 L hv

end SoundDecoding
