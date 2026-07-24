"""Priced width/depth (size) selection for AllGraph (mixin).

The "degrees of freedom" stage: select_architecture (sequential width-then-depth) and
select_architecture_by_area (joint width x depth area) sweep candidate sizes on a held-out split and pick
by the priced marginal-value rule; _select_size_variable does per-layer variable-width selection. Split out
of allgraph.py to keep the controller focused; AllGraph mixes this in so `self` resolves as before."""
import numpy as np

from .allgraph_types import _SweepCtx


class _SizeSelectionMixin:
    def _sweep_setup(self, data, task, n_out, contract, val_frac, sweep_epochs, sweep_epochs_floor):
        """Shared prologue for the size-selection sweeps (select_architecture / _by_area): resolve the
        contract (tiebreak -> route -> default 'set'), hold out a seeded validation split, infer n_out, and
        compute the capped sweep epoch budget (measurement-trustworthy, independent of the deployment budget).
        Returns (contract, tr, va, n_out, ep). Callers save/restore self.width/depth/seed around their own
        score closures."""
        if contract is None:
            cands = self._tiebreak_candidates(data)
            if cands is not None:
                contract, _, _ = self.tiebreak(data, task=task, n_out=n_out)
            else:
                routed, _ = self.route(data)
                contract = routed if routed in ("set", "graph", "equivariant") else "set"
        nf = getattr(data, "node_feats", None)
        n = len(nf) if nf is not None else len(data.y)
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n); nval = max(1, int(val_frac * n))
        va, tr = perm[:nval], perm[nval:]
        n_out = self._infer_nout(data.y, task, n_out)
        ep = sweep_epochs if sweep_epochs is not None else self._search_ep(max(self.epochs, sweep_epochs_floor))
        return contract, tr, va, n_out, ep

    def select_architecture(self, data, task="classification", n_out=None, contract=None,
                            widths=(8, 16, 32, 64, 128), depths=(1, 2, 3), width_mu=0.0, depth_mu=0.0,
                            seeds=(0,), val_frac=0.25, sweep_epochs=None, sweep_epochs_floor=40,
                            edge_cutoff=None, min_signal_r2=0.05, record=True,
                            extend_width=True, max_width_cap=1024):
        """OPT-IN minimal-architecture selection: choose WIDTH (K*) and DEPTH (L*) by the priced
        marginal-value criterion, so architecture size becomes an OUTPUT of metaoptimality rather than a
        fixed hyperparameter. This trains O(#widths + #depths) nets -- a sweep -- so it is a SEPARATE,
        explicitly invoked method (not part of the single-train `fit`), mirroring the tie-break bake-off.

        Protocol (bilevel/honest): a validation split is held out from the data; weights train on the train
        part and both marginal-value curves are measured on validation. WIDTH is swept and the smallest
        width whose per-neuron marginal val-gain falls below `width_mu` is chosen (significant-elbow when
        width_mu==0); DEPTH is then swept at K* via the priced_depth machinery. The chosen (K*, L*) are
        written back to self.width/self.depth.

        Two robustness safeguards (learned from rMD17, where the sweep at a tiny budget saw only noise):
          * SWEEP_EPOCHS FLOOR: the sweep is a MEASUREMENT and must be trustworthy independent of how
            briefly the final model will train. sweep_epochs defaults to max(self.epochs, sweep_epochs_floor)
            so that even under a tiny deployment budget the candidates train enough to reveal their
            representational reach (capacity signal), not transient under-training. Width is also swept at
            the deployment depth (or depth 1 if unset) rather than always depth 1.
          * UNINFORMATIVE-SWEEP GUARD: if even the best width's validation fit stays below min_signal_r2,
            every candidate is in the noise regime and the marginal-value ranking is meaningless. Rather
            than return a noise-driven K*, fall back to the LARGEST width (most capacity, the safe choice
            when the signal to prune is absent) and flag the sweep uninformative in the detail.
        Returns a detail dict with both curves.
        """
        from ..machinery.priced_depth import measure_depth_curve, select_depth, significant_elbow
        contract, tr, va, n_out, ep = self._sweep_setup(
            data, task, n_out, contract, val_frac, sweep_epochs, sweep_epochs_floor)
        sweep_depth = self.depth if self.depth and self.depth >= 1 else 1
        w0, d0, s0 = self.width, self.depth, self.seed
        ctx = _SweepCtx(data, contract, tr, va, task, n_out, ep, edge_cutoff)   # sweep-invariant (size varies via self)

        def score_at(width, depth, seed):
            self.width, self.depth, self.seed = width, depth, seed
            return float(self._train_candidate_contract(ctx))

        wscores = [np.mean([score_at(w, sweep_depth, s) for s in seeds]) for w in widths]
        widths = list(widths)
        n_base_widths = len(widths)
        wlosses = [1.0 - s for s in wscores]
        best_width_score = max(wscores)
        uninformative = best_width_score < min_signal_r2

        def _pick_width(widths, wlosses):
            """Apply the priced marginal-value / significant-elbow rule to a width curve. Returns
            (Kstar, hit_ceiling): hit_ceiling is True iff the rule selected the LARGEST width because the
            marginal never fell below the price (ceiling binding), as opposed to a genuine interior stop."""
            if width_mu > 0:
                for i in range(1, len(widths)):
                    marg = (wlosses[i - 1] - wlosses[i]) / (widths[i] - widths[i - 1])
                    if marg < width_mu:
                        return widths[i - 1], False           # interior stop: marginal dropped below price
                return widths[-1], True                        # never dropped -> ceiling binding
            else:
                Kstar = widths[0]; interior = False
                for i in range(1, len(widths)):
                    if wlosses[i - 1] - wlosses[i] > 1e-4:
                        Kstar = widths[i]
                    else:
                        interior = True                        # a width stopped clearly improving -> elbow found
                return Kstar, (Kstar == widths[-1] and not interior)

        if uninformative:
            # no capacity separates from noise -> do NOT prune on noise; take the most capacity available.
            Kstar = widths[-1]
        else:
            Kstar, hit_ceiling = _pick_width(widths, wlosses)
            # BOUNDARY-EXTENSION GUARD: if K* landed on the largest candidate because the marginal-value
            # rule never fired (the ceiling is binding, not a genuine interior optimum), the fixed grid is
            # cutting off capacity the task would use (the rMD17 signature: R2 still climbing steeply at the
            # top width). Extend the sweep by DOUBLING the ceiling and re-measuring, until the rule actually
            # stops at an interior width or a hard cap is reached -- so K* is chosen by the priced criterion,
            # not by where the grid happened to end.
            if extend_width and hit_ceiling:
                while Kstar == widths[-1] and widths[-1] * 2 <= max_width_cap:
                    w_new = widths[-1] * 2
                    s_new = np.mean([score_at(w_new, sweep_depth, s) for s in seeds])
                    gain = s_new - wscores[-1]                 # improvement from the current top width
                    widths.append(w_new); wscores.append(s_new); wlosses.append(1.0 - s_new)
                    best_width_score = max(best_width_score, s_new)
                    # SATURATION FLOOR: if doubling the width barely moved the score, the curve has
                    # plateaued -- stop extending even if the formal marginal-value rule hasn't fired, so we
                    # do not double indefinitely into a flat region on a noisy curve.
                    if gain < max(width_mu, 1e-3):
                        Kstar = widths[-2] if gain <= 0 else w_new
                        break
                    Kstar, hit_ceiling = _pick_width(widths, wlosses)
                    if not hit_ceiling:
                        break

        ceiling_extended = len(widths) > n_base_widths

        def depth_eval(L, sd):
            self.width, self.depth, self.seed = Kstar, L, sd
            score = self._train_candidate_contract(ctx)
            return 1.0 - float(score), float(score)   # (val_loss, val_score) as measure_depth_curve expects
        curve = measure_depth_curve(depth_eval, list(depths), list(seeds))
        Lstar = select_depth(curve, depth_mu) if depth_mu > 0 else significant_elbow(curve)

        self.seed = s0
        self.width, self.depth = int(Kstar), int(Lstar)
        detail = {"contract": contract, "width_star": int(Kstar), "depth_star": int(Lstar),
                  "sweep_epochs": ep, "sweep_depth": sweep_depth,
                  "best_width_val_score": round(float(best_width_score), 4),
                  "uninformative_sweep": bool(uninformative),
                  "ceiling_extended": bool(ceiling_extended),
                  "width_curve": list(zip(list(widths), [round(float(x), 4) for x in wlosses])),
                  "depth_marginals": [(m, round(float(v), 4), round(float(e), 4)) for (m, v, e) in curve.marginals],
                  "prev": {"width": w0, "depth": d0}}
        if record:
            note = " [uninformative -> max width]" if uninformative else ""
            note += f" [ceiling extended to {widths[-1]}]" if ceiling_extended else ""
            self._log(f"[AllGraph] select_architecture -> width K*={Kstar}, depth L*={Lstar} "
                      f"(contract={contract}, sweep_epochs={ep}, best_val={best_width_score:.3f}){note}")
        return detail

    def select_architecture_by_area(self, data, task="classification", n_out=None, contract=None,
                                    widths=(16, 32, 48, 64), depths=(1, 2, 3), tol=0.05,
                                    seeds=(0,), val_frac=0.25, sweep_epochs=None, sweep_epochs_floor=40,
                                    edge_cutoff=None, extend_depth=True, max_depth_cap=8, record=True):
        """OPT-IN joint width/depth selection by AREA (= width*depth = total neurons) minimization.

        select_architecture chooses width and depth SEQUENTIALLY (width first at a fixed sweep depth, then
        depth at that width), so it cannot trade depth against width and is biased toward wide-shallow nets;
        its depth grid is also hard-capped with no boundary extension. This method instead evaluates the full
        (width, depth) grid and returns the MINIMUM-AREA configuration whose validation score is within `tol`
        of the best score on the grid. Rationale: at matched accuracy the physically meaningful cost is the
        total number of hidden units (area), not the depth alone; minimizing area lets a narrow-deep net beat
        a wide-shallow one when it is genuinely smaller (validated on ESOL: at tol~0.08-0.12 this selects
        w32/L3 (area 96) where the shallow-first rule takes w64/L2 (area 128)).

        DEPTH BOUNDARY EXTENSION: mirroring the width guard in select_architecture, if the best score on the
        grid is achieved at the LARGEST depth (the cap is binding, not an interior optimum), the depth grid is
        extended (depth += 1) and the new column measured, until the optimum is interior or max_depth_cap is
        reached. This removes the fixed depth-3 ceiling, which real datasets exceed (ESOL peaks at depth 4).

        Returns a detail dict; also sets self.width, self.depth to the selected (K*, L*).
        """
        contract, tr, va, n_out, ep = self._sweep_setup(
            data, task, n_out, contract, val_frac, sweep_epochs, sweep_epochs_floor)
        w0, d0, s0 = self.width, self.depth, self.seed
        ctx = _SweepCtx(data, contract, tr, va, task, n_out, ep, edge_cutoff)   # sweep-invariant (size varies via self)

        grid = {}      # (w, L) -> mean val score
        grid_std = {}  # (w, L) -> std across seeds (noise estimate for significance gating)

        def score_at(width, depth):
            vals = []
            for sd in seeds:
                self.width, self.depth, self.seed = width, depth, sd
                vals.append(float(self._train_candidate_contract(ctx)))
            return float(np.mean(vals)), float(np.std(vals))

        widths = list(widths); depths = list(depths)
        for w in widths:
            for L in depths:
                grid[(w, L)], grid_std[(w, L)] = score_at(w, L)

        # noise floor: typical across-seed std on the grid (used to gate depth extension and to widen the
        # accuracy tolerance so we never chase differences smaller than measurement noise -- the ESOL lesson,
        # where depths 3/4/5 differ by less than the ~0.03 seed std).
        noise = float(np.median(list(grid_std.values()))) if len(seeds) > 1 else 0.0

        # DEPTH BOUNDARY EXTENSION (significance-gated): if the grid optimum sits at the largest depth, the
        # cap may be binding. Extend depth only while the deeper column improves the best score by MORE than
        # the noise floor -- so we do not extend into noise (the single-seed depth-4 mirage).
        def best_depth_of_best():
            return max(grid, key=lambda k: grid[k])[1]
        depth_extended = False
        if extend_depth:
            while best_depth_of_best() == depths[-1] and depths[-1] < max_depth_cap:
                Lnew = depths[-1] + 1
                prev_best = max(grid.values())
                col = {}
                for w in widths:
                    grid[(w, Lnew)], grid_std[(w, Lnew)] = score_at(w, Lnew)
                    col[w] = grid[(w, Lnew)]
                depths.append(Lnew)
                if max(grid.values()) - prev_best <= noise + 1e-3:
                    # the deeper column did not beat the incumbent beyond noise -> stop (and the interior
                    # optimum stands; the just-added column simply loses the area comparison anyway).
                    break
                depth_extended = True

        best = max(grid.values())
        # tolerance is the max of the user tol and the noise floor, so configs within noise of the best are
        # ALL treated as equally accurate and the tie is broken by area -- the whole point of the reframing.
        eff_tol = max(tol, noise)
        target = best - eff_tol
        feasible = [(w, L, grid[(w, L)]) for (w, L) in grid if grid[(w, L)] >= target]
        # minimum area, tie-broken by smaller width (fewer params at equal area for these contracts)
        Kstar, Lstar, star_score = sorted(feasible, key=lambda t: (t[0] * t[1], t[0]))[0]
        # the shallow-first pick, for comparison/reporting (what sequential selection is biased toward)
        sw, sL, ss = sorted(feasible, key=lambda t: (t[1], t[0]))[0]

        self.seed = s0
        self.width, self.depth = int(Kstar), int(Lstar)
        detail = {"contract": contract, "width_star": int(Kstar), "depth_star": int(Lstar),
                  "area_star": int(Kstar * Lstar), "score_star": round(float(star_score), 4),
                  "best_score": round(float(best), 4), "tol": tol, "noise": round(float(noise), 4),
                  "eff_tol": round(float(eff_tol), 4), "sweep_epochs": ep,
                  "depth_extended": bool(depth_extended), "max_depth_reached": int(depths[-1]),
                  "shallow_first": {"width": int(sw), "depth": int(sL), "area": int(sw * sL),
                                    "score": round(float(ss), 4)},
                  "area_beats_shallow": bool(Kstar * Lstar < sw * sL),
                  "grid": {f"w{w}_L{L}": round(float(grid[(w, L)]), 4) for (w, L) in grid},
                  "prev": {"width": w0, "depth": d0}}
        if record:
            note = f" [depth extended to {depths[-1]}]" if depth_extended else ""
            note += " [area < shallow-first]" if detail["area_beats_shallow"] else ""
            self._log(f"[AllGraph] select_architecture_by_area -> width K*={Kstar}, depth L*={Lstar}, "
                      f"area={Kstar*Lstar} (contract={contract}, tol={tol}, best_val={best:.3f}){note}")
        return detail

    def _select_size_variable(self, data, task="classification", n_out=None, max_width=None,
                              max_depth=3, lam=None, epochs=400, seeds=(0, 1),
                              extend_depth=True, max_depth_cap=8):
        """Bridge the generalized-area variable-width selector (variable_width_area) into the size-selection
        stage. The variable-width net is dense, so we featurize each example into a fixed vector (relational:
        permutation-invariant mean||max pool over nodes; dense: flattened / pooled grid), run the certificate-
        calibrated area minimization to get per-layer widths and the emergent effective depth, then SET
        self.width to the largest kept per-layer width and self.depth to the effective depth, so the contract
        schema is subsequently built and trained at that size.

        PRINCIPLED DEPTH CEILING (no hard-coded max): max_depth starts small and is EXTENDED while the extra
        layer is actually used AND buys accuracy beyond the seed-noise floor. Concretely, after fitting at the
        current max_depth we check two conditions: (i) the ceiling is BINDING -- the emergent effective depth
        equals max_depth (the net wants all the layers it was given); and (ii) adding depth at the previous
        step improved the probe accuracy by more than its across-seed std. While both hold, max_depth += 1 and
        we refit, up to max_depth_cap. This mirrors the width boundary-extension guard and the marginal-value
        depth rule: depth grows until the marginal accuracy benefit becomes insignificant, rather than being
        capped by hand."""
        from .variable_width_area import certificate_lambda_scale, fit_variable_width_area
        # featurize to a fixed vector per example (relational node_feats OR dense grid).
        if getattr(data, "node_feats", None) is not None:
            feats = []
            for nf in data.node_feats:
                a = np.asarray(nf, dtype=np.float32)
                feats.append(np.concatenate([a.mean(0), a.max(0)]))
            X = np.stack(feats).astype(np.float32)
        elif getattr(data, "dense", None) is not None:
            D = np.asarray(data.dense, dtype=np.float32)
            nex = D.shape[0]
            flat = D.reshape(nex, -1)
            if flat.shape[1] <= 512:
                X = flat
            else:
                lead = D.reshape(nex, -1, D.shape[-1])
                pooled = np.concatenate([lead.mean(1), lead.max(1), lead.std(1)], axis=1)
                nblk = 32
                blk = np.stack([flat[:, i * (flat.shape[1] // nblk):(i + 1) * (flat.shape[1] // nblk)].mean(1)
                                for i in range(nblk)], axis=1)
                X = np.concatenate([pooled, blk], axis=1).astype(np.float32)
        else:
            raise ValueError("variable size selection needs node_feats (relational) or dense (grid) data")
        y = np.asarray(data.y)
        if task == "regression":
            y = y.astype(np.float32)
            y = (y - y.mean()) / (y.std() + 1e-8)
        mw = int(max_width) if max_width is not None else int(max(self.width, 16))
        if lam is None:
            scale, _ = certificate_lambda_scale(X, y if task == "regression" else y.astype(np.float32))
            lam = 0.006 * scale

        def fit_at(md):
            """Fit the variable-width net at ceiling md over seeds; return (mean_widths, mean_val, val_std)."""
            wr, vs = [], []
            for sd in seeds:
                r = fit_variable_width_area(X, y, task=task, lam=lam, max_width=mw, max_depth=md,
                                            epochs=epochs, seed=sd)
                wr.append(r["widths"]); vs.append(r["value"])
            return np.array(wr).mean(0), float(np.mean(vs)), float(np.std(vs))

        cur = max_depth
        mean_widths, val, vstd = fit_at(cur)
        prev_val = val
        depth_extended = False
        # PRINCIPLED CEILING EXTENSION: grow depth only while BOTH (i) the ceiling is binding (the net uses
        # all layers it was given) AND (ii) going one deeper improves the probe accuracy by more than the
        # combined seed-noise. Crucially the marginal-gain test applies from the FIRST extension: a task that
        # already fits at max_depth (e.g. a linear signal) yields no significant gain and stops immediately,
        # so depth does not climb to the cap just because every layer is nominally "used".
        while (extend_depth and cur < max_depth_cap
               and int(np.sum(mean_widths > 0.5)) >= cur):          # ceiling binding
            new_widths, new_val, new_std = fit_at(cur + 1)
            gain = new_val - prev_val
            noise = max(vstd, new_std)
            if gain <= noise:                                        # marginal gain below noise -> stop
                break
            cur += 1; mean_widths, val, vstd = new_widths, new_val, new_std
            prev_val = new_val; depth_extended = True

        eff_depth = int(max(1, np.sum(mean_widths > 0.5)))
        chosen_width = int(max(1, round(float(mean_widths[mean_widths > 0.5].max()) if np.any(mean_widths > 0.5)
                                        else mean_widths.max())))
        w0, d0 = self.width, self.depth
        self.width, self.depth = chosen_width, eff_depth
        detail = {"mode": "variable", "contract": self.contract,
                  "lam": float(lam), "width_profile_mean": [round(float(x), 2) for x in mean_widths],
                  "effective_depth": eff_depth, "chosen_width": chosen_width,
                  "generalized_area_mean": float(mean_widths.sum()),
                  "probe_val_mean": float(val), "probe_val_std": float(vstd),
                  "depth_extended": bool(depth_extended), "final_max_depth": int(cur),
                  "max_width": mw, "prev": {"width": w0, "depth": d0}}
        return detail
