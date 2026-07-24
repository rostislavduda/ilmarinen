"""Diagnostic report read-outs for AllGraph (mixin).

Opt-in observables assembled after a fit (LLC / developmental LLC / thermodynamic potential / response
spectroscopy / effective-dimension ledger / equivariance-breaking probe). Each reuses already-computed
quantities and never changes the fit result. Split out of allgraph.py to keep the controller focused;
AllGraph mixes this in, so every method still resolves against the same instance via `self`."""
import numpy as np
import torch
import torch.nn as nn


class _ReportsMixin:
    def _attach_diagnostics(self, result, data, task):
        """STAGE 4 (observables): attach the opt-in diagnostic read-outs (LLC, developmental LLC,
        thermodynamic potential, response spectroscopy, effective-dimension ledger, equivariance-breaking
        probe) to `result`. Each reuses already-computed quantities and never changes the fit."""
        # OPT-IN singularity-aware complexity read-out (B2): the LLC (approx. RLCT) of the deployed net,
        # the principled replacement for the parameter-count complexity. Diagnostic; does not change results.
        if getattr(self, "report_llc", False) and self.net is not None and hasattr(self.net, "parameters"):
            llc = self._llc_report(self.net, data, task)
            if llc is not None:
                result["llc"] = llc
        # OPT-IN developmental read-out (D4): lambda_hat(t) over one retrained trajectory of the selected
        # architecture -- the located convergence onset (where usable capacity turns on) and staged-learning
        # structure. Diagnostic; does not change results.
        if getattr(self, "developmental_llc", False) and self.net is not None and hasattr(self.net, "parameters"):
            dev = self._developmental_report(data, task)
            if dev is not None:
                result["developmental_llc"] = dev
        # OPT-IN thermodynamic-potential read-out (D2): record the single free-energy form's three-level
        # temperature hierarchy and assert the temperatures are at their principled values (and not coupled).
        # Diagnostic / conceptual-hygiene; does not change results.
        if getattr(self, "report_thermo", False):
            thermo = self._thermo_report(data, result)
            if thermo is not None:
                result["thermodynamic_potential"] = thermo
        # OPT-IN response / susceptibility spectroscopy (D5): report the CURVATURE of the selection
        # objective at the chosen point -- the specific heat of the primitive readout and the first-order
        # transition structure of the contract choice (critical price mu*, margins to flip, slope jump).
        # Reuses already-computed quantities; no retraining. Pure observable; changes nothing.
        if getattr(self, "report_response", False):
            resp = self._response_report(data, result)
            if resp is not None:
                result["response_spectroscopy"] = resp
        # OPT-IN effective-dimension ledger (D3): assemble the package's several effective-dimension
        # measures (participation ratio at the data-covariance and alpha-simplex levels, plus lambda at
        # the model level) onto one coarse-graining axis. Reuses already-computed pieces; changes nothing.
        if getattr(self, "report_ledger", False):
            ledger = self._ledger_report(data, result)
            if ledger is not None:
                result["effective_dimension_ledger"] = ledger
        # B5 (main equivariant path): when price_equivariance is set and we deployed the ordinary equivariant
        # contract, run the CHEAP breaking probe to report whether the symmetry appears broken (and hence
        # whether the expensive priced-relaxation ladder would be worth running). Reporting-only here; the
        # actual priced relaxation on discovered latent symmetries runs via _deploy_approx_equivariant.
        if getattr(self, "price_equivariance", False) and self.contract == "equivariant" and self.net is not None:
            probe = self._equivariance_breaking_probe(data, task)
            if probe is not None:
                result["equivariance_breaking"] = probe
                if probe.get("symmetry_broken"):
                    self._log(f"[AllGraph] equivariance-breaking probe: symmetry appears BROKEN "
                              f"(signal={probe['breaking_signal']}); priced relaxation may help.")
                else:
                    self._log(f"[AllGraph] equivariance-breaking probe: symmetry appears intact "
                              f"(signal={probe['breaking_signal']}); strict equivariance is appropriate.")
        return result

    def _llc_closure(self, net, data, task, batch=128):
        """Build a mean-loss closure()->scalar for `net` at its CURRENT params over a FIXED training minibatch
        (minibatch is standard for SGLD-LLC), matching the contract's forward. Returns (closure, n_total).
        Shared by the static LLC read-out (_llc_report) and the developmental read-out (_developmental_report);
        the closure closes over `net` by reference, so as `net` trains the SAME closure tracks its trajectory.
        The fixed minibatch is seeded (RandomState(self.seed)) so every checkpoint's probe uses the same
        subsample -- the developmental curve is then a clean trajectory, not confounded by minibatch churn."""
        mod = self.contract
        y = np.asarray(data.y)
        lossf = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        n_total = (len(data.dense) if mod in ("sequence", "spatial", "volumetric", "4d", "operator")
                   else len(data.node_feats))
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n_total)[:min(batch, n_total)]
        yt = torch.as_tensor(y)

        if mod in ("sequence", "spatial", "volumetric", "4d"):
            X = data.dense
            if mod == "sequence" and X.dim() == 2:
                X = X.unsqueeze(-1)
            Xb = X[idx].to(self.device)
            tb = (yt[idx].long().to(self.device) if task == "classification"
                  else yt[idx].float().unsqueeze(1).to(self.device))

            def closure():
                out = (net.forward_seq_readout(Xb, 1).squeeze(1) if mod == "sequence" else net(Xb))
                return lossf(out, tb)
        elif mod == "operator":
            # operator forward is net(a, coords) -> field; the loss is a per-grid-point field MSE, matching
            # _fit_operator. LLC on the operator net makes the mode-budget/width complexity singularity-aware
            # like the other contracts.
            a = data.dense if isinstance(data.dense, torch.Tensor) else torch.as_tensor(np.asarray(data.dense), dtype=torch.float32)
            gx = data.grid if isinstance(data.grid, torch.Tensor) else torch.as_tensor(np.asarray(data.grid), dtype=torch.float32)
            ab = a[idx].to(self.device); gb = gx[idx].to(self.device)
            tb = yt[idx].float().to(self.device)

            def closure():
                return lossf(net(ab, gb), tb)
        else:
            with_pos = (mod == "equivariant"); use_edges = mod in ("graph", "equivariant")
            ids = np.asarray(idx)
            tb = (yt[ids].long().to(self.device) if task == "classification"
                  else yt[ids].float().unsqueeze(1).to(self.device))

            def closure():
                out = self._forward_contract(net, data, ids, mod, with_pos, use_edges, 3.0)
                return lossf(out, tb)

        return closure, n_total

    def _thermo_report(self, data, result):
        """D2: record the single thermodynamic potential's three-level temperature hierarchy and assert the
        temperatures sit at their principled values (beta_W=1/log n, beta_C=1, beta_A=gibbs_beta a free knob)
        without being coupled. Diagnostic; returns the hierarchy dict + the consistency result, or None on
        failure (best-effort). Does not change selection."""
        try:
            from ..machinery.thermodynamic_potential import (assert_temperature_consistency, wbic_beta,
                                                             POTENTIAL_LEVELS)
            # n = training-set size (fixes beta_W); resolve the actual Level-A temperature in use.
            n_total = (len(data.dense) if self.contract in ("sequence", "spatial", "volumetric", "4d", "operator")
                       else (len(data.node_feats) if data.node_feats is not None else len(data.y)))
            # gibbs_beta may be 'auto' (resolved per-fit); if the gibbs path ran it is echoed in the result,
            # else fall back to the configured value (numeric default 8.0). 'auto' with no gibbs run -> use 8.0.
            gb = self.gibbs_beta
            if isinstance(gb, str):
                gb = 8.0  # 'auto' resolves to an elbow value only inside a gibbs reselect; report the default
            check = assert_temperature_consistency(n_total, float(gb), contract_beta=1.0, raise_on_fail=False)
            if not check["ok"]:
                self._log(f"[AllGraph] thermodynamic consistency WARNING: {' | '.join(check['issues'])}")
            else:
                self._log(f"[AllGraph] thermodynamic potential: beta_W=1/log n={check['beta_W']:.4f} (weights), "
                          f"beta_A=gibbs_beta={check['beta_A']:.2f} (primitives), beta_C=1 (contracts) -- one "
                          f"free-energy form, three decoupled temperatures.")
            return {"levels": list(POTENTIAL_LEVELS), "consistency": check, "n": int(n_total),
                    "beta_W": wbic_beta(n_total), "beta_A": float(gb), "beta_C": 1.0}
        except Exception as e:
            self._log(f"[AllGraph] thermodynamic report skipped ({str(e)[:70]})")
            return None

    def _response_report(self, data, result):
        """D5: report the curvature of the selection objective at the chosen point -- how sharply the
        argmin is preferred and how far to the nearest selection boundary. Two channels, each reusing
        quantities the fit ALREADY computed (NO retraining):

          * readout (Level A): the primitive Gibbs-alpha's specific heat chi = Var_alpha(Psi) and the
            monotone entropy sharpness, from result["gibbs_energies"] + the resolved gibbs_beta.
          * contract (Level C): the first-order transition spectroscopy (critical price mu*, price
            margins to flip, slope jump) from the tie-break detail {risk, omega_struct, mu_c} stashed
            in self.route_detail["tiebreak"] when a contract bake-off ran.

        A run may populate one channel, both, or neither (a rule-routed run with a single primitive has
        nothing perturbable). Diagnostic; returns the spectrum dict, or None on failure/empty."""
        try:
            from ..machinery.response_spectroscopy import response_spectrum
            # --- readout channel: solo energies + the beta actually used ---
            energies = result.get("gibbs_energies")
            beta = None
            if energies:
                # the readout ran; recover the temperature it used (mirrors _resolve_gibbs_beta's output
                # which is echoed nowhere, so recompute from the same scores it saw)
                gb = self.gibbs_beta
                if isinstance(gb, str):
                    # 'auto' resolves to an elbow on the solo scores; reuse the resolver on the energies
                    scores_from_e = {p: -float(v) for p, v in energies.items()}
                    prims = list(scores_from_e)
                    beta = self._resolve_gibbs_beta(scores_from_e, prims)
                else:
                    beta = float(gb)
            # --- contract channel: the tie-break detail, if a PRICED bake-off ran ---
            # The bake-off only prices contracts (populating an MDL sub-detail with risk/omega/mu_c) when
            # the learned router does NOT short-circuit it; a router-shortcut or single-candidate route
            # leaves nothing to compute a susceptibility of, and this channel then stays empty (honest).
            scores = omegas = mu_c = None
            rd = getattr(self, "route_detail", None)
            if isinstance(rd, dict):
                tb = rd.get("tiebreak")
                if isinstance(tb, dict):
                    # the priced objective lives in tb["mdl"] (the select_contract_mdl return): it carries
                    # risk, omega_struct, mu_c directly. Fall back to top-level keys if ever present.
                    src = tb.get("mdl") if isinstance(tb.get("mdl"), dict) else tb
                    if isinstance(src, dict) and "omega_struct" in src and "risk" in src \
                            and len(src["omega_struct"]) >= 2:
                        scores = {c: 1.0 - float(r) for c, r in src["risk"].items()}
                        omegas = {c: float(o) for c, o in src["omega_struct"].items()}
                        mu_c = float(src.get("mu_c", self.sparsity_mu or 0.05))
            spec = response_spectrum(energies=energies, beta=beta,
                                     scores=scores, omegas=omegas, mu_c=mu_c)
            if spec.get("readout") is None and spec.get("contract") is None:
                return None  # nothing perturbable to report
            self._log(f"[AllGraph] response spectroscopy: {spec['summary']}")
            return spec
        except Exception as e:
            self._log(f"[AllGraph] response report skipped ({str(e)[:70]})")
            return None

    def _ledger_report(self, data, result):
        """D3: assemble the effective-dimension ledger from already-computed pieces onto one
        coarse-graining axis (machinery.effective_dimension_ledger). Reuses: the deployed alpha (mixture
        level), an LLC if report_llc ran (model level), and a cheap data covariance spectrum (data-modes
        level). No selection is changed; a level is omitted when its input is unavailable. Best-effort;
        returns the ledger dict or None."""
        try:
            from ..machinery.effective_dimension_ledger import effective_dimension_ledger
            import numpy as _np
            # --- data-modes level: covariance spectrum of the input features (cheap) ---
            cov_spectrum = None
            X = None
            if getattr(data, "dense", None) is not None:
                X = _np.asarray(data.dense, dtype=_np.float64)
                X = X.reshape(len(X), -1) if X.ndim > 2 else X
            elif getattr(data, "node_feats", None) is not None:
                # stack node features across the batch into (total_nodes, feat) for a covariance
                try:
                    X = _np.concatenate([_np.asarray(nf, dtype=_np.float64) for nf in data.node_feats], axis=0)
                except Exception:
                    X = None
            if X is not None and X.ndim == 2 and X.shape[0] > 2 and X.shape[1] >= 1:
                Xc = X - X.mean(axis=0, keepdims=True)
                # covariance eigenvalues via SVD (numerically stable), same convention as effective_dimension
                s = _np.linalg.svd(Xc, compute_uv=False)
                cov_spectrum = (s ** 2) / max(len(X) - 1, 1)
            # --- primitive-mixture level: the deployed mixture weights w = softmax(alpha) ---
            weights = result.get("gibbs_weights")
            alpha_vec = None
            if isinstance(weights, dict) and len(weights) > 0:
                alpha_vec = _np.asarray(list(weights.values()), dtype=_np.float64)
            # --- model level: an LLC if report_llc attached one ---
            llc = result.get("llc") if isinstance(result.get("llc"), dict) else None
            ledger = effective_dimension_ledger(cov_spectrum=cov_spectrum, alpha=alpha_vec, llc=llc)
            if not ledger["levels"]:
                return None
            summary = "; ".join(f"{lv['level']}={lv['value']:.2f} {lv['unit']}"
                                for lv in ledger["levels"] if lv.get("value") is not None)
            self._log(f"[AllGraph] effective-dimension ledger: {summary}")
            return ledger
        except Exception as e:
            self._log(f"[AllGraph] ledger report skipped ({str(e)[:70]})")
            return None

    def _developmental_report(self, data, task, batch=128, chains=3, steps=100, burn=40):
        """D4: developmental read-out of the deployed architecture. Re-trains a FRESH copy of the selected
        net for one trajectory and probes lambda_hat at checkpoints (machinery.developmental_llc), returning
        the developmental curve + located transitions, or None on failure (best-effort, like _llc_report).

        Whereas _llc_report reads lambda ONCE at convergence, this reads lambda(t) OVER training: the located
        negative->positive onset marks where the architecture's usable capacity turns on; plateaus/jumps mark
        staged learning (advisory -- present only when the data induces separated phases). Diagnostic only;
        does not change selection. Guarded to the standard contracts with a rebuildable net + .parameters();
        skips generated-equivariant dict-net paths (no single rebuildable module). The retrain is independent
        of the deployed weights -- this is a read-out of the SELECTED architecture's learning dynamics."""
        try:
            from ..machinery.developmental_llc import developmental_llc as _dev_llc
            mod = self.contract
            builder = self._selected_net_builder(data, task)
            if builder is None:
                self._log(f"[AllGraph] developmental LLC skipped (no rebuildable net for contract '{mod}')")
                return None

            # per-checkpoint SGLD probe budget: modest (the curve is probed many times); relational forwards
            # are expensive so keep it lean, matching _llc_report's economy.
            def make_closure(net):
                closure, _ = self._llc_closure(net, data, task, batch)
                return closure

            def train_step(net, opt):
                closure, _ = self._llc_closure(net, data, task, batch)
                opt.zero_grad()
                loss = closure()
                loss.backward()
                opt.step()
                return float(loss.item())

            # n_total from a throwaway closure build (cheap; just reads lengths)
            _, n_total = self._llc_closure(builder(), data, task, batch)
            # trajectory length: a convergence-scale budget (LLC needs a converged w* to become valid). Default
            # to a healthy multiple of the fit budget so the onset is actually reached within the trajectory.
            te = self.developmental_llc_epochs or max(self.epochs * 4, 120)

            out = _dev_llc(builder, make_closure, train_step, n_total,
                           total_epochs=te, checkpoints=self.developmental_llc_checkpoints,
                           chains=chains, steps=steps, burn=burn, eps=2e-5, gamma=100.0, seed=self.seed)
            tr = out.get("transitions", {})
            onset = tr.get("convergence_onset_epoch")
            njumps = len(tr.get("candidate_staged_jumps", []))
            if onset is not None:
                self._log(f"[AllGraph] developmental LLC: convergence onset at epoch {onset} "
                          f"(lambda flips negative->positive; usable capacity turns on); "
                          f"{njumps} candidate staged jump(s); final lambda={out['final']['lambda']:.2f} "
                          f"(k/2={out['half_params']:.0f}).")
            else:
                self._log(f"[AllGraph] developmental LLC: no stable convergence onset within "
                          f"{out['checkpoints'][-1]} epochs (net may need a longer trajectory to converge); "
                          f"curve reported for inspection.")
            return out
        except Exception as e:
            self._log(f"[AllGraph] developmental LLC skipped ({str(e)[:70]})")
            return None

    def _llc_report(self, net, data, task, batch=128, chains=3, steps=120, burn=50):
        """Estimate the deployed net's Local Learning Coefficient (approx. RLCT) -- a singularity-aware
        complexity <= n_params/2 -- via SGLD (machinery.singular_complexity). Builds a mean-loss closure over
        a fixed minibatch of the training data (minibatch is standard for SGLD-LLC) matching the contract's
        forward, then reads lambda. Diagnostic only; returns the estimator dict augmented with the singular
        free energy, or None on failure (LLC is best-effort). The SGLD budget (chains/steps/batch) is kept
        modest by default since relational forwards are expensive; raise it for a lower-variance estimate."""
        try:
            from ..machinery.singular_complexity import estimate_llc, free_energy
            closure, n_total = self._llc_closure(net, data, task, batch)
            r = estimate_llc(net, closure, n_total, chains=chains, steps=steps, burn=burn, eps=2e-5,
                             gamma=100.0, seed=self.seed)
            r["free_energy_singular"] = free_energy(r["L_star"], r["lambda"], n_total)
            r["free_energy_bic"] = free_energy(r["L_star"], r["half_params"], n_total)
            if r.get("valid", True):
                self._log(f"[AllGraph] LLC lambda={r['lambda']:.2f}+/-{r['lambda_std']:.2f} "
                          f"(k/2={r['half_params']:.0f}, ratio={r['ratio']:.4f}) -> singular complexity << param count")
            else:
                self._log(f"[AllGraph] LLC lambda={r['lambda']:.2f} is negative -> deployed net not at a "
                          f"converged minimum at this epoch budget; LLC needs a converged w* (train longer). "
                          f"Reported as invalid.")
            return r
        except Exception as e:
            self._log(f"[AllGraph] LLC report skipped ({str(e)[:70]})")
            return None

    def _equivariance_breaking_probe(self, data, task):
        """Cheap diagnostic (direction B5): after a strict-equivariant fit, test whether the residual the
        equivariant model CANNOT represent is predictable from a NON-invariant feature (the coordinate sum, a
        vector whose components break SO(3)) beyond what an invariant feature (radius) explains. A positive
        'breaking_signal' flags that the data's symmetry is broken -- computed from the already-trained net by
        a single linear least-squares probe, so it is nearly free and can gate the expensive priced-relaxation
        ladder. Returns a dict, or None if positions are unavailable / on failure."""
        try:
            if getattr(data, "positions", None) is None:
                return None
            y = np.asarray(data.y, dtype=np.float32)
            ids = np.arange(len(data.node_feats))
            with torch.no_grad():
                pred = self._forward_contract(self.net, data, ids, self.contract, True, True, 3.0).squeeze(-1).cpu().numpy()
            if task == "classification":
                return None                       # probe is defined for regression residuals
            resid = y[:len(pred)] - pred
            resid = (resid - resid.mean()) / (resid.std() + 1e-9)
            positions = [np.asarray(p, dtype=np.float32) for p in data.positions][:len(pred)]
            dip = np.stack([p.sum(0) for p in positions])                      # non-invariant vector
            inv = np.stack([np.sqrt((p ** 2).sum()) for p in positions]).reshape(-1, 1)  # invariant control

            def explained_var(X):
                X = (X - X.mean(0)) / (X.std(0) + 1e-9)
                Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
                beta, _, _, _ = np.linalg.lstsq(Xb, resid, rcond=None)
                pr = Xb @ beta
                return float(1 - ((resid - pr) ** 2).sum() / ((resid ** 2).sum() + 1e-9))

            ev_ni = explained_var(dip); ev_i = explained_var(inv)
            signal = ev_ni - ev_i
            broken = signal > 0.1                  # non-invariant explains residual appreciably beyond invariant
            return {"breaking_signal": round(signal, 3), "resid_explained_noninvariant": round(ev_ni, 3),
                    "resid_explained_invariant": round(ev_i, 3), "symmetry_broken": bool(broken),
                    "note": "residual of the strict-equivariant fit probed by a non-invariant vs invariant "
                            "feature; positive breaking_signal => symmetry appears broken (consider "
                            "price_equivariance). Cheap: one linear probe on the trained net."}
        except Exception as e:
            self._log(f"[AllGraph] equivariance-breaking probe skipped ({str(e)[:60]})")
            return None
