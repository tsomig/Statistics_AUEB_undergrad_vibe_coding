"""
diafaneia_data.py
=================
Δομικό υπόσχεμα της εργασίας «Δράση Διαφάνεια — Βραχυχρόνια Μίσθωση».
Δημιουργεί τα προσωποποιημένα δεδομένα και τις σταθερές της εκφώνησης
για κάθε ΑΜ φοιτητή/τριας.

Χρήση από φοιτητή (μέσα στο Colab notebook):

    from diafaneia_data import load_assignment
    AM = "1250072"
    df, constants = load_assignment(AM)

Επιστρέφει:
    df         — pandas DataFrame με στήλες property_id, declared_revenue,
                 electricity_kwh
    constants  — dict με τις σταθερές της εκφώνησης (N, TAU, K, W0, PI, P, Q)

Σημείωση: ο σπόρος SHA-256(AM) είναι ντετερμινιστικός, οπότε ο καθηγητής
μπορεί να αναπαράγει τα ίδια ακριβώς δεδομένα και απαντήσεις από την
πλευρά του για τους σκοπούς της βαθμολόγησης.
"""

import hashlib
import numpy as np
import pandas as pd


def _seed_from_am(am: str) -> int:
    """SHA-256 του ΑΜ → ακέραιος 32-bit για χρήση ως seed."""
    h = hashlib.sha256(str(am).strip().encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _parameters_for_am(am: str) -> dict:
    """
    Παράγει όλες τις παραμέτρους της εργασίας για δοθέν ΑΜ.
    Όλα τα εύρη είναι έτσι επιλεγμένα ώστε να εξασφαλίζεται αριθμητική
    ευστάθεια και διδακτικά ενδιαφέροντα αποτελέσματα.
    """
    rng = np.random.default_rng(_seed_from_am(am))

    # --- Πληθυσμιακές παράμετροι (Y, W) = (log-έσοδα, log-κατανάλωση) ---
    mu_Y    = rng.uniform(9.20, 9.55)
    sigma_Y = rng.uniform(0.55, 0.72)
    mu_W    = rng.uniform(8.45, 8.95)
    sigma_W = rng.uniform(0.50, 0.65)
    rho     = rng.uniform(0.58, 0.78)

    N = int(rng.choice([200, 230, 260, 290]))

    # σταθερές που παρουσιάζονται στον φοιτητή
    tau = int(rng.choice([6500, 7000, 7500, 8000, 8500, 9000]))
    k_chebyshev = float(rng.choice([1.5, 1.75, 2.0, 2.25, 2.5]))
    w0_log_target = mu_W + 0.5 * sigma_W
    w0_kwh = int(round(np.exp(w0_log_target) / 100) * 100)

    pi_prior    = round(float(rng.uniform(0.20, 0.32)), 2)
    sensitivity = round(float(rng.uniform(0.78, 0.90)), 2)
    specificity = round(float(rng.uniform(0.65, 0.80)), 2)

    return {
        "AM": str(am),
        "_seed": _seed_from_am(am),
        "_mu_Y": mu_Y, "_sigma_Y": sigma_Y,
        "_mu_W": mu_W, "_sigma_W": sigma_W,
        "_rho": rho,
        "N": N,
        "TAU": tau,
        "K": k_chebyshev,
        "W0": w0_kwh,
        "PI": pi_prior,
        "P": sensitivity,
        "Q": specificity,
    }


def _generate_data(params: dict) -> pd.DataFrame:
    """Δειγματοληψία (Y, W) από διμεταβλητή κανονική, μετατροπή σε ευρώ/kWh."""
    rng = np.random.default_rng(params["_seed"] + 1)
    mean = np.array([params["_mu_Y"], params["_mu_W"]])
    cov = np.array([
        [params["_sigma_Y"]**2,
         params["_rho"] * params["_sigma_Y"] * params["_sigma_W"]],
        [params["_rho"] * params["_sigma_Y"] * params["_sigma_W"],
         params["_sigma_W"]**2]
    ])
    samples = rng.multivariate_normal(mean, cov, size=params["N"])
    Y, W = samples[:, 0], samples[:, 1]

    return pd.DataFrame({
        "property_id":      [f"P{i+1:04d}" for i in range(params["N"])],
        "declared_revenue": np.round(np.exp(Y), 2),
        "electricity_kwh":  np.round(np.exp(W), 1),
    })


def load_assignment(am):
    """
    Κύρια συνάρτηση. Φορτώνει τα δεδομένα και τις σταθερές για δοθέν ΑΜ.

    Parameters
    ----------
    am : str
        Ο Αριθμός Μητρώου του φοιτητή/τριας.

    Returns
    -------
    df : pandas.DataFrame
        Δεδομένα ακινήτων (στήλες: property_id, declared_revenue, electricity_kwh)
    constants : dict
        Σταθερές της εκφώνησης: AM, N, TAU, K, W0, PI, P, Q
    """
    am_str = str(am).strip()
    if not am_str or am_str == "XXXXX":
        raise ValueError(
            "Παρακαλώ συμπληρώστε τον ΑΜ σας στη μεταβλητή AM στην αρχή του notebook."
        )

    params = _parameters_for_am(am_str)
    df = _generate_data(params)

    # Εξαγωγή μόνο των δημόσιων σταθερών (όχι των κρυφών παραμέτρων)
    constants = {k: v for k, v in params.items() if not k.startswith("_")}

    return df, constants


# Συνάρτηση ground truth — χρησιμοποιείται ΜΟΝΟ από τον καθηγητή
def _compute_ground_truth(am):
    """Υπολογισμός σωστών απαντήσεων (μόνο για βαθμολογητή)."""
    params = _parameters_for_am(str(am).strip())
    df = _generate_data(params)

    X = df["declared_revenue"].values
    E = df["electricity_kwh"].values
    Y = np.log(X)
    W = np.log(E)

    mean_X = float(np.mean(X))
    mean_Y = float(np.mean(Y))
    var_Y  = float(np.var(Y, ddof=1))
    cdf_X_at_tau = float(np.mean(X <= params["TAU"]))

    k = params["K"]
    chebyshev_bound = float(1.0 / k**2)
    sd_Y = float(np.std(Y, ddof=1))
    empirical_fraction = float(np.mean(np.abs(Y - mean_Y) >= k * sd_Y))

    pi, p, q = params["PI"], params["P"], params["Q"]
    posterior = (pi * p) / (pi * p + (1 - pi) * (1 - q))

    cov_YW = float(np.cov(Y, W, ddof=1)[0, 1])
    var_W  = float(np.var(W, ddof=1))
    rho_YW = float(np.corrcoef(Y, W)[0, 1])
    slope  = cov_YW / var_W

    W0 = float(np.log(params["W0"]))
    mean_W = float(np.mean(W))
    cond_mean = mean_Y + (cov_YW / var_W) * (W0 - mean_W)
    cond_var  = var_Y - cov_YW**2 / var_W
    cond_sd   = float(np.sqrt(cond_var))
    audit_threshold = cond_mean - 1.645 * cond_sd

    beta1   = slope
    beta0   = mean_Y - beta1 * mean_W
    resids  = Y - (beta0 + beta1 * W)
    emp_q05 = float(np.quantile(resids, 0.05))

    threshold_per_property = beta0 + beta1 * W - 1.645 * cond_sd
    n_flagged_model     = int(np.sum(Y < threshold_per_property))
    n_flagged_empirical = int(np.sum(resids < emp_q05))

    return {
        "q1_mean_X": mean_X, "q1_mean_Y": mean_Y, "q1_var_Y": var_Y,
        "q1_cdf_X_at_tau": cdf_X_at_tau,
        "q2_chebyshev_bound": chebyshev_bound,
        "q2_empirical_frac": empirical_fraction,
        "q3_posterior": posterior,
        "q4_cov_YW": cov_YW, "q4_rho_YW": rho_YW, "q4_slope": slope,
        "q5_cond_mean": cond_mean, "q5_cond_sd": cond_sd,
        "q5_audit_threshold_w0": audit_threshold,
        "q5_emp_quantile_residuals": emp_q05,
        "q5_n_flagged_model": n_flagged_model,
        "q5_n_flagged_empirical": n_flagged_empirical,
    }
