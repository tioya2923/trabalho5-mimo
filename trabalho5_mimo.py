#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho 5 — Capacidade de canais MIMO
=======================================

Simulação de Monte Carlo da capacidade de canais MIMO com desvanecimento de
Rayleigh (i.i.d. e com correlação no emissor), calculada por decomposição em
valores singulares (SVD), e comparação com o canal SISO equivalente.

Estrutura do script
-------------------
  Parte 1  Funções de base: geração do canal, capacidade (det e SVD)
  Parte 2  Verificação numa única realização (passos 1 a 3 do enunciado)
  Parte 3  Capacidade ergódica SISO, 2x2 e 4x4 (passos 4 e 5; questões 1 e 2)
  Parte 4  Canal correlacionado — modelo de Kronecker (questão 3)
  Parte 5  Tabelas de resultados e figuras

Requisitos: Python >= 3.9, numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
Execução:
    python trabalho5_mimo.py
Saídas: figuras fig*.png e ficheiro resultados.json, gravados na pasta deste
        script (seja qual for a pasta do terminal); no fim, as figuras são
        também abertas em janelas.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma

# Pasta onde este script está: é aí que se gravam as figuras e o JSON,
# independentemente da pasta em que o terminal se encontra.
PASTA = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Parâmetros globais
# ---------------------------------------------------------------------------
SEMENTE = 2026                             # semente -> resultados reprodutíveis
N_MC = 20_000                              # n.º de realizações de Monte Carlo
SNR_DB = np.arange(-10, 41, 2)             # gama de SNR (dB): -10, -8, ..., 40
SNR = 10.0 ** (SNR_DB / 10.0)              # SNR em escala linear
RHOS = [0.0, 0.5, 0.9, 0.99, 1.0]          # coeficientes de correlação (Q3)

rng = np.random.default_rng(SEMENTE)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.35,
                     "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


# ===========================================================================
# Parte 1 — Funções de base
# ===========================================================================
def canal_rayleigh(n_r, n_t, n_real=1):
    """Gera `n_real` matrizes n_r x n_t com entradas i.i.d. CN(0, 1).

    Cada entrada é (a + j b)/sqrt(2), com a, b ~ N(0, 1) independentes, o que
    garante E[|h_ij|^2] = 1 (mesmo ganho médio de potência que o canal SISO).
    """
    forma = (n_real, n_r, n_t)
    return (rng.standard_normal(forma)
            + 1j * rng.standard_normal(forma)) / np.sqrt(2.0)


def capacidade_det(H, snr_lin):
    """Capacidade (bit/s/Hz) de UMA matriz H pela fórmula do determinante:
        C = log2 det( I_Nr + (SNR/Nt) H H^H )
    """
    n_r, n_t = H.shape
    M = np.eye(n_r) + (snr_lin / n_t) * (H @ H.conj().T)
    # slogdet é numericamente mais estável do que log(det(.))
    sinal, logdet = np.linalg.slogdet(M)
    return logdet / np.log(2.0)


def capacidade_svd(sigma, n_t, snr_lin):
    """Capacidade a partir dos valores singulares (soma dos modos próprios):
        C = sum_i log2( 1 + (SNR/Nt) sigma_i^2 )
    `sigma` pode ter dimensões arbitrárias; a soma faz-se no último eixo.
    """
    return np.sum(np.log2(1.0 + (snr_lin / n_t) * sigma ** 2), axis=-1)


def raiz_matriz_hermitiana(R):
    """Raiz quadrada (R^{1/2}) de uma matriz hermitiana semidefinida positiva,
    via decomposição em valores próprios (funciona também quando R é
    singular, como para rho = 1)."""
    w, V = np.linalg.eigh(R)
    return V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T


def simular(n_r, n_t, R_t=None):
    """Monte Carlo: devolve (C, sigma) com
         C     -> matriz (N_MC, n_SNR) de capacidades instantâneas
         sigma -> matriz (N_MC, min(n_r, n_t)) de valores singulares
    Se R_t for dado, usa o modelo de Kronecker H = H_w R_t^{1/2}.
    A mesma realização de H é usada para todas as SNR (curvas mais suaves).
    """
    H = canal_rayleigh(n_r, n_t, N_MC)
    if R_t is not None:
        H = H @ raiz_matriz_hermitiana(R_t)
    sigma = np.linalg.svd(H, compute_uv=False)
    C = capacidade_svd(sigma[:, None, :], n_t, SNR[None, :, None])
    return C, sigma


def resumo(C):
    """Média (capacidade ergódica), semi-amplitude do IC a 95 % e percentis."""
    media = C.mean(axis=0)
    ic95 = 1.96 * C.std(axis=0, ddof=1) / np.sqrt(C.shape[0])
    p10, p90 = np.percentile(C, [10, 90], axis=0)
    return media, ic95, p10, p90


def assintota_alta_snr(n):
    """Aproximação de alta SNR da capacidade ergódica de um canal n x n i.i.d.:
        C ≈ n log2(SNR/n) + E[log2 det(H H^H)],
    com E[ln det W] = sum_{i=0}^{n-1} psi(n - i) para W Wishart complexa
    n x n com n graus de liberdade (psi = função digama)."""
    e_logdet = sum(digamma(n - i) for i in range(n)) / np.log(2.0)
    return n * np.log2(SNR / n) + e_logdet, e_logdet


def idx(snr_db):
    """Índice de um valor de SNR (dB) no vetor SNR_DB."""
    return int(np.where(SNR_DB == snr_db)[0][0])


# ===========================================================================
# Parte 2 — Uma única realização (passos 1 a 3)
# ===========================================================================
H1 = canal_rayleigh(2, 2)[0]
sig1 = np.linalg.svd(H1, compute_uv=False)
C1_det = capacidade_det(H1, SNR[idx(30)])
C1_svd = capacidade_svd(sig1, 2, SNR[idx(30)])
print("=== Realização única (2x2) ===")
print("H =\n", np.round(H1, 4))
print("sigma =", np.round(sig1, 4), "| kappa =", round(sig1[0] / sig1[1], 3))
print("soma sigma^2 =", round(float(np.sum(sig1 ** 2)), 6),
      "| ||H||_F^2 =", round(float(np.sum(np.abs(H1) ** 2)), 6))
print(f"C(30 dB): det = {C1_det:.6f} | SVD = {C1_svd:.6f} bit/s/Hz")

# Validação sistemática: det vs SVD em 2000 realizações e todas as SNR
Hv = canal_rayleigh(2, 2, 2000)
sv = np.linalg.svd(Hv, compute_uv=False)
err_max = max(abs(capacidade_det(Hv[k], s) - capacidade_svd(sv[k], 2, s))
              for k in range(2000) for s in SNR)
print(f"Erro máximo |C_det - C_svd| em 2000 x {len(SNR)} casos: {err_max:.2e}")

# ===========================================================================
# Parte 3 — Capacidade ergódica SISO, 2x2 e 4x4
# ===========================================================================
C_siso_awgn = np.log2(1.0 + SNR)                       # referência do enunciado
C_s, _ = simular(1, 1)                                 # SISO com Rayleigh
C_22, sig22 = simular(2, 2)
C_44, sig44 = simular(4, 4)

m_s, ic_s, p10_s, p90_s = resumo(C_s)
m22, ic22, p10_22, p90_22 = resumo(C_22)
m44, ic44, p10_44, p90_44 = resumo(C_44)
ass22, e22 = assintota_alta_snr(2)
ass44, e44 = assintota_alta_snr(4)

def declive(c, a=30, b=40):
    """Declive em bit/s/Hz por 3 dB entre a e b dB (= graus de liberdade)."""
    return (c[idx(b)] - c[idx(a)]) / ((b - a) / 10 * np.log2(10))

# ===========================================================================
# Parte 4 — Canal correlacionado (modelo de Kronecker, correlação no emissor)
# ===========================================================================
corr = {}
for rho in RHOS:
    R_t = np.array([[1.0, rho], [rho, 1.0]])
    C_r, sig_r = simular(2, 2, R_t)
    with np.errstate(divide="ignore"):
        kappa_db = 20 * np.log10(sig_r[:, 0] / sig_r[:, 1])
    corr[rho] = dict(C=C_r, media=C_r.mean(axis=0), kappa_db=kappa_db,
                     lam2_lam1=np.mean(sig_r[:, 1] ** 2 / sig_r[:, 0] ** 2))

# ===========================================================================
# Parte 5 — Tabelas e figuras
# ===========================================================================
print("\n=== Capacidade ergódica (bit/s/Hz), média ± IC 95 % ===")
print(" SNR   AWGN   SISO-Ray     2x2            4x4       2x2/AWGN 4x4/AWGN")
for s in (-10, 0, 10, 20, 30, 40):
    i = idx(s)
    print(f"{s:4d} {C_siso_awgn[i]:6.2f} {m_s[i]:7.2f}  {m22[i]:6.2f}±{ic22[i]:.2f}"
          f"  {m44[i]:6.2f}±{ic44[i]:.2f}  {m22[i]/C_siso_awgn[i]:6.2f}"
          f"  {m44[i]/C_siso_awgn[i]:6.2f}")
print("\nDeclive 30-40 dB (bit/s/Hz por 3 dB):",
      f"AWGN {declive(C_siso_awgn):.3f} | 2x2 {declive(m22):.3f} | "
      f"4x4 {declive(m44):.3f}")
print(f"Assíntota 2x2 a 30 dB: {ass22[idx(30)]:.2f} (E log2det = {e22:.4f})")
print(f"Assíntota 4x4 a 30 dB: {ass44[idx(30)]:.2f} (E log2det = {e44:.4f})")
print("\n=== Correlação (2x2) ===")
print(" rho  kappa_med(dB)  E[l2/l1]  C10   C20   C30   decl.30-40")
for rho in RHOS:
    d = corr[rho]
    kd = np.median(d["kappa_db"])
    print(f"{rho:4.2f}  {kd:10.1f}  {d['lam2_lam1']:8.4f}  "
          f"{d['media'][idx(10)]:5.2f} {d['media'][idx(20)]:5.2f} "
          f"{d['media'][idx(30)]:5.2f}  {declive(d['media']):6.3f}")

# Outage a 10 dB: capacidade garantida em 90 % das realizações (percentil 10)
i10 = idx(10)
print("\nCapacidade de outage 10 % a 10 dB:",
      f"SISO {p10_s[i10]:.2f} | 2x2 {p10_22[i10]:.2f} | 4x4 {p10_44[i10]:.2f}")

# --- Guarda resultados numéricos para o relatório ---------------------------
res = {
    "semente": SEMENTE, "N_MC": N_MC,
    "unica": {"H": [[str(np.round(z, 4)) for z in row] for row in H1],
              "sigma": sig1.tolist(), "C_det": C1_det, "C_svd": C1_svd,
              "normaF2": float(np.sum(np.abs(H1) ** 2))},
    "err_max": err_max,
    "tabela": {int(s): {"awgn": C_siso_awgn[idx(s)], "siso": m_s[idx(s)],
                        "m22": m22[idx(s)], "ic22": ic22[idx(s)],
                        "m44": m44[idx(s)], "ic44": ic44[idx(s)],
                        "a22": ass22[idx(s)], "a44": ass44[idx(s)]}
               for s in SNR_DB},
    "declive": {"awgn": declive(C_siso_awgn), "siso": declive(m_s),
                "m22": declive(m22), "m44": declive(m44)},
    "e_logdet": {"2": e22, "4": e44},
    "outage10": {"siso": p10_s[i10], "m22": p10_22[i10], "m44": p10_44[i10]},
    "corr": {str(r): {"kappa_med_db": float(np.median(corr[r]["kappa_db"])),
                      "lam_ratio": float(corr[r]["lam2_lam1"]),
                      "C": {int(s): float(corr[r]["media"][idx(s)])
                            for s in (0, 10, 20, 30, 40)},
                      "declive": float(declive(corr[r]["media"]))}
             for r in RHOS},
}
with open(PASTA / "resultados.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, default=float)

# --- Figura 1: SISO vs MIMO 2x2 (com faixa 10–90 %) --------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(SNR_DB, C_siso_awgn, "k--", label="SISO AWGN: log$_2$(1+SNR)")
ax.plot(SNR_DB, m_s, "o-", ms=3.5, c="tab:blue", label="SISO Rayleigh (ergódica)")
ax.plot(SNR_DB, m22, "s-", ms=3.5, c="tab:orange", label="MIMO 2×2 (ergódica)")
ax.fill_between(SNR_DB, p10_22, p90_22, color="tab:orange", alpha=0.18,
                label="MIMO 2×2: percentis 10–90 %")
ax.set(xlabel="SNR (dB)", ylabel="Capacidade (bit/s/Hz)", xlim=(-10, 40))
ax.legend(loc="upper left")
fig.savefig(PASTA / "fig1_siso_vs_2x2.png")

# --- Figura 2: razão C_MIMO / C_SISO ----------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(SNR_DB, m22 / C_siso_awgn, "s-", ms=3.5, c="tab:orange", label="2×2 / SISO")
ax.plot(SNR_DB, m44 / C_siso_awgn, "^-", ms=3.5, c="tab:green", label="4×4 / SISO")
ax.axhline(2, ls=":", c="tab:orange"); ax.axhline(4, ls=":", c="tab:green")
ax.set(xlabel="SNR (dB)", ylabel="C$_{\\mathrm{MIMO}}$ / C$_{\\mathrm{SISO}}$",
       xlim=(-10, 40), ylim=(1, 4.5))
ax.legend(loc="lower right")
fig.savefig(PASTA / "fig2_razao.png")

# --- Figura 3: SISO, 2x2, 4x4 + assíntotas de alta SNR -----------------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(SNR_DB, C_siso_awgn, "o-", ms=3.5, c="tab:blue", label="SISO (AWGN)")
ax.plot(SNR_DB, m22, "s-", ms=3.5, c="tab:orange", label="MIMO 2×2")
ax.plot(SNR_DB, m44, "^-", ms=3.5, c="tab:green", label="MIMO 4×4")
m = SNR_DB >= 0
ax.plot(SNR_DB[m], ass22[m], ":", c="tab:orange", lw=1.6, label="Assíntota 2×2 (teórica)")
ax.plot(SNR_DB[m], ass44[m], ":", c="tab:green", lw=1.6, label="Assíntota 4×4 (teórica)")
ax.set(xlabel="SNR (dB)", ylabel="Capacidade ergódica (bit/s/Hz)", xlim=(-10, 40),
       ylim=(0, None))
ax.legend(loc="upper left")
fig.savefig(PASTA / "fig3_numero_antenas.png")

# --- Figura 4: CDF da capacidade a 10 dB (capacidade de outage) --------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for C, nome, cor in [(C_s, "SISO Rayleigh", "tab:blue"),
                     (C_22, "MIMO 2×2", "tab:orange"),
                     (C_44, "MIMO 4×4", "tab:green")]:
    x = np.sort(C[:, i10]); y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, c=cor, label=nome)
ax.axhline(0.10, ls=":", c="gray")
ax.text(0.2, 0.12, "outage 10 %", color="gray", fontsize=8)
ax.set(xlabel="Capacidade instantânea a SNR = 10 dB (bit/s/Hz)",
       ylabel="Probabilidade P(C ≤ x)", xlim=(0, None), ylim=(0, 1))
ax.legend(loc="lower right")
fig.savefig(PASTA / "fig4_cdf_outage.png")

# --- Figura 5: capacidade 2x2 em função da correlação -------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
cores = plt.cm.viridis(np.linspace(0, 0.9, len(RHOS)))
ax.plot(SNR_DB, C_siso_awgn, "k--", label="SISO (AWGN)")
for rho, cor in zip(RHOS, cores):
    ax.plot(SNR_DB, corr[rho]["media"], c=cor, label=f"2×2, ρ = {rho:g}")
ax.set(xlabel="SNR (dB)", ylabel="Capacidade ergódica (bit/s/Hz)", xlim=(-10, 40))
ax.legend(loc="upper left")
fig.savefig(PASTA / "fig5_correlacao.png")

# --- Figura 6: CDF do número de condição -------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for rho, cor in zip(RHOS[:-1], cores[:-1]):           # rho = 1 -> kappa infinito
    x = np.sort(corr[rho]["kappa_db"]); y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, c=cor, label=f"ρ = {rho:g}")
ax.set(xlabel="Número de condição κ = σ$_1$/σ$_2$ (dB)",
       ylabel="Probabilidade P(κ ≤ x)", xlim=(0, 60), ylim=(0, 1))
ax.legend(loc="lower right")
fig.savefig(PASTA / "fig6_condicao.png")

print(f"\nFiguras e resultados.json guardados em {PASTA}")

# Abre as seis figuras em janelas; o script termina quando forem fechadas.
plt.show()
