/**
 * Phase 2B: generalization + sampler-reliability (protocol_2b.md, frozen).
 * Modes: mapping | confirm | pilot
 *  mapping: deterministic orthonormal-hull basis tests for B_3,B_4,B_5,B_8
 *           (isometry required before any Birkhoff feature is trusted).
 *  confirm: cube/ball/matched-ball at intrinsic dims {4,9,16,49,100},
 *           new seed 20261201, P1=50 (+P1=20 sensitivity), AUC + 95% HM CI.
 *  pilot:   multi-chain CDHR pilot (seed 20261301) on Birkhoff B_5 (d=16)
 *           and random V-polytope d=20/M=40; ESS + split-half PSRF (existing,
 *           limits recorded) + F1/F2 chain diagnostics + membership validity.
 * Exit 0 iff the mode's gate passes.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Eigen"
#include "lp_lib.h"

#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/vpolytope.h"
#include "generators/known_polytope_generators.h"
#include "random_walks/random_walks.hpp"
#include "sampling/sampling.hpp"
#include "diagnostics/effective_sample_size.hpp"
#include "diagnostics/univariate_psrf.hpp"
#include "diagnostics/multivariate_psrf.hpp"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope<Point> Hpolytope;
typedef VPolytope<Point> Vpolytope;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;
typedef BoostRandomNumberGenerator<boost::mt19937, NT> RNGType;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const std::string &msg) {
    if (cond) { ++g_pass; std::cout << "  [ok] " << msg << "\n"; }
    else { ++g_fail; std::cout << "  [FAIL] " << msg << "\n"; }
}
static bool close(double a, double b, double tol) { return std::abs(a - b) <= tol; }

static Eigen::VectorXd gauss_vec(int d, std::mt19937 &gen) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd v(d);
    for (int i = 0; i < d; ++i) v(i) = nd(gen);
    return v;
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static void proj_moments(const Eigen::RowVectorXd &z, double &skew, double &kurt) {
    int n = (int)z.size();
    double m = z.mean(), m2 = 0, m3 = 0, m4 = 0;
    for (int j = 0; j < n; ++j) { double c = z(j) - m, c2 = c * c; m2 += c2; m3 += c2 * c; m4 += c2 * c2; }
    m2 /= n; m3 /= n; m4 /= n;
    if (!(m2 > 0)) { skew = 0; kurt = -3; return; }
    skew = m3 / std::pow(m2, 1.5);
    kurt = m4 / (m2 * m2) - 3.0;
}
// F1 = rel radial std, F2 = median 1-D excess kurtosis over P1 projections
static void F1F2(const Eigen::MatrixXd &Xc, std::mt19937 &gen, int P1,
                 double &rms, double &f1, double &f2, double &relmean,
                 std::vector<double> *ku_out = nullptr) {
    int d = (int)Xc.rows(), n = (int)Xc.cols();
    rms = std::sqrt(Xc.array().square().sum() / n);
    Eigen::VectorXd rad(n);
    for (int j = 0; j < n; ++j) rad(j) = Xc.col(j).norm();
    double rm = rad.mean();
    double rs = std::sqrt((rad.array() - rm).square().sum() / (n - 1));
    f1 = rs / rms; relmean = rm / rms;
    std::vector<double> ku;
    for (int p = 0; p < P1; ++p) {
        Eigen::VectorXd u = gauss_vec(d, gen);
        Eigen::RowVectorXd z = (u.transpose() * Xc).eval();
        double s, k; proj_moments(z, s, k); ku.push_back(k);
    }
    f2 = median(ku);
    if (ku_out) *ku_out = ku;
}
// Bootstrap SE of the median over P1 projection statistics (covers projection
// resampling only, NOT chain autocorrelation — documented partial uncertainty).
static double boot_se_median(const std::vector<double> &v, std::mt19937 &gen, int B = 500) {
    int n = (int)v.size();
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();
    double med = median(v), s = 0;
    std::uniform_int_distribution<int> u(0, n - 1);
    for (int b = 0; b < B; ++b) {
        std::vector<double> r; r.reserve(n);
        for (int i = 0; i < n; ++i) r.push_back(v[u(gen)]);
        double m = median(r); s += (m - med) * (m - med);
    }
    return std::sqrt(s / (B - 1));
}
// Cross-chain agreement: max deviation from grand mean in pooled-SE units.
// Returns false when no finite SE is available.
static bool chain_agree(const std::vector<double> &vals, const std::vector<double> &ses, double &maxdev) {
    double grand = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
    maxdev = 0;
    for (size_t k = 0; k < vals.size(); ++k) {
        if (!std::isfinite(ses[k]) || ses[k] <= 0) return false;
        maxdev = std::max(maxdev, std::abs(vals[k] - grand) / ses[k]);
    }
    return maxdev <= 2.0;
}
// Orthonormal hull basis for Birkhoff B_m (shared by mapping + discrimination).
static bool birkhoff_basis(int m, Eigen::MatrixXd &Q, Eigen::VectorXd &xc, std::string &err) {
    int d = (m - 1) * (m - 1), m2 = m * m;
    Eigen::MatrixXd C(2 * m - 1, m2);
    C.setZero();
    for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) C(i, i * m + j) = 1.0;
    for (int j = 0; j < m - 1; ++j) for (int i = 0; i < m; ++i) C(m + j, i * m + j) = 1.0;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(C);
    if (lu.rank() != 2 * m - 1) { err = "constraint rank"; return false; }
    Eigen::MatrixXd K = lu.kernel();
    if ((int)K.cols() != d) { err = "nullspace dim"; return false; }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(K);
    Q = qr.householderQ() * Eigen::MatrixXd::Identity(m2, d);
    xc = Eigen::VectorXd::Constant(m2, 1.0 / m);
    if (((Q.transpose() * Q) - Eigen::MatrixXd::Identity(d, d)).cwiseAbs().maxCoeff() > 1e-9) { err = "orthonormality"; return false; }
    if ((C * Q).cwiseAbs().maxCoeff() > 1e-9) { err = "nullspace"; return false; }
    return true;
}
static double auc_mw(const std::vector<double> &a, const std::vector<double> &b) {
    double w = 0; int na = 0, nb = 0;
    for (double u : a) if (std::isfinite(u)) ++na;
    for (double v : b) if (std::isfinite(v)) ++nb;
    if (na < 2 || nb < 2) return std::numeric_limits<double>::quiet_NaN();
    for (double u : a) if (std::isfinite(u))
        for (double v : b) if (std::isfinite(v)) w += (u > v) ? 1.0 : ((u == v) ? 0.5 : 0.0);
    return w / (na * nb);
}
// Hanley-McNeil 95% CI for AUC from m positive / n negative samples.
static std::pair<double, double> hm_ci(double A, int m, int n) {
    if (!std::isfinite(A)) return {A, A};
    if (A <= 0.0 || A >= 1.0) return {A, A}; // degenerate at extremes
    double Q1 = A / (2 - A), Q2 = 2 * A * A / (1 + A);
    double v = (A * (1 - A) + (m - 1) * (Q1 - A * A) + (n - 1) * (Q2 - A * A)) / (m * n);
    double se = std::sqrt(std::max(v, 0.0));
    return {std::max(0.0, A - 1.96 * se), std::min(1.0, A + 1.96 * se)};
}
// Independent V-polytope membership: V^T l = x, 1^T l = 1, l >= 0 (not P.is_in).
static bool vpoly_contains(const Eigen::MatrixXd &V, const Eigen::VectorXd &x, double tol) {
    int M = (int)V.rows(), d = (int)V.cols();
    lprec *lp = make_lp(d + 1, M);
    if (!lp) return false;
    std::vector<REAL> row(M); std::vector<int> colno(M);
    for (int j = 0; j < M; ++j) colno[j] = j + 1;
    set_add_rowmode(lp, TRUE);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < M; ++j) row[j] = V(j, i);
        if (!add_constraintex(lp, M, row.data(), colno.data(), EQ, (REAL)x(i))) { delete_lp(lp); return false; }
    }
    for (int j = 0; j < M; ++j) row[j] = 1.0;
    if (!add_constraintex(lp, M, row.data(), colno.data(), EQ, 1.0)) { delete_lp(lp); return false; }
    set_add_rowmode(lp, FALSE);
    for (int j = 0; j < M; ++j) set_bounds(lp, j + 1, 0.0, 1.0);
    for (int j = 0; j < M; ++j) row[j] = (j == 0) ? 1.0 : 0.0;
    if (!set_obj_fnex(lp, M, row.data(), colno.data())) { delete_lp(lp); return false; }
    set_minim(lp); set_verbose(lp, NEUTRAL);
    int st = solve(lp);
    bool feas = (st == OPTIMAL);
    if (feas) {
        std::vector<REAL> lam(M); get_variables(lp, lam.data());
        Eigen::VectorXd l(M);
        for (int j = 0; j < M; ++j) l(j) = lam[j];
        feas = (std::abs(l.sum() - 1.0) <= tol) && (l.minCoeff() >= -tol)
            && ((V.transpose() * l - x).norm() <= tol);
    }
    delete_lp(lp);
    return feas;
}

// ---- Birkhoff helpers (frozen Phase-1 affine map) ----
static Eigen::VectorXd y_of_perm(const std::vector<int> &perm, int m) {
    Eigen::VectorXd y((m - 1) * (m - 1));
    for (int i = 0; i < m - 1; ++i) for (int j = 0; j < m - 1; ++j)
        y(i * (m - 1) + j) = (perm[i] == j) ? 1.0 : 0.0;
    return y;
}
static Eigen::MatrixXd X_of_y(const Eigen::VectorXd &y, int m) {
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < m - 1; ++i) for (int j = 0; j < m - 1; ++j) X(i, j) = y(i * (m - 1) + j);
    for (int i = 0; i < m - 1; ++i) X(i, m - 1) = 1.0 - X.row(i).head(m - 1).sum();
    for (int j = 0; j < m - 1; ++j) X(m - 1, j) = 1.0 - X.col(j).head(m - 1).sum();
    X(m - 1, m - 1) = y.sum() - (m - 2);
    return X;
}

static int mode_mapping() {
    std::cout << "2B mapping: orthonormal hull bases + isometry (B_11 on documented sampled set, seed 811)\n";
    int base = g_fail;
    for (int m : {3, 4, 5, 8, 11}) {
        int d = (m - 1) * (m - 1), m2 = m * m;
        Eigen::MatrixXd Q; Eigen::VectorXd xc; std::string err;
        bool bok = birkhoff_basis(m, Q, xc, err);
        check(bok, "B_" + std::to_string(m) + " basis (rank/null/orthonormal)" + (bok ? "" : ": " + err));
        if (!bok) continue;
        Eigen::MatrixXd C(2 * m - 1, m2);
        C.setZero();
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) C(i, i * m + j) = 1.0;
        for (int j = 0; j < m - 1; ++j) for (int i = 0; i < m; ++i) C(m + j, i * m + j) = 1.0;
        Eigen::VectorXd e = Eigen::VectorXd::Ones(2 * m - 1);
        check((C * xc - e).cwiseAbs().maxCoeff() < 1e-12, "B_" + std::to_string(m) + " centroid satisfies sums");
        // documented permutation set: exhaustive for m<=8, sampled for m=11
        std::vector<std::vector<int>> perms;
        std::vector<int> id(m);
        std::iota(id.begin(), id.end(), 0);
        if (m <= 8) {
            auto p = id;
            do { perms.push_back(p); } while (std::next_permutation(p.begin(), p.end()));
        } else {
            perms.push_back(id);
            auto rev = id;
            std::reverse(rev.begin(), rev.end());
            perms.push_back(rev);
            std::mt19937 gs(811);
            for (int t = 0; t < 20; ++t) { // random transpositions
                auto p = id;
                std::uniform_int_distribution<int> u(0, m - 1);
                int a = u(gs), b = u(gs);
                std::swap(p[a], p[b]);
                perms.push_back(p);
            }
            for (int t = 0; t < 300; ++t) { // random perms
                auto p = id;
                std::shuffle(p.begin(), p.end(), gs);
                perms.push_back(p);
            }
        }
        bool vok = true;
        for (auto &pm : perms) {
            Eigen::MatrixXd X = X_of_y(y_of_perm(pm, m), m);
            for (int i = 0; i < m && vok; ++i) for (int j = 0; j < m; ++j)
                vok = vok && close(X(i, j), (pm[i] == j) ? 1.0 : 0.0, 1e-9);
        }
        check(vok, "B_" + std::to_string(m) + " y->X exact on " + std::to_string(perms.size()) +
                         (m <= 8 ? " (exhaustive)" : " (documented sample)"));
        std::vector<Eigen::VectorXd> Z, Apts;
        for (auto &pm : perms) {
            Eigen::VectorXd xv(m2);
            for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j)
                xv(i * m + j) = (pm[i] == j) ? 1.0 : 0.0;
            Apts.push_back(xv - xc);
            Z.push_back(Q.transpose() * (xv - xc));
        }
        double iso = 0;
        size_t npair = 0;
        for (size_t a = 0; a < Z.size() && npair < 20000; ++a) {
            for (size_t b = a + 1; b < Z.size() && npair < 20000; ++b) {
                iso = std::max(iso, std::abs((Z[a] - Z[b]).norm() - (Apts[a] - Apts[b]).norm()));
                ++npair;
            }
        }
        check(iso < 1e-9, "B_" + std::to_string(m) + " hull-coord isometry over " +
                             std::to_string(npair) + " pairs (err=" + std::to_string(iso) + ")");
    }
    std::cout << "mapping gate: " << (g_fail == base ? "PASS" : "FAIL") << "\n";
    return (g_fail == base) ? 0 : 1;
}

static Eigen::MatrixXd sample_body2(int body, int d, int n, double R, std::mt19937 &gen) {
    Eigen::MatrixXd X(d, n);
    if (body == 0) {
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (int j = 0; j < n; ++j) for (int i = 0; i < d; ++i) X(i, j) = u(gen);
    } else {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int j = 0; j < n; ++j) {
            Eigen::VectorXd g = gauss_vec(d, gen); g.normalize();
            X.col(j) = g * (R * std::pow(u(gen), 1.0 / d));
        }
    }
    return X;
}

static int mode_confirm(const std::string &outdir) {
    const int SEED = 20261201, REPS = 30;
    const std::vector<int> dims = {4, 9, 16, 49, 100};
    const std::vector<double> ratios = {0.5, 1.0, 2.0, 5.0, 10.0};
    std::cout << "2B confirm: seed=" << SEED << " P1=50/20\n";
    struct Cell {
        int d, n; std::vector<double> rms[3], f1[3], f2a[3], f2b[3], relm[3];
        std::vector<double> rank[3], gap[3];
    };
    std::vector<Cell> cells;
    for (int d : dims) for (double ra : ratios) {
        int n = std::max(3, (int)std::lround(d * ra));
        Cell c; c.d = d; c.n = n;
        double Rmatch = std::sqrt((d + 2.0) / 3.0);
        for (int b = 0; b < 3; ++b) for (int r = 0; r < REPS; ++r) {
            std::seed_seq ss{SEED, b, d, n, r};
            std::mt19937 gen(ss);
            double R = (b == 2) ? Rmatch : 1.0;
            Eigen::MatrixXd X = sample_body2(b, d, n, R, gen);
            Eigen::VectorXd mu = X.rowwise().mean();
            Eigen::MatrixXd Xc = X.colwise() - mu;
            double rms, f1, f2, relm;
            F1F2(Xc, gen, 50, rms, f1, f2, relm);
            double r2, f2b, dum, dum2;
            { std::mt19937 g2(ss); F1F2(Xc, g2, 20, dum, dum2, f2b, dum2); }
            Eigen::MatrixXd S = (Xc * Xc.transpose()) / double(n - 1);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
            double lmax = es.eigenvalues().maxCoeff();
            double tol = lmax * std::max(d, n) * 2.2e-16;
            int rk = 0;
            for (int i = 0; i < d; ++i) if (es.eigenvalues()(i) > tol) ++rk;
            double sg = 0, mt = 0;
            for (int t = 0; t < 30; ++t) {
                Eigen::VectorXd g = gauss_vec(d, gen); g.normalize();
                double ht = (b == 0) ? g.array().abs().sum() : R * g.norm();
                sg += std::max(0.0, ht - (X.transpose() * g).maxCoeff()); mt += ht;
            }
            c.rms[b].push_back(rms); c.f1[b].push_back(f1); c.f2a[b].push_back(f2);
            c.f2b[b].push_back(f2b); c.relm[b].push_back(relm);
            c.rank[b].push_back(rk); c.gap[b].push_back(sg / mt);
        }
        cells.push_back(std::move(c));
    }
    { std::ofstream o(outdir + "/phase2b_confirm.csv");
      o << "body,d,n,rep,rms,f1,f2_50,f2_20,relmean,rank,gapmean\n";
      const char *nm[3] = {"cube", "ball", "ball_matched"};
      for (auto &c : cells) for (int b = 0; b < 3; ++b) for (int r = 0; r < REPS; ++r)
          o << nm[b] << "," << c.d << "," << c.n << "," << r << "," << c.rms[b][r] << ","
            << c.f1[b][r] << "," << c.f2a[b][r] << "," << c.f2b[b][r] << "," << c.relm[b][r] << ","
            << c.rank[b][r] << "," << c.gap[b][r] << "\n"; }
    printf("\n[A] raw cube vs ball @Birkhoff dims (AUC [95%% CI])\n%4s %5s %18s %18s %18s\n",
           "d", "n", "rms", "F1", "F2_50");
    for (auto &c : cells) {
        double a1 = auc_mw(c.rms[0], c.rms[1]), a2 = auc_mw(c.f1[0], c.f1[1]), a3 = auc_mw(c.f2a[0], c.f2a[1]);
        auto c1 = hm_ci(a1, REPS, REPS), c2 = hm_ci(a2, REPS, REPS), c3 = hm_ci(a3, REPS, REPS);
        printf("%4d %5d %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f]\n",
               c.d, c.n, a1, c1.first, c1.second, a2, c2.first, c2.second, a3, c3.first, c3.second);
    }
    printf("\n[B] normalized cube vs ball\n%4s %5s %18s %18s %18s %18s\n",
           "d", "n", "F1", "relmean", "F2_50", "F2_20");
    for (auto &c : cells) {
        double a1 = auc_mw(c.f1[0], c.f1[1]), a2 = auc_mw(c.relm[0], c.relm[1]);
        double a3 = auc_mw(c.f2a[0], c.f2a[1]), a4 = auc_mw(c.f2b[0], c.f2b[1]);
        auto c1 = hm_ci(a1, REPS, REPS), c2 = hm_ci(a2, REPS, REPS);
        auto c3 = hm_ci(a3, REPS, REPS), c4 = hm_ci(a4, REPS, REPS);
        printf("%4d %5d %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f]\n",
               c.d, c.n, a1, c1.first, c1.second, a2, c2.first, c2.second,
               a3, c3.first, c3.second, a4, c4.first, c4.second);
    }
    printf("\n[C] matched cube vs scaled ball\n%4s %5s %18s %18s %18s %18s\n",
           "d", "n", "rms", "F1", "F2_50", "rank");
    for (auto &c : cells) {
        double a1 = auc_mw(c.rms[0], c.rms[2]), a2 = auc_mw(c.f1[0], c.f1[2]);
        double a3 = auc_mw(c.f2a[0], c.f2a[2]), a4 = auc_mw(c.rank[0], c.rank[2]);
        auto c1 = hm_ci(a1, REPS, REPS), c2 = hm_ci(a2, REPS, REPS);
        auto c3 = hm_ci(a3, REPS, REPS), c4 = hm_ci(a4, REPS, REPS);
        printf("%4d %5d %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f] %5.3f[%.2f,%.2f]\n",
               c.d, c.n, a1, c1.first, c1.second, a2, c2.first, c2.second,
               a3, c3.first, c3.second, a4, c4.first, c4.second);
    }
    printf("\nCSV: phase2b_confirm.csv. Directions frozen: cube HIGHER for F1/F2.\n");
    return 0;
}

// Hungarian maximum-weight perfect matching (validates vs exhaustive in Phase 1).
static std::pair<std::vector<int>, double> hungarian_max(const Eigen::MatrixXd &a) {
    int n = (int)a.rows();
    const double INF = 1e100;
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double cur = -a(i0 - 1, j - 1) - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<int> col(n, -1);
    for (int j = 1; j <= n; ++j) col[p[j] - 1] = j - 1;
    double val = 0.0;
    for (int i = 0; i < n; ++i) val += a(i, col[i]);
    return {col, val};
}

static Eigen::MatrixXd draws_to_mat(const std::list<Point> &pts, int d) {    Eigen::MatrixXd M(d, pts.size());
    int j = 0;
    for (auto &p : pts) {
        for (int i = 0; i < d; ++i) M(i, j) = p[i];
        ++j;
    }
    return M;
}

static int mode_pilot() {
    const int SEED = 20261301, CHAINS = 4, DRAWS = 200, NBURNS = 50;
    const std::vector<int> budgets = {10, 50, 200};
    std::cout << "2B pilot: seed=" << SEED << " chains=" << CHAINS << " draws=" << DRAWS << "\n";
    std::cout << "exit status: 0 iff B_5 reaches an ADEQUATE budget (V20 exploratory, human-readable only)\n";
    bool b5_ok = false;
    // ---- Birkhoff B_5 (d=16) ----
    {
        Hpolytope P = generate_birkhoff<Hpolytope>(5);
        int d = P.dimension();
        auto ib = P.ComputeInnerBall();
        std::cout << "B_5: dim=" << d << " inner_r=" << ib.second << "\n";
        std::mt19937 gs(SEED);
        std::vector<Point> starts;
        for (int k = 0; k < CHAINS; ++k) {
            for (int att = 0; att < 100; ++att) {
                Eigen::VectorXd u = gauss_vec(d, gs); u.normalize();
                Eigen::VectorXd c = ib.first.getCoefficients() + 0.25 * ib.second * u;
                Point p(c);
                if (P.is_in(p) != 0) { starts.push_back(p); break; }
            }
        }
        check((int)starts.size() == CHAINS, "B_5 dispersed interior starts");
        for (int W : budgets) {
            std::vector<double> f1, se1, f2, se2, miness;
            double maxpsrf = 0, meanpsrf = 0;
            int inside = 0, total = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < CHAINS; ++k) {
                std::seed_seq ss{SEED, 51, k, W};
                std::mt19937 tmp(ss); RNGType rng(d); rng.set_seed(tmp());
                std::list<Point> pts;
                uniform_sampling<CDHRWalk>(pts, P, rng, (unsigned)W, (unsigned)DRAWS, starts[k], (unsigned)NBURNS);
                Eigen::MatrixXd M = draws_to_mat(pts, d);
                for (auto &p : pts) { total++; inside += (P.is_in(p) != 0); }
                unsigned int me = 0;
                VT e = effective_sample_size<NT, VT, MT>(M, me);
                miness.push_back(e.minCoeff());
                VT pr = univariate_psrf<NT, VT, MT>(M);
                maxpsrf = std::max(maxpsrf, pr.maxCoeff()); meanpsrf += pr.mean() / CHAINS;
                Eigen::VectorXd mu = M.rowwise().mean();
                Eigen::MatrixXd Xc = M.colwise() - mu;
                double rms, a, b, rm;
                std::vector<double> ku;
                std::seed_seq sf{SEED, 52, k, W}; std::mt19937 gf(sf);
                F1F2(Xc, gf, 20, rms, a, b, rm, &ku);
                f1.push_back(a); f2.push_back(b);
                Eigen::VectorXd rad(DRAWS);
                for (int j = 0; j < DRAWS; ++j) rad(j) = Xc.col(j).norm();
                double rs = std::sqrt((rad.array() - rad.mean()).square().sum() / (DRAWS - 1));
                se1.push_back(rs / std::sqrt(std::max<double>(miness.back(), 1.0)) / rms);
                std::seed_seq sb{SEED, 53, k, W}; std::mt19937 gb(sb);
                se2.push_back(boot_se_median(ku, gb));
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            double dev1, dev2;
            bool ag1 = chain_agree(f1, se1, dev1), ag2 = chain_agree(f2, se2, dev2);
            double mnE = *std::min_element(miness.begin(), miness.end());
            bool ok = (mnE >= 100.0) && ag1 && ag2 && (inside == total);
            b5_ok = b5_ok || ok;
            printf("B_5 W=%-4d minESS=%7.1f maxPSRF1=%.3f meanPSRF1=%.3f F1dev=%.2f F2dev=%.2f memb=%d/%d %ldms %s\n",
                   W, mnE, maxpsrf, meanpsrf, dev1, dev2, inside, total, ms,
                   ok ? "ADEQUATE" : "INADEQUATE");
            std::cout << "  F1 chains:";
            for (double v : f1) std::cout << " " << v;
            std::cout << "\n  F2 chains:";
            for (double v : f2) std::cout << " " << v;
            std::cout << "\n";
        }
        // multivariate PSRF limitation demo (singular-W expected at small n)
        {
            std::seed_seq ss{SEED, 51, 0, 10}; std::mt19937 tmp(ss);
            RNGType rng(d); rng.set_seed(tmp());
            std::list<Point> pts;
            uniform_sampling<CDHRWalk>(pts, P, rng, 10, 10, starts[0], 10); // n=10 < d=16
            Eigen::MatrixXd M = draws_to_mat(pts, d);
            Eigen::MatrixXd H = M.colwise() - M.rowwise().mean();
            Eigen::MatrixXd Wm = (H * H.transpose()) / double(M.cols() - 1);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Wm);
            std::cout << "  short-chain demo n=10<d=16: min eig(W)=" << es.eigenvalues().minCoeff()
                      << " -> multivariate PSRF SKIPPED-SINGULAR (W.inverse() invalid)\n";
        }
    }
    // ---- random V-polytope d=20/M=40 ----
    {
        int d = 20, M = 40;
        Eigen::MatrixXd V(M, d);
        std::mt19937 gv(61301);
        for (int j = 0; j < M; ++j) { Eigen::VectorXd g = gauss_vec(d, gv); g.normalize(); V.row(j) = g.transpose(); }
        VT bb = VT::Ones(M);
        Vpolytope P(d, V, bb);
        auto ib = P.ComputeInnerBall();
        std::cout << "V20: inner_r=" << ib.second << "\n";
        std::mt19937 gs(SEED + 1);
        std::vector<Point> starts;
        for (int k = 0; k < CHAINS; ++k)
            for (int att = 0; att < 200; ++att) {
                Eigen::VectorXd u = gauss_vec(d, gs); u.normalize();
                Eigen::VectorXd c = ib.first.getCoefficients() + 0.25 * ib.second * u;
                if (vpoly_contains(V, c, 1e-6)) { starts.push_back(Point(c)); break; }
            }
        check((int)starts.size() == CHAINS, "V20 dispersed interior starts (LP-verified)");
        for (int W : budgets) {
            std::vector<double> f1, se1, f2, se2, miness;
            int inside = 0, total = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < CHAINS; ++k) {
                std::seed_seq ss{SEED, 61, k, W};
                std::mt19937 tmp(ss); RNGType rng(d); rng.set_seed(tmp());
                std::list<Point> pts;
                uniform_sampling<CDHRWalk>(pts, P, rng, (unsigned)W, (unsigned)DRAWS, starts[k], (unsigned)NBURNS);
                Eigen::MatrixXd Mc = draws_to_mat(pts, d);
                for (auto &p : pts) {
                    Eigen::VectorXd x(d);
                    for (int i = 0; i < d; ++i) x(i) = p[i];
                    total++; inside += vpoly_contains(V, x, 1e-6) ? 1 : 0;
                }
                unsigned int me = 0;
                VT e = effective_sample_size<NT, VT, MT>(Mc, me);
                miness.push_back(e.minCoeff());
                Eigen::VectorXd mu = Mc.rowwise().mean();
                Eigen::MatrixXd Xc = Mc.colwise() - mu;
                double rms, a, b, rm;
                std::vector<double> ku;
                std::seed_seq sf{SEED, 62, k, W}; std::mt19937 gf(sf);
                F1F2(Xc, gf, 20, rms, a, b, rm, &ku);
                f1.push_back(a); f2.push_back(b);
                Eigen::VectorXd rad(DRAWS);
                for (int j = 0; j < DRAWS; ++j) rad(j) = Xc.col(j).norm();
                double rs = std::sqrt((rad.array() - rad.mean()).square().sum() / (DRAWS - 1));
                se1.push_back(rs / std::sqrt(std::max<double>(miness.back(), 1.0)) / rms);
                std::seed_seq sb{SEED, 63, k, W}; std::mt19937 gb(sb);
                se2.push_back(boot_se_median(ku, gb));
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            double dev1, dev2;
            bool ag1 = chain_agree(f1, se1, dev1), ag2 = chain_agree(f2, se2, dev2);
            double mnE = *std::min_element(miness.begin(), miness.end());
            bool ok = (mnE >= 100.0) && ag1 && ag2 && (inside == total);
            printf("V20 W=%-4d minESS=%7.1f F1dev=%.2f F2dev=%.2f memb=%d/%d %ldms %s\n",
                   W, mnE, dev1, dev2, inside, total, ms, ok ? "ADEQUATE" : "INADEQUATE");
            std::cout << "  F1 chains:";
            for (double v : f1) std::cout << " " << v;
            std::cout << "\n  F2 chains:";
            for (double v : f2) std::cout << " " << v;
            std::cout << "\n";
        }
    }
    // ---- B_8 costing note (construction + inner ball only) ----
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        Hpolytope P = generate_birkhoff<Hpolytope>(8);
        auto ib = P.ComputeInnerBall();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "B_8 costing: dim=" << P.dimension() << " inner_r=" << ib.second
                  << " ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "\n";
    }
    std::cout << "pilot gate: classification runs justified only at ADEQUATE budgets\n";
    std::cout << "pilot exit: " << (b5_ok ? 0 : 1) << " (B_5 adequate budget " << (b5_ok ? "found" : "NOT found") << ")\n";
    return b5_ok ? 0 : 1;
}

// B_5 budget top-up + first Birkhoff-vs-control discrimination at d=16.
// Budgets: (W=200,D=500) then (W=400,D=500). First ADEQUATE budget (frozen
// gate incl. F1+F2 agreement at P1=50) feeds the discrimination run.
// Seeds: draws 20261501, controls 20261601, oracle 20261701.
static int mode_b5campaign() {
    const int CHAINS = 4, P1 = 50;
    const int m = 5, d = 16;
    std::cout << "2B b5campaign: top-up (200,500)->(400,500), then discriminate at first ADEQUATE\n";
    Eigen::MatrixXd Q; Eigen::VectorXd xc; std::string err;
    if (!birkhoff_basis(m, Q, xc, err)) { std::cout << "basis FAILED: " << err << "\n"; return 1; }
    Hpolytope P = generate_birkhoff<Hpolytope>((unsigned)m);
    auto ib = P.ComputeInnerBall();
    std::mt19937 gs(20261501);
    std::vector<Point> starts;
    for (int k = 0; k < CHAINS; ++k)
        for (int att = 0; att < 200; ++att) {
            Eigen::VectorXd u = gauss_vec(d, gs); u.normalize();
            Point p(ib.first.getCoefficients() + 0.25 * ib.second * u);
            if (P.is_in(p) != 0) { starts.push_back(p); break; }
        }
    if ((int)starts.size() != CHAINS) { std::cout << "starts FAILED\n"; return 1; }
    auto to_hull = [&](const Point &p) -> Eigen::VectorXd {
        Eigen::VectorXd y(d);
        for (int i = 0; i < d; ++i) y(i) = p[i];
        Eigen::MatrixXd X = X_of_y(y, m);
        Eigen::VectorXd xv(m * m);
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) xv(i * m + j) = X(i, j);
        return Q.transpose() * (xv - xc);
    };
    struct Win { int W, D; std::vector<Eigen::MatrixXd> chains; };
    Win win{0, 0, {}};
    for (auto [W, D] : {std::pair<int,int>{200, 500}, {400, 500}}) {
        std::vector<double> f1, se1, f2, se2, miness;
        std::vector<Eigen::MatrixXd> Hc;
        int inside = 0, total = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < CHAINS; ++k) {
            std::seed_seq ss{20261501, k, W, D};
            std::mt19937 tmp(ss); RNGType rng(d); rng.set_seed(tmp());
            std::list<Point> pts;
            uniform_sampling<CDHRWalk>(pts, P, rng, (unsigned)W, (unsigned)D, starts[k], 50);
            Eigen::MatrixXd Z(d, pts.size());
            int j = 0;
            for (auto &p : pts) {
                total++; inside += (P.is_in(p) != 0);
                Z.col(j++) = to_hull(p);
            }
            Hc.push_back(Z);
            Eigen::MatrixXd Mr = draws_to_mat(pts, d);
            unsigned int me = 0;
            VT e = effective_sample_size<NT, VT, MT>(Mr, me);
            miness.push_back(e.minCoeff());
            Eigen::VectorXd mu = Z.rowwise().mean();
            Eigen::MatrixXd Zc = Z.colwise() - mu;
            double rms, a, b, rm;
            std::vector<double> ku;
            std::seed_seq sf{20261501, 7, k, W}; std::mt19937 gf(sf);
            F1F2(Zc, gf, P1, rms, a, b, rm, &ku);
            f1.push_back(a); f2.push_back(b);
            Eigen::VectorXd rad(Z.cols());
            for (int t = 0; t < Z.cols(); ++t) rad(t) = Zc.col(t).norm();
            double rs = std::sqrt((rad.array() - rad.mean()).square().sum() / (Z.cols() - 1));
            se1.push_back(rs / std::sqrt(std::max<double>(miness.back(), 1.0)) / rms);
            std::seed_seq sb{20261501, 8, k, W}; std::mt19937 gb(sb);
            se2.push_back(boot_se_median(ku, gb));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double dev1, dev2;
        bool ag1 = chain_agree(f1, se1, dev1), ag2 = chain_agree(f2, se2, dev2);
        double mnE = *std::min_element(miness.begin(), miness.end());
        bool ok = (mnE >= 100.0) && ag1 && ag2 && (inside == total);
        printf("top-up (W=%d,D=%d) minESS=%.1f F1dev=%.2f F2dev=%.2f memb=%d/%d %ldms %s\n",
               W, D, mnE, dev1, dev2, inside, total,
               std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
               ok ? "ADEQUATE" : "INADEQUATE");
        std::cout << "  F1:";
        for (double v : f1) std::cout << " " << v;
        std::cout << "\n  F2:";
        for (double v : f2) std::cout << " " << v;
        std::cout << "\n";
        if (ok) { win = {W, D, Hc}; break; }
    }
    if (win.W == 0) { std::cout << "b5campaign: no ADEQUATE budget — no discrimination run\n"; return 1; }
    const int NCTRL = 30, D = win.D;
    std::vector<double> cF1[2], cF2[2];
    for (int b = 0; b < 2; ++b) for (int r = 0; r < NCTRL; ++r) {
        std::seed_seq ss{20261601, b, r, D};
        std::mt19937 gen(ss);
        Eigen::MatrixXd X = sample_body2(b, d, D, 1.0, gen);
        Eigen::VectorXd mu = X.rowwise().mean();
        Eigen::MatrixXd Xc = X.colwise() - mu;
        double rms, a, bb, rm;
        F1F2(Xc, gen, P1, rms, a, bb, rm);
        cF1[b].push_back(a); cF2[b].push_back(bb);
    }
    std::vector<double> bF1, bF2;
    for (size_t k = 0; k < win.chains.size(); ++k) {
        Eigen::VectorXd mu = win.chains[k].rowwise().mean();
        Eigen::MatrixXd Zc = win.chains[k].colwise() - mu;
        std::seed_seq sf{20261501, 9, (int)k, win.W}; std::mt19937 gf(sf);
        double rms, a, bb, rm;
        F1F2(Zc, gf, P1, rms, a, bb, rm);
        bF1.push_back(a); bF2.push_back(bb);
    }
    auto rep = [](const std::vector<double> &v) {
        double mm = std::accumulate(v.begin(), v.end(), 0.0) / v.size(), s = 0;
        for (double x : v) s += (x - mm) * (x - mm);
        return std::pair<double,double>(mm, std::sqrt(s / (v.size() - 1)));
    };
    printf("discrimination at (W=%d,D=%d), hull coords, P1=%d\n", win.W, D, P1);
    for (int b = 0; b < 2; ++b) {
        const char *nm = b ? "ball" : "cube";
        double A1 = auc_mw(bF1, cF1[b]), A2 = auc_mw(bF2, cF2[b]);
        auto i1 = hm_ci(A1, CHAINS, NCTRL), i2 = hm_ci(A2, CHAINS, NCTRL);
        auto bc = rep(bF1), cc = rep(cF1[b]), bf = rep(bF2), cf = rep(cF2[b]);
        printf("B_5 vs %-4s F1: birkhoff %.4f±%.4f, control %.4f±%.4f, AUC=%.3f[%.2f,%.2f]\n",
               nm, bc.first, bc.second, cc.first, cc.second, A1, i1.first, i1.second);
        printf("B_5 vs %-4s F2: birkhoff %.4f±%.4f, control %.4f±%.4f, AUC=%.3f[%.2f,%.2f]\n",
               nm, bf.first, bf.second, cf.first, cf.second, A2, i2.first, i2.second);
    }
    {
        Eigen::MatrixXd Sall(d, CHAINS * D);
        int j = 0;
        for (auto &Z : win.chains) { Sall.block(0, j, d, D) = Z; j += D; }
        std::mt19937 go(20261701);
        std::uniform_real_distribution<double> un(-1.0, 1.0);
        double sg = 0, mt = 0, mg = 0;
        const int T = 30;
        for (int t = 0; t < T; ++t) {
            Eigen::MatrixXd C(m, m);
            for (int i = 0; i < m; ++i) for (int jj = 0; jj < m; ++jj) C(i, jj) = un(go);
            Eigen::VectorXd cv(m * m);
            for (int i = 0; i < m; ++i) for (int jj = 0; jj < m; ++jj) cv(i * m + jj) = C(i, jj);
            Eigen::VectorXd ch = Q.transpose() * cv;
            double ht = hungarian_max(C).second;
            double hh = (Sall.transpose() * ch).maxCoeff() + cv.dot(xc);
            double gap = std::max(0.0, ht - hh);
            sg += gap; mg = std::max(mg, gap); mt += std::abs(ht);
        }
        printf("B_5 support-gap oracle (T=%d ambient matrices): mean=%.4f max=%.4f (normalized by mean|h|)\n",
               T, (sg / T) / (mt / T), mg / (mt / T));
    }
    std::cout << "b5campaign: body-specific d=16 result; NOT a generalization claim\n";
    return 0;
}

static double se_of_mean(const std::vector<double> &v) {
    if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size(), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1) / v.size());
}
// Per-block median kurtosis with one fixed projection set G (P1 x d), K blocks.
static std::vector<double> block_medians(const Eigen::MatrixXd &Xc, const Eigen::MatrixXd &G, int K) {
    int n = (int)Xc.cols(), nb = n / K, P1 = (int)G.rows();
    std::vector<double> out;
    for (int k = 0; k < K; ++k) {
        Eigen::MatrixXd Xb = Xc.middleCols(k * nb, nb);
        Eigen::MatrixXd Zp = G * Xb;
        std::vector<double> ku;
        for (int p = 0; p < P1; ++p) { double s, kk; proj_moments(Zp.row(p), s, kk); ku.push_back(kk); }
        out.push_back(median(ku));
    }
    return out;
}
static std::vector<double> full_ku(const Eigen::MatrixXd &Xc, const Eigen::MatrixXd &G) {
    Eigen::MatrixXd Zp = G * Xc;
    std::vector<double> ku;
    for (int p = 0; p < G.rows(); ++p) { double s, k; proj_moments(Zp.row(p), s, k); ku.push_back(k); }
    return ku;
}

struct AdeqResult {
    bool ok = false;
    double minEssRed = 0, minEssHull = 0, maxPsrf = 0, f1dev = 0, f2devBoot = 0, f2devBatch = 0;
    int inside = 0, total = 0;
    long ms = 0;
    std::vector<double> f1, f2;
    std::vector<Eigen::MatrixXd> hull;
};
// Four-chain adequacy at (W,D): dual-coordinate ESS + F1/F2 agreement
// (bootstrap AND batch-means for F2) + membership. seedBase isolates tuning.
static AdeqResult run_adequacy(int m, int W, int D, int seedBase, bool verbose) {
    AdeqResult R;
    const int CHAINS = 4, P1 = 50, K = 5, d = (m - 1) * (m - 1);
    Eigen::MatrixXd Q; Eigen::VectorXd xc; std::string err;
    if (!birkhoff_basis(m, Q, xc, err)) { if (verbose) std::cout << "basis FAILED: " << err << "\n"; return R; }
    Hpolytope P = generate_birkhoff<Hpolytope>((unsigned)m);
    auto ib = P.ComputeInnerBall();
    std::mt19937 gs(seedBase);
    std::vector<Point> starts;
    for (int k = 0; k < CHAINS; ++k)
        for (int att = 0; att < 500; ++att) {
            Eigen::VectorXd u = gauss_vec(d, gs); u.normalize();
            Point p(ib.first.getCoefficients() + 0.25 * ib.second * u);
            if (P.is_in(p) != 0) { starts.push_back(p); break; }
        }
    if ((int)starts.size() != CHAINS) { if (verbose) std::cout << "starts FAILED\n"; return R; }
    std::vector<double> f1, se1, f2, se2b, se2t, eR, eH;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < CHAINS; ++k) {
        std::seed_seq ss{seedBase + 1, k, W, D};
        std::mt19937 tmp(ss); RNGType rng(d); rng.set_seed(tmp());
        std::list<Point> pts;
        uniform_sampling<CDHRWalk>(pts, P, rng, (unsigned)W, (unsigned)D, starts[k], 50);
        Eigen::MatrixXd Mr = draws_to_mat(pts, d);
        Eigen::MatrixXd Z(d, pts.size());
        int j = 0;
        for (auto &p : pts) {
            R.total++; R.inside += (P.is_in(p) != 0);
            Eigen::VectorXd y(d);
            for (int i = 0; i < d; ++i) y(i) = p[i];
            Eigen::MatrixXd X = X_of_y(y, m);
            Eigen::VectorXd xv(m * m);
            for (int i = 0; i < m; ++i) for (int jj = 0; jj < m; ++jj) xv(i * m + jj) = X(i, jj);
            Eigen::VectorXd z = Q.transpose() * (xv - xc);
            Z.col(j++) = z;
        }
        R.hull.push_back(Z);
        unsigned int me = 0;
        eR.push_back(effective_sample_size<NT, VT, MT>(Mr, me).minCoeff());
        eH.push_back(effective_sample_size<NT, VT, MT>(Z, me).minCoeff());
        VT pr = univariate_psrf<NT, VT, MT>(Mr);
        R.maxPsrf = std::max(R.maxPsrf, pr.maxCoeff());
        Eigen::VectorXd mu = Z.rowwise().mean();
        Eigen::MatrixXd Zc = Z.colwise() - mu;
        Eigen::VectorXd rad(Z.cols());
        for (int t = 0; t < Z.cols(); ++t) rad(t) = Zc.col(t).norm();
        double rms = std::sqrt(Zc.array().square().sum() / Z.cols());
        double rs = std::sqrt((rad.array() - rad.mean()).square().sum() / (Z.cols() - 1));
        f1.push_back(rs / rms);
        se1.push_back(rs / std::sqrt(std::max<double>(std::min(eR.back(), eH.back()), 1.0)) / rms);
        std::seed_seq sg{seedBase + 2, k, W, D}; std::mt19937 gg(sg);
        Eigen::MatrixXd G(P1, d);
        for (int p = 0; p < P1; ++p) { Eigen::VectorXd g = gauss_vec(d, gg); G.row(p) = g.transpose(); }
        std::vector<double> ku = full_ku(Zc, G);
        f2.push_back(median(ku));
        std::seed_seq sb{seedBase + 3, k, W, D}; std::mt19937 gb(sb);
        se2b.push_back(boot_se_median(ku, gb));
        se2t.push_back(se_of_mean(block_medians(Zc, G, K)));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    R.ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    R.minEssRed = *std::min_element(eR.begin(), eR.end());
    R.minEssHull = *std::min_element(eH.begin(), eH.end());
    bool ag1 = chain_agree(f1, se1, R.f1dev);
    double db_; bool ag2b = chain_agree(f2, se2b, db_);
    bool ag2t = chain_agree(f2, se2t, R.f2devBatch);
    R.f2devBoot = db_;
    R.f1 = f1; R.f2 = f2;
    R.ok = (R.minEssRed >= 100.0) && (R.minEssHull >= 100.0) && ag1 && ag2b && ag2t &&
           (R.inside == R.total);
    if (verbose) {
        printf("B_%d (W=%d,D=%d) ESSred=%.1f ESShull=%.1f maxPSRF1=%.3f F1dev=%.2f F2boot=%.2f F2batch=%.2f memb=%d/%d %ldms %s\n",
               m, W, D, R.minEssRed, R.minEssHull, R.maxPsrf, R.f1dev, R.f2devBoot, R.f2devBatch,
               R.inside, R.total, R.ms, R.ok ? "ADEQUATE" : "INADEQUATE");
        std::cout << "  F1:";
        for (double v : f1) std::cout << " " << v;
        std::cout << "\n  F2:";
        for (double v : f2) std::cout << " " << v;
        std::cout << "\n";
    }
    return R;
}

static int mode_adequacy(int m, int W, int D) {
    std::cout << "2B.2 adequacy B_" << m << " seedBase=20261801\n";
    AdeqResult R = run_adequacy(m, W, D, 20261801, true);
    return R.ok ? 0 : 1;
}

// Independent 30+30+30 discrimination at an adequate (W,D). Eval seeds
// disjoint from tuning. Exit 0 iff every eval dataset passes spot checks.
static int mode_discriminate(int m, int W, int D, const std::string &outdir) {
    const int N = 30, P1 = 50, d = (m - 1) * (m - 1);
    std::cout << "2B.2 discriminate B_" << m << " (W=" << W << ",D=" << D << ") evalSeeds=20261901/20261902\n";
    Eigen::MatrixXd Q; Eigen::VectorXd xc; std::string err;
    if (!birkhoff_basis(m, Q, xc, err)) { std::cout << "basis FAILED\n"; return 1; }
    Hpolytope P = generate_birkhoff<Hpolytope>((unsigned)m);
    auto ib = P.ComputeInnerBall();
    struct DS { double f1, f2, essR, essH; };
    std::vector<DS> B;
    std::vector<Eigen::MatrixXd> Bm; // hull-coord draws per dataset (for ESS-matched arm)
    std::vector<int> failed;
    for (int r = 0; r < N; ++r) {
        std::seed_seq ss{20261901, r, W, D};
        std::mt19937 gen(ss);
        Point start(d);
        bool placed = false;
        for (int att = 0; att < 500 && !placed; ++att) {
            Eigen::VectorXd u = gauss_vec(d, gen); u.normalize();
            Point p(ib.first.getCoefficients() + 0.25 * ib.second * u);
            if (P.is_in(p) != 0) { start = p; placed = true; }
        }
        if (!placed) { failed.push_back(r); continue; }
        std::mt19937 tmp(ss); RNGType rng(d); rng.set_seed(tmp());
        std::list<Point> pts;
        uniform_sampling<CDHRWalk>(pts, P, rng, (unsigned)W, (unsigned)D, start, 50);
        Eigen::MatrixXd Mr = draws_to_mat(pts, d);
        int in = 0;
        Eigen::MatrixXd Z(d, pts.size());
        int j = 0;
        for (auto &p : pts) {
            in += (P.is_in(p) != 0);
            Eigen::VectorXd y(d);
            for (int i = 0; i < d; ++i) y(i) = p[i];
            Eigen::MatrixXd X = X_of_y(y, m);
            Eigen::VectorXd xv(m * m);
            for (int i = 0; i < m; ++i) for (int jj = 0; jj < m; ++jj) xv(i * m + jj) = X(i, jj);
            Eigen::VectorXd z = Q.transpose() * (xv - xc);
            Z.col(j++) = z;
        }
        unsigned int me = 0;
        double eR = effective_sample_size<NT, VT, MT>(Mr, me).minCoeff();
        double eH = effective_sample_size<NT, VT, MT>(Z, me).minCoeff();
        if (in != (int)pts.size() || eR < 50.0 || eH < 50.0) { failed.push_back(r); continue; }
        Eigen::VectorXd mu = Z.rowwise().mean();
        Eigen::MatrixXd Zc = Z.colwise() - mu;
        double rms, a, b, rm;
        std::seed_seq sp{20261903, 2, r, D}; std::mt19937 gp(sp);
        F1F2(Zc, gp, P1, rms, a, b, rm);
        B.push_back({a, b, eR, eH});
        Bm.push_back(Z);
    }
    std::vector<DS> C[2];
    std::vector<Eigen::MatrixXd> Cm[2];
    for (int b = 0; b < 2; ++b) for (int r = 0; r < N; ++r) {
        std::seed_seq ss{20261902, b, r, D};
        std::mt19937 gen(ss);
        Eigen::MatrixXd X = sample_body2(b, d, D, 1.0, gen);
        Eigen::VectorXd mu = X.rowwise().mean();
        Eigen::MatrixXd Xc = X.colwise() - mu;
        double rms, a, bb, rm;
        std::seed_seq sp{20261903, b, r, D}; std::mt19937 gp(sp);
        F1F2(Xc, gp, P1, rms, a, bb, rm);
        C[b].push_back({a, bb, (double)D, (double)D});
        Cm[b].push_back(X);
    }
    { std::ofstream o(outdir + "/phase2b_disc_m" + std::to_string(m) + ".csv");
      o << "body,rep,F1,F2,minEssRed,minEssHull,W,D\n";
      for (int r = 0; r < (int)B.size(); ++r)
          o << "birkhoff," << r << "," << B[r].f1 << "," << B[r].f2 << "," << B[r].essR << "," << B[r].essH << "," << W << "," << D << "\n";
      const char *nm[2] = {"cube", "ball"};
      for (int b = 0; b < 2; ++b) for (int r = 0; r < N; ++r)
          o << nm[b] << "," << r << "," << C[b][r].f1 << "," << C[b][r].f2 << "," << C[b][r].essR << "," << C[b][r].essH << "," << W << "," << D << "\n"; }
    auto col = [](const std::vector<DS> &v, double DS::*f) {
        std::vector<double> o; for (auto &x : v) o.push_back(x.*f); return o; };
    auto rep = [](const std::vector<double> &v) {
        double mm = std::accumulate(v.begin(), v.end(), 0.0) / v.size(), s = 0;
        for (double x : v) s += (x - mm) * (x - mm);
        return std::pair<double,double>(mm, std::sqrt(s / (v.size() - 1))); };
    // ESS-matched arm: thin all datasets to cross-body min minESS_red
    double nT = D;
    for (auto &x : B) nT = std::min(nT, std::floor(x.essR));
    int nEff = std::max(10, (int)nT);
    printf("B_%d discrimination: %d/30 Birkhoff datasets passed spot checks; ESS-matched n=%d\n",
           m, (int)B.size(), nEff);
    if (!failed.empty()) {
        std::cout << "FAILED eval datasets (budget must be increased, none excluded):";
        for (int r : failed) std::cout << " " << r;
        std::cout << "\n";
        return 1;
    }
    for (int b = 0; b < 2; ++b) {
        const char *nm = b ? "ball" : "cube";
        double A1 = auc_mw(col(B, &DS::f1), col(C[b], &DS::f1));
        double A2 = auc_mw(col(B, &DS::f2), col(C[b], &DS::f2));
        auto i1 = hm_ci(A1, (int)B.size(), N), i2 = hm_ci(A2, (int)B.size(), N);
        auto b1 = rep(col(B, &DS::f1)), c1 = rep(col(C[b], &DS::f1));
        auto b2 = rep(col(B, &DS::f2)), c2 = rep(col(C[b], &DS::f2));
        printf("B_%d vs %-4s F1: %.4f±%.4f vs %.4f±%.4f AUC=%.3f[%.2f,%.2f] (frozen dir: birkhoff higher)\n",
               m, nm, b1.first, b1.second, c1.first, c1.second, A1, i1.first, i1.second);
        printf("B_%d vs %-4s F2: %.4f±%.4f vs %.4f±%.4f AUC=%.3f[%.2f,%.2f]\n",
               m, nm, b2.first, b2.second, c2.first, c2.second, A2, i2.first, i2.second);
    }
    // ESS-matched arm: thin every dataset to cross-body min minESS_red, same projections
    {
        auto thin = [](const Eigen::MatrixXd &X, int t) {
            int n = (int)X.cols();
            Eigen::MatrixXd Y(X.rows(), t);
            for (int j = 0; j < t; ++j) Y.col(j) = X.col((j * (n - 1)) / (t - 1));
            return Y;
        };
        auto feat = [&](const Eigen::MatrixXd &X, int bodyid, int r) {
            Eigen::VectorXd mu = X.rowwise().mean();
            Eigen::MatrixXd Xc = X.colwise() - mu;
            double rms, a, b, rm;
            std::seed_seq sp{20261903, bodyid, r, D}; std::mt19937 gp(sp);
            F1F2(Xc, gp, P1, rms, a, b, rm);
            return std::pair<double,double>(a, b);
        };
        printf("ESS-matched arm (n=%d):\n", nEff);
        for (int b = 0; b < 2; ++b) {
            const char *nm = b ? "ball" : "cube";
            std::vector<double> b1, b2, c1, c2;
            for (int r = 0; r < (int)Bm.size(); ++r) {
                auto f = feat(thin(Bm[r], nEff), 2, r);
                b1.push_back(f.first); b2.push_back(f.second);
            }
            for (int r = 0; r < N; ++r) {
                auto f = feat(thin(Cm[b][r], nEff), b, r);
                c1.push_back(f.first); c2.push_back(f.second);
            }
            double A1 = auc_mw(b1, c1), A2 = auc_mw(b2, c2);
            auto i1 = hm_ci(A1, (int)b1.size(), N), i2 = hm_ci(A2, (int)b1.size(), N);
            printf("B_%d vs %-4s matched F1 AUC=%.3f[%.2f,%.2f] F2 AUC=%.3f[%.2f,%.2f]\n",
                   m, nm, A1, i1.first, i1.second, A2, i2.first, i2.second);
        }
    }
    std::cout << "body-specific d=" << d << " result; NOT a generalization claim\n";
    return 0;
}

// V-polytope reliability pilot (Gate A): d/M from argv, 4 chains, full gates.
// Ambient coords throughout (no reduced/hull split). Membership by independent
// LP on every draw. Exit 0 iff ADEQUATE. Vertices seed 61301 (continuity);
// sampling seeds 20262001-family (disjoint from all earlier runs).
static int mode_vpilot(int d, int M, int W, int D, const std::string &saveprefix = "", bool sharedG = false) {
    const int CHAINS = 4, P1 = 50, K = 5;
    std::cout << "2B.3 Gate A vpilot d=" << d << " M=" << M << " W=" << W << " D=" << D
              << (sharedG ? " shared-G" : " independent-G") << "\n";
    Eigen::MatrixXd V(M, d);
    std::mt19937 gv(61301);
    for (int j = 0; j < M; ++j) { Eigen::VectorXd g = gauss_vec(d, gv); g.normalize(); V.row(j) = g.transpose(); }
    VT bb = VT::Ones(M);
    Vpolytope P(d, V, bb);
    auto ib = P.ComputeInnerBall();
    std::cout << "inner_r=" << ib.second << "\n";
    std::mt19937 gs(20262001);
    std::vector<Point> starts;
    for (int k = 0; k < CHAINS; ++k)
        for (int att = 0; att < 500; ++att) {
            Eigen::VectorXd u = gauss_vec(d, gs); u.normalize();
            Eigen::VectorXd c = ib.first.getCoefficients() + 0.25 * ib.second * u;
            if (vpoly_contains(V, c, 1e-6)) { starts.push_back(Point(c)); break; }
        }
    if ((int)starts.size() != CHAINS) { std::cout << "starts FAILED\n"; return 1; }
    std::vector<double> f1, se1, f2, se2b, se2t, eA;
    int inside = 0, total = 0;
    long samp_ms = 0, lp_ms = 0;
    for (int k = 0; k < CHAINS; ++k) {
        std::seed_seq ss{20262001, k, W, D};
        std::mt19937 tmp(ss); RNGType rng(d); rng.set_seed(tmp());
        std::list<Point> pts;
        auto t0 = std::chrono::high_resolution_clock::now();
        uniform_sampling<CDHRWalk>(pts, P, rng, (unsigned)W, (unsigned)D, starts[k], 50);
        auto t1 = std::chrono::high_resolution_clock::now();
        samp_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        Eigen::MatrixXd Mc = draws_to_mat(pts, d);
        if (!saveprefix.empty()) {
            std::ofstream o(saveprefix + "_chain" + std::to_string(k) + ".csv");
            for (int i = 0; i < d; ++i) { if (i) o << ","; o << "x" << i; }
            o << "\n";
            for (int j = 0; j < (int)pts.size(); ++j) {
                for (int i = 0; i < d; ++i) { if (i) o << ","; o << Mc(i, j); }
                o << "\n";
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        for (auto &p : pts) {
            Eigen::VectorXd x(d);
            for (int i = 0; i < d; ++i) x(i) = p[i];
            total++; inside += vpoly_contains(V, x, 1e-6) ? 1 : 0;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        lp_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        unsigned int me = 0;
        eA.push_back(effective_sample_size<NT, VT, MT>(Mc, me).minCoeff());
        VT pr = univariate_psrf<NT, VT, MT>(Mc);
        Eigen::VectorXd mu = Mc.rowwise().mean();
        Eigen::MatrixXd Xc = Mc.colwise() - mu;
        Eigen::VectorXd rad(Mc.cols());
        for (int t = 0; t < Mc.cols(); ++t) rad(t) = Xc.col(t).norm();
        double rms = std::sqrt(Xc.array().square().sum() / Mc.cols());
        double rs = std::sqrt((rad.array() - rad.mean()).square().sum() / (Mc.cols() - 1));
        f1.push_back(rs / rms);
        se1.push_back(rs / std::sqrt(std::max<double>(eA.back(), 1.0)) / rms);
        std::seed_seq sg{20262002, sharedG ? 0 : k, W, D}; std::mt19937 gg(sg);
        Eigen::MatrixXd G(P1, d);
        for (int p = 0; p < P1; ++p) { Eigen::VectorXd g = gauss_vec(d, gg); G.row(p) = g.transpose(); }
        std::vector<double> ku = full_ku(Xc, G);
        f2.push_back(median(ku));
        std::seed_seq sb{20262003, k, W, D}; std::mt19937 gb(sb);
        se2b.push_back(boot_se_median(ku, gb));
        se2t.push_back(se_of_mean(block_medians(Xc, G, K)));
        printf("chain %d minESS=%.1f maxPSRF1=%.3f F1=%.4f F2=%.4f\n",
               k, eA.back(), pr.maxCoeff(), f1.back(), f2.back());
    }
    double dev1, dev2b, dev2t;
    bool ag1 = chain_agree(f1, se1, dev1), ag2b = chain_agree(f2, se2b, dev2b), ag2t = chain_agree(f2, se2t, dev2t);
    double mnE = *std::min_element(eA.begin(), eA.end());
    bool ok = (mnE >= 100.0) && ag1 && ag2b && ag2t && (inside == total);
    printf("vpilot (W=%d,D=%d) minESS=%.1f F1dev=%.2f F2boot=%.2f F2batch=%.2f memb=%d/%d samp=%ldms lpcheck=%ldms %s\n",
           W, D, mnE, dev1, dev2b, dev2t, inside, total, samp_ms, lp_ms, ok ? "ADEQUATE" : "INADEQUATE");
    return ok ? 0 : 1;
}

// Gate B: held-out polytope instances. ninst fresh vertex sets (seeds disjoint
// from pilot), ntrain for orientation fitting, rest held out; nds datasets each.
// Orientation fit on train vs controls; scored on held-out vs FRESH controls.
// Exit 0 iff every dataset passes spot checks (membership 100%, minESS>=50).
static int mode_vdisc(int d, int M, int ninst, int ntrain, int nds) {
    const int W = 200, D = 500, P1 = 50, NCTRL = 30;
    std::cout << "2B.3 Gate B vdisc d=" << d << " M=" << M << " inst=" << ninst
              << " train=" << ntrain << " ds=" << nds << " (W=" << W << ",D=" << D << ")\n";
    struct DS { double f1, f2, ess; int inst; };
    std::vector<std::vector<DS>> PI(ninst);
    std::vector<int> failed;
    for (int ii = 0; ii < ninst; ++ii) {
        Eigen::MatrixXd V(M, d);
        std::mt19937 gv(20262101 + ii);
        for (int j = 0; j < M; ++j) { Eigen::VectorXd g = gauss_vec(d, gv); g.normalize(); V.row(j) = g.transpose(); }
        VT bb = VT::Ones(M);
        Vpolytope P(d, V, bb);
        auto ib = P.ComputeInnerBall();
        for (int s = 0; s < nds; ++s) {
            std::seed_seq ss{20262201, ii, s};
            std::mt19937 gen(ss);
            Point start(d);
            bool placed = false;
            for (int att = 0; att < 500 && !placed; ++att) {
                Eigen::VectorXd u = gauss_vec(d, gen); u.normalize();
                Eigen::VectorXd c = ib.first.getCoefficients() + 0.25 * ib.second * u;
                if (vpoly_contains(V, c, 1e-6)) { start = Point(c); placed = true; }
            }
            if (!placed) { failed.push_back(ii * 100 + s); continue; }
            std::mt19937 tmp(ss); RNGType rng(d); rng.set_seed(tmp());
            std::list<Point> pts;
            auto t0 = std::chrono::high_resolution_clock::now();
            uniform_sampling<CDHRWalk>(pts, P, rng, (unsigned)W, (unsigned)D, start, 50);
            auto t1 = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXd Mc = draws_to_mat(pts, d);
            int in = 0;
            for (auto &p : pts) {
                Eigen::VectorXd x(d);
                for (int i = 0; i < d; ++i) x(i) = p[i];
                in += vpoly_contains(V, x, 1e-6) ? 1 : 0;
            }
            unsigned int me = 0;
            double eR = effective_sample_size<NT, VT, MT>(Mc, me).minCoeff();
            if (in != (int)pts.size() || eR < 50.0) { failed.push_back(ii * 100 + s); continue; }
            Eigen::VectorXd mu = Mc.rowwise().mean();
            Eigen::MatrixXd Xc = Mc.colwise() - mu;
            double rms, a, b, rm;
            std::seed_seq sp{20262203, ii, s}; std::mt19937 gp(sp);
            F1F2(Xc, gp, P1, rms, a, b, rm);
            PI[ii].push_back({a, b, eR, ii});
            printf("inst %d ds %d F1=%.4f F2=%.4f ess=%.0f %ldms\n", ii, s, a, b, eR,
                   std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        }
    }
    // controls: train-orientation set (20262202) + fresh held-out set (20262204)
    std::vector<DS> Ctr[2][2]; // [split][body]; split 0=train-orient, 1=fresh
    for (int sp = 0; sp < 2; ++sp) for (int b = 0; b < 2; ++b) for (int r = 0; r < NCTRL; ++r) {
        std::seed_seq ss{20262202 + 2 * sp, b, r};
        std::mt19937 gen(ss);
        Eigen::MatrixXd X = sample_body2(b, d, D, 1.0, gen);
        Eigen::VectorXd mu = X.rowwise().mean();
        Eigen::MatrixXd Xc = X.colwise() - mu;
        double rms, a, bb, rm;
        std::seed_seq q{20262203, 100 + sp, 10 * b + r}; std::mt19937 gq(q);
        F1F2(Xc, gq, P1, rms, a, bb, rm);
        Ctr[sp][b].push_back({a, bb, (double)D, -1});
    }
    auto col = [](const std::vector<DS> &v, double DS::*f) {
        std::vector<double> o; for (auto &x : v) o.push_back(x.*f); return o; };
    auto meanof = [](const std::vector<double> &v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    printf("per-instance F1/F2 (mean over %d datasets):\n", nds);
    for (int ii = 0; ii < ninst; ++ii) {
        auto a = col(PI[ii], &DS::f1), b = col(PI[ii], &DS::f2);
        printf("inst %d [%s] F1=%.4f F2=%.4f\n", ii, ii < ntrain ? "train" : "heldout", meanof(a), meanof(b));
    }
    // orientation on train vs orientation-controls; score on held-out vs fresh
    std::vector<DS> TR, HO;
    for (int ii = 0; ii < ninst; ++ii) for (auto &x : PI[ii]) (ii < ntrain ? TR : HO).push_back(x);
    const char *bnm[2] = {"cube", "ball"};
    for (int b = 0; b < 2; ++b) {
        for (int f = 0; f < 2; ++f) {
            double DS::*mf = f ? &DS::f2 : &DS::f1;
            const char *fn = f ? "F2" : "F1";
            double s = (meanof(col(TR, mf)) >= meanof(col(Ctr[0][b], mf))) ? 1.0 : -1.0;
            double Atr = auc_mw(col(TR, mf), col(Ctr[0][b], mf));
            double A = auc_mw(col(HO, mf), col(Ctr[1][b], mf));
            double Ao = (s > 0) ? A : 1.0 - A;
            auto ci = hm_ci(A, (int)col(HO, mf).size(), NCTRL);
            std::pair<double,double> cio = (s > 0) ? ci : std::make_pair(1.0 - ci.second, 1.0 - ci.first);
            printf("poly vs %-4s %s: trainAUC=%.3f dir=%s heldoutAUC=%.3f[%.2f,%.2f]\n",
                   bnm[b], fn, Atr, s > 0 ? "poly>ctrl" : "poly<ctrl", Ao, cio.first, cio.second);
        }
    }
    if (!failed.empty()) {
        std::cout << "FAILED datasets (escalate budget, none excluded):";
        for (int r : failed) std::cout << " " << r;
        std::cout << "\n";
        return 1;
    }
    std::cout << "Gate B d=" << d << ": directions fit on train instances, scored on held-out\n";
    return 0;
}

int main(int argc, char **argv) {
    std::string mode = (argc > 1) ? argv[1] : "mapping";
    std::string out = (argc > 2) ? argv[2] : ".";
    if (mode == "mapping") return mode_mapping();
    if (mode == "confirm") return mode_confirm(out);
    if (mode == "pilot") return mode_pilot();
    if (mode == "b5campaign") return mode_b5campaign();
    if (mode == "adequacy") {
        if (argc < 5) { std::cerr << "usage: phase2b adequacy m W D\n"; return 2; }
        return mode_adequacy(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    }
    if (mode == "vpilot") {
        if (argc < 6) { std::cerr << "usage: phase2b vpilot d M W D [saveprefix] [shared]\n"; return 2; }
        return mode_vpilot(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                           argc > 6 ? argv[6] : "", argc > 7 ? std::string(argv[7]) == "shared" : false);
    }
    if (mode == "vdisc") {
        if (argc < 7) { std::cerr << "usage: phase2b vdisc d M ninst ntrain nds\n"; return 2; }
        return mode_vdisc(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    }
    if (mode == "discriminate") {
        if (argc < 6) { std::cerr << "usage: phase2b discriminate m W D outdir\n"; return 2; }
        return mode_discriminate(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argv[5]);
    }
    std::cerr << "usage: phase2b [mapping|confirm outdir|pilot|b5campaign|adequacy m W D|discriminate m W D outdir|vpilot d M W D [saveprefix] [shared]|vdisc d M ninst ntrain nds]\n";
    return 2;
}
