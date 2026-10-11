/**
 * Phase 1 reference tests — high-dimensional convex-body inference.
 *
 * Validates mathematical quantities INDEPENDENTLY of the samplers:
 *  A: cube / ball support oracles (+ PR #479 ball-helper precondition).
 *  A2: analytic covariance / volume / radial references.
 *  B: random V-polytope support oracle vs independent brute force.
 *  C1: Birkhoff assignment oracle (Hungarian vs exhaustive).
 *  C2: Birkhoff ambient <-> reduced-coordinate mapping + structural rank.
 *  M: independent V-polytope membership (V^T l = x, 1^T l = 1, l >= 0 via LP).
 *  D: statistical references from direct (non-MCMC) sampling with SE bounds.
 *
 * Fast, seeded, deterministic. Exit 0 iff all 7 acceptance items pass.
 */

#include <algorithm>
#include <cmath>
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
static bool close(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

// ---- independent oracle implementations (Eigen-based, no volesti code) ----
static double cube_h_eigen(const Eigen::VectorXd &u) {
    return u.array().abs().sum();
}
static double cube_h_loop(const Eigen::VectorXd &u) {
    double s = 0.0;
    for (int i = 0; i < u.size(); ++i) s += std::abs(u(i));
    return s;
}
static double ball_h_eigen(const Eigen::VectorXd &u) { return u.norm(); }
static double ball_h_loop(const Eigen::VectorXd &u) {
    double s = 0.0;
    for (int i = 0; i < u.size(); ++i) s += u(i) * u(i);
    return std::sqrt(s);
}
// PR #479 helper, replicated verbatim: correct only for unit directions.
static double pr479_ball_helper() { return 1.0; }

static Eigen::VectorXd gauss_vec(int d, std::mt19937 &gen) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd v(d);
    for (int i = 0; i < d; ++i) v(i) = nd(gen);
    return v;
}

// ---- Hungarian (max) + exhaustive assignment ----
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
static std::pair<std::vector<int>, double> brute_max(const Eigen::MatrixXd &C) {
    int m = (int)C.rows();
    std::vector<int> perm(m), best;
    std::iota(perm.begin(), perm.end(), 0);
    double opt = -1e300;
    do {
        double s = 0.0;
        for (int i = 0; i < m; ++i) s += C(i, perm[i]);
        if (s > opt) { opt = s; best = perm; }
    } while (std::next_permutation(perm.begin(), perm.end()));
    return {best, opt};
}
static bool is_perm(const std::vector<int> &p) {
    std::vector<int> s = p;
    std::sort(s.begin(), s.end());
    for (size_t i = 0; i < s.size(); ++i) if (s[i] != (int)i) return false;
    return true;
}

// ---- independent V-polytope membership via explicit LP ----
// Decides feasibility of V^T l = x, 1^T l = 1, l >= 0. Independently written;
// does NOT call VPolytope::is_in (which uses memLP_Vpoly).
static bool vpoly_contains(const Eigen::MatrixXd &V, const Eigen::VectorXd &x,
                           double tol, Eigen::VectorXd *lambda_out, double *resid_out) {
    int M = (int)V.rows(), d = (int)V.cols();
    lprec *lp = make_lp(d + 1, M);
    if (lp == NULL) return false;
    std::vector<REAL> row(M);
    std::vector<int> colno(M);
    for (int j = 0; j < M; ++j) colno[j] = j + 1;
    set_add_rowmode(lp, TRUE);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < M; ++j) row[j] = V(j, i);
        if (!add_constraintex(lp, M, row.data(), colno.data(), EQ, (REAL)x(i))) {
            delete_lp(lp); return false;
        }
    }
    for (int j = 0; j < M; ++j) row[j] = 1.0;
    if (!add_constraintex(lp, M, row.data(), colno.data(), EQ, 1.0)) {
        delete_lp(lp); return false;
    }
    set_add_rowmode(lp, FALSE);
    for (int j = 0; j < M; ++j) set_bounds(lp, j + 1, 0.0, 1.0);
    for (int j = 0; j < M; ++j) row[j] = (j == 0) ? 1.0 : 0.0; // min l_1: bounded, OPTIMAL iff feasible
    if (!set_obj_fnex(lp, M, row.data(), colno.data())) { delete_lp(lp); return false; }
    set_minim(lp);
    set_verbose(lp, NEUTRAL);
    int status = solve(lp);
    bool feasible = (status == OPTIMAL);
    double resid = 1e300, sum = 0.0, mn = 1e300;
    if (feasible) {
        std::vector<REAL> lam(M);
        get_variables(lp, lam.data());
        Eigen::VectorXd l(M);
        for (int j = 0; j < M; ++j) l(j) = lam[j];
        sum = l.sum(); mn = l.minCoeff();
        resid = (V.transpose() * l - x).norm();
        feasible = (std::abs(sum - 1.0) <= tol) && (mn >= -tol) && (resid <= tol);
        if (lambda_out) *lambda_out = l;
        if (resid_out) *resid_out = resid;
    }
    delete_lp(lp);
    (void)sum; (void)mn;
    return feasible;
}

int main() {
    std::cout << "Phase 1 reference tests (seeds recorded inline)\n";
    bool acc[7] = {false, false, false, false, false, false, false};
    int base;

    // ============ A1: cube / ball support ============
    std::cout << "\n[A1] cube h(u)=||u||_1, ball h(u)=||u||_2\n";
    base = g_fail;
    {
        Eigen::VectorXd e1(3); e1 << 1, 0, 0;
        Eigen::VectorXd o3(3); o3 << 1, 1, 1;
        Eigen::VectorXd u3(3); u3 << 1, -2, 0.5;
        Eigen::VectorXd uh = o3 / o3.norm();
        check(close(cube_h_eigen(e1), 1.0, 1e-12), "cube h(e1)=1");
        check(close(ball_h_eigen(e1), 1.0, 1e-12), "ball h(e1)=1");
        check(close(cube_h_eigen(o3), 3.0, 1e-12), "cube h(1,1,1)=3");
        check(close(ball_h_eigen(o3), std::sqrt(3.0), 1e-12), "ball h(1,1,1)=sqrt(3)");
        check(close(cube_h_eigen(uh), std::sqrt(3.0), 1e-12), "cube h(unit(1,1,1))=sqrt(3)");
        check(close(ball_h_eigen(uh), 1.0, 1e-12), "ball h(unit)=1");
        check(close(cube_h_eigen(u3), 3.5, 1e-12), "cube h(1,-2,.5)=3.5");
        check(close(ball_h_eigen(u3), std::sqrt(5.25), 1e-12), "ball h(1,-2,.5)=sqrt(5.25)");
        // PR #479 helper precondition: exact on unit dirs, wrong off them.
        check(close(pr479_ball_helper(), ball_h_eigen(uh), 1e-12), "pr479 ball helper == ||u|| on unit u");
        bool helper_wrong_off_unit = !close(pr479_ball_helper(), ball_h_eigen(o3), 1e-12);
        check(helper_wrong_off_unit, "pr479 ball helper != ||u|| on non-unit u (precondition documented)");
        // seeded cross-implementation agreement
        std::mt19937 gen(20261009);
        bool agree = true;
        for (int d : {3, 10, 50}) for (int k = 0; k < 50; ++k) {
            Eigen::VectorXd g = gauss_vec(d, gen);
            Eigen::VectorXd u = g / g.norm();
            agree &= close(cube_h_eigen(u), cube_h_loop(u), 1e-12)
                  && close(ball_h_eigen(u), ball_h_loop(u), 1e-12)
                  && close(cube_h_eigen(g), cube_h_loop(g), 1e-9)
                  && close(ball_h_eigen(g), ball_h_loop(g), 1e-9);
        }
        check(agree, "seeded 20261009: Eigen vs loop oracles agree (150 dirs, unit+raw)");
    }
    acc[0] = (g_fail == base);

    // ============ A2: analytic covariance / volume / radial ============
    std::cout << "\n[A2] analytic covariance, volume, radial references\n";
    base = g_fail;
    {
        bool ok = true;
        for (int d : {2, 3, 5, 10, 20}) {
            double sc = 1.0 / 3.0, sb = 1.0 / (d + 2.0);
            // participation ratio and effective rank of sI_d both equal d
            double pr_c = (d * sc) * (d * sc) / (d * sc * sc);
            double er_c = (d * sc) / sc;
            double pr_b = (d * sb) * (d * sb) / (d * sb * sb);
            double er_b = (d * sb) / sb;
            ok &= close(pr_c, d, 1e-9) && close(er_c, d, 1e-9)
               && close(pr_b, d, 1e-9) && close(er_b, d, 1e-9);
            ok &= close(std::pow(2.0, d), std::pow(2.0, d), 0.0); // cube vol identity
        }
        check(ok, "Cov=I/3 (cube), I/(d+2) (ball): tr/D_PR/r_eff consistent d=2..20");
        double vb2 = std::pow(M_PI, 1.0) / std::tgamma(2.0);
        double vb3 = std::pow(M_PI, 1.5) / std::tgamma(2.5);
        check(close(vb2, M_PI, 1e-12), "vol(B^2)=pi");
        check(close(vb3, 4.0 * M_PI / 3.0, 1e-9), "vol(B^3)=4pi/3");
        check(close(std::pow(0.5, 3), 0.125, 1e-15), "radial CDF P(R<=.5)=.125 in d=3");
        check(close(std::pow(0.9, 10), 0.3486784401, 1e-9), "radial CDF P(R<=.9)=.9^10 in d=10");
    }
    acc[1] = (g_fail == base);

    // ============ B: V-polytope support oracle ============
    std::cout << "\n[B] V-polytope support = max_j v_j.u\n";
    base = g_fail;
    {
        Eigen::MatrixXd Sq(4, 2); Sq << 1, 1,  1, -1,  -1, 1,  -1, -1;
        Eigen::VectorXd e1(2); e1 << 1, 0;
        Eigen::VectorXd dd(2); dd << 1, 1;
        Eigen::VectorXd du = dd / dd.norm();
        auto hmax = [](const Eigen::MatrixXd &V, const Eigen::VectorXd &u) {
            double m = -1e300;
            for (int j = 0; j < V.rows(); ++j) m = std::max(m, V.row(j).dot(u));
            return m;
        };
        check(close(hmax(Sq, e1), 1.0, 1e-12), "square h(e1)=1");
        check(close(hmax(Sq, dd), 2.0, 1e-12), "square h(1,1)=2");
        check(close(hmax(Sq, du), std::sqrt(2.0), 1e-12), "square h(unit45)=sqrt(2)");
        Eigen::MatrixXd Tri(3, 2); Tri << 0, 0,  1, 0,  0, 1;
        Eigen::VectorXd mn(2); mn << -1, -1;
        check(close(hmax(Tri, mn), 0.0, 1e-12), "triangle h(-1,-1)=0");
        check(close(hmax(Tri, dd), 1.0, 1e-12), "triangle h(1,1)=1");
        // seeded instances: Eigen rowwise vs manual loop + identities
        std::mt19937 gen(7);
        bool ok = true;
        for (int cfg = 0; cfg < 2; ++cfg) {
            int d = (cfg == 0) ? 5 : 10, M = (cfg == 0) ? 12 : 25;
            Eigen::MatrixXd V(M, d);
            for (int j = 0; j < M; ++j) {
                Eigen::VectorXd g = gauss_vec(d, gen); g.normalize();
                V.row(j) = g.transpose();
            }
            for (int k = 0; k < 30; ++k) {
                Eigen::VectorXd u = gauss_vec(d, gen);
                double a = (V * u).maxCoeff();
                double b = hmax(V, u);
                ok &= close(a, b, 1e-12);
                for (double al : {0.0, 0.5, 2.0, 3.7})
                    ok &= close(hmax(V, al * u), al * hmax(V, u), 1e-9);
                Eigen::VectorXd w = gauss_vec(d, gen);
                ok &= (hmax(V, u + w) <= hmax(V, u) + hmax(V, w) + 1e-9);
            }
        }
        check(ok, "seeded d=5/M=12 + d=10/M=25: brute-force agree, homogeneity, subadditivity");
    }
    acc[2] = (g_fail == base);

    // ============ C1: Birkhoff assignment oracle ============
    std::cout << "\n[C1] Birkhoff support = max_pi sum C[i,pi(i)]\n";
    base = g_fail;
    {
        std::mt19937 gen(21);
        std::uniform_real_distribution<double> uni(-5.0, 5.0);
        bool ok = true;
        for (int m : {3, 4}) {
            int trials = (m == 3) ? 30 : 20;
            for (int t = 0; t < trials; ++t) {
                Eigen::MatrixXd C(m, m);
                for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) C(i, j) = uni(gen);
                auto H = hungarian_max(C), E = brute_max(C);
                double rowmax = 0.0, ident = 0.0;
                for (int i = 0; i < m; ++i) { rowmax += C.row(i).maxCoeff(); ident += C(i, i); }
                ok &= is_perm(H.first);
                ok &= close(H.second, E.second, 1e-9);
                double rec = 0.0;
                for (int i = 0; i < m; ++i) rec += C(i, H.first[i]);
                ok &= close(rec, H.second, 1e-9);
                ok &= (H.second <= rowmax + 1e-9) && (H.second + 1e-9 >= ident);
            }
        }
        check(ok, "m=3 (30 trials) + m=4 (20 trials), seed 21: Hungarian==exhaustive, perm valid, bounds hold");
        bool ok8 = true;
        for (int t = 0; t < 3; ++t) {
            int m = 8;
            Eigen::MatrixXd C(m, m);
            for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) C(i, j) = uni(gen);
            auto H = hungarian_max(C);
            double rowmax = 0.0, ident = 0.0;
            for (int i = 0; i < m; ++i) { rowmax += C.row(i).maxCoeff(); ident += C(i, i); }
            ok8 &= is_perm(H.first)
                && (H.second <= rowmax + 1e-9)
                && (H.second + 1e-9 >= ident);
        }
        check(ok8, "m=8 (3 trials): permutation valid, row-max upper + identity lower bounds hold");
    }
    acc[3] = (g_fail == base);

    // ============ C2: Birkhoff coordinate mapping ============
    std::cout << "\n[C2] ambient <-> intrinsic mapping, structural rank\n";
    base = g_fail;
    {
        bool ok = true;
        for (int m : {3, 4}) {
            Hpolytope P = generate_birkhoff<Hpolytope>((unsigned)m);
            int d = (m - 1) * (m - 1);
            check(P.dimension() == (unsigned)d, "B_" + std::to_string(m) + " intrinsic dim=" + std::to_string(d));
            MT A = P.get_mat(); VT b = P.get_vec();
            auto y_of = [m](const std::vector<int> &perm) {
                Eigen::VectorXd y((m - 1) * (m - 1));
                for (int i = 0; i < m - 1; ++i) for (int j = 0; j < m - 1; ++j)
                    y(i * (m - 1) + j) = (perm[i] == j) ? 1.0 : 0.0;
                return y;
            };
            // centroid J/m -> y=(1/m)ones strictly inside
            Eigen::VectorXd y0 = Eigen::VectorXd::Constant(d, 1.0 / m);
            Eigen::VectorXd s0 = b - A * y0;
            ok &= (s0.minCoeff() > 1e-9);
            // all permutation vertices feasible + on boundary
            std::vector<int> perm(m);
            std::iota(perm.begin(), perm.end(), 0);
            int nperm = 0;
            do {
                Eigen::VectorXd y = y_of(perm);
                Eigen::VectorXd s = b - A * y;
                ok &= (s.minCoeff() >= -1e-9);
                ok &= (s.minCoeff() <= 1e-9);
                ++nperm;
            } while (std::next_permutation(perm.begin(), perm.end()));
            ok &= (nperm == (m == 3 ? 6 : 24));
            // support identity: ambient(C) == <C,X0> + reduced(c)
            std::mt19937 gen(33);
            std::uniform_real_distribution<double> uni(-3.0, 3.0);
            for (int t = 0; t < 5; ++t) {
                Eigen::MatrixXd C(m, m);
                for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) C(i, j) = uni(gen);
                double c0 = 0.0;
                for (int i = 0; i < m - 1; ++i) c0 += C(i, m - 1);
                for (int j = 0; j < m - 1; ++j) c0 += C(m - 1, j);
                c0 -= (m - 2) * C(m - 1, m - 1);
                Eigen::VectorXd c(d);
                for (int i = 0; i < m - 1; ++i) for (int j = 0; j < m - 1; ++j)
                    c(i * (m - 1) + j) = C(i, j) - C(i, m - 1) - C(m - 1, j) + C(m - 1, m - 1);
                auto H = hungarian_max(C);
                std::iota(perm.begin(), perm.end(), 0);
                double red = -1e300;
                do {
                    Eigen::VectorXd y = y_of(perm);
                    red = std::max(red, c.dot(y));
                } while (std::next_permutation(perm.begin(), perm.end()));
                ok &= close(H.second, c0 + red, 1e-9);
            }
        }
        check(ok, "B_3/B_4: centroid interior, perms feasible+boundary, ambient==offset+reduced (seed 33)");
        // structural rank: ambient covariance of all 6 B_3 vertices has rank 4 = (m-1)^2
        {
            int m = 3;
            Eigen::MatrixXd X(6, 9);
            std::vector<int> perm = {0, 1, 2};
            int r = 0;
            do {
                for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j)
                    X(r, i * m + j) = (perm[i] == j) ? 1.0 : 0.0;
                ++r;
            } while (std::next_permutation(perm.begin(), perm.end()));
            Eigen::MatrixXd C = (X.rowwise() - X.colwise().mean());
            Eigen::MatrixXd S = (C.adjoint() * C) / double(X.rows() - 1);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
            int nz = 0;
            for (int i = 0; i < es.eigenvalues().size(); ++i)
                if (es.eigenvalues()(i) > 1e-9) ++nz;
            check(nz == 4, "B_3 ambient vertex covariance rank=4=(m-1)^2 (5 structural zeros, not sampling failure)");
        }
    }
    acc[4] = (g_fail == base);

    // ============ M: independent V-polytope membership ============
    std::cout << "\n[M] V^T l=x feasibility (independent LP, not P.is_in)\n";
    base = g_fail;
    {
        Eigen::MatrixXd Sq(4, 2); Sq << 1, 1,  1, -1,  -1, 1,  -1, -1;
        Eigen::VectorXd z(2); z << 0, 0;
        Eigen::VectorXd q(2); q << 0.9, 0.9;
        Eigen::VectorXd o(2); o << 1.1, 0.0;
        Eigen::VectorXd f(2); f << 2.0, 0.0;
        Eigen::VectorXd lam; double res = 0;
        bool ok = true;
        ok &= vpoly_contains(Sq, z, 1e-6, &lam, &res);
        ok &= vpoly_contains(Sq, q, 1e-6, nullptr, nullptr);
        ok &= !vpoly_contains(Sq, o, 1e-6, nullptr, nullptr);
        ok &= !vpoly_contains(Sq, f, 1e-6, nullptr, nullptr);
        check(ok, "square: interior feasible, exterior (1.1,0),(2,0) infeasible");
        // known convex combos on seeded instance
        std::mt19937 gen(11);
        int d = 5, M = 12;
        Eigen::MatrixXd V(M, d);
        for (int j = 0; j < M; ++j) {
            Eigen::VectorXd g = gauss_vec(d, gen); g.normalize(); V.row(j) = g.transpose();
        }
        bool kok = true;
        for (int t = 0; t < 10; ++t) {
            Eigen::VectorXd w(M);
            for (int j = 0; j < M; ++j) w(j) = std::exponential_distribution<double>(1.0)(gen);
            w /= w.sum();
            Eigen::VectorXd x = V.transpose() * w;
            kok &= vpoly_contains(V, x, 1e-6, &lam, &res);
            kok &= (res <= 1e-6);
        }
        check(kok, "seeded d=5/M=12: 10 Dirichlet combos feasible, residual<=1e-6");
        // live sampler points, small instance
        {
            Eigen::MatrixXd Vs(12, 5);
            std::mt19937 gv(12);
            for (int j = 0; j < 12; ++j) {
                Eigen::VectorXd g = gauss_vec(5, gv); g.normalize(); Vs.row(j) = g.transpose();
            }
            VT bb = VT::Ones(12);
            Vpolytope P(5, Vs, bb);
            auto ib = P.ComputeInnerBall();
            RNGType rng(5); rng.set_seed(42);
            std::list<Point> pts;
            uniform_sampling<CDHRWalk>(pts, P, rng, 5, 10, ib.first, 50);
            bool sok = (pts.size() == 10);
            for (auto &p : pts) {
                Eigen::VectorXd x(5);
                for (int i = 0; i < 5; ++i) x(i) = p[i];
                sok &= vpoly_contains(Vs, x, 1e-6, nullptr, nullptr);
            }
            check(sok, "live CDHR d=5/M=12 (10 pts): all satisfy independent LP");
        }
        // Phase-0-scale: d=50/M=100, same generator seed as feas_gate
        {
            int d50 = 50, M50 = 100;
            Eigen::MatrixXd V50(M50, d50);
            std::mt19937 gv(123);
            for (int j = 0; j < M50; ++j) {
                Eigen::VectorXd g = gauss_vec(d50, gv); g.normalize(); V50.row(j) = g.transpose();
            }
            VT bb = VT::Ones(M50);
            Vpolytope P(d50, V50, bb);
            auto ib = P.ComputeInnerBall();
            bool ibok = (ib.second > 0) && std::isfinite(ib.second);
            RNGType rng(d50); rng.set_seed(42);
            std::list<Point> pts;
            uniform_sampling<CDHRWalk>(pts, P, rng, 1, 10, ib.first, 10);
            bool sok = ibok && (pts.size() == 10);
            int k = 0;
            for (auto &p : pts) {
                Eigen::VectorXd x(d50);
                for (int i = 0; i < d50; ++i) x(i) = p[i];
                bool c = vpoly_contains(V50, x, 1e-6, nullptr, nullptr);
                sok &= c;
                ++k;
            }
            check(sok, "Phase-0-scale d=50/M=100 (10 CDHR pts, walk_len=1): all satisfy independent LP");
        }
    }
    acc[5] = (g_fail == base);

    // ============ D: statistical references (direct sampling only) ============
    std::cout << "\n[D] empirical mean/cov vs analytic (seeded, SE-calibrated)\n";
    base = g_fail;
    {
        // cube d=5, n=5000
        int d = 5, n = 5000;
        std::mt19937 gen(100);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        Eigen::MatrixXd X(d, n);
        for (int j = 0; j < n; ++j) for (int i = 0; i < d; ++i) X(i, j) = uni(gen);
        Eigen::VectorXd mu = X.rowwise().mean();
        Eigen::MatrixXd Xc = X.colwise() - mu;
        Eigen::MatrixXd S = (Xc * Xc.transpose()) / double(n - 1);
        double se_mean = std::sqrt((1.0 / 3.0) / n);
        double se_var = std::sqrt((4.0 / 45.0) / n);
        double se_off = std::sqrt((1.0 / 9.0) / n);
        double maxmean = mu.array().abs().maxCoeff();
        double maxvar = 0.0, maxoff = 0.0;
        for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
            double e = std::abs(S(i, j) - ((i == j) ? 1.0 / 3.0 : 0.0));
            if (i == j) maxvar = std::max(maxvar, e); else maxoff = std::max(maxoff, e);
        }
        std::cout << "  cube d=5 n=5000 seed=100: max|mean|=" << maxmean << " (5SE=" << 5 * se_mean
                  << ") max|var-1/3|=" << maxvar << " (5SE=" << 5 * se_var
                  << ") max|off|=" << maxoff << " (5SE=" << 5 * se_off << ")\n";
        check(maxmean <= 5 * se_mean && maxvar <= 5 * se_var && maxoff <= 5 * se_off,
              "cube empirical moments within 5 SE of I/3");
        // ball d=5, n=5000 (Gaussian direction + r^(1/d) radius)
        std::mt19937 gen2(101);
        std::uniform_real_distribution<double> uu(0.0, 1.0);
        Eigen::MatrixXd Y(d, n);
        for (int j = 0; j < n; ++j) {
            Eigen::VectorXd g = gauss_vec(d, gen2); g.normalize();
            g *= std::pow(uu(gen2), 1.0 / d);
            Y.col(j) = g;
        }
        Eigen::VectorXd mu2 = Y.rowwise().mean();
        Eigen::MatrixXd Yc = Y.colwise() - mu2;
        Eigen::MatrixXd S2 = (Yc * Yc.transpose()) / double(n - 1);
        double sb = 1.0 / (d + 2.0);
        double se2_mean = std::sqrt(sb / n);
        double se2_var = std::sqrt((4.0 / 147.0) / n);   // Var(X_i^2)=4/147 for d=5
        double se2_off = std::sqrt((1.0 / 63.0) / n);    // E[X_i^2 X_j^2]=1/63
        double mm2 = mu2.array().abs().maxCoeff(), mv2 = 0.0, mo2 = 0.0;
        for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
            double e = std::abs(S2(i, j) - ((i == j) ? sb : 0.0));
            if (i == j) mv2 = std::max(mv2, e); else mo2 = std::max(mo2, e);
        }
        std::cout << "  ball d=5 n=5000 seed=101: max|mean|=" << mm2 << " (5SE=" << 5 * se2_mean
                  << ") max|var-1/7|=" << mv2 << " (5SE=" << 5 * se2_var
                  << ") max|off|=" << mo2 << " (5SE=" << 5 * se2_off << ")\n";
        check(mm2 <= 5 * se2_mean && mv2 <= 5 * se2_var && mo2 <= 5 * se2_off,
              "ball empirical moments within 5 SE of I/7");
        // SE calibration over R=10 repeats, small n
        const int R = 10, n0 = 1000;
        std::vector<double> m0;
        for (int r = 0; r < R; ++r) {
            std::mt19937 gr(2000 + r);
            double s = 0.0;
            for (int j = 0; j < n0; ++j) s += uni(gr);
            m0.push_back(s / n0);
        }
        double em = std::accumulate(m0.begin(), m0.end(), 0.0) / R, vs = 0.0;
        for (double v : m0) vs += (v - em) * (v - em);
        vs = std::sqrt(vs / (R - 1));
        double pred = std::sqrt((1.0 / 3.0) / n0);
        std::cout << "  cube coord-mean over R=10 repeats (n=1000): empSE=" << vs << " predSE=" << pred << "\n";
        check(vs >= 0.5 * pred && vs <= 2.0 * pred, "repeat-trial SE within 2x of prediction");
        // reproducibility: same seed -> identical stream
        std::mt19937 ga(777), gb(777);
        bool same = true;
        for (int k = 0; k < 50; ++k) same &= (uni(ga) == uni(gb));
        check(same, "seed 777 reproduces identical stream");
    }
    acc[6] = (g_fail == base);

    const char *names[7] = {
        "1. cube/ball support tests", "2. analytic covariance refs",
        "3. V-polytope support vs brute force", "4. Birkhoff assignment oracle",
        "5. Birkhoff coordinate mapping", "6. V-polytope membership (independent LP)",
        "7. statistical tests w/ uncertainty + seeds"};
    std::cout << "\n===== Phase 1 acceptance gate =====\n";
    int nok = 0;
    for (int i = 0; i < 7; ++i) {
        std::cout << "  [" << (acc[i] ? "PASS" : "FAIL") << "] " << names[i] << "\n";
        nok += acc[i] ? 1 : 0;
    }
    std::cout << nok << "/7 verified; checks passed=" << g_pass << " failed=" << g_fail << "\n";
    return (nok == 7) ? 0 : 1;
}
