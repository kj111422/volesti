/**
 * Phase 2A: scale-controlled identifiability screening (direct sampling only).
 *
 * Three experiments, cube vs ball family, d in {5,10,20,50,100},
 * n/d in {0.5,1,2,5,10}, 30 reps, seeded:
 *  A: raw bodies (scale intact) — reference baseline.
 *  B: sample-only RMS normalization — which discrimination survives in
 *     scale-free features (verified numerically invariant); radial stats
 *     become relative (radmean/rms, radstd/rms).
 *  C: population-matched control — ball scaled to radius sqrt((d+2)/3) so its
 *     population covariance equals the cube's (I/3); residual discrimination
 *     must come from higher-order geometry.
 *
 * Support-function gaps are ground-truth oracles (known body required),
 * never classifier features. No MCMC here. Exit 0 on successful completion.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Eigen"

static const int MASTER_SEED = 20261101;
static const int REPS = 30;
static const int DIRS_ORACLE = 30;
static const int P1 = 20;   // 1-D projections for skew/kurtosis
static const int PK = 10;   // k-D projections for k in {2,5}

static Eigen::VectorXd gauss_vec(int d, std::mt19937 &gen) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd v(d);
    for (int i = 0; i < d; ++i) v(i) = nd(gen);
    return v;
}

// sample n points in R^d; body 0=cube[-1,1]^d, 1=ball(R=1), 2=ball(R=scaled)
static Eigen::MatrixXd sample_body(int body, int d, int n, double R, std::mt19937 &gen) {
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

struct Feats {
    double rms, radmean, radstd;
    double rank;
    double pr, reff, topshare;
    double askew, kurt;       // median over P1 1-D projections
    double pr2, pr5;          // median projected PR over PK maps
    double gapmean, gapmax;   // oracle, normalized by mean true support
};

static void moments1234(const Eigen::RowVectorXd &z, double &skew, double &kurt) {
    int n = (int)z.size();
    double m = z.mean();
    double m2 = 0, m3 = 0, m4 = 0;
    for (int j = 0; j < n; ++j) {
        double c = z(j) - m, c2 = c * c;
        m2 += c2; m3 += c2 * c; m4 += c2 * c2;
    }
    m2 /= n; m3 /= n; m4 /= n;
    if (!(m2 > 0)) { skew = 0; kurt = -3; return; }
    skew = m3 / std::pow(m2, 1.5);
    kurt = m4 / (m2 * m2) - 3.0;
}
static double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// shape-only features of centered data Xc (d x n); projections drawn from gen
static void shape_feats(const Eigen::MatrixXd &Xc, std::mt19937 &gen,
                        double &pr, double &reff, double &topshare, double &rank,
                        double &askew, double &kurt, double &pr2, double &pr5) {
    int d = (int)Xc.rows(), n = (int)Xc.cols();
    Eigen::MatrixXd S = (Xc * Xc.transpose()) / double(n - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd ev = es.eigenvalues(); // ascending
    double lmax = ev.maxCoeff();
    double tol = lmax * std::max(d, n) * 2.2e-16;
    int r = 0;
    for (int i = 0; i < d; ++i) if (ev(i) > tol) ++r;
    rank = r;
    double tr = S.trace(), tr2 = (S * S).trace();
    pr = (tr2 > 0) ? tr * tr / tr2 : 0;
    reff = (lmax > 0) ? tr / lmax : 0;
    topshare = (tr > 0) ? lmax / tr : 0;
    std::vector<double> sk, ku;
    for (int p = 0; p < P1; ++p) {
        Eigen::VectorXd u = gauss_vec(d, gen);
        Eigen::RowVectorXd z = (u.transpose() * Xc).eval();
        double s, k; moments1234(z, s, k);
        sk.push_back(std::abs(s)); ku.push_back(k);
    }
    askew = median(sk); kurt = median(ku);
    for (int kk = 0; kk < 2; ++kk) {
        int k = (kk == 0) ? 2 : 5;
        std::vector<double> pv;
        if (k <= d) for (int p = 0; p < PK; ++p) {
            Eigen::MatrixXd G(k, d);
            for (int i = 0; i < k; ++i) { Eigen::VectorXd g = gauss_vec(d, gen); G.row(i) = g.transpose(); }
            Eigen::MatrixXd Y = G * Xc;
            Eigen::VectorXd m = Y.rowwise().mean();
            Eigen::MatrixXd Yc = Y.colwise() - m;
            Eigen::MatrixXd Sp = (Yc * Yc.transpose()) / double(n - 1);
            double t = Sp.trace(), t2 = (Sp * Sp).trace();
            pv.push_back((t2 > 0) ? t * t / t2 : 0);
        }
        else pv.push_back(std::numeric_limits<double>::quiet_NaN());
        if (kk == 0) pr2 = median(pv); else pr5 = median(pv);
    }
}

static double auc_mw(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> x, y;
    for (double v : a) if (std::isfinite(v)) x.push_back(v);
    for (double v : b) if (std::isfinite(v)) y.push_back(v);
    if (x.size() < 2 || y.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    double wins = 0;
    for (double u : x) for (double v : y) wins += (u > v) ? 1.0 : ((u == v) ? 0.5 : 0.0);
    return wins / (x.size() * y.size());
}

int main(int argc, char **argv) {
    std::string outdir = (argc > 1) ? argv[1] : ".";
    const std::vector<int> dims = {5, 10, 20, 50, 100};
    const std::vector<double> ratios = {0.5, 1.0, 2.0, 5.0, 10.0};
    std::cout << "Phase 2A screening: master_seed=" << MASTER_SEED << " reps=" << REPS
              << " oracle_dirs=" << DIRS_ORACLE << "\n";

    struct Cell { int d, n; std::vector<Feats> f[3]; std::vector<double> relmean[3], relstd[3]; };
    std::vector<Cell> cells;
    double max_pr_invar = 0, max_ku_invar = 0;

    for (int d : dims) for (double ra : ratios) {
        int n = std::max(3, (int)std::lround(d * ra));
        Cell c; c.d = d; c.n = n;
        double Rmatch = std::sqrt((d + 2.0) / 3.0);
        for (int b = 0; b < 3; ++b) {
            for (int r = 0; r < REPS; ++r) {
                std::seed_seq ss{MASTER_SEED, b, d, n, r};
                std::mt19937 gen(ss);
                double R = (b == 2) ? Rmatch : 1.0;
                Eigen::MatrixXd X = sample_body(b, d, n, R, gen);
                Eigen::VectorXd mu = X.rowwise().mean();
                Eigen::MatrixXd Xc = X.colwise() - mu;
                double rms = std::sqrt((Xc.array().square().sum()) / n);
                Eigen::VectorXd rad(n);
                for (int j = 0; j < n; ++j) rad(j) = Xc.col(j).norm();
                double rmean = rad.mean();
                double rstd = std::sqrt(((rad.array() - rmean).square().sum()) / (n - 1));
                Feats f;
                f.rms = rms; f.radmean = rmean; f.radstd = rstd;
                // B-invariance self-check: same projection draws on RMS-normalized data
                Eigen::MatrixXd Xn = Xc / rms;
                std::mt19937 gsave = gen; // state before shape_feats draws projections
                shape_feats(Xc, gen, f.pr, f.reff, f.topshare, f.rank,
                            f.askew, f.kurt, f.pr2, f.pr5);
                double pr2_, re2_, top2_, rank2_, ask2_, ku2_, p22_, p52_;
                shape_feats(Xn, gsave, pr2_, re2_, top2_, rank2_, ask2_, ku2_, p22_, p52_);
                max_pr_invar = std::max({max_pr_invar, std::abs(f.pr - pr2_), std::abs(f.reff - re2_),
                                           std::abs(f.topshare - top2_), std::abs(f.pr2 - p22_), std::abs(f.pr5 - p52_)});
                max_ku_invar = std::max({max_ku_invar, std::abs(f.kurt - ku2_), std::abs(f.askew - ask2_)});
                c.relmean[b].push_back(rmean / rms);
                c.relstd[b].push_back(rstd / rms);
                // oracle gaps (true body known): cube ||u||_1, ball R*||u||_2
                double sg = 0, mg = 0, mtrue = 0;
                for (int t = 0; t < DIRS_ORACLE; ++t) {
                    Eigen::VectorXd g = gauss_vec(d, gen); g.normalize();
                    double ht = (b == 0) ? g.array().abs().sum() : R * g.norm();
                    double hh = (X.transpose() * g).maxCoeff();
                    double gap = std::max(0.0, ht - hh);
                    mg = std::max(mg, gap); sg += gap; mtrue += ht;
                }
                f.gapmean = (sg / DIRS_ORACLE) / (mtrue / DIRS_ORACLE);
                f.gapmax = mg / (mtrue / DIRS_ORACLE);
                c.f[b].push_back(f);
            }
        }
        cells.push_back(std::move(c));
    }
    std::cout << "scale-invariance self-check: max|pr-pr_norm|=" << max_pr_invar
              << " max|kurt-kurt_norm|=" << max_ku_invar << "\n";

    // CSV
    {
        std::ofstream o(outdir + "/phase2a_features.csv");
        o << "body,d,n,rep,rms,radmean,radstd,relradmean,relradstd,rank,pr,reff,topshare,askew,kurt,pr2,pr5,gapmean,gapmax\n";
        const char *nm[3] = {"cube", "ball", "ball_matched"};
        for (auto &c : cells) for (int b = 0; b < 3; ++b) for (int r = 0; r < REPS; ++r) {
            Feats &f = c.f[b][r];
            o << nm[b] << "," << c.d << "," << c.n << "," << r << "," << f.rms << ","
              << f.radmean << "," << f.radstd << "," << c.relmean[b][r] << "," << c.relstd[b][r] << ","
              << f.rank << "," << f.pr << "," << f.reff << "," << f.topshare << ","
              << f.askew << "," << f.kurt << "," << f.pr2 << "," << f.pr5 << ","
              << f.gapmean << "," << f.gapmax << "\n";
        }
    }

    auto col = [](const std::vector<Feats> &v, double Feats::*m) {
        std::vector<double> o; for (auto &f : v) o.push_back(f.*m); return o;
    };
    printf("\n[A] raw cube vs ball — AUC per feature (rms carries scale)\n");
    printf("%4s %5s %6s %6s %6s %6s %6s %6s %6s\n", "d", "n", "rms", "pr", "reff", "kurt", "askew", "pr5", "orac*");
    for (auto &c : cells) {
        printf("%4d %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n", c.d, c.n,
               auc_mw(col(c.f[0], &Feats::rms), col(c.f[1], &Feats::rms)),
               auc_mw(col(c.f[0], &Feats::pr), col(c.f[1], &Feats::pr)),
               auc_mw(col(c.f[0], &Feats::reff), col(c.f[1], &Feats::reff)),
               auc_mw(col(c.f[0], &Feats::kurt), col(c.f[1], &Feats::kurt)),
               auc_mw(col(c.f[0], &Feats::askew), col(c.f[1], &Feats::askew)),
               auc_mw(col(c.f[0], &Feats::pr5), col(c.f[1], &Feats::pr5)),
               auc_mw(col(c.f[0], &Feats::gapmean), col(c.f[1], &Feats::gapmean)));
    }
    printf("\n[B] RMS-normalized cube vs ball — AUC (scale-free + relative radial)\n");
    printf("%4s %5s %6s %6s %6s %6s %6s %6s\n", "d", "n", "relmn", "relsd", "pr", "kurt", "askew", "pr5");
    for (auto &c : cells) {
        printf("%4d %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n", c.d, c.n,
               auc_mw(c.relmean[0], c.relmean[1]), auc_mw(c.relstd[0], c.relstd[1]),
               auc_mw(col(c.f[0], &Feats::pr), col(c.f[1], &Feats::pr)),
               auc_mw(col(c.f[0], &Feats::kurt), col(c.f[1], &Feats::kurt)),
               auc_mw(col(c.f[0], &Feats::askew), col(c.f[1], &Feats::askew)),
               auc_mw(col(c.f[0], &Feats::pr5), col(c.f[1], &Feats::pr5)));
    }
    printf("\n[C] population-matched cube vs scaled ball — AUC (scale removed by design)\n");
    printf("%4s %5s %6s %6s %6s %6s %6s %6s\n", "d", "n", "rms", "pr", "kurt", "askew", "pr5", "rank");
    for (auto &c : cells) {
        printf("%4d %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n", c.d, c.n,
               auc_mw(col(c.f[0], &Feats::rms), col(c.f[2], &Feats::rms)),
               auc_mw(col(c.f[0], &Feats::pr), col(c.f[2], &Feats::pr)),
               auc_mw(col(c.f[0], &Feats::kurt), col(c.f[2], &Feats::kurt)),
               auc_mw(col(c.f[0], &Feats::askew), col(c.f[2], &Feats::askew)),
               auc_mw(col(c.f[0], &Feats::pr5), col(c.f[2], &Feats::pr5)),
               auc_mw(col(c.f[0], &Feats::rank), col(c.f[2], &Feats::rank)));
    }
    printf("\n[R] mean numerical rank (sample-limited: expect min(d,n-1)) + oracle gapmean per body\n");
    printf("%4s %5s %5s %5s %5s %7s %7s %7s\n", "d", "n", "rk_cb", "rk_b", "rk_m", "gap_cb", "gap_b", "gap_m");
    for (auto &c : cells) {
        auto meanrk = [](const std::vector<Feats> &v) {
            double s = 0; for (auto &f : v) s += f.rank; return s / v.size();
        };
        auto meang = [](const std::vector<Feats> &v) {
            double s = 0; for (auto &f : v) s += f.gapmean; return s / v.size();
        };
        printf("%4d %5d %5.1f %5.1f %5.1f %7.4f %7.4f %7.4f\n", c.d, c.n,
               meanrk(c.f[0]), meanrk(c.f[1]), meanrk(c.f[2]),
               meang(c.f[0]), meang(c.f[1]), meang(c.f[2]));
    }
    printf("\n* orac = support-gap oracle AUC shown for context only (requires known body).\n");
    printf("CSV: phase2a_features.csv (%d rows). Gate: which features carry info beyond scale/noise.\n",
           (int)(cells.size() * 3 * REPS));
    return 0;
}
