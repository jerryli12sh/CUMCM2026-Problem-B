#include <algorithm>
#include <cmath>
static double integral(const double *a, const double *pref, int bins,
                       double x) {
  const double pi = 3.14159265358979323846, step = 2 * pi / bins;
  double y = (x + pi) / step;
  long long k = (long long)floor(y);
  double f = y - k;
  long long cycle = (long long)floor((double)k / bins), j = k - cycle * bins;
  return cycle * pref[bins] + pref[j] + f * step * a[j];
}
extern "C" void expected_moments(const double *xy, int n, double qx, double qy,
                                 double delta, const double *low,
                                 const double *high, const double *plow,
                                 const double *phigh, int bins, double sl,
                                 double sh, double kl, double kh, double D,
                                 double Dlo, double *result) {
  const double A = 3.14159265358979323846 / 180;
  double mxlo = 0, mxhi = 0;
  for (int j = 0; j < bins; j++) {
    mxlo = std::max(mxlo, low[j]);
    mxhi = std::max(mxhi, high[j]);
  }
  double meanlo = 0, meanhi = 0, m2lo = 0, m2hi = 0;
  auto clip = [](double x, double a, double b) {
    return std::max(a, std::min(b, x));
  };
  for (int i = 0; i < n; i++) {
    double x = xy[2 * i], y = xy[2 * i + 1], r = hypot(x, y), dx = x - qx,
           dy = y - qy, d = hypot(dx, dy), b = atan2(dy, dx);
    double dn = std::max(0., d - delta), du = d + delta, a = std::max(1000., r),
           den = std::max(1e-12, 1500 - a);
    double p = clip((dn - a) / den, 0, 1), pp = clip((du - a) / den, 0, 1);
    double nl = clip(
        (integral(low, plow, bins, b + A) - integral(low, plow, bins, b - A)) /
            (2 * A),
        0, mxlo);
    double nh = clip((integral(high, phigh, bins, b + A) -
                      integral(high, phigh, bins, b - A)) /
                         (2 * A),
                     0, mxhi);
    double l = std::min(p * sl + (1 - p) * nl, pp * sl + (1 - pp) * nl),
           h = std::max(p * sh + (1 - p) * nh, pp * sh + (1 - pp) * nh);
    if (du <= 5) {
      l = kl;
      h = kh;
    } else if (dn <= 5) {
      l = std::min(l, kl);
      h = std::max(h, kh);
    }
    if (hypot(qx, qy) <= delta) {
      h = D;
      if (delta == 0)
        l = Dlo;
    }
    l = std::max(0., l);
    h = std::min(D, h);
    double dl = l - meanlo;
    meanlo += dl / (i + 1);
    m2lo += dl * (l - meanlo);
    double dh = h - meanhi;
    meanhi += dh / (i + 1);
    m2hi += dh * (h - meanhi);
  }
  result[0] = meanlo;
  result[1] = meanhi;
  result[2] = m2lo / (n - 1);
  result[3] = m2hi / (n - 1);
}
