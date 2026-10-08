#include <algorithm>
#include <cmath>
extern "C" void witness_lower(const double *xy, int n, const double *rr,
                              double qx, double qy, double delta, double cx,
                              double cy, double silent, double *out) {
  const double A = 3.14159265358979323846 / 180;
  auto clip = [](double v) { return std::max(0., std::min(1., v)); };
  for (int i = 0; i < n; i++) {
    double x = xy[2 * i], y = xy[2 * i + 1], r = sqrt(x * x + y * y),
           ux = x / r, uy = y / r;
    double zx = x - qx, zy = y - qy, d = sqrt(zx * zx + zy * zy),
           a = std::max(1000., r), den = std::max(1e-12, 1500 - a);
    double pmin = clip((std::max(0., d - delta) - a) / den),
           pmax = clip((d + delta - a) / den), best = 0;
    if (pmax < 1 && d - delta > 5 && std::abs(atan2(y, x)) <= A + 1e-12) {
      for (int j = 0; j < 13; j++) {
        double hx = ux * rr[j], hy = uy * rr[j], tx = hx - cx, ty = hy - cy;
        if (sqrt(tx * tx + ty * ty) > 1800 + 1e-9)
          continue;
        double vx = hx - qx, vy = hy - qy, dh = sqrt(vx * vx + vy * vy);
        if (dh - delta <= 5 || dh + delta > 1500)
          continue;
        double sep = std::abs(r - rr[j]);
        double angle = std::abs(atan2(zx * vy - zy * vx, zx * vx + zy * vy));
        double perturb =
            delta * sep /
            (std::max(d - delta, 1e-12) * std::max(dh - delta, 1e-12));
        double pe = std::max(0., 1 - (angle + 2 * perturb) / (2 * A));
        best = std::max(best, sep * pe);
      }
    }
    out[i] = best * (1 - pmax) + silent * pmin;
  }
}
