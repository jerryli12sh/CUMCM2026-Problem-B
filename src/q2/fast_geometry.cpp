#include <algorithm>
#include <cmath>
#include <vector>
struct P {
  double x, y;
};
static std::vector<P> clip(const std::vector<P> &p, double a, double b,
                           double c) {
  std::vector<P> o;
  if (p.empty())
    return o;
  P t = p.back();
  double ft = a * t.x + b * t.y - c;
  for (P q : p) {
    double fq = a * q.x + b * q.y - c;
    if ((ft <= 0) != (fq <= 0)) {
      double u = ft / (ft - fq);
      o.push_back({t.x + u * (q.x - t.x), t.y + u * (q.y - t.y)});
    }
    if (fq <= 0)
      o.push_back(q);
    t = q;
    ft = fq;
  }
  return o;
}
extern "C" double wedge_diameter(const double *xy, int n, double qx, double qy,
                                 double beta, double alpha, double radius) {
  if (n == 0 || alpha <= 0)
    return 0;
  std::vector<P> p;
  for (int i = 0; i < n; i++)
    p.push_back({xy[2 * i], xy[2 * i + 1]});
  if (alpha < 1.5707963267948966) {
    double a = sin(beta - alpha), b = -cos(beta - alpha);
    p = clip(p, a, b, a * qx + b * qy);
    a = -sin(beta + alpha);
    b = cos(beta + alpha);
    p = clip(p, a, b, a * qx + b * qy);
  }
  if (p.empty())
    return 0;
  std::vector<P> out;
  double rr = radius * radius;
  for (P a : p)
    if ((a.x - qx) * (a.x - qx) + (a.y - qy) * (a.y - qy) >= rr)
      out.push_back(a);
  for (unsigned i = 0; i < p.size(); i++) {
    P a = p[i], b = p[(i + 1) % p.size()];
    double dx = b.x - a.x, dy = b.y - a.y, ux = a.x - qx, uy = a.y - qy,
           aa = dx * dx + dy * dy;
    if (aa == 0)
      continue;
    double bb = 2 * (ux * dx + uy * dy), cc = ux * ux + uy * uy - rr,
           disc = bb * bb - 4 * aa * cc;
    if (disc < 0)
      continue;
    double root = sqrt(disc);
    for (double f : {(-bb - root) / (2 * aa), (-bb + root) / (2 * aa)})
      if (f >= 0 && f <= 1)
        out.push_back({a.x + f * dx, a.y + f * dy});
  }
  double d = 0;
  for (unsigned i = 0; i < out.size(); i++)
    for (unsigned j = 0; j < i; j++) {
      double dx = out[i].x - out[j].x, dy = out[i].y - out[j].y;
      d = std::max(d, dx * dx + dy * dy);
    }
  return sqrt(d);
}
