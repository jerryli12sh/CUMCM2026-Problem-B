#include "fast_geometry.cpp"
extern "C" void wedge_table(const double *xy, int n, double qx, double qy,
                            double alpha, double radius, int bins,
                            double *output) {
  const double pi = 3.14159265358979323846, twopi = 2 * pi;
  if (n == 0 || alpha <= 0) {
    std::fill(output, output + bins, 0.0);
    return;
  }
  // Convex polygon direction support; skip only directions strictly outside its
  // padded arc.
  std::vector<double> angles;
  angles.reserve(n);
  for (int i = 0; i < n; i++) {
    double a = atan2(xy[2 * i + 1] - qy, xy[2 * i] - qx);
    if (a < 0)
      a += twopi;
    angles.push_back(a);
  }
  std::sort(angles.begin(), angles.end());
  double gap = -1, start = 0;
  for (int i = 0; i < n; i++) {
    double next = (i + 1 < n ? angles[i + 1] : angles[0] + twopi);
    double g = next - angles[i];
    if (g > gap) {
      gap = g;
      start = fmod(next, twopi);
    }
  }
  double width = twopi - gap;
  bool skip = (width < pi - 1e-10) && (width + 2 * alpha < twopi - 1e-10);
  for (int j = 0; j < bins; j++) {
    double beta = -pi + (j + .5) * twopi / bins;
    double rel = fmod(beta - start + alpha + 2 * twopi, twopi);
    if (skip && rel > width + 2 * alpha + 1e-10 && rel < twopi - 1e-10)
      output[j] = 0;
    else
      output[j] = wedge_diameter(xy, n, qx, qy, beta, alpha, radius);
  }
}
