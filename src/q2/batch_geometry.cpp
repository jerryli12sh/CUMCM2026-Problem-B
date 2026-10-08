#include "fast_geometry.cpp"
extern "C" void wedge_table(const double *xy, int n, double qx, double qy,
                            double alpha, double radius, int bins,
                            double *output) {
  const double pi = 3.14159265358979323846;
  for (int j = 0; j < bins; j++)
    output[j] = wedge_diameter(xy, n, qx, qy, -pi + (j + .5) * 2 * pi / bins,
                               alpha, radius);
}
