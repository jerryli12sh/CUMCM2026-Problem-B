"""Optional compiled rejection filter; every acceptance still uses Python proof.

The embedded integer routine mirrors the quadtree predicate with signed128-bit
products. Input bounds below keep every product within that range. If a local
C++ compiler is unavailable, the original exact-integer certificate is used.
No network, scene information, or precomputed performance labels are accessed.
"""

import ctypes, shutil, subprocess, tempfile
from pathlib import Path
from certify_layout_integer import certify as python_certify

SOURCE = r"""
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstdlib>
using I=__int128_t;
using L=int64_t;
struct P { L x,y; bool operator<(const P& b)const{return x<b.x || (x==b.x && y<b.y);} bool operator==(const P& b)const{return x==b.x && y==b.y;} };
I cross(P a,P b,P c){return I(b.x-a.x)*(c.y-a.y)-I(b.y-a.y)*(c.x-a.x);}
std::vector<P> hull(std::vector<P> p){
 std::sort(p.begin(),p.end());p.erase(std::unique(p.begin(),p.end()),p.end());
 if(p.size()<3)return {};
 std::vector<P> lo,hi;
 for(auto q:p){while(lo.size()>1 && cross(lo[lo.size()-2],lo.back(),q)<=0)lo.pop_back();lo.push_back(q);}
 for(auto it=p.rbegin();it!=p.rend();++it){auto q=*it;while(hi.size()>1 && cross(hi[hi.size()-2],hi.back(),q)<=0)hi.pop_back();hi.push_back(q);}
 lo.pop_back();hi.pop_back();lo.insert(lo.end(),hi.begin(),hi.end());return lo;
}
extern "C" int covered(const L* data,int n,L domain,L range,int depth){
 std::vector<P> sites;for(int i=0;i<n;i++)sites.push_back({data[2*i],data[2*i+1]});
 struct Cell{L x,y;int level;};std::vector<Cell> stack{{0,0,0}};
 while(!stack.empty()){
  Cell z=stack.back();stack.pop_back();L scale=L(1)<<z.level;
  L nx=std::max(std::abs(z.x)-domain,L(0)),ny=std::max(std::abs(z.y)-domain,L(0));
  I d=I(domain)*scale;if(I(nx)*nx+I(ny)*ny>d*d)continue;
  std::vector<P> near;I r=I(range)*scale;
  for(P p:sites){L dx=std::abs(p.x*scale-z.x)+domain,dy=std::abs(p.y*scale-z.y)+domain;if(I(dx)*dx+I(dy)*dy<=r*r)near.push_back(p);}
  auto poly=hull(near);bool good=poly.size()>=3;
  if(good)for(size_t k=0;k<poly.size() && good;k++){
   P a=poly[k],b=poly[(k+1)%poly.size()];
   for(L dx:{-domain,domain})for(L dy:{-domain,domain}){
    if(I(b.x-a.x)*(z.y+dy-a.y*scale)-I(b.y-a.y)*(z.x+dx-a.x*scale)<0)good=false;
   }
  }
  if(good)continue;
  if(z.level>=depth)return 0;
  for(L dx:{-domain,domain})for(L dy:{-domain,domain})stack.push_back({2*z.x+dx,2*z.y+dy,z.level+1});
 }
 return 1;
}
"""
_FUNCTION = None
_INITIALIZED = False
_BUILD = None
_LIBRARY = None


def initialize():
    global _FUNCTION, _INITIALIZED, _BUILD, _LIBRARY
    if _INITIALIZED:
        return _FUNCTION
    _INITIALIZED = True
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        return None
    try:
        _BUILD = tempfile.TemporaryDirectory(prefix="q4-integer-reject-")
        root = Path(_BUILD.name)
        src = root / "filter.cpp"
        library = root / "filter.so"
        src.write_text(SOURCE)
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(src),
                "-o",
                str(library),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=45,
        )
        _LIBRARY = ctypes.CDLL(str(library))
        _FUNCTION = _LIBRARY.covered
        _FUNCTION.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        _FUNCTION.restype = ctypes.c_int
    except (OSError, subprocess.SubprocessError):
        _FUNCTION = None
    return _FUNCTION


def verdict(points_mm, domain_mm=1800000, range_mm=1000000, max_depth=23):
    points = [tuple(map(int, p)) for p in points_mm]
    if not (
        1 <= domain_mm <= 10000000
        and 1 <= range_mm <= 10000000
        and 0 <= max_depth <= 25
        and len(points) <= 1000
        and all(len(p) == 2 and max(abs(v) for v in p) <= 10000000 for p in points)
    ):
        return None
    function = initialize()
    if function is None:
        return None
    flat = [v for p in points for v in p]
    data = (ctypes.c_int64 * len(flat))(*flat)
    return bool(function(data, len(points), domain_mm, range_mm, max_depth))


def certify(points_mm, domain_mm=1800000, range_mm=1000000, max_depth=23, record=False):
    points = [tuple(map(int, p)) for p in points_mm]
    accepted = verdict(points, domain_mm, range_mm, max_depth)
    if accepted is False:
        return dict(
            certified=False,
            arithmetic="bounded128-bit rejection only",
            domain_mm=domain_mm,
            range_mm=range_mm,
            max_depth=max_depth,
        )
    proof = python_certify(
        points,
        domain_mm=domain_mm,
        range_mm=range_mm,
        max_depth=max_depth,
        record=record,
    )
    if accepted is True:
        assert proof[
            "certified"
        ], "Compiled rejection predicate disagrees with original proof"
    return proof
