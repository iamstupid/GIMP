#include "config.h"
#include "gmp-impl.h"
#include "experimental/gmp_fft_dp/gmp_fft_dp_r22_clean.h"
#include <algorithm>

#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) && GMP_NUMB_BITS == 64
#define GMP_FFT_DP_REAL_BACKEND 1

static int
gmp_fft_dp_runtime_available (void)
{
  static int cached = -1;

  if (cached < 0)
    {
      __builtin_cpu_init ();
      cached = (__builtin_cpu_supports ("avx2")
                && __builtin_cpu_supports ("fma"));
    }
  return cached;
}

static int
gmp_fft_dp_mul_imbalance_ok (mp_size_t an, mp_size_t bn)
{
  mp_size_t un, vn;

  if (an <= 0 || bn <= 0)
    return 0;

  un = an;
  vn = bn;
  if (vn > un)
    std::swap (un, vn);

  if (un == vn)
    return 0;
  if (un + vn > 32768)
    return 0;
  if (vn * 16 < un)
    return 0;
  if (vn * 8 < un)
    return vn >= 80;
  return vn >= MUL_FFT_DP_IMBALANCE_THRESHOLD;
}

extern "C" int
mpn_fft_dp_mul_balanced (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  if (UNLIKELY (n <= 0 || n > MUL_FFT_DP_MAX_BALANCED))
    return 0;
  if (UNLIKELY (!gmp_fft_dp_runtime_available ()))
    return 0;
  return gmp_fft_dp_r22_clean_mul_pq_fusednocopy_balanced (rp, ap, bp, n);
}

extern "C" int
mpn_fft_dp_mul (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                mp_srcptr bp, mp_size_t bn)
{
  if (UNLIKELY (!gmp_fft_dp_runtime_available ()))
    return 0;
  if (an == bn)
    {
      if (UNLIKELY (an <= 0 || an > MUL_FFT_DP_MAX_BALANCED))
        return 0;
      if (BELOW_THRESHOLD (an, MUL_FFT_DP_THRESHOLD))
        return 0;
      return gmp_fft_dp_r22_clean_mul_pq_fusednocopy_balanced (rp, ap, bp, an);
    }
  if (UNLIKELY (!gmp_fft_dp_mul_imbalance_ok (an, bn)))
    return 0;
  return gmp_fft_dp_r22_clean_mul_pq_fusednocopy (rp, ap, an, bp, bn);
}

extern "C" int
mpn_fft_dp_sqr_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  if (UNLIKELY (n <= 0 || n > SQR_FFT_DP_MAX_BALANCED))
    return 0;
  if (UNLIKELY (!gmp_fft_dp_runtime_available ()))
    return 0;
  return gmp_fft_dp_r22_clean_sqr_pq_fusednocopy_balanced (rp, ap, n);
}

#else
#define GMP_FFT_DP_REAL_BACKEND 0
#endif
#else
#define GMP_FFT_DP_REAL_BACKEND 0
#endif
