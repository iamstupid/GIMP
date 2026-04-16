#include "config.h"
#include "gmp-impl.h"

#if defined(__GNUC__) || defined(__clang__)
#define GMP_FFT_DP_WEAK __attribute__((weak))
#else
#define GMP_FFT_DP_WEAK
#endif

int GMP_FFT_DP_WEAK
mpn_fft_dp_mul_balanced (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n)
{
  (void) rp;
  (void) ap;
  (void) bp;
  (void) n;
  return 0;
}

int GMP_FFT_DP_WEAK
mpn_fft_dp_mul (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                mp_srcptr bp, mp_size_t bn)
{
  (void) rp;
  (void) ap;
  (void) an;
  (void) bp;
  (void) bn;
  return 0;
}

int GMP_FFT_DP_WEAK
mpn_fft_dp_sqr_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  (void) rp;
  (void) ap;
  (void) n;
  return 0;
}
