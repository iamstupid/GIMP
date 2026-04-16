#ifndef GMP_FFT_DP_R22_CLEAN_H
#define GMP_FFT_DP_R22_CLEAN_H

#include "gmp-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

int gmp_fft_dp_r22_clean_supported (mp_size_t an, mp_size_t bn);
int gmp_fft_dp_r22_clean_mul (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                              mp_srcptr bp, mp_size_t bn);
int gmp_fft_dp_r22_clean_mul_balanced (mp_ptr rp, mp_srcptr ap,
                                       mp_srcptr bp, mp_size_t n);
int gmp_fft_dp_r22_clean_mul_pq (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                                 mp_srcptr bp, mp_size_t bn);
int gmp_fft_dp_r22_clean_mul_pq_balanced (mp_ptr rp, mp_srcptr ap,
                                          mp_srcptr bp, mp_size_t n);
int gmp_fft_dp_r22_clean_mul_pq_fusedexp (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                                          mp_srcptr bp, mp_size_t bn);
int gmp_fft_dp_r22_clean_mul_pq_fusedexp_balanced (mp_ptr rp, mp_srcptr ap,
                                                   mp_srcptr bp, mp_size_t n);
int gmp_fft_dp_r22_clean_mul_pq_fusednocopy (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                                             mp_srcptr bp, mp_size_t bn);
int gmp_fft_dp_r22_clean_mul_pq_fusednocopy_balanced (mp_ptr rp, mp_srcptr ap,
                                                      mp_srcptr bp, mp_size_t n);
int gmp_fft_dp_r22_clean_mul_pq_fusedmagic (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                                            mp_srcptr bp, mp_size_t bn);
int gmp_fft_dp_r22_clean_mul_pq_fusedmagic_balanced (mp_ptr rp, mp_srcptr ap,
                                                     mp_srcptr bp, mp_size_t n);
int gmp_fft_dp_r22_clean_sqr (mp_ptr rp, mp_srcptr ap, mp_size_t an);
int gmp_fft_dp_r22_clean_sqr_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n);
int gmp_fft_dp_r22_clean_sqr_pq (mp_ptr rp, mp_srcptr ap, mp_size_t an);
int gmp_fft_dp_r22_clean_sqr_pq_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n);
int gmp_fft_dp_r22_clean_sqr_pq_fusedexp (mp_ptr rp, mp_srcptr ap, mp_size_t an);
int gmp_fft_dp_r22_clean_sqr_pq_fusedexp_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n);
int gmp_fft_dp_r22_clean_sqr_pq_fusednocopy (mp_ptr rp, mp_srcptr ap, mp_size_t an);
int gmp_fft_dp_r22_clean_sqr_pq_fusednocopy_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n);
int gmp_fft_dp_r22_clean_sqr_pq_fusedmagic (mp_ptr rp, mp_srcptr ap, mp_size_t an);
int gmp_fft_dp_r22_clean_sqr_pq_fusedmagic_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n);
int mpn_fft_dp_r22_clean_mul_balanced (mp_ptr rp, mp_srcptr ap,
                                       mp_srcptr bp, mp_size_t n);
int mpn_fft_dp_r22_clean_sqr_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n);

#ifdef __cplusplus
}

static inline int
mpn_fft_dp_sqr (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr_pq_fusednocopy_balanced (rp, ap, n);
}
#endif

#endif
