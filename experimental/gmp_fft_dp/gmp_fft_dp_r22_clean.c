#include "config.h"

#include "experimental/gmp_fft_dp/gmp_fft_dp_r22_clean.h"
#include "experimental/gmp_ntt_avx2/gmp_ntt_avx2_simd_v4.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

typedef struct
{
  double *data;
  size_t data_cap;
  double *data2;
  size_t data2_cap;
  uint64_t *ri_u64;
  size_t ri_cap;
} gmp_fft_dp_r22_workspace;

typedef struct
{
  uint32_t max_n;
  double *tw;
  size_t tw_cap;
  double *root;
  size_t root_cap;
  uint32_t root_n;
  double *pq_omega_br;
  size_t pq_omega_br_cap;
  uint32_t pq_omega_br_n;
} gmp_fft_dp_r22_plan_cache;

static GMP_NTT_THREAD_LOCAL gmp_fft_dp_r22_workspace
gmp_fft_dp_r22_ws = { NULL, 0, NULL, 0, NULL, 0 };
static GMP_NTT_THREAD_LOCAL gmp_fft_dp_r22_plan_cache
gmp_fft_dp_r22_plan = { 0, NULL, 0, NULL, 0, 0, NULL, 0, 0 };

GMP_NTT_FORCEINLINE static uint32_t
gmp_fft_dp_r22_bitreverse_u32 (uint32_t x, unsigned bits);

GMP_NTT_FORCEINLINE static int
gmp_fft_dp_r22_is_aligned32 (const void *p)
{
  return ((((uintptr_t) p) & 31u) == 0);
}

static void *
gmp_fft_dp_r22_aligned_alloc (size_t bytes)
{
  void *p;

  if (bytes == 0)
    return NULL;
#if defined(_WIN32)
  p = _aligned_malloc (bytes, 32);
  if (p == NULL)
    abort ();
#else
  p = NULL;
  if (posix_memalign (&p, 32, bytes) != 0)
    abort ();
#endif
  return p;
}

static void
gmp_fft_dp_r22_aligned_free (void *p)
{
#if defined(_WIN32)
  _aligned_free (p);
#else
  free (p);
#endif
}

static void
gmp_fft_dp_r22_workspace_ensure (gmp_fft_dp_r22_workspace *ws, uint32_t n)
{
  if (ws->data_cap < 2u * (size_t) n)
    {
      gmp_fft_dp_r22_aligned_free (ws->data);
      ws->data_cap = 2u * (size_t) n;
      ws->data = (double *) gmp_fft_dp_r22_aligned_alloc (ws->data_cap
                                                           * sizeof (double));
    }
  if (ws->data2_cap < 2u * (size_t) n)
    {
      gmp_fft_dp_r22_aligned_free (ws->data2);
      ws->data2_cap = 2u * (size_t) n;
      ws->data2 = (double *) gmp_fft_dp_r22_aligned_alloc (ws->data2_cap
                                                            * sizeof (double));
    }
  if (ws->ri_cap < ((size_t) n >> 1))
    {
      gmp_fft_dp_r22_aligned_free (ws->ri_u64);
      ws->ri_cap = ((size_t) n >> 1);
      ws->ri_u64 = (uint64_t *) gmp_fft_dp_r22_aligned_alloc (ws->ri_cap
                                                              * sizeof (uint64_t));
    }
}

static uint32_t
gmp_fft_dp_r22_ceil_pow2_u32 (uint32_t x)
{
  uint32_t r;
  r = 1;
  while (r < x)
    r <<= 1;
  return r;
}

GMP_NTT_FORCEINLINE static size_t
gmp_fft_dp_r22_stage_offset (unsigned lg_len)
{
  ASSERT (lg_len >= 2);
  if (lg_len == 2)
    return 0;
  if (lg_len == 3)
    return 16;
  return ((size_t) 1u << lg_len) + 16u;
}

static void
gmp_fft_dp_r22_plan_ensure (gmp_fft_dp_r22_plan_cache *plan, uint32_t n)
{
  uint32_t len;
  size_t total_tw;
  double *root_re;
  double *root_im;

  if (n <= plan->max_n)
    goto ensure_pq_omega_br;

  total_tw = 0;
  for (len = 4; len <= n; len <<= 1)
    {
      uint32_t l = len >> 2;
      size_t tile_count = ((size_t) l + 3u) >> 2;
      total_tw += tile_count * 16u;
    }

  gmp_fft_dp_r22_aligned_free (plan->tw);
  plan->tw = (double *) gmp_fft_dp_r22_aligned_alloc (total_tw * sizeof (double));
  plan->tw_cap = total_tw;
  if (plan->root_cap < (size_t) n)
    {
      gmp_fft_dp_r22_aligned_free (plan->root);
      plan->root = (double *) gmp_fft_dp_r22_aligned_alloc ((size_t) n
                                                            * sizeof (double));
      plan->root_cap = (size_t) n;
    }

  root_re = plan->root;
  root_im = plan->root + ((size_t) n >> 1);
  for (len = 0; len < (n >> 1); ++len)
    {
      double angle = (2.0 * M_PI * (double) len) / (double) n;
      root_re[len] = cos (angle);
      root_im[len] = sin (angle);
    }
  plan->root_n = n;

  for (len = 4; len <= n; len <<= 1)
    {
      uint32_t l = len >> 2;
      uint32_t step = n / len;
      size_t t;
      double *dst = plan->tw + gmp_fft_dp_r22_stage_offset (gmp_ntt_ctz (len));

      for (t = 0; t < ((size_t) l + 3u) >> 2; ++t)
        {
          uint32_t lane;
          uint32_t j = (uint32_t) (t << 2);
          for (lane = 0; lane < 4; ++lane)
            {
              uint32_t idx = j + lane;
              if (idx < l)
                {
                  uint32_t root_idx = idx * step;
                  uint32_t root_idx2 = root_idx << 1;
                  dst[16 * t + lane] = root_re[root_idx];
                  dst[16 * t + 4 + lane] = root_im[root_idx];
                  dst[16 * t + 8 + lane] = root_re[root_idx2];
                  dst[16 * t + 12 + lane] = root_im[root_idx2];
                }
              else
                {
                  dst[16 * t + lane] = 1.0;
                  dst[16 * t + 4 + lane] = 0.0;
                  dst[16 * t + 8 + lane] = 1.0;
                  dst[16 * t + 12 + lane] = 0.0;
                }
            }
        }
    }
  plan->max_n = n;

ensure_pq_omega_br:
  if (plan->pq_omega_br_cap < ((size_t) plan->max_n >> 1))
    {
      gmp_fft_dp_r22_aligned_free (plan->pq_omega_br);
      plan->pq_omega_br = (double *) gmp_fft_dp_r22_aligned_alloc (((size_t) plan->max_n >> 1)
                                                                    * sizeof (double));
      plan->pq_omega_br_cap = ((size_t) plan->max_n >> 1);
    }

  if (plan->pq_omega_br_n == plan->max_n)
    return;

  root_re = plan->root;
  root_im = plan->root + ((size_t) plan->root_n >> 1);
  for (len = 0; len < (plan->max_n >> 2); ++len)
    {
      uint32_t e = gmp_fft_dp_r22_bitreverse_u32 (len,
                                                  gmp_ntt_ctz (plan->max_n) - 2u);
      plan->pq_omega_br[len] = root_re[e];
      plan->pq_omega_br[(plan->max_n >> 2) + len] = root_im[e];
    }
  plan->pq_omega_br_n = plan->max_n;
}

GMP_NTT_FORCEINLINE static const double *
gmp_fft_dp_r22_stage_tw (const gmp_fft_dp_r22_plan_cache *plan, unsigned lg_len)
{
  return plan->tw + gmp_fft_dp_r22_stage_offset (lg_len);
}

GMP_NTT_FORCEINLINE static const double *
gmp_fft_dp_r22_stage_tw_if_present (const gmp_fft_dp_r22_plan_cache *plan,
                                    unsigned lg_len)
{
  if (((uint32_t) 1u << lg_len) > plan->max_n)
    return NULL;
  return gmp_fft_dp_r22_stage_tw (plan, lg_len);
}

GMP_NTT_FORCEINLINE static double *
gmp_fft_dp_r22_tile_ptr (double *data, uint32_t complex_index)
{
  return data + 8u * (size_t) (complex_index >> 2);
}

GMP_NTT_FORCEINLINE static const double *
gmp_fft_dp_r22_tile_ptr_const (const double *data, uint32_t complex_index)
{
  return data + 8u * (size_t) (complex_index >> 2);
}

GMP_NTT_FORCEINLINE static double
gmp_fft_dp_r22_get_re (const double *data, uint32_t i)
{
  return data[8u * (size_t) (i >> 2) + (i & 3u)];
}

GMP_NTT_FORCEINLINE static double
gmp_fft_dp_r22_get_im (const double *data, uint32_t i)
{
  return data[8u * (size_t) (i >> 2) + 4u + (i & 3u)];
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_set_complex (double *data, uint32_t i, double re, double im)
{
  size_t off = 8u * (size_t) (i >> 2) + (size_t) (i & 3u);
  data[off] = re;
  data[off + 4u] = im;
}

GMP_NTT_FORCEINLINE static uint32_t
gmp_fft_dp_r22_bitreverse_u32 (uint32_t x, unsigned bits)
{
  if (bits == 0u)
    return 0u;
  x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
  x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
  x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
  x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
  x = (x << 16) | (x >> 16);
  return x >> (32u - bits);
}

GMP_NTT_FORCEINLINE static uint32_t
gmp_fft_dp_r22_get_u16_digit (mp_srcptr ap, size_t an, uint32_t digit_idx)
{
  if ((size_t) digit_idx >= 4u * an)
    return 0u;
  return (uint32_t) ((ap[digit_idx >> 2] >> (16u * (digit_idx & 3u))) & 0xFFFFu);
}

static void
gmp_fft_dp_r22_bitrev_permute_inplace (double *data, uint32_t n)
{
  unsigned bits = (unsigned) gmp_ntt_ctz (n);
  uint32_t i;

  for (i = 0; i < n; ++i)
    {
      uint32_t j = gmp_fft_dp_r22_bitreverse_u32 (i, bits);
      if (i < j)
        {
          double ir = gmp_fft_dp_r22_get_re (data, i);
          double ii = gmp_fft_dp_r22_get_im (data, i);
          double jr = gmp_fft_dp_r22_get_re (data, j);
          double ji = gmp_fft_dp_r22_get_im (data, j);

          gmp_fft_dp_r22_set_complex (data, i, jr, ji);
          gmp_fft_dp_r22_set_complex (data, j, ir, ii);
        }
    }
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pair_load (gmp_ntt_v4 *re, gmp_ntt_v4 *im, const double *p)
{
  ASSERT (gmp_fft_dp_r22_is_aligned32 (p));
  *re = gmp_ntt_v4_load (p + 0);
  *im = gmp_ntt_v4_load (p + 4);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pair_store (double *p, gmp_ntt_v4 re, gmp_ntt_v4 im)
{
  ASSERT (gmp_fft_dp_r22_is_aligned32 (p));
  gmp_ntt_v4_store (p + 0, re);
  gmp_ntt_v4_store (p + 4, im);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pair_cmul (gmp_ntt_v4 *re, gmp_ntt_v4 *im,
                          gmp_ntt_v4 ar, gmp_ntt_v4 ai,
                          gmp_ntt_v4 wr, gmp_ntt_v4 wi)
{
#if defined(__FMA__) || defined(__AVX2__)
  *re = _mm256_fmsub_pd (ar, wr, gmp_ntt_v4_mul (ai, wi));
  *im = _mm256_fmadd_pd (ar, wi, gmp_ntt_v4_mul (ai, wr));
#else
  *re = gmp_ntt_v4_sub (gmp_ntt_v4_mul (ar, wr), gmp_ntt_v4_mul (ai, wi));
  *im = gmp_ntt_v4_add (gmp_ntt_v4_mul (ar, wi), gmp_ntt_v4_mul (ai, wr));
#endif
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pair_cmul_inplace (gmp_ntt_v4 *xr, gmp_ntt_v4 *xi,
                                  gmp_ntt_v4 wr, gmp_ntt_v4 wi)
{
#if defined(__FMA__) || defined(__AVX2__)
  gmp_ntt_v4 tr = *xr;
  gmp_ntt_v4 ti = gmp_ntt_v4_mul (*xi, wr);
  *xr = _mm256_fmsub_pd (tr, wr, gmp_ntt_v4_mul (*xi, wi));
  *xi = _mm256_fmadd_pd (tr, wi, ti);
#else
  gmp_ntt_v4 tr = *xr;
  gmp_ntt_v4 ti = *xi;
  *xr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (tr, wr), gmp_ntt_v4_mul (ti, wi));
  *xi = gmp_ntt_v4_add (gmp_ntt_v4_mul (tr, wi), gmp_ntt_v4_mul (ti, wr));
#endif
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pair_cmul_conj (gmp_ntt_v4 *re, gmp_ntt_v4 *im,
                               gmp_ntt_v4 ar, gmp_ntt_v4 ai,
                               gmp_ntt_v4 wr, gmp_ntt_v4 wi)
{
#if defined(__FMA__) || defined(__AVX2__)
  *re = _mm256_fmadd_pd (ai, wi, gmp_ntt_v4_mul (ar, wr));
  *im = _mm256_fmsub_pd (ai, wr, gmp_ntt_v4_mul (ar, wi));
#else
  *re = gmp_ntt_v4_add (gmp_ntt_v4_mul (ar, wr), gmp_ntt_v4_mul (ai, wi));
  *im = gmp_ntt_v4_sub (gmp_ntt_v4_mul (ai, wr), gmp_ntt_v4_mul (ar, wi));
#endif
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_load_u16_pair (gmp_ntt_v4 *re, gmp_ntt_v4 *im,
                              const uint64_t *ri, uint32_t digit_idx)
{
  __m128i pair128;
  __m256i d32;

  ASSERT ((digit_idx & 3u) == 0);
  pair128 = _mm_loadu_si128 ((const __m128i_u *) (ri + 2u * (digit_idx >> 2)));
  d32 = _mm256_cvtepu16_epi32 (pair128);
  *re = _mm256_cvtepi32_pd (_mm256_castsi256_si128 (d32));
  *im = _mm256_cvtepi32_pd (_mm256_extracti128_si256 (d32, 1));
}

GMP_NTT_FORCEINLINE static __m128d
gmp_fft_dp_r22_addsub2 (__m128d x)
{
  __m128d xs;
  xs = _mm_shuffle_pd (x, x, 1);
  return _mm_unpacklo_pd (_mm_add_pd (x, xs), _mm_sub_pd (x, xs));
}

GMP_NTT_FORCEINLINE static __m128d
gmp_fft_dp_r22_dup_lo (__m128d x)
{
  return _mm_unpacklo_pd (x, x);
}

GMP_NTT_FORCEINLINE static __m128d
gmp_fft_dp_r22_dup_hi (__m128d x)
{
  return _mm_unpackhi_pd (x, x);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_transpose4x4 (gmp_ntt_v4 *x0, gmp_ntt_v4 *x1,
                             gmp_ntt_v4 *x2, gmp_ntt_v4 *x3)
{
  gmp_ntt_v4 t0, t1, t2, t3;

  t0 = _mm256_unpacklo_pd (*x0, *x1);
  t1 = _mm256_unpackhi_pd (*x0, *x1);
  t2 = _mm256_unpacklo_pd (*x2, *x3);
  t3 = _mm256_unpackhi_pd (*x2, *x3);

  *x0 = _mm256_permute2f128_pd (t0, t2, 0x20);
  *x1 = _mm256_permute2f128_pd (t1, t3, 0x20);
  *x2 = _mm256_permute2f128_pd (t0, t2, 0x31);
  *x3 = _mm256_permute2f128_pd (t1, t3, 0x31);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_radix22_dif_bfly_store (double *d0, double *d1,
                                       double *d2, double *d3,
                                       const double *s0, const double *s1,
                                       const double *s2, const double *s3,
                                       gmp_ntt_v4 ur, gmp_ntt_v4 ui,
                                       gmp_ntt_v4 vr, gmp_ntt_v4 vi)
{
  gmp_ntt_v4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
  gmp_ntt_v4 b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
  gmp_ntt_v4 tr, ti;

  gmp_fft_dp_r22_pair_load (&a0r, &a0i, s0);
  gmp_fft_dp_r22_pair_load (&a1r, &a1i, s1);
  gmp_fft_dp_r22_pair_load (&a2r, &a2i, s2);
  gmp_fft_dp_r22_pair_load (&a3r, &a3i, s3);

  b0r = gmp_ntt_v4_add (a0r, a2r);
  b0i = gmp_ntt_v4_add (a0i, a2i);
  tr = gmp_ntt_v4_sub (a0r, a2r);
  ti = gmp_ntt_v4_sub (a0i, a2i);
  gmp_fft_dp_r22_pair_cmul (&b2r, &b2i, tr, ti, ur, ui);

  b1r = gmp_ntt_v4_add (a1r, a3r);
  b1i = gmp_ntt_v4_add (a1i, a3i);
  tr = gmp_ntt_v4_sub (a1r, a3r);
  ti = gmp_ntt_v4_sub (a1i, a3i);
  gmp_fft_dp_r22_pair_cmul (&b3r, &b3i, tr, ti, gmp_ntt_v4_neg (ui), ur);

  gmp_fft_dp_r22_pair_store (d0, gmp_ntt_v4_add (b0r, b1r),
                             gmp_ntt_v4_add (b0i, b1i));
  tr = gmp_ntt_v4_sub (b0r, b1r);
  ti = gmp_ntt_v4_sub (b0i, b1i);
  gmp_fft_dp_r22_pair_cmul (&tr, &ti, tr, ti, vr, vi);
  gmp_fft_dp_r22_pair_store (d1, tr, ti);

  gmp_fft_dp_r22_pair_store (d2, gmp_ntt_v4_add (b2r, b3r),
                             gmp_ntt_v4_add (b2i, b3i));
  tr = gmp_ntt_v4_sub (b2r, b3r);
  ti = gmp_ntt_v4_sub (b2i, b3i);
  gmp_fft_dp_r22_pair_cmul (&tr, &ti, tr, ti, vr, vi);
  gmp_fft_dp_r22_pair_store (d3, tr, ti);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_radix22_dit_bfly_store (double *d0, double *d1,
                                       double *d2, double *d3,
                                       const double *s0, const double *s1,
                                       const double *s2, const double *s3,
                                       gmp_ntt_v4 ur, gmp_ntt_v4 ui,
                                       gmp_ntt_v4 vr, gmp_ntt_v4 vi)
{
  gmp_ntt_v4 y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
  gmp_ntt_v4 b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
  gmp_ntt_v4 tr, ti;

  gmp_fft_dp_r22_pair_load (&y0r, &y0i, s0);
  gmp_fft_dp_r22_pair_load (&y1r, &y1i, s1);
  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, y1r, y1i, vr, vi);
  b0r = gmp_ntt_v4_add (y0r, tr);
  b0i = gmp_ntt_v4_add (y0i, ti);
  b1r = gmp_ntt_v4_sub (y0r, tr);
  b1i = gmp_ntt_v4_sub (y0i, ti);

  gmp_fft_dp_r22_pair_load (&y2r, &y2i, s2);
  gmp_fft_dp_r22_pair_load (&y3r, &y3i, s3);
  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, y3r, y3i, vr, vi);
  b2r = gmp_ntt_v4_add (y2r, tr);
  b2i = gmp_ntt_v4_add (y2i, ti);
  b3r = gmp_ntt_v4_sub (y2r, tr);
  b3i = gmp_ntt_v4_sub (y2i, ti);

  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, b2r, b2i, ur, ui);
  gmp_fft_dp_r22_pair_store (d0, gmp_ntt_v4_add (b0r, tr),
                             gmp_ntt_v4_add (b0i, ti));
  gmp_fft_dp_r22_pair_store (d2, gmp_ntt_v4_sub (b0r, tr),
                             gmp_ntt_v4_sub (b0i, ti));

  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, b3r, b3i, gmp_ntt_v4_neg (ui), ur);
  gmp_fft_dp_r22_pair_store (d1, gmp_ntt_v4_add (b1r, tr),
                             gmp_ntt_v4_add (b1i, ti));
gmp_fft_dp_r22_pair_store (d3, gmp_ntt_v4_sub (b1r, tr),
                             gmp_ntt_v4_sub (b1i, ti));
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_forward_tile (double *p)
{
  const __m128d sign = _mm_set1_pd (-0.0);
  __m128d r01, r23, i01, i23;
  __m128d sumr, sumi, diffr, diffi;
  __m128d o01r, o01i, o23r, o23i;

  r01 = _mm_loadu_pd (p + 0);
  r23 = _mm_loadu_pd (p + 2);
  i01 = _mm_loadu_pd (p + 4);
  i23 = _mm_loadu_pd (p + 6);

  sumr = _mm_add_pd (r01, r23);
  sumi = _mm_add_pd (i01, i23);
  diffr = _mm_sub_pd (r01, r23);
  diffi = _mm_sub_pd (i01, i23);

  o01r = gmp_fft_dp_r22_addsub2 (sumr);
  o01i = gmp_fft_dp_r22_addsub2 (sumi);
  o23r = gmp_fft_dp_r22_addsub2 (_mm_unpacklo_pd (gmp_fft_dp_r22_dup_lo (diffr),
                                                  _mm_xor_pd (gmp_fft_dp_r22_dup_hi (diffi), sign)));
  o23i = gmp_fft_dp_r22_addsub2 (_mm_unpacklo_pd (gmp_fft_dp_r22_dup_lo (diffi),
                                                  gmp_fft_dp_r22_dup_hi (diffr)));

  _mm_storeu_pd (p + 0, o01r);
  _mm_storeu_pd (p + 2, o23r);
  _mm_storeu_pd (p + 4, o01i);
  _mm_storeu_pd (p + 6, o23i);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_inverse_tile (double *p)
{
  const __m128d sign = _mm_set1_pd (-0.0);
  __m128d o01r, o01i, o23r, o23i;
  __m128d e01r, e01i, e23r, e23i;
  __m128d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

  o01r = _mm_loadu_pd (p + 0);
  o23r = _mm_loadu_pd (p + 2);
  o01i = _mm_loadu_pd (p + 4);
  o23i = _mm_loadu_pd (p + 6);

  e01r = gmp_fft_dp_r22_addsub2 (o01r);
  e01i = gmp_fft_dp_r22_addsub2 (o01i);
  e23r = gmp_fft_dp_r22_addsub2 (o23r);
  e23i = gmp_fft_dp_r22_addsub2 (o23i);

  x0r = _mm_add_pd (gmp_fft_dp_r22_dup_lo (e01r), gmp_fft_dp_r22_dup_lo (e23r));
  x0i = _mm_add_pd (gmp_fft_dp_r22_dup_lo (e01i), gmp_fft_dp_r22_dup_lo (e23i));
  x2r = _mm_sub_pd (gmp_fft_dp_r22_dup_lo (e01r), gmp_fft_dp_r22_dup_lo (e23r));
  x2i = _mm_sub_pd (gmp_fft_dp_r22_dup_lo (e01i), gmp_fft_dp_r22_dup_lo (e23i));
  x1r = _mm_add_pd (gmp_fft_dp_r22_dup_hi (e01r), gmp_fft_dp_r22_dup_hi (e23i));
  x1i = _mm_add_pd (gmp_fft_dp_r22_dup_hi (e01i),
                    _mm_xor_pd (gmp_fft_dp_r22_dup_hi (e23r), sign));
  x3r = _mm_sub_pd (gmp_fft_dp_r22_dup_hi (e01r), gmp_fft_dp_r22_dup_hi (e23i));
  x3i = _mm_sub_pd (gmp_fft_dp_r22_dup_hi (e01i),
                    _mm_xor_pd (gmp_fft_dp_r22_dup_hi (e23r), sign));

  _mm_storeu_pd (p + 0, _mm_unpacklo_pd (x0r, x1r));
  _mm_storeu_pd (p + 2, _mm_unpacklo_pd (x2r, x3r));
  _mm_storeu_pd (p + 4, _mm_unpacklo_pd (x0i, x1i));
  _mm_storeu_pd (p + 6, _mm_unpacklo_pd (x2i, x3i));
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_forward_pair_store (double *p, gmp_ntt_v4 re, gmp_ntt_v4 im)
{
  const __m128d sign = _mm_set1_pd (-0.0);
  __m128d r01, r23, i01, i23;
  __m128d sumr, sumi, diffr, diffi;
  __m128d o01r, o01i, o23r, o23i;

  r01 = _mm256_castpd256_pd128 (re);
  r23 = _mm256_extractf128_pd (re, 1);
  i01 = _mm256_castpd256_pd128 (im);
  i23 = _mm256_extractf128_pd (im, 1);

  sumr = _mm_add_pd (r01, r23);
  sumi = _mm_add_pd (i01, i23);
  diffr = _mm_sub_pd (r01, r23);
  diffi = _mm_sub_pd (i01, i23);

  o01r = gmp_fft_dp_r22_addsub2 (sumr);
  o01i = gmp_fft_dp_r22_addsub2 (sumi);
  o23r = gmp_fft_dp_r22_addsub2 (_mm_unpacklo_pd (gmp_fft_dp_r22_dup_lo (diffr),
                                                  _mm_xor_pd (gmp_fft_dp_r22_dup_hi (diffi), sign)));
  o23i = gmp_fft_dp_r22_addsub2 (_mm_unpacklo_pd (gmp_fft_dp_r22_dup_lo (diffi),
                                                  gmp_fft_dp_r22_dup_hi (diffr)));

  _mm_storeu_pd (p + 0, o01r);
  _mm_storeu_pd (p + 2, o23r);
  _mm_storeu_pd (p + 4, o01i);
  _mm_storeu_pd (p + 6, o23i);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_inverse_pair (gmp_ntt_v4 *re_out, gmp_ntt_v4 *im_out,
                                   gmp_ntt_v4 re, gmp_ntt_v4 im)
{
  const __m128d sign = _mm_set1_pd (-0.0);
  __m128d o01r, o01i, o23r, o23i;
  __m128d e01r, e01i, e23r, e23i;
  __m128d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
  __m128d rlo, rhi, ilo, ihi;

  o01r = _mm256_castpd256_pd128 (re);
  o23r = _mm256_extractf128_pd (re, 1);
  o01i = _mm256_castpd256_pd128 (im);
  o23i = _mm256_extractf128_pd (im, 1);

  e01r = gmp_fft_dp_r22_addsub2 (o01r);
  e01i = gmp_fft_dp_r22_addsub2 (o01i);
  e23r = gmp_fft_dp_r22_addsub2 (o23r);
  e23i = gmp_fft_dp_r22_addsub2 (o23i);

  x0r = _mm_add_pd (gmp_fft_dp_r22_dup_lo (e01r), gmp_fft_dp_r22_dup_lo (e23r));
  x0i = _mm_add_pd (gmp_fft_dp_r22_dup_lo (e01i), gmp_fft_dp_r22_dup_lo (e23i));
  x2r = _mm_sub_pd (gmp_fft_dp_r22_dup_lo (e01r), gmp_fft_dp_r22_dup_lo (e23r));
  x2i = _mm_sub_pd (gmp_fft_dp_r22_dup_lo (e01i), gmp_fft_dp_r22_dup_lo (e23i));
  x1r = _mm_add_pd (gmp_fft_dp_r22_dup_hi (e01r), gmp_fft_dp_r22_dup_hi (e23i));
  x1i = _mm_add_pd (gmp_fft_dp_r22_dup_hi (e01i),
                    _mm_xor_pd (gmp_fft_dp_r22_dup_hi (e23r), sign));
  x3r = _mm_sub_pd (gmp_fft_dp_r22_dup_hi (e01r), gmp_fft_dp_r22_dup_hi (e23i));
  x3i = _mm_sub_pd (gmp_fft_dp_r22_dup_hi (e01i),
                    _mm_xor_pd (gmp_fft_dp_r22_dup_hi (e23r), sign));

  rlo = _mm_unpacklo_pd (x0r, x1r);
  rhi = _mm_unpacklo_pd (x2r, x3r);
  ilo = _mm_unpacklo_pd (x0i, x1i);
  ihi = _mm_unpacklo_pd (x2i, x3i);

  *re_out = _mm256_insertf128_pd (_mm256_castpd128_pd256 (rlo), rhi, 1);
  *im_out = _mm256_insertf128_pd (_mm256_castpd128_pd256 (ilo), ihi, 1);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_forward_pair4_store (double *p,
                                          gmp_ntt_v4 a0r, gmp_ntt_v4 a0i,
                                          gmp_ntt_v4 a1r, gmp_ntt_v4 a1i,
                                          gmp_ntt_v4 a2r, gmp_ntt_v4 a2i,
                                          gmp_ntt_v4 a3r, gmp_ntt_v4 a3i)
{
  gmp_ntt_v4 e0r, e0i, e1r, e1i, o0r, o0i, o1r, o1i;
  gmp_ntt_v4 y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
  gmp_ntt_v4 s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i;

  gmp_fft_dp_r22_transpose4x4 (&a0r, &a1r, &a2r, &a3r);
  gmp_fft_dp_r22_transpose4x4 (&a0i, &a1i, &a2i, &a3i);

  e0r = gmp_ntt_v4_add (a0r, a2r);
  e0i = gmp_ntt_v4_add (a0i, a2i);
  e1r = gmp_ntt_v4_sub (a0r, a2r);
  e1i = gmp_ntt_v4_sub (a0i, a2i);
  o0r = gmp_ntt_v4_add (a1r, a3r);
  o0i = gmp_ntt_v4_add (a1i, a3i);
  o1r = gmp_ntt_v4_sub (a1r, a3r);
  o1i = gmp_ntt_v4_sub (a1i, a3i);

  y0r = gmp_ntt_v4_add (e0r, o0r);
  y0i = gmp_ntt_v4_add (e0i, o0i);
  y1r = gmp_ntt_v4_sub (e1r, o1i);
  y1i = gmp_ntt_v4_add (e1i, o1r);
  y2r = gmp_ntt_v4_sub (e0r, o0r);
  y2i = gmp_ntt_v4_sub (e0i, o0i);
  y3r = gmp_ntt_v4_add (e1r, o1i);
  y3i = gmp_ntt_v4_sub (e1i, o1r);

  s0r = y0r; s0i = y0i;
  s1r = y2r; s1i = y2i;
  s2r = y1r; s2i = y1i;
  s3r = y3r; s3i = y3i;
  gmp_fft_dp_r22_transpose4x4 (&s0r, &s1r, &s2r, &s3r);
  gmp_fft_dp_r22_transpose4x4 (&s0i, &s1i, &s2i, &s3i);

  gmp_fft_dp_r22_pair_store (p + 0, s0r, s0i);
  gmp_fft_dp_r22_pair_store (p + 8, s1r, s1i);
  gmp_fft_dp_r22_pair_store (p + 16, s2r, s2i);
  gmp_fft_dp_r22_pair_store (p + 24, s3r, s3i);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_inverse_pair4 (gmp_ntt_v4 *o0r, gmp_ntt_v4 *o0i,
                                    gmp_ntt_v4 *o1r, gmp_ntt_v4 *o1i,
                                    gmp_ntt_v4 *o2r, gmp_ntt_v4 *o2i,
                                    gmp_ntt_v4 *o3r, gmp_ntt_v4 *o3i,
                                    gmp_ntt_v4 y0r, gmp_ntt_v4 y0i,
                                    gmp_ntt_v4 y1r, gmp_ntt_v4 y1i,
                                    gmp_ntt_v4 y2r, gmp_ntt_v4 y2i,
                                    gmp_ntt_v4 y3r, gmp_ntt_v4 y3i)
{
  gmp_ntt_v4 a0r, a0i, a1r, a1i, b0r, b0i, b1r, b1i;
  gmp_ntt_v4 x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

  gmp_fft_dp_r22_transpose4x4 (&y0r, &y1r, &y2r, &y3r);
  gmp_fft_dp_r22_transpose4x4 (&y0i, &y1i, &y2i, &y3i);

  a0r = gmp_ntt_v4_add (y0r, y1r);
  a0i = gmp_ntt_v4_add (y0i, y1i);
  a1r = gmp_ntt_v4_sub (y0r, y1r);
  a1i = gmp_ntt_v4_sub (y0i, y1i);
  b0r = gmp_ntt_v4_add (y2r, y3r);
  b0i = gmp_ntt_v4_add (y2i, y3i);
  b1r = gmp_ntt_v4_sub (y2r, y3r);
  b1i = gmp_ntt_v4_sub (y2i, y3i);

  x0r = gmp_ntt_v4_add (a0r, b0r);
  x0i = gmp_ntt_v4_add (a0i, b0i);
  x1r = gmp_ntt_v4_add (a1r, b1i);
  x1i = gmp_ntt_v4_sub (a1i, b1r);
  x2r = gmp_ntt_v4_sub (a0r, b0r);
  x2i = gmp_ntt_v4_sub (a0i, b0i);
  x3r = gmp_ntt_v4_sub (a1r, b1i);
  x3i = gmp_ntt_v4_add (a1i, b1r);

  gmp_fft_dp_r22_transpose4x4 (&x0r, &x1r, &x2r, &x3r);
  gmp_fft_dp_r22_transpose4x4 (&x0i, &x1i, &x2i, &x3i);
  *o0r = x0r; *o0i = x0i;
  *o1r = x1r; *o1i = x1i;
  *o2r = x2r; *o2i = x2i;
  *o3r = x3r; *o3i = x3i;
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_forward_tile4_batch (double *p)
{
  gmp_ntt_v4 r0, r1, r2, r3, i0, i1, i2, i3;
  gmp_ntt_v4 e0r, e0i, e1r, e1i, o0r, o0i, o1r, o1i;
  gmp_ntt_v4 y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;

  gmp_fft_dp_r22_pair_load (&r0, &i0, p + 0);
  gmp_fft_dp_r22_pair_load (&r1, &i1, p + 8);
  gmp_fft_dp_r22_pair_load (&r2, &i2, p + 16);
  gmp_fft_dp_r22_pair_load (&r3, &i3, p + 24);

  gmp_fft_dp_r22_transpose4x4 (&r0, &r1, &r2, &r3);
  gmp_fft_dp_r22_transpose4x4 (&i0, &i1, &i2, &i3);

  e0r = gmp_ntt_v4_add (r0, r2);
  e0i = gmp_ntt_v4_add (i0, i2);
  e1r = gmp_ntt_v4_sub (r0, r2);
  e1i = gmp_ntt_v4_sub (i0, i2);
  o0r = gmp_ntt_v4_add (r1, r3);
  o0i = gmp_ntt_v4_add (i1, i3);
  o1r = gmp_ntt_v4_sub (r1, r3);
  o1i = gmp_ntt_v4_sub (i1, i3);

  y0r = gmp_ntt_v4_add (e0r, o0r);
  y0i = gmp_ntt_v4_add (e0i, o0i);
  y1r = gmp_ntt_v4_sub (e1r, o1i);
  y1i = gmp_ntt_v4_add (e1i, o1r);
  y2r = gmp_ntt_v4_sub (e0r, o0r);
  y2i = gmp_ntt_v4_sub (e0i, o0i);
  y3r = gmp_ntt_v4_add (e1r, o1i);
  y3i = gmp_ntt_v4_sub (e1i, o1r);

  gmp_fft_dp_r22_transpose4x4 (&y0r, &y2r, &y1r, &y3r);
  gmp_fft_dp_r22_transpose4x4 (&y0i, &y2i, &y1i, &y3i);

  gmp_fft_dp_r22_pair_store (p + 0, y0r, y0i);
  gmp_fft_dp_r22_pair_store (p + 8, y2r, y2i);
  gmp_fft_dp_r22_pair_store (p + 16, y1r, y1i);
  gmp_fft_dp_r22_pair_store (p + 24, y3r, y3i);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_final_inverse_tile4_batch (double *p)
{
  gmp_ntt_v4 y0r, y1r, y2r, y3r, y0i, y1i, y2i, y3i;
  gmp_ntt_v4 a0r, a0i, a1r, a1i, b0r, b0i, b1r, b1i;
  gmp_ntt_v4 x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
  gmp_ntt_v4 tr, ti;

  gmp_fft_dp_r22_pair_load (&y0r, &y0i, p + 0);
  gmp_fft_dp_r22_pair_load (&y1r, &y1i, p + 8);
  gmp_fft_dp_r22_pair_load (&y2r, &y2i, p + 16);
  gmp_fft_dp_r22_pair_load (&y3r, &y3i, p + 24);

  gmp_fft_dp_r22_transpose4x4 (&y0r, &y1r, &y2r, &y3r);
  gmp_fft_dp_r22_transpose4x4 (&y0i, &y1i, &y2i, &y3i);
  tr = y1r; y1r = y2r; y2r = tr;
  ti = y1i; y1i = y2i; y2i = ti;

  a0r = gmp_ntt_v4_add (y0r, y2r);
  a0i = gmp_ntt_v4_add (y0i, y2i);
  a1r = gmp_ntt_v4_sub (y0r, y2r);
  a1i = gmp_ntt_v4_sub (y0i, y2i);
  b0r = gmp_ntt_v4_add (y1r, y3r);
  b0i = gmp_ntt_v4_add (y1i, y3i);
  b1r = gmp_ntt_v4_sub (y1r, y3r);
  b1i = gmp_ntt_v4_sub (y1i, y3i);

  x0r = gmp_ntt_v4_add (a0r, b0r);
  x0i = gmp_ntt_v4_add (a0i, b0i);
  x1r = gmp_ntt_v4_add (a1r, b1i);
  x1i = gmp_ntt_v4_sub (a1i, b1r);
  x2r = gmp_ntt_v4_sub (a0r, b0r);
  x2i = gmp_ntt_v4_sub (a0i, b0i);
  x3r = gmp_ntt_v4_sub (a1r, b1i);
  x3i = gmp_ntt_v4_add (a1i, b1r);

  gmp_fft_dp_r22_transpose4x4 (&x0r, &x1r, &x2r, &x3r);
  gmp_fft_dp_r22_transpose4x4 (&x0i, &x1i, &x2i, &x3i);

  gmp_fft_dp_r22_pair_store (p + 0, x0r, x0i);
  gmp_fft_dp_r22_pair_store (p + 8, x1r, x1i);
  gmp_fft_dp_r22_pair_store (p + 16, x2r, x2i);
  gmp_fft_dp_r22_pair_store (p + 24, x3r, x3i);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_forward_tail_block16 (const double *tw, double *p)
{
  gmp_ntt_v4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
  gmp_ntt_v4 b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
  gmp_ntt_v4 tr, ti, wr, wi, wr2, wi2;

  gmp_fft_dp_r22_pair_load (&a0r, &a0i, p + 0);
  gmp_fft_dp_r22_pair_load (&a1r, &a1i, p + 8);
  gmp_fft_dp_r22_pair_load (&a2r, &a2i, p + 16);
  gmp_fft_dp_r22_pair_load (&a3r, &a3i, p + 24);

  ASSERT (gmp_fft_dp_r22_is_aligned32 (tw));
  wr = gmp_ntt_v4_load (tw + 0);
  wi = gmp_ntt_v4_load (tw + 4);
  wr2 = gmp_ntt_v4_load (tw + 8);
  wi2 = gmp_ntt_v4_load (tw + 12);

  b0r = gmp_ntt_v4_add (a0r, a2r);
  b0i = gmp_ntt_v4_add (a0i, a2i);
  tr = gmp_ntt_v4_sub (a0r, a2r);
  ti = gmp_ntt_v4_sub (a0i, a2i);
  gmp_fft_dp_r22_pair_cmul (&b2r, &b2i, tr, ti, wr, wi);

  b1r = gmp_ntt_v4_add (a1r, a3r);
  b1i = gmp_ntt_v4_add (a1i, a3i);
  tr = gmp_ntt_v4_sub (a1r, a3r);
  ti = gmp_ntt_v4_sub (a1i, a3i);
  gmp_fft_dp_r22_pair_cmul (&b3r, &b3i, tr, ti, gmp_ntt_v4_neg (wi), wr);

  a0r = gmp_ntt_v4_add (b0r, b1r);
  a0i = gmp_ntt_v4_add (b0i, b1i);
  tr = gmp_ntt_v4_sub (b0r, b1r);
  ti = gmp_ntt_v4_sub (b0i, b1i);
  gmp_fft_dp_r22_pair_cmul (&a1r, &a1i, tr, ti, wr2, wi2);
  a2r = gmp_ntt_v4_add (b2r, b3r);
  a2i = gmp_ntt_v4_add (b2i, b3i);
  tr = gmp_ntt_v4_sub (b2r, b3r);
  ti = gmp_ntt_v4_sub (b2i, b3i);
  gmp_fft_dp_r22_pair_cmul (&a3r, &a3i, tr, ti, wr2, wi2);
  gmp_fft_dp_r22_final_forward_pair4_store (p, a0r, a0i, a1r, a1i,
                                            a2r, a2i, a3r, a3i);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_inverse_tail_block16 (const double *tw, double *p)
{
  gmp_ntt_v4 y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
  gmp_ntt_v4 b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
  gmp_ntt_v4 tr, ti, wr, wi, wr2, wi2;

  gmp_fft_dp_r22_pair_load (&y0r, &y0i, p + 0);
  gmp_fft_dp_r22_pair_load (&y1r, &y1i, p + 8);
  gmp_fft_dp_r22_pair_load (&y2r, &y2i, p + 16);
  gmp_fft_dp_r22_pair_load (&y3r, &y3i, p + 24);

  gmp_fft_dp_r22_final_inverse_pair4 (&y0r, &y0i, &y1r, &y1i,
                                      &y2r, &y2i, &y3r, &y3i,
                                      y0r, y0i, y1r, y1i,
                                      y2r, y2i, y3r, y3i);

  ASSERT (gmp_fft_dp_r22_is_aligned32 (tw));
  wr = gmp_ntt_v4_load (tw + 0);
  wi = gmp_ntt_v4_load (tw + 4);
  wr2 = gmp_ntt_v4_load (tw + 8);
  wi2 = gmp_ntt_v4_load (tw + 12);

  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, y1r, y1i, wr2, wi2);
  b0r = gmp_ntt_v4_add (y0r, tr);
  b0i = gmp_ntt_v4_add (y0i, ti);
  b1r = gmp_ntt_v4_sub (y0r, tr);
  b1i = gmp_ntt_v4_sub (y0i, ti);

  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, y3r, y3i, wr2, wi2);
  b2r = gmp_ntt_v4_add (y2r, tr);
  b2i = gmp_ntt_v4_add (y2i, ti);
  b3r = gmp_ntt_v4_sub (y2r, tr);
  b3i = gmp_ntt_v4_sub (y2i, ti);

  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, b2r, b2i, wr, wi);
  gmp_fft_dp_r22_pair_store (p + 0, gmp_ntt_v4_add (b0r, tr),
                             gmp_ntt_v4_add (b0i, ti));
  gmp_fft_dp_r22_pair_store (p + 16, gmp_ntt_v4_sub (b0r, tr),
                             gmp_ntt_v4_sub (b0i, ti));

  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, b3r, b3i, gmp_ntt_v4_neg (wi), wr);
  gmp_fft_dp_r22_pair_store (p + 8, gmp_ntt_v4_add (b1r, tr),
                             gmp_ntt_v4_add (b1i, ti));
  gmp_fft_dp_r22_pair_store (p + 24, gmp_ntt_v4_sub (b1r, tr),
                             gmp_ntt_v4_sub (b1i, ti));
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_radix23_final_forward_block (double *p)
{
  const double c = 0.70710678118654752440;
  gmp_ntt_v4 wr, wi, ar, ai, br, bi, sr, si;

  wr = gmp_ntt_v4_set_d4 (1.0, c, 0.0, -c);
  wi = gmp_ntt_v4_set_d4 (0.0, c, 1.0, c);
  gmp_fft_dp_r22_pair_load (&ar, &ai, p);
  gmp_fft_dp_r22_pair_load (&br, &bi, p + 8);
  sr = gmp_ntt_v4_add (ar, br);
  si = gmp_ntt_v4_add (ai, bi);
  ar = gmp_ntt_v4_sub (ar, br);
  ai = gmp_ntt_v4_sub (ai, bi);
  gmp_fft_dp_r22_pair_cmul (&br, &bi, ar, ai, wr, wi);
  gmp_fft_dp_r22_final_forward_pair_store (p, sr, si);
  gmp_fft_dp_r22_final_forward_pair_store (p + 8, br, bi);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_radix23_final_inverse_block (double *p)
{
  const double c = 0.70710678118654752440;
  gmp_ntt_v4 wr, wi, ar, ai, br, bi, tr, ti;

  wr = gmp_ntt_v4_set_d4 (1.0, c, 0.0, -c);
  wi = gmp_ntt_v4_set_d4 (0.0, c, 1.0, c);
  gmp_fft_dp_r22_pair_load (&ar, &ai, p);
  gmp_fft_dp_r22_pair_load (&br, &bi, p + 8);
  gmp_fft_dp_r22_final_inverse_pair (&ar, &ai, ar, ai);
  gmp_fft_dp_r22_final_inverse_pair (&br, &bi, br, bi);
  gmp_fft_dp_r22_pair_cmul_conj (&tr, &ti, br, bi, wr, wi);
  gmp_fft_dp_r22_pair_store (p, gmp_ntt_v4_add (ar, tr), gmp_ntt_v4_add (ai, ti));
  gmp_fft_dp_r22_pair_store (p + 8, gmp_ntt_v4_sub (ar, tr), gmp_ntt_v4_sub (ai, ti));
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_radix23_final_forward_block2_batch (double *p)
{
  const double c = 0.70710678118654752440;
  gmp_ntt_v4 wr, wi;
  gmp_ntt_v4 a0r, a0i, b0r, b0i, a1r, a1i, b1r, b1i;
  gmp_ntt_v4 s0r, s0i, d0r, d0i, s1r, s1i, d1r, d1i;

  wr = gmp_ntt_v4_set_d4 (1.0, c, 0.0, -c);
  wi = gmp_ntt_v4_set_d4 (0.0, c, 1.0, c);

  gmp_fft_dp_r22_pair_load (&a0r, &a0i, p + 0);
  gmp_fft_dp_r22_pair_load (&b0r, &b0i, p + 8);
  gmp_fft_dp_r22_pair_load (&a1r, &a1i, p + 16);
  gmp_fft_dp_r22_pair_load (&b1r, &b1i, p + 24);

  s0r = gmp_ntt_v4_add (a0r, b0r);
  s0i = gmp_ntt_v4_add (a0i, b0i);
  a0r = gmp_ntt_v4_sub (a0r, b0r);
  a0i = gmp_ntt_v4_sub (a0i, b0i);
  gmp_fft_dp_r22_pair_cmul (&d0r, &d0i, a0r, a0i, wr, wi);

  s1r = gmp_ntt_v4_add (a1r, b1r);
  s1i = gmp_ntt_v4_add (a1i, b1i);
  a1r = gmp_ntt_v4_sub (a1r, b1r);
  a1i = gmp_ntt_v4_sub (a1i, b1i);
  gmp_fft_dp_r22_pair_cmul (&d1r, &d1i, a1r, a1i, wr, wi);

  gmp_fft_dp_r22_final_forward_pair4_store (p, s0r, s0i, d0r, d0i,
                                            s1r, s1i, d1r, d1i);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_radix23_final_inverse_block2_batch (double *p)
{
  const double c = 0.70710678118654752440;
  gmp_ntt_v4 wr, wi;
  gmp_ntt_v4 s0r, s0i, d0r, d0i, s1r, s1i, d1r, d1i;
  gmp_ntt_v4 t0r, t0i, t1r, t1i;

  wr = gmp_ntt_v4_set_d4 (1.0, c, 0.0, -c);
  wi = gmp_ntt_v4_set_d4 (0.0, c, 1.0, c);

  gmp_fft_dp_r22_pair_load (&s0r, &s0i, p + 0);
  gmp_fft_dp_r22_pair_load (&d0r, &d0i, p + 8);
  gmp_fft_dp_r22_pair_load (&s1r, &s1i, p + 16);
  gmp_fft_dp_r22_pair_load (&d1r, &d1i, p + 24);

  gmp_fft_dp_r22_final_inverse_pair4 (&s0r, &s0i, &d0r, &d0i,
                                      &s1r, &s1i, &d1r, &d1i,
                                      s0r, s0i, d0r, d0i,
                                      s1r, s1i, d1r, d1i);

  gmp_fft_dp_r22_pair_cmul_conj (&t0r, &t0i, d0r, d0i, wr, wi);
  gmp_fft_dp_r22_pair_store (p + 0, gmp_ntt_v4_add (s0r, t0r),
                             gmp_ntt_v4_add (s0i, t0i));
  gmp_fft_dp_r22_pair_store (p + 8, gmp_ntt_v4_sub (s0r, t0r),
                             gmp_ntt_v4_sub (s0i, t0i));

  gmp_fft_dp_r22_pair_cmul_conj (&t1r, &t1i, d1r, d1i, wr, wi);
  gmp_fft_dp_r22_pair_store (p + 16, gmp_ntt_v4_add (s1r, t1r),
                             gmp_ntt_v4_add (s1i, t1i));
  gmp_fft_dp_r22_pair_store (p + 24, gmp_ntt_v4_sub (s1r, t1r),
                             gmp_ntt_v4_sub (s1i, t1i));
}

static void
gmp_fft_dp_r22_build_interleaved_ri_u64 (uint64_t *dst, size_t limb_count,
                                         mp_srcptr ap, size_t an,
                                         mp_srcptr bp, size_t bn)
{
  size_t i;
  size_t both = (an < bn) ? an : bn;
  size_t maxab = (an > bn) ? an : bn;
  size_t max_copy = (maxab < limb_count) ? maxab : limb_count;

  memset (dst, 0, 2u * limb_count * sizeof (uint64_t));

  for (i = 0; i + 4 <= both; i += 4)
    {
      __m256i va = _mm256_loadu_si256 ((const __m256i_u *) (ap + i));
      __m256i vb = _mm256_loadu_si256 ((const __m256i_u *) (bp + i));
      __m256i lo = _mm256_unpacklo_epi64 (va, vb);
      __m256i hi = _mm256_unpackhi_epi64 (va, vb);
      __m256i ab01 = _mm256_permute2x128_si256 (lo, hi, 0x20);
      __m256i ab23 = _mm256_permute2x128_si256 (lo, hi, 0x31);

      _mm256_storeu_si256 ((__m256i_u *) (dst + 2u * i), ab01);
      _mm256_storeu_si256 ((__m256i_u *) (dst + 2u * i + 4u), ab23);
    }

  for (; i < max_copy; ++i)
    {
      if (i < an)
        dst[2u * i + 0] = (uint64_t) ap[i];
      if (i < bn)
        dst[2u * i + 1] = (uint64_t) bp[i];
    }
}

static void
gmp_fft_dp_r22_build_pq_ri_u64 (uint64_t *dst, uint32_t n,
                                mp_srcptr ap, size_t an)
{
  uint32_t groups = (n + 3u) >> 2;
  uint32_t g = 0;
  uint32_t avx2_blocks = (uint32_t) (an >> 2);
  uint32_t full_groups = (uint32_t) (an >> 1);
  const __m256i deinterleave_mask = _mm256_setr_epi8 (
      0, 1, 4, 5, 8, 9, 12, 13,
      2, 3, 6, 7, 10, 11, 14, 15,
      0, 1, 4, 5, 8, 9, 12, 13,
      2, 3, 6, 7, 10, 11, 14, 15);
  const __m128i even_mask = _mm_setr_epi8 (0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80);
  const __m128i odd_mask = _mm_setr_epi8 (2, 3, 6, 7, 10, 11, 14, 15,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80);

  memset (dst, 0, (size_t) n / 2u * sizeof (uint64_t));

  if (avx2_blocks > (groups >> 1))
    avx2_blocks = groups >> 1;

  for (; g + 1u < 2u * avx2_blocks; g += 2u)
    {
      __m256i v = _mm256_loadu_si256 ((const __m256i_u *) (ap + 2u * g));
      __m256i packed = _mm256_shuffle_epi8 (v, deinterleave_mask);

      _mm256_storeu_si256 ((__m256i_u *) (dst + 2u * g), packed);
    }

  if (full_groups > groups)
    full_groups = groups;

  for (; g < full_groups; ++g)
    {
      __m128i v = _mm_loadu_si128 ((const __m128i_u *) (ap + 2u * g));
      __m128i vr = _mm_shuffle_epi8 (v, even_mask);
      __m128i vi = _mm_shuffle_epi8 (v, odd_mask);

      dst[2u * g + 0u] = (uint64_t) _mm_cvtsi128_si64 (vr);
      dst[2u * g + 1u] = (uint64_t) _mm_cvtsi128_si64 (vi);
    }

  for (; g < groups; ++g)
    {
      uint64_t ru64 = 0;
      uint64_t iu64 = 0;
      uint32_t lane;

      for (lane = 0; lane < 4u; ++lane)
        {
          uint32_t base_digit = 8u * g + 2u * lane;
          uint64_t re = (uint64_t) gmp_fft_dp_r22_get_u16_digit (ap, an, base_digit + 0u);
          uint64_t im = (uint64_t) gmp_fft_dp_r22_get_u16_digit (ap, an, base_digit + 1u);
          ru64 |= re << (16u * lane);
          iu64 |= im << (16u * lane);
        }

      dst[2u * g + 0u] = ru64;
      dst[2u * g + 1u] = iu64;
    }
}

static void
gmp_fft_dp_r22_build_pq_padded_limbs (uint64_t *dst, uint32_t n,
                                      mp_srcptr ap, size_t an)
{
  size_t limb_count = (size_t) n >> 1;
  size_t copy_count = (an < limb_count) ? an : limb_count;

  memset (dst, 0, limb_count * sizeof (uint64_t));
  memcpy (dst, ap, copy_count * sizeof (uint64_t));
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_load_pq_u16_pair_from_limbs (gmp_ntt_v4 *re, gmp_ntt_v4 *im,
                                            const uint64_t *src, uint32_t digit_idx)
{
  const __m128i even_mask = _mm_setr_epi8 (0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80);
  const __m128i odd_mask = _mm_setr_epi8 (2, 3, 6, 7, 10, 11, 14, 15,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80);
  __m128i v;
  __m128i vr;
  __m128i vi;
  __m256i d32;

  ASSERT ((digit_idx & 3u) == 0);
  v = _mm_loadu_si128 ((const __m128i_u *) (src + (digit_idx >> 1)));
  vr = _mm_shuffle_epi8 (v, even_mask);
  vi = _mm_shuffle_epi8 (v, odd_mask);
  d32 = _mm256_cvtepu16_epi32 (vr);
  *re = _mm256_cvtepi32_pd (_mm256_castsi256_si128 (d32));
  d32 = _mm256_cvtepu16_epi32 (vi);
  *im = _mm256_cvtepi32_pd (_mm256_castsi256_si128 (d32));
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked (gmp_ntt_v4 *re, gmp_ntt_v4 *im,
                                                    mp_srcptr src, size_t limb_count,
                                                    uint32_t digit_idx)
{
  const __m128i even_mask = _mm_setr_epi8 (0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80);
  const __m128i odd_mask = _mm_setr_epi8 (2, 3, 6, 7, 10, 11, 14, 15,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80);
  size_t limb_idx = (size_t) (digit_idx >> 1);
  __m128i v;
  __m128i vr;
  __m128i vi;
  __m256i d32;

  ASSERT ((digit_idx & 3u) == 0);

  if (limb_idx >= limb_count)
    {
      *re = gmp_ntt_v4_zero ();
      *im = gmp_ntt_v4_zero ();
      return;
    }
  else if (limb_idx + 1u < limb_count)
    v = _mm_loadu_si128 ((const __m128i_u *) (src + limb_idx));
  else
    v = _mm_set_epi64x (0, (long long) src[limb_idx]);

  vr = _mm_shuffle_epi8 (v, even_mask);
  vi = _mm_shuffle_epi8 (v, odd_mask);
  d32 = _mm256_cvtepu16_epi32 (vr);
  *re = _mm256_cvtepi32_pd (_mm256_castsi256_si128 (d32));
  d32 = _mm256_cvtepu16_epi32 (vi);
  *im = _mm256_cvtepi32_pd (_mm256_castsi256_si128 (d32));
}

GMP_NTT_FORCEINLINE static gmp_ntt_v4
gmp_fft_dp_r22_u16x4_to_f64_magic (__m128i v16)
{
  const __m128i zero128 = _mm_setzero_si128 ();
  const __m256i bias_bits = _mm256_set1_epi64x (0x4330000000000000ULL);
  const gmp_ntt_v4 bias52 = _mm256_castsi256_pd (bias_bits);
  __m128i d32 = _mm_unpacklo_epi16 (v16, zero128);
  __m128i lo64 = _mm_unpacklo_epi32 (d32, zero128);
  __m128i hi64 = _mm_unpackhi_epi32 (d32, zero128);
  __m256i u64 = _mm256_set_m128i (hi64, lo64);

  return _mm256_sub_pd (_mm256_castsi256_pd (_mm256_or_si256 (u64, bias_bits)),
                        bias52);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked_magic (gmp_ntt_v4 *re,
                                                          gmp_ntt_v4 *im,
                                                          mp_srcptr src,
                                                          size_t limb_count,
                                                          uint32_t digit_idx)
{
  const __m128i even_mask = _mm_setr_epi8 (0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80);
  const __m128i odd_mask = _mm_setr_epi8 (2, 3, 6, 7, 10, 11, 14, 15,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80);
  size_t limb_idx = (size_t) (digit_idx >> 1);
  __m128i v;

  ASSERT ((digit_idx & 3u) == 0);

  if (limb_idx >= limb_count)
    {
      *re = gmp_ntt_v4_zero ();
      *im = gmp_ntt_v4_zero ();
      return;
    }
  else if (limb_idx + 1u < limb_count)
    v = _mm_loadu_si128 ((const __m128i_u *) (src + limb_idx));
  else
    v = _mm_set_epi64x (0, (long long) src[limb_idx]);

  *re = gmp_fft_dp_r22_u16x4_to_f64_magic (_mm_shuffle_epi8 (v, even_mask));
  *im = gmp_fft_dp_r22_u16x4_to_f64_magic (_mm_shuffle_epi8 (v, odd_mask));
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_load_pq_u16_pair_from_digits_checked (gmp_ntt_v4 *re,
                                                     gmp_ntt_v4 *im,
                                                     const uint16_t *src,
                                                     size_t digit_count,
                                                     uint32_t complex_idx)
{
  const __m128i even_mask = _mm_setr_epi8 (0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80,
                                           (char) 0x80, (char) 0x80);
  const __m128i odd_mask = _mm_setr_epi8 (2, 3, 6, 7, 10, 11, 14, 15,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80,
                                          (char) 0x80, (char) 0x80);
  size_t raw_idx = 2u * (size_t) complex_idx;
  __m128i v;
  __m128i vr;
  __m128i vi;
  __m256i d32;

  ASSERT ((complex_idx & 3u) == 0);

  if (raw_idx >= digit_count)
    {
      *re = gmp_ntt_v4_zero ();
      *im = gmp_ntt_v4_zero ();
      return;
    }
  else if (raw_idx + 8u <= digit_count)
    v = _mm_loadu_si128 ((const __m128i_u *) (src + raw_idx));
  else
    {
      uint16_t tmp[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      size_t rem = digit_count - raw_idx;
      memcpy (tmp, src + raw_idx, rem * sizeof (uint16_t));
      v = _mm_loadu_si128 ((const __m128i_u *) tmp);
    }

  vr = _mm_shuffle_epi8 (v, even_mask);
  vi = _mm_shuffle_epi8 (v, odd_mask);
  d32 = _mm256_cvtepu16_epi32 (vr);
  *re = _mm256_cvtepi32_pd (_mm256_castsi256_si128 (d32));
  d32 = _mm256_cvtepu16_epi32 (vi);
  *im = _mm256_cvtepi32_pd (_mm256_castsi256_si128 (d32));
}

static void
gmp_fft_dp_r22_unpack_inputs_to_aosov (double *data, uint32_t n,
                                       const uint64_t *ri_u64)
{
  uint32_t tiles = (n + 3u) >> 2;
  uint32_t t;

  for (t = 0; t < tiles; ++t)
    {
      gmp_ntt_v4 re, im;
      gmp_fft_dp_r22_load_u16_pair (&re, &im, ri_u64, 4u * t);
      gmp_fft_dp_r22_pair_store (data + 8u * (size_t) t, re, im);
    }
}

static void
gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_limbs (double *data, uint32_t n,
                                                     const uint64_t *src)
{
  uint32_t tiles = (n + 3u) >> 2;
  uint32_t t;

  for (t = 0; t < tiles; ++t)
    {
      gmp_ntt_v4 re, im;
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs (&re, &im, src, 4u * t);
      gmp_fft_dp_r22_pair_store (data + 8u * (size_t) t, re, im);
    }
}

static void
gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_limbs_checked (double *data, uint32_t n,
                                                             mp_srcptr src,
                                                             size_t limb_count)
{
  uint32_t tiles = (n + 3u) >> 2;
  uint32_t t;

  for (t = 0; t < tiles; ++t)
    {
      gmp_ntt_v4 re, im;
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked (&re, &im, src,
                                                          limb_count, 4u * t);
      gmp_fft_dp_r22_pair_store (data + 8u * (size_t) t, re, im);
    }
}

static void
gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_limbs_checked_magic (double *data,
                                                                   uint32_t n,
                                                                   mp_srcptr src,
                                                                   size_t limb_count)
{
  uint32_t tiles = (n + 3u) >> 2;
  uint32_t t;

  for (t = 0; t < tiles; ++t)
    {
      gmp_ntt_v4 re, im;
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked_magic (&re, &im, src,
                                                                limb_count,
                                                                4u * t);
      gmp_fft_dp_r22_pair_store (data + 8u * (size_t) t, re, im);
    }
}

static void
gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_digits_checked (double *data,
                                                              uint32_t n,
                                                              const uint16_t *src,
                                                              size_t digit_count)
{
  uint32_t tiles = (n + 3u) >> 2;
  uint32_t t;

  for (t = 0; t < tiles; ++t)
    {
      gmp_ntt_v4 re, im;
      gmp_fft_dp_r22_load_pq_u16_pair_from_digits_checked (&re, &im, src,
                                                           digit_count, 4u * t);
      gmp_fft_dp_r22_pair_store (data + 8u * (size_t) t, re, im);
    }
}

static void
gmp_fft_dp_r22_forward_stage_radix2 (double *data, uint32_t n)
{
  uint32_t base;
  for (base = 0; base < n; base += 2u)
    {
      double ar = gmp_fft_dp_r22_get_re (data, base + 0u);
      double ai = gmp_fft_dp_r22_get_im (data, base + 0u);
      double br = gmp_fft_dp_r22_get_re (data, base + 1u);
      double bi = gmp_fft_dp_r22_get_im (data, base + 1u);

      gmp_fft_dp_r22_set_complex (data, base + 0u, ar + br, ai + bi);
      gmp_fft_dp_r22_set_complex (data, base + 1u, ar - br, ai - bi);
    }
}

static void
gmp_fft_dp_r22_inverse_stage_radix2 (double *data, uint32_t n)
{
  uint32_t base;
  for (base = 0; base < n; base += 2u)
    {
      double ar = gmp_fft_dp_r22_get_re (data, base + 0u);
      double ai = gmp_fft_dp_r22_get_im (data, base + 0u);
      double br = gmp_fft_dp_r22_get_re (data, base + 1u);
      double bi = gmp_fft_dp_r22_get_im (data, base + 1u);

      gmp_fft_dp_r22_set_complex (data, base + 0u, ar + br, ai + bi);
      gmp_fft_dp_r22_set_complex (data, base + 1u, ar - br, ai - bi);
    }
}

static uint32_t
gmp_fft_dp_r22_tail_block_complex (uint32_t n)
{
  ASSERT ((n & (n - 1u)) == 0);
  if (n >= 16u && ((gmp_ntt_ctz (n) & 1u) == 0))
    return 16u;
  if (n >= 8u)
    return 8u;
  return 4u;
}

static void
gmp_fft_dp_r22_forward_tail_range (double *data, uint32_t start, uint32_t stop,
                                   const double *tw16)
{
  uint32_t blk = gmp_fft_dp_r22_tail_block_complex (stop - start);
  uint32_t t;

  if (blk == 16u)
    {
      for (t = start; t < stop; t += 16u)
        gmp_fft_dp_r22_forward_tail_block16 (tw16,
                                             gmp_fft_dp_r22_tile_ptr (data, t));
      return;
    }

  if (blk == 8u)
    {
      for (t = start; t + 16u <= stop; t += 16u)
        gmp_fft_dp_r22_radix23_final_forward_block2_batch (
          gmp_fft_dp_r22_tile_ptr (data, t));
      for (; t < stop; t += 8u)
        gmp_fft_dp_r22_radix23_final_forward_block (
          gmp_fft_dp_r22_tile_ptr (data, t));
      return;
    }

  for (t = start; t + 16u <= stop; t += 16u)
    gmp_fft_dp_r22_final_forward_tile4_batch (gmp_fft_dp_r22_tile_ptr (data, t));
  for (; t < stop; t += 4u)
    gmp_fft_dp_r22_final_forward_tile (gmp_fft_dp_r22_tile_ptr (data, t));
}

static void
gmp_fft_dp_r22_inverse_tail_range (double *data, uint32_t start, uint32_t stop,
                                   const double *tw16)
{
  uint32_t blk = gmp_fft_dp_r22_tail_block_complex (stop - start);
  uint32_t t;

  if (blk == 16u)
    {
      for (t = start; t < stop; t += 16u)
        gmp_fft_dp_r22_inverse_tail_block16 (tw16,
                                             gmp_fft_dp_r22_tile_ptr (data, t));
      return;
    }

  if (blk == 8u)
    {
      for (t = start; t + 16u <= stop; t += 16u)
        gmp_fft_dp_r22_radix23_final_inverse_block2_batch (
          gmp_fft_dp_r22_tile_ptr (data, t));
      for (; t < stop; t += 8u)
        gmp_fft_dp_r22_radix23_final_inverse_block (
          gmp_fft_dp_r22_tile_ptr (data, t));
      return;
    }

  for (t = start; t + 16u <= stop; t += 16u)
    gmp_fft_dp_r22_final_inverse_tile4_batch (gmp_fft_dp_r22_tile_ptr (data, t));
  for (; t < stop; t += 4u)
    gmp_fft_dp_r22_final_inverse_tile (gmp_fft_dp_r22_tile_ptr (data, t));
}

static void
gmp_fft_dp_r22_unpack_fwd (double *data, uint32_t n, const uint64_t *ri_u64,
                           const double *tw)
{
  uint32_t l = n >> 2;
  uint32_t j;
  double *p0 = gmp_fft_dp_r22_tile_ptr (data, 0);
  double *p1 = gmp_fft_dp_r22_tile_ptr (data, l);
  double *p2 = gmp_fft_dp_r22_tile_ptr (data, 2u * l);
  double *p3 = gmp_fft_dp_r22_tile_ptr (data, 3u * l);

  ASSERT ((l & 3u) == 0);

  for (j = 0; j < l; j += 4u)
    {
      gmp_ntt_v4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
      gmp_ntt_v4 wr, wi, wr2, wi2, tr, ti;

      gmp_fft_dp_r22_load_u16_pair (&a0r, &a0i, ri_u64, j + 0u * l);
      gmp_fft_dp_r22_load_u16_pair (&a1r, &a1i, ri_u64, j + 1u * l);
      gmp_fft_dp_r22_load_u16_pair (&a2r, &a2i, ri_u64, j + 2u * l);
      gmp_fft_dp_r22_load_u16_pair (&a3r, &a3i, ri_u64, j + 3u * l);

      wr = gmp_ntt_v4_load (tw + 0);
      wi = gmp_ntt_v4_load (tw + 4);
      wr2 = gmp_ntt_v4_load (tw + 8);
      wi2 = gmp_ntt_v4_load (tw + 12);

      tr = gmp_ntt_v4_add (a0r, a2r);
      a2r = gmp_ntt_v4_sub (a0r, a2r);
      a0r = tr;
      ti = gmp_ntt_v4_add (a0i, a2i);
      a2i = gmp_ntt_v4_sub (a0i, a2i);
      a0i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a2r, &a2i, wr, wi);

      tr = gmp_ntt_v4_add (a1r, a3r);
      a3r = gmp_ntt_v4_sub (a1r, a3r);
      a1r = tr;
      ti = gmp_ntt_v4_add (a1i, a3i);
      a3i = gmp_ntt_v4_sub (a1i, a3i);
      a1i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, gmp_ntt_v4_neg (wi), wr);

      gmp_fft_dp_r22_pair_store (p0, gmp_ntt_v4_add (a0r, a1r),
                                 gmp_ntt_v4_add (a0i, a1i));
      a1r = gmp_ntt_v4_sub (a0r, a1r);
      a1i = gmp_ntt_v4_sub (a0i, a1i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a1r, &a1i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p1, a1r, a1i);

      gmp_fft_dp_r22_pair_store (p2, gmp_ntt_v4_add (a2r, a3r),
                                 gmp_ntt_v4_add (a2i, a3i));
      a3r = gmp_ntt_v4_sub (a2r, a3r);
      a3i = gmp_ntt_v4_sub (a2i, a3i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p3, a3r, a3i);

      p0 += 8;
      p1 += 8;
      p2 += 8;
      p3 += 8;
      tw += 16;
    }
}

static void
gmp_fft_dp_r22_unpack_fwd_pq_from_limbs (double *data, uint32_t n,
                                         const uint64_t *src,
                                         const double *tw)
{
  uint32_t l = n >> 2;
  uint32_t j;
  double *p0 = gmp_fft_dp_r22_tile_ptr (data, 0);
  double *p1 = gmp_fft_dp_r22_tile_ptr (data, l);
  double *p2 = gmp_fft_dp_r22_tile_ptr (data, 2u * l);
  double *p3 = gmp_fft_dp_r22_tile_ptr (data, 3u * l);

  ASSERT ((l & 3u) == 0);

  for (j = 0; j < l; j += 4u)
    {
      gmp_ntt_v4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
      gmp_ntt_v4 wr, wi, wr2, wi2, tr, ti;

      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs (&a0r, &a0i, src, j + 0u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs (&a1r, &a1i, src, j + 1u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs (&a2r, &a2i, src, j + 2u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs (&a3r, &a3i, src, j + 3u * l);

      wr = gmp_ntt_v4_load (tw + 0);
      wi = gmp_ntt_v4_load (tw + 4);
      wr2 = gmp_ntt_v4_load (tw + 8);
      wi2 = gmp_ntt_v4_load (tw + 12);

      tr = gmp_ntt_v4_add (a0r, a2r);
      a2r = gmp_ntt_v4_sub (a0r, a2r);
      a0r = tr;
      ti = gmp_ntt_v4_add (a0i, a2i);
      a2i = gmp_ntt_v4_sub (a0i, a2i);
      a0i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a2r, &a2i, wr, wi);

      tr = gmp_ntt_v4_add (a1r, a3r);
      a3r = gmp_ntt_v4_sub (a1r, a3r);
      a1r = tr;
      ti = gmp_ntt_v4_add (a1i, a3i);
      a3i = gmp_ntt_v4_sub (a1i, a3i);
      a1i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, gmp_ntt_v4_neg (wi), wr);

      gmp_fft_dp_r22_pair_store (p0, gmp_ntt_v4_add (a0r, a1r),
                                 gmp_ntt_v4_add (a0i, a1i));
      a1r = gmp_ntt_v4_sub (a0r, a1r);
      a1i = gmp_ntt_v4_sub (a0i, a1i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a1r, &a1i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p1, a1r, a1i);

      gmp_fft_dp_r22_pair_store (p2, gmp_ntt_v4_add (a2r, a3r),
                                 gmp_ntt_v4_add (a2i, a3i));
      a3r = gmp_ntt_v4_sub (a2r, a3r);
      a3i = gmp_ntt_v4_sub (a2i, a3i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p3, a3r, a3i);

      p0 += 8;
      p1 += 8;
      p2 += 8;
      p3 += 8;
      tw += 16;
    }
}

static void
gmp_fft_dp_r22_unpack_fwd_pq_from_limbs_checked (double *data, uint32_t n,
                                                 mp_srcptr src, size_t limb_count,
                                                 const double *tw)
{
  uint32_t l = n >> 2;
  uint32_t j;
  double *p0 = gmp_fft_dp_r22_tile_ptr (data, 0);
  double *p1 = gmp_fft_dp_r22_tile_ptr (data, l);
  double *p2 = gmp_fft_dp_r22_tile_ptr (data, 2u * l);
  double *p3 = gmp_fft_dp_r22_tile_ptr (data, 3u * l);

  ASSERT ((l & 3u) == 0);

  for (j = 0; j < l; j += 4u)
    {
      gmp_ntt_v4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
      gmp_ntt_v4 wr, wi, wr2, wi2, tr, ti;

      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked (&a0r, &a0i, src,
                                                          limb_count, j + 0u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked (&a1r, &a1i, src,
                                                          limb_count, j + 1u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked (&a2r, &a2i, src,
                                                          limb_count, j + 2u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked (&a3r, &a3i, src,
                                                          limb_count, j + 3u * l);

      wr = gmp_ntt_v4_load (tw + 0);
      wi = gmp_ntt_v4_load (tw + 4);
      wr2 = gmp_ntt_v4_load (tw + 8);
      wi2 = gmp_ntt_v4_load (tw + 12);

      tr = gmp_ntt_v4_add (a0r, a2r);
      a2r = gmp_ntt_v4_sub (a0r, a2r);
      a0r = tr;
      ti = gmp_ntt_v4_add (a0i, a2i);
      a2i = gmp_ntt_v4_sub (a0i, a2i);
      a0i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a2r, &a2i, wr, wi);

      tr = gmp_ntt_v4_add (a1r, a3r);
      a3r = gmp_ntt_v4_sub (a1r, a3r);
      a1r = tr;
      ti = gmp_ntt_v4_add (a1i, a3i);
      a3i = gmp_ntt_v4_sub (a1i, a3i);
      a1i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, gmp_ntt_v4_neg (wi), wr);

      gmp_fft_dp_r22_pair_store (p0, gmp_ntt_v4_add (a0r, a1r),
                                 gmp_ntt_v4_add (a0i, a1i));
      a1r = gmp_ntt_v4_sub (a0r, a1r);
      a1i = gmp_ntt_v4_sub (a0i, a1i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a1r, &a1i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p1, a1r, a1i);

      gmp_fft_dp_r22_pair_store (p2, gmp_ntt_v4_add (a2r, a3r),
                                 gmp_ntt_v4_add (a2i, a3i));
      a3r = gmp_ntt_v4_sub (a2r, a3r);
      a3i = gmp_ntt_v4_sub (a2i, a3i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p3, a3r, a3i);

      p0 += 8;
      p1 += 8;
      p2 += 8;
      p3 += 8;
      tw += 16;
    }
}

static void
gmp_fft_dp_r22_unpack_fwd_pq_from_limbs_checked_magic (double *data, uint32_t n,
                                                       mp_srcptr src,
                                                       size_t limb_count,
                                                       const double *tw)
{
  uint32_t l = n >> 2;
  uint32_t j;
  double *p0 = gmp_fft_dp_r22_tile_ptr (data, 0);
  double *p1 = gmp_fft_dp_r22_tile_ptr (data, l);
  double *p2 = gmp_fft_dp_r22_tile_ptr (data, 2u * l);
  double *p3 = gmp_fft_dp_r22_tile_ptr (data, 3u * l);

  ASSERT ((l & 3u) == 0);

  for (j = 0; j < l; j += 4u)
    {
      gmp_ntt_v4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
      gmp_ntt_v4 wr, wi, wr2, wi2, tr, ti;

      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked_magic (&a0r, &a0i, src,
                                                                limb_count, j + 0u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked_magic (&a1r, &a1i, src,
                                                                limb_count, j + 1u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked_magic (&a2r, &a2i, src,
                                                                limb_count, j + 2u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_limbs_checked_magic (&a3r, &a3i, src,
                                                                limb_count, j + 3u * l);

      wr = gmp_ntt_v4_load (tw + 0);
      wi = gmp_ntt_v4_load (tw + 4);
      wr2 = gmp_ntt_v4_load (tw + 8);
      wi2 = gmp_ntt_v4_load (tw + 12);

      tr = gmp_ntt_v4_add (a0r, a2r);
      a2r = gmp_ntt_v4_sub (a0r, a2r);
      a0r = tr;
      ti = gmp_ntt_v4_add (a0i, a2i);
      a2i = gmp_ntt_v4_sub (a0i, a2i);
      a0i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a2r, &a2i, wr, wi);

      tr = gmp_ntt_v4_add (a1r, a3r);
      a3r = gmp_ntt_v4_sub (a1r, a3r);
      a1r = tr;
      ti = gmp_ntt_v4_add (a1i, a3i);
      a3i = gmp_ntt_v4_sub (a1i, a3i);
      a1i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, gmp_ntt_v4_neg (wi), wr);

      gmp_fft_dp_r22_pair_store (p0, gmp_ntt_v4_add (a0r, a1r),
                                 gmp_ntt_v4_add (a0i, a1i));
      a1r = gmp_ntt_v4_sub (a0r, a1r);
      a1i = gmp_ntt_v4_sub (a0i, a1i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a1r, &a1i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p1, a1r, a1i);

      gmp_fft_dp_r22_pair_store (p2, gmp_ntt_v4_add (a2r, a3r),
                                 gmp_ntt_v4_add (a2i, a3i));
      a3r = gmp_ntt_v4_sub (a2r, a3r);
      a3i = gmp_ntt_v4_sub (a2i, a3i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p3, a3r, a3i);

      p0 += 8;
      p1 += 8;
      p2 += 8;
      p3 += 8;
      tw += 16;
    }
}

static void
gmp_fft_dp_r22_unpack_fwd_pq_from_digits_checked (double *data, uint32_t n,
                                                  const uint16_t *src,
                                                  size_t digit_count,
                                                  const double *tw)
{
  uint32_t l = n >> 2;
  uint32_t j;
  double *p0 = gmp_fft_dp_r22_tile_ptr (data, 0);
  double *p1 = gmp_fft_dp_r22_tile_ptr (data, l);
  double *p2 = gmp_fft_dp_r22_tile_ptr (data, 2u * l);
  double *p3 = gmp_fft_dp_r22_tile_ptr (data, 3u * l);

  ASSERT ((l & 3u) == 0);

  for (j = 0; j < l; j += 4u)
    {
      gmp_ntt_v4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
      gmp_ntt_v4 wr, wi, wr2, wi2, tr, ti;

      gmp_fft_dp_r22_load_pq_u16_pair_from_digits_checked (&a0r, &a0i, src,
                                                           digit_count, j + 0u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_digits_checked (&a1r, &a1i, src,
                                                           digit_count, j + 1u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_digits_checked (&a2r, &a2i, src,
                                                           digit_count, j + 2u * l);
      gmp_fft_dp_r22_load_pq_u16_pair_from_digits_checked (&a3r, &a3i, src,
                                                           digit_count, j + 3u * l);

      wr = gmp_ntt_v4_load (tw + 0);
      wi = gmp_ntt_v4_load (tw + 4);
      wr2 = gmp_ntt_v4_load (tw + 8);
      wi2 = gmp_ntt_v4_load (tw + 12);

      tr = gmp_ntt_v4_add (a0r, a2r);
      a2r = gmp_ntt_v4_sub (a0r, a2r);
      a0r = tr;
      ti = gmp_ntt_v4_add (a0i, a2i);
      a2i = gmp_ntt_v4_sub (a0i, a2i);
      a0i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a2r, &a2i, wr, wi);

      tr = gmp_ntt_v4_add (a1r, a3r);
      a3r = gmp_ntt_v4_sub (a1r, a3r);
      a1r = tr;
      ti = gmp_ntt_v4_add (a1i, a3i);
      a3i = gmp_ntt_v4_sub (a1i, a3i);
      a1i = ti;
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, gmp_ntt_v4_neg (wi), wr);

      gmp_fft_dp_r22_pair_store (p0, gmp_ntt_v4_add (a0r, a1r),
                                 gmp_ntt_v4_add (a0i, a1i));
      a1r = gmp_ntt_v4_sub (a0r, a1r);
      a1i = gmp_ntt_v4_sub (a0i, a1i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a1r, &a1i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p1, a1r, a1i);

      gmp_fft_dp_r22_pair_store (p2, gmp_ntt_v4_add (a2r, a3r),
                                 gmp_ntt_v4_add (a2i, a3i));
      a3r = gmp_ntt_v4_sub (a2r, a3r);
      a3i = gmp_ntt_v4_sub (a2i, a3i);
      gmp_fft_dp_r22_pair_cmul_inplace (&a3r, &a3i, wr2, wi2);
      gmp_fft_dp_r22_pair_store (p3, a3r, a3i);

      p0 += 8;
      p1 += 8;
      p2 += 8;
      p3 += 8;
      tw += 16;
    }
}

static void
gmp_fft_dp_r22_fwd_range (double *data, uint32_t start, uint32_t stop,
                          uint32_t len, const double *tw)
{
  uint32_t l = len >> 2;
  size_t tile_stride = 2u * (size_t) l;
  uint32_t base;

  ASSERT ((l & 3u) == 0);

  for (base = start; base < stop; base += len)
    {
      const double *twp = tw;
      uint32_t blocks = l >> 2;
      double *p0 = gmp_fft_dp_r22_tile_ptr (data, base);
      double *p1 = p0 + tile_stride;
      double *p2 = p1 + tile_stride;
      double *p3 = p2 + tile_stride;

      while (blocks-- != 0u)
        {
          gmp_ntt_v4 wr = gmp_ntt_v4_load (twp + 0);
          gmp_ntt_v4 wi = gmp_ntt_v4_load (twp + 4);
          gmp_ntt_v4 wr2 = gmp_ntt_v4_load (twp + 8);
          gmp_ntt_v4 wi2 = gmp_ntt_v4_load (twp + 12);

          gmp_fft_dp_r22_radix22_dif_bfly_store (p0, p1, p2, p3,
                                                 p0, p1, p2, p3,
                                                 wr, wi, wr2, wi2);
          p0 += 8;
          p1 += 8;
          p2 += 8;
          p3 += 8;
          twp += 16;
        }
    }
}

static void
gmp_fft_dp_r22_inv_range (double *data, uint32_t start, uint32_t stop,
                          uint32_t len, const double *tw)
{
  uint32_t l = len >> 2;
  size_t tile_stride = 2u * (size_t) l;
  uint32_t base;

  ASSERT ((l & 3u) == 0);

  for (base = start; base < stop; base += len)
    {
      const double *twp = tw;
      uint32_t blocks = l >> 2;
      double *p0 = gmp_fft_dp_r22_tile_ptr (data, base);
      double *p1 = p0 + tile_stride;
      double *p2 = p1 + tile_stride;
      double *p3 = p2 + tile_stride;

      while (blocks-- != 0u)
        {
          gmp_ntt_v4 wr = gmp_ntt_v4_load (twp + 0);
          gmp_ntt_v4 wi = gmp_ntt_v4_load (twp + 4);
          gmp_ntt_v4 wr2 = gmp_ntt_v4_load (twp + 8);
          gmp_ntt_v4 wi2 = gmp_ntt_v4_load (twp + 12);

          gmp_fft_dp_r22_radix22_dit_bfly_store (p0, p1, p2, p3,
                                                 p0, p1, p2, p3,
                                                 wr, wi, wr2, wi2);
          p0 += 8;
          p1 += 8;
          p2 += 8;
          p3 += 8;
          twp += 16;
        }
    }
}

static void
gmp_fft_dp_r22_fwd (double *data, const uint64_t *limbs, uint32_t n,
                    const gmp_fft_dp_r22_plan_cache *plan)
{
  uint32_t lgn = (uint32_t) gmp_ntt_ctz (n);
  uint32_t blk = 256u << (lgn & 1u);
  uint32_t tail_blk = 16u >> (lgn & 1u);
  const double *tw16 = gmp_fft_dp_r22_stage_tw_if_present (plan, 4);

  if (blk > n)
    blk = n;
  if (n < 32u)
    {
      gmp_fft_dp_r22_unpack_inputs_to_aosov (data, n, limbs);
      if (n == 2u)
        gmp_fft_dp_r22_forward_stage_radix2 (data, 2u);
      else
        gmp_fft_dp_r22_forward_tail_range (data, 0, n, tw16);
      return;
    }

  gmp_fft_dp_r22_unpack_fwd (data, n, limbs, gmp_fft_dp_r22_stage_tw (plan, lgn));
  {
    uint32_t len;
    unsigned lg_len;

    for (len = n >> 2, lg_len = lgn - 2u; len > blk; len >>= 2, lg_len -= 2u)
      gmp_fft_dp_r22_fwd_range (data, 0, n, len,
                                gmp_fft_dp_r22_stage_tw (plan, lg_len));

    for (uint32_t base = 0; base < n; base += blk)
      {
        uint32_t cur = len;
        unsigned cur_lg = lg_len;

        for (; cur > tail_blk; cur >>= 2, cur_lg -= 2u)
          gmp_fft_dp_r22_fwd_range (data, base, base + blk, cur,
                                    gmp_fft_dp_r22_stage_tw (plan, cur_lg));
        gmp_fft_dp_r22_forward_tail_range (data, base, base + blk, tw16);
      }
  }
}

static void
gmp_fft_dp_r22_inv (double *data, uint32_t n,
                    const gmp_fft_dp_r22_plan_cache *plan)
{
  uint32_t lgn = (uint32_t) gmp_ntt_ctz (n);
  uint32_t blk = 256u << (lgn & 1u);
  uint32_t tail_blk = 16u >> (lgn & 1u);
  const double *tw16 = gmp_fft_dp_r22_stage_tw_if_present (plan, 4);

  if (blk > n)
    blk = n;
  if (n < 32u)
    {
      if (n == 2u)
        gmp_fft_dp_r22_inverse_stage_radix2 (data, 2u);
      else
        gmp_fft_dp_r22_inverse_tail_range (data, 0, n, tw16);
      return;
    }

  for (uint32_t base = 0; base < n; base += blk)
    {
      gmp_fft_dp_r22_inverse_tail_range (data, base, base + blk, tw16);
      for (uint32_t cur = tail_blk << 2; cur <= blk; cur <<= 2)
        gmp_fft_dp_r22_inv_range (data, base, base + blk, cur,
                                  gmp_fft_dp_r22_stage_tw (plan, gmp_ntt_ctz (cur)));
    }

  for (uint32_t len = blk << 2; len <= n; len <<= 2)
    gmp_fft_dp_r22_inv_range (data, 0, n, len,
                              gmp_fft_dp_r22_stage_tw (plan, gmp_ntt_ctz (len)));
}

GMP_NTT_FORCEINLINE static gmp_ntt_v4
gmp_fft_dp_r22_scale_pow2 (gmp_ntt_v4 x, unsigned shift)
{
  const __m256i abs_mask = _mm256_set1_epi64x (0x7FFFFFFFFFFFFFFFULL);
  __m256i bits = _mm256_castpd_si256 (x);
  __m256i abs_bits = _mm256_and_si256 (bits, abs_mask);
  __m256i nz = _mm256_cmpgt_epi64 (abs_bits, _mm256_setzero_si256 ());
  __m256i sub = _mm256_set1_epi64x ((long long) shift << 52);
  __m256i scaled = _mm256_sub_epi64 (bits, _mm256_and_si256 (nz, sub));
  return _mm256_castsi256_pd (scaled);
}

static void
gmp_fft_dp_r22_pointwise_packed_mul_bitrev (double *data, uint32_t n)
{
  static const unsigned char partner[8] = { 0, 1, 3, 2, 7, 6, 5, 4 };
  unsigned scale_shift = (unsigned) gmp_ntt_ctz (n) + 2u;
  uint32_t i, base;

  for (i = 0; i < ((n < 8u) ? n : 8u); ++i)
    {
      uint32_t j = partner[i];
      double zr, zi, pr, pi;
      double sr, si, dr, di, tr, ti;
      double hr, hi;

      if (i > j)
        continue;

      zr = gmp_fft_dp_r22_get_re (data, i);
      zi = gmp_fft_dp_r22_get_im (data, i);
      pr = gmp_fft_dp_r22_get_re (data, j);
      pi = -gmp_fft_dp_r22_get_im (data, j);

      sr = zr + pr;
      si = zi + pi;
      dr = zr - pr;
      di = zi - pi;
      tr = sr * dr - si * di;
      ti = sr * di + si * dr;
      hr = ldexp (ti, -(int) scale_shift);
      hi = -ldexp (tr, -(int) scale_shift);

      gmp_fft_dp_r22_set_complex (data, i, hr, hi);
      if (i != j)
        gmp_fft_dp_r22_set_complex (data, j, hr, -hi);
    }

  for (base = 8u; base < n; base <<= 1)
    {
      uint32_t grp = base;

      for (i = 0; i < (grp >> 1); i += 4u)
        {
          uint32_t li = base + i;
          uint32_t ri = (base + grp) - 4u - i;
          gmp_ntt_v4 zlr, zli, zrr, zri, pr, pi;
          gmp_ntt_v4 sr, si, dr, di, tr, ti, hr, hi;

          gmp_fft_dp_r22_pair_load (&zlr, &zli, gmp_fft_dp_r22_tile_ptr (data, li));
          gmp_fft_dp_r22_pair_load (&zrr, &zri, gmp_fft_dp_r22_tile_ptr (data, ri));

          pr = gmp_ntt_v4_reverse (zrr);
          pi = gmp_ntt_v4_neg (gmp_ntt_v4_reverse (zri));

          sr = gmp_ntt_v4_add (zlr, pr);
          si = gmp_ntt_v4_add (zli, pi);
          dr = gmp_ntt_v4_sub (zlr, pr);
          di = gmp_ntt_v4_sub (zli, pi);
#if defined(__FMA__) || defined(__AVX2__)
          tr = _mm256_fmsub_pd (sr, dr, gmp_ntt_v4_mul (si, di));
          ti = _mm256_fmadd_pd (sr, di, gmp_ntt_v4_mul (si, dr));
#else
          tr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (sr, dr), gmp_ntt_v4_mul (si, di));
          ti = gmp_ntt_v4_add (gmp_ntt_v4_mul (sr, di), gmp_ntt_v4_mul (si, dr));
#endif
          hr = gmp_fft_dp_r22_scale_pow2 (ti, scale_shift);
          hi = gmp_ntt_v4_neg (gmp_fft_dp_r22_scale_pow2 (tr, scale_shift));

          gmp_fft_dp_r22_pair_store (gmp_fft_dp_r22_tile_ptr (data, li), hr, hi);
          gmp_fft_dp_r22_pair_store (gmp_fft_dp_r22_tile_ptr (data, ri),
                                     gmp_ntt_v4_reverse (hr),
                                     gmp_ntt_v4_neg (gmp_ntt_v4_reverse (hi)));
        }
    }
}

static void
gmp_fft_dp_r22_fwd_pq_fusedexp (double *data, const uint64_t *limbs, uint32_t n,
                                const gmp_fft_dp_r22_plan_cache *plan)
{
  uint32_t lgn = (uint32_t) gmp_ntt_ctz (n);
  uint32_t blk = 256u << (lgn & 1u);
  uint32_t tail_blk = 16u >> (lgn & 1u);
  const double *tw16 = gmp_fft_dp_r22_stage_tw_if_present (plan, 4);

  if (blk > n)
    blk = n;
  if (n < 32u)
    {
      gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_limbs (data, n, limbs);
      if (n == 2u)
        gmp_fft_dp_r22_forward_stage_radix2 (data, 2u);
      else
        gmp_fft_dp_r22_forward_tail_range (data, 0, n, tw16);
      return;
    }

  gmp_fft_dp_r22_unpack_fwd_pq_from_limbs (data, n, limbs,
                                           gmp_fft_dp_r22_stage_tw (plan, lgn));
  {
    uint32_t len;
    unsigned lg_len;

    for (len = n >> 2, lg_len = lgn - 2u; len > blk; len >>= 2, lg_len -= 2u)
      gmp_fft_dp_r22_fwd_range (data, 0, n, len,
                                gmp_fft_dp_r22_stage_tw (plan, lg_len));

    for (uint32_t base = 0; base < n; base += blk)
      {
        uint32_t cur = len;
        unsigned cur_lg = lg_len;

        for (; cur > tail_blk; cur >>= 2, cur_lg -= 2u)
          gmp_fft_dp_r22_fwd_range (data, base, base + blk, cur,
                                    gmp_fft_dp_r22_stage_tw (plan, cur_lg));
        gmp_fft_dp_r22_forward_tail_range (data, base, base + blk, tw16);
      }
  }
}

static void
gmp_fft_dp_r22_fwd_pq_fusednocopy (double *data, mp_srcptr limbs, size_t limb_count,
                                   uint32_t n,
                                   const gmp_fft_dp_r22_plan_cache *plan)
{
  uint32_t lgn = (uint32_t) gmp_ntt_ctz (n);
  uint32_t blk = 256u << (lgn & 1u);
  uint32_t tail_blk = 16u >> (lgn & 1u);
  const double *tw16 = gmp_fft_dp_r22_stage_tw_if_present (plan, 4);

  if (blk > n)
    blk = n;
  if (n < 32u)
    {
      gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_limbs_checked (data, n,
                                                                   limbs,
                                                                   limb_count);
      if (n == 2u)
        gmp_fft_dp_r22_forward_stage_radix2 (data, 2u);
      else
        gmp_fft_dp_r22_forward_tail_range (data, 0, n, tw16);
      return;
    }

  gmp_fft_dp_r22_unpack_fwd_pq_from_limbs_checked (data, n, limbs, limb_count,
                                                   gmp_fft_dp_r22_stage_tw (plan, lgn));
  {
    uint32_t len;
    unsigned lg_len;

    for (len = n >> 2, lg_len = lgn - 2u; len > blk; len >>= 2, lg_len -= 2u)
      gmp_fft_dp_r22_fwd_range (data, 0, n, len,
                                gmp_fft_dp_r22_stage_tw (plan, lg_len));

    for (uint32_t base = 0; base < n; base += blk)
      {
        uint32_t cur = len;
        unsigned cur_lg = lg_len;

        for (; cur > tail_blk; cur >>= 2, cur_lg -= 2u)
          gmp_fft_dp_r22_fwd_range (data, base, base + blk, cur,
                                    gmp_fft_dp_r22_stage_tw (plan, cur_lg));
        gmp_fft_dp_r22_forward_tail_range (data, base, base + blk, tw16);
      }
  }
}

static void
gmp_fft_dp_r22_fwd_pq_fusedmagic (double *data, mp_srcptr limbs, size_t limb_count,
                                  uint32_t n,
                                  const gmp_fft_dp_r22_plan_cache *plan)
{
  uint32_t lgn = (uint32_t) gmp_ntt_ctz (n);
  uint32_t blk = 256u << (lgn & 1u);
  uint32_t tail_blk = 16u >> (lgn & 1u);
  const double *tw16 = gmp_fft_dp_r22_stage_tw_if_present (plan, 4);

  if (blk > n)
    blk = n;
  if (n < 32u)
    {
      gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_limbs_checked_magic (data, n,
                                                                         limbs,
                                                                         limb_count);
      if (n == 2u)
        gmp_fft_dp_r22_forward_stage_radix2 (data, 2u);
      else
        gmp_fft_dp_r22_forward_tail_range (data, 0, n, tw16);
      return;
    }

  gmp_fft_dp_r22_unpack_fwd_pq_from_limbs_checked_magic (data, n, limbs,
                                                         limb_count,
                                                         gmp_fft_dp_r22_stage_tw (plan, lgn));
  {
    uint32_t len;
    unsigned lg_len;

    for (len = n >> 2, lg_len = lgn - 2u; len > blk; len >>= 2, lg_len -= 2u)
      gmp_fft_dp_r22_fwd_range (data, 0, n, len,
                                gmp_fft_dp_r22_stage_tw (plan, lg_len));

    for (uint32_t base = 0; base < n; base += blk)
      {
        uint32_t cur = len;
        unsigned cur_lg = lg_len;

        for (; cur > tail_blk; cur >>= 2, cur_lg -= 2u)
          gmp_fft_dp_r22_fwd_range (data, base, base + blk, cur,
                                    gmp_fft_dp_r22_stage_tw (plan, cur_lg));
        gmp_fft_dp_r22_forward_tail_range (data, base, base + blk, tw16);
      }
  }
}

static void
gmp_fft_dp_r22_fwd_pq_digits_fusednocopy (double *data, const uint16_t *digits,
                                          size_t digit_count, uint32_t n,
                                          const gmp_fft_dp_r22_plan_cache *plan)
{
  uint32_t lgn = (uint32_t) gmp_ntt_ctz (n);
  uint32_t blk = 256u << (lgn & 1u);
  uint32_t tail_blk = 16u >> (lgn & 1u);
  const double *tw16 = gmp_fft_dp_r22_stage_tw_if_present (plan, 4);

  if (blk > n)
    blk = n;
  if (n < 32u)
    {
      gmp_fft_dp_r22_unpack_inputs_to_aosov_pq_from_digits_checked (data, n,
                                                                    digits,
                                                                    digit_count);
      if (n == 2u)
        gmp_fft_dp_r22_forward_stage_radix2 (data, 2u);
      else
        gmp_fft_dp_r22_forward_tail_range (data, 0, n, tw16);
      return;
    }

  gmp_fft_dp_r22_unpack_fwd_pq_from_digits_checked (data, n, digits,
                                                    digit_count,
                                                    gmp_fft_dp_r22_stage_tw (plan, lgn));
  {
    uint32_t len;
    unsigned lg_len;

    for (len = n >> 2, lg_len = lgn - 2u; len > blk; len >>= 2, lg_len -= 2u)
      gmp_fft_dp_r22_fwd_range (data, 0, n, len,
                                gmp_fft_dp_r22_stage_tw (plan, lg_len));

    for (uint32_t base = 0; base < n; base += blk)
      {
        uint32_t cur = len;
        unsigned cur_lg = lg_len;

        for (; cur > tail_blk; cur >>= 2, cur_lg -= 2u)
          gmp_fft_dp_r22_fwd_range (data, base, base + blk, cur,
                                    gmp_fft_dp_r22_stage_tw (plan, cur_lg));
        gmp_fft_dp_r22_forward_tail_range (data, base, base + blk, tw16);
      }
  }
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pointwise_pq_eval (double *zr, double *zi,
                                  double xr, double xi,
                                  double xnr, double xni,
                                  double yr, double yi,
                                  double ynr, double yni,
                                  double wr, double wi,
                                  unsigned scale_shift)
{
  double pqr, pqi;
  double dpr, dpi, dqr, dqi;
  double tr, ti, cr, ci;

  pqr = xr * yr - xi * yi;
  pqi = xr * yi + xi * yr;

  dpr = xr - xnr;
  dpi = xi + xni;
  dqr = yr - ynr;
  dqi = yi + yni;

  tr = dpr * dqr - dpi * dqi;
  ti = dpr * dqi + dpi * dqr;

  cr = tr * wr - ti * wi;
  ci = tr * wi + ti * wr;

  *zr = ldexp (pqr - 0.25 * cr, -(int) scale_shift);
  *zi = ldexp (pqi - 0.25 * ci, -(int) scale_shift);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pointwise_pq_eval_v4 (gmp_ntt_v4 *zr, gmp_ntt_v4 *zi,
                                     gmp_ntt_v4 xr, gmp_ntt_v4 xi,
                                     gmp_ntt_v4 xnr, gmp_ntt_v4 xni,
                                     gmp_ntt_v4 yr, gmp_ntt_v4 yi,
                                     gmp_ntt_v4 ynr, gmp_ntt_v4 yni,
                                     gmp_ntt_v4 wr, gmp_ntt_v4 wi,
                                     unsigned scale_shift)
{
  gmp_ntt_v4 pqr, pqi;
  gmp_ntt_v4 dpr, dpi, dqr, dqi;
  gmp_ntt_v4 tr, ti, cr, ci;
  gmp_ntt_v4 quarter = gmp_ntt_v4_set1 (0.25);

#if defined(__FMA__) || defined(__AVX2__)
  pqr = _mm256_fmsub_pd (xr, yr, gmp_ntt_v4_mul (xi, yi));
  pqi = _mm256_fmadd_pd (xr, yi, gmp_ntt_v4_mul (xi, yr));
#else
  pqr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (xr, yr), gmp_ntt_v4_mul (xi, yi));
  pqi = gmp_ntt_v4_add (gmp_ntt_v4_mul (xr, yi), gmp_ntt_v4_mul (xi, yr));
#endif

  dpr = gmp_ntt_v4_sub (xr, xnr);
  dpi = gmp_ntt_v4_add (xi, xni);
  dqr = gmp_ntt_v4_sub (yr, ynr);
  dqi = gmp_ntt_v4_add (yi, yni);

#if defined(__FMA__) || defined(__AVX2__)
  tr = _mm256_fmsub_pd (dpr, dqr, gmp_ntt_v4_mul (dpi, dqi));
  ti = _mm256_fmadd_pd (dpr, dqi, gmp_ntt_v4_mul (dpi, dqr));
  cr = _mm256_fmsub_pd (tr, wr, gmp_ntt_v4_mul (ti, wi));
  ci = _mm256_fmadd_pd (tr, wi, gmp_ntt_v4_mul (ti, wr));
#else
  tr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (dpr, dqr), gmp_ntt_v4_mul (dpi, dqi));
  ti = gmp_ntt_v4_add (gmp_ntt_v4_mul (dpr, dqi), gmp_ntt_v4_mul (dpi, dqr));
  cr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (tr, wr), gmp_ntt_v4_mul (ti, wi));
  ci = gmp_ntt_v4_add (gmp_ntt_v4_mul (tr, wi), gmp_ntt_v4_mul (ti, wr));
#endif

  *zr = gmp_fft_dp_r22_scale_pow2 (gmp_ntt_v4_sub (pqr, gmp_ntt_v4_mul (quarter, cr)),
                                   scale_shift);
  *zi = gmp_fft_dp_r22_scale_pow2 (gmp_ntt_v4_sub (pqi, gmp_ntt_v4_mul (quarter, ci)),
                                   scale_shift);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pointwise_pq_sqr_eval (double *zr, double *zi,
                                      double xr, double xi,
                                      double xnr, double xni,
                                      double wr, double wi,
                                      unsigned scale_shift)
{
  double pqr, pqi;
  double dpr, dpi;
  double tr, ti, cr, ci;

  pqr = xr * xr - xi * xi;
  pqi = (xr + xr) * xi;

  dpr = xr - xnr;
  dpi = xi + xni;

  tr = dpr * dpr - dpi * dpi;
  ti = (dpr + dpr) * dpi;

  cr = tr * wr - ti * wi;
  ci = tr * wi + ti * wr;

  *zr = ldexp (pqr - 0.25 * cr, -(int) scale_shift);
  *zi = ldexp (pqi - 0.25 * ci, -(int) scale_shift);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pointwise_pq_sqr_eval_v4 (gmp_ntt_v4 *zr, gmp_ntt_v4 *zi,
                                         gmp_ntt_v4 xr, gmp_ntt_v4 xi,
                                         gmp_ntt_v4 xnr, gmp_ntt_v4 xni,
                                         gmp_ntt_v4 wr, gmp_ntt_v4 wi,
                                         unsigned scale_shift)
{
  gmp_ntt_v4 pqr, pqi;
  gmp_ntt_v4 dpr, dpi;
  gmp_ntt_v4 tr, ti, cr, ci;
  gmp_ntt_v4 quarter = gmp_ntt_v4_set1 (0.25);

#if defined(__FMA__) || defined(__AVX2__)
  pqr = _mm256_fmsub_pd (xr, xr, gmp_ntt_v4_mul (xi, xi));
#else
  pqr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (xr, xr), gmp_ntt_v4_mul (xi, xi));
#endif
  pqi = gmp_ntt_v4_mul (gmp_ntt_v4_add (xr, xr), xi);

  dpr = gmp_ntt_v4_sub (xr, xnr);
  dpi = gmp_ntt_v4_add (xi, xni);

#if defined(__FMA__) || defined(__AVX2__)
  tr = _mm256_fmsub_pd (dpr, dpr, gmp_ntt_v4_mul (dpi, dpi));
#else
  tr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (dpr, dpr), gmp_ntt_v4_mul (dpi, dpi));
#endif
  ti = gmp_ntt_v4_mul (gmp_ntt_v4_add (dpr, dpr), dpi);

#if defined(__FMA__) || defined(__AVX2__)
  cr = _mm256_fmsub_pd (tr, wr, gmp_ntt_v4_mul (ti, wi));
  ci = _mm256_fmadd_pd (tr, wi, gmp_ntt_v4_mul (ti, wr));
#else
  cr = gmp_ntt_v4_sub (gmp_ntt_v4_mul (tr, wr), gmp_ntt_v4_mul (ti, wi));
  ci = gmp_ntt_v4_add (gmp_ntt_v4_mul (tr, wi), gmp_ntt_v4_mul (ti, wr));
#endif

  *zr = gmp_fft_dp_r22_scale_pow2 (gmp_ntt_v4_sub (pqr, gmp_ntt_v4_mul (quarter, cr)),
                                   scale_shift);
  *zi = gmp_fft_dp_r22_scale_pow2 (gmp_ntt_v4_sub (pqi, gmp_ntt_v4_mul (quarter, ci)),
                                   scale_shift);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pq_twiddle_block_v4 (gmp_ntt_v4 *wr, gmp_ntt_v4 *wi,
                                    const gmp_fft_dp_r22_plan_cache *plan,
                                    uint32_t block4_index)
{
  const double *pq_re = plan->pq_omega_br;
  const double *pq_im = plan->pq_omega_br + ((size_t) plan->pq_omega_br_n >> 2);
  double gr = pq_re[block4_index];
  double gi = pq_im[block4_index];

  *wr = gmp_ntt_v4_set_d4 (1.0 + gr, 1.0 - gr, 1.0 - gi, 1.0 + gi);
  *wi = gmp_ntt_v4_set_d4 (gi, -gi, gr, -gr);
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pq_twiddle_scalar (double *wr, double *wi,
                                  const gmp_fft_dp_r22_plan_cache *plan,
                                  uint32_t i)
{
  uint32_t lane = i & 3u;
  const double *pq_re = plan->pq_omega_br;
  const double *pq_im = plan->pq_omega_br + ((size_t) plan->pq_omega_br_n >> 2);
  double gr = pq_re[i >> 2];
  double gi = pq_im[i >> 2];

  switch (lane)
    {
    case 0u:
      *wr = 1.0 + gr;
      *wi = gi;
      break;
    case 1u:
      *wr = 1.0 - gr;
      *wi = -gi;
      break;
    case 2u:
      *wr = 1.0 - gi;
      *wi = gr;
      break;
    default:
      *wr = 1.0 + gi;
      *wi = -gr;
      break;
    }
}

static void
gmp_fft_dp_r22_pointwise_pq_bitrev (double *data,
                                    double *other,
                                    uint32_t n,
                                    const gmp_fft_dp_r22_plan_cache *plan)
{
  static const unsigned char partner[8] = { 0, 1, 3, 2, 7, 6, 5, 4 };
  unsigned scale_shift = (unsigned) gmp_ntt_ctz (n);
  uint32_t i, base;

  for (i = 0; i < ((n < 8u) ? n : 8u); ++i)
    {
      uint32_t j = partner[i];
      double wr, wi;
      double xr, xi, xjr, xji, yr, yi, yjr, yji;
      double zr, zi;

      if (i > j)
        continue;

      xr = gmp_fft_dp_r22_get_re (data, i);
      xi = gmp_fft_dp_r22_get_im (data, i);
      xjr = gmp_fft_dp_r22_get_re (data, j);
      xji = gmp_fft_dp_r22_get_im (data, j);
      yr = gmp_fft_dp_r22_get_re (other, i);
      yi = gmp_fft_dp_r22_get_im (other, i);
      yjr = gmp_fft_dp_r22_get_re (other, j);
      yji = gmp_fft_dp_r22_get_im (other, j);

      gmp_fft_dp_r22_pq_twiddle_scalar (&wr, &wi, plan, i);
      gmp_fft_dp_r22_pointwise_pq_eval (&zr, &zi,
                                        xr, xi, xjr, xji,
                                        yr, yi, yjr, yji,
                                        wr, wi, scale_shift);
      gmp_fft_dp_r22_set_complex (data, i, zr, zi);

      if (i != j)
        {
          gmp_fft_dp_r22_pq_twiddle_scalar (&wr, &wi, plan, j);
          gmp_fft_dp_r22_pointwise_pq_eval (&zr, &zi,
                                            xjr, xji, xr, xi,
                                            yjr, yji, yr, yi,
                                            wr, wi, scale_shift);
          gmp_fft_dp_r22_set_complex (data, j, zr, zi);
        }
    }

  for (base = 8u; base < n; base <<= 1)
    {
      uint32_t grp = base;

      for (i = 0; i < (grp >> 1); i += 4u)
        {
          uint32_t li = base + i;
          uint32_t ri = (base + grp) - 4u - i;
          gmp_ntt_v4 zlr, zli, zrr, zri, xnr, xni;
          gmp_ntt_v4 ylr, yli, yrr, yri, ynr, yni;
          gmp_ntt_v4 wr, wi, wi_conj;
          gmp_ntt_v4 outlr, outli, outr, outi;

          gmp_fft_dp_r22_pair_load (&zlr, &zli, gmp_fft_dp_r22_tile_ptr (data, li));
          gmp_fft_dp_r22_pair_load (&zrr, &zri, gmp_fft_dp_r22_tile_ptr (data, ri));
          gmp_fft_dp_r22_pair_load (&ylr, &yli, gmp_fft_dp_r22_tile_ptr (other, li));
          gmp_fft_dp_r22_pair_load (&yrr, &yri, gmp_fft_dp_r22_tile_ptr (other, ri));

          xnr = gmp_ntt_v4_reverse (zrr);
          xni = gmp_ntt_v4_reverse (zri);
          ynr = gmp_ntt_v4_reverse (yrr);
          yni = gmp_ntt_v4_reverse (yri);

          gmp_fft_dp_r22_pq_twiddle_block_v4 (&wr, &wi, plan, li >> 2);
          wi_conj = gmp_ntt_v4_neg (wi);

          gmp_fft_dp_r22_pointwise_pq_eval_v4 (&outlr, &outli,
                                               zlr, zli, xnr, xni,
                                               ylr, yli, ynr, yni,
                                               wr, wi, scale_shift);
          gmp_fft_dp_r22_pointwise_pq_eval_v4 (&outr, &outi,
                                               xnr, xni, zlr, zli,
                                               ynr, yni, ylr, yli,
                                               wr, wi_conj, scale_shift);

          gmp_fft_dp_r22_pair_store (gmp_fft_dp_r22_tile_ptr (data, li), outlr, outli);
          gmp_fft_dp_r22_pair_store (gmp_fft_dp_r22_tile_ptr (data, ri),
                                     gmp_ntt_v4_reverse (outr),
                                     gmp_ntt_v4_reverse (outi));
        }
    }
}

static void
gmp_fft_dp_r22_pointwise_pq_sqr_bitrev (double *data,
                                        uint32_t n,
                                        const gmp_fft_dp_r22_plan_cache *plan)
{
  static const unsigned char partner[8] = { 0, 1, 3, 2, 7, 6, 5, 4 };
  unsigned scale_shift = (unsigned) gmp_ntt_ctz (n);
  uint32_t i, base;

  for (i = 0; i < ((n < 8u) ? n : 8u); ++i)
    {
      uint32_t j = partner[i];
      double wr, wi;
      double xr, xi, xjr, xji;
      double zr, zi;

      if (i > j)
        continue;

      xr = gmp_fft_dp_r22_get_re (data, i);
      xi = gmp_fft_dp_r22_get_im (data, i);
      xjr = gmp_fft_dp_r22_get_re (data, j);
      xji = gmp_fft_dp_r22_get_im (data, j);

      gmp_fft_dp_r22_pq_twiddle_scalar (&wr, &wi, plan, i);
      gmp_fft_dp_r22_pointwise_pq_sqr_eval (&zr, &zi,
                                            xr, xi, xjr, xji,
                                            wr, wi, scale_shift);
      gmp_fft_dp_r22_set_complex (data, i, zr, zi);

      if (i != j)
        {
          gmp_fft_dp_r22_pq_twiddle_scalar (&wr, &wi, plan, j);
          gmp_fft_dp_r22_pointwise_pq_sqr_eval (&zr, &zi,
                                                xjr, xji, xr, xi,
                                                wr, wi, scale_shift);
          gmp_fft_dp_r22_set_complex (data, j, zr, zi);
        }
    }

  for (base = 8u; base < n; base <<= 1)
    {
      uint32_t grp = base;

      for (i = 0; i < (grp >> 1); i += 4u)
        {
          uint32_t li = base + i;
          uint32_t ri = (base + grp) - 4u - i;
          gmp_ntt_v4 zlr, zli, zrr, zri, xnr, xni;
          gmp_ntt_v4 wr, wi, wi_conj;
          gmp_ntt_v4 outlr, outli, outr, outi;

          gmp_fft_dp_r22_pair_load (&zlr, &zli, gmp_fft_dp_r22_tile_ptr (data, li));
          gmp_fft_dp_r22_pair_load (&zrr, &zri, gmp_fft_dp_r22_tile_ptr (data, ri));

          xnr = gmp_ntt_v4_reverse (zrr);
          xni = gmp_ntt_v4_reverse (zri);

          gmp_fft_dp_r22_pq_twiddle_block_v4 (&wr, &wi, plan, li >> 2);
          wi_conj = gmp_ntt_v4_neg (wi);

          gmp_fft_dp_r22_pointwise_pq_sqr_eval_v4 (&outlr, &outli,
                                                   zlr, zli, xnr, xni,
                                                   wr, wi, scale_shift);
          gmp_fft_dp_r22_pointwise_pq_sqr_eval_v4 (&outr, &outi,
                                                   xnr, xni, zlr, zli,
                                                   wr, wi_conj, scale_shift);

          gmp_fft_dp_r22_pair_store (gmp_fft_dp_r22_tile_ptr (data, li), outlr, outli);
          gmp_fft_dp_r22_pair_store (gmp_fft_dp_r22_tile_ptr (data, ri),
                                     gmp_ntt_v4_reverse (outr),
                                     gmp_ntt_v4_reverse (outi));
        }
    }
}

static int
gmp_fft_dp_r22_recover_limbs_from_fft (mp_ptr rp, const double *data,
                                       mp_size_t an, mp_size_t bn)
{
  const gmp_ntt_v4 zero = gmp_ntt_v4_zero ();
  const __m256i bias_bits = _mm256_set1_epi64x (0x4330000000000000ULL);
  const gmp_ntt_v4 bias52 = _mm256_castsi256_pd (bias_bits);
  size_t result_limbs = (size_t) (an + bn);
  size_t i;
  mp_limb_t carry = 0;

  for (i = 0; i < result_limbs; ++i)
    {
      gmp_ntt_v4 re = gmp_ntt_v4_load (data + 8u * i);
      gmp_ntt_v4 clipped;
      gmp_ntt_v4 biased;
      mp_limb_t limb;

      if (i + 1 == result_limbs)
        re = _mm256_blend_pd (re, zero, 0x8);

      clipped = _mm256_max_pd (re, zero);
      biased = gmp_ntt_v4_add (clipped, bias52);
      {
        __m256i ints = _mm256_sub_epi64 (_mm256_castpd_si256 (biased), bias_bits);
        __m128i lo = _mm256_castsi256_si128 (ints);
        __m128i hi = _mm256_extracti128_si256 (ints, 1);
        __m128i tlo = _mm_add_epi64 (lo,
                                     _mm_slli_epi64 (_mm_unpackhi_epi64 (lo, lo), 16));
        __m128i thi = _mm_add_epi64 (hi,
                                     _mm_slli_epi64 (_mm_unpackhi_epi64 (hi, hi), 16));
        __m128i packed = _mm_unpacklo_epi64 (tlo, thi);
        mp_limb_t packed0 = (mp_limb_t) _mm_cvtsi128_si64 (packed);
        mp_limb_t packed1 = (mp_limb_t) _mm_extract_epi64 (packed, 1);
        mp_limb_t carry0, carry1;

        ADDC_LIMB (carry0, limb, packed0, carry);
        ADDC_LIMB (carry1, limb, limb, packed1 << 32);
        carry = (packed1 >> 32) + carry0 + carry1;
        rp[i] = limb;
      }
    }

  return carry == 0;
}

GMP_NTT_FORCEINLINE static void
gmp_fft_dp_r22_pq_u64_to_partials (unsigned __int128 part[2], __m256i u_i)
{
  const __m256i even_mask = _mm256_setr_epi64x (-1, 0, -1, 0);
  const __m256i one_mask = _mm256_setr_epi64x (1, 0, 1, 0);
  const __m256i sign_mask = _mm256_set1_epi64x (0x8000000000000000ULL);
  uint64_t lo_tmp[4];
  uint64_t hi_tmp[4];
  __m256i even_u;
  __m256i odd_dup;
  __m256i odd_lo;
  __m256i odd_hi;
  __m256i sum_lo;
  __m256i carry_cmp;
  __m256i carry64;
  __m256i sum_hi;

  even_u = _mm256_and_si256 (u_i, even_mask);
  odd_dup = _mm256_permute4x64_epi64 (u_i, 0xF5);
  odd_lo = _mm256_and_si256 (_mm256_slli_epi64 (odd_dup, 32), even_mask);
  odd_hi = _mm256_and_si256 (_mm256_srli_epi64 (odd_dup, 32), even_mask);
  sum_lo = _mm256_add_epi64 (even_u, odd_lo);
  carry_cmp = _mm256_cmpgt_epi64 (_mm256_xor_si256 (even_u, sign_mask),
                                  _mm256_xor_si256 (sum_lo, sign_mask));
  carry64 = _mm256_and_si256 (carry_cmp, one_mask);
  sum_hi = _mm256_add_epi64 (odd_hi, carry64);

  _mm256_storeu_si256 ((__m256i_u *) lo_tmp, sum_lo);
  _mm256_storeu_si256 ((__m256i_u *) hi_tmp, sum_hi);
  part[0] = (unsigned __int128) lo_tmp[0]
            + ((unsigned __int128) hi_tmp[0] << 64);
  part[1] = (unsigned __int128) lo_tmp[2]
            + ((unsigned __int128) hi_tmp[2] << 64);
}

static int
gmp_fft_dp_r22_recover_limbs_from_pq_fft (mp_ptr rp, const double *data,
                                          mp_size_t an, mp_size_t bn)
{
  const gmp_ntt_v4 zero = gmp_ntt_v4_zero ();
  const __m256i bias_bits = _mm256_set1_epi64x (0x4330000000000000ULL);
  const gmp_ntt_v4 bias52 = _mm256_castsi256_pd (bias_bits);
  size_t result_limbs = (size_t) (an + bn);
  size_t total_digits = 4u * (size_t) (an + bn);
  size_t d;
  unsigned __int128 carry = 0;

  memset (rp, 0, result_limbs * sizeof (mp_limb_t));

  for (d = 0; d < total_digits; d += 8u)
    {
      uint64_t u64[4];
      unsigned __int128 part[2];
      size_t rem = total_digits - d;
      size_t limb_idx;
      const double *tile = gmp_fft_dp_r22_tile_ptr_const (data, (uint32_t) (d >> 1));
      gmp_ntt_v4 re = gmp_ntt_v4_load (tile + 0);
      gmp_ntt_v4 im = gmp_ntt_v4_load (tile + 4);
      __m256i re_i, im_i, u_i;

      if (rem > 8u)
        rem = 8u;

      re = _mm256_max_pd (re, zero);
      im = _mm256_max_pd (im, zero);
      re_i = _mm256_sub_epi64 (_mm256_castpd_si256 (gmp_ntt_v4_add (re, bias52)),
                               bias_bits);
      im_i = _mm256_sub_epi64 (_mm256_castpd_si256 (gmp_ntt_v4_add (im, bias52)),
                               bias_bits);
      u_i = _mm256_add_epi64 (re_i, _mm256_slli_epi64 (im_i, 16));

      limb_idx = d >> 2;
      if (rem >= 8u)
        {
          gmp_fft_dp_r22_pq_u64_to_partials (part, u_i);

          carry += part[0];
          rp[limb_idx] = (mp_limb_t) carry;
          carry >>= 64;
          ++limb_idx;

          carry += part[1];
          rp[limb_idx] = (mp_limb_t) carry;
          carry >>= 64;
        }
      else
        {
          _mm256_storeu_si256 ((__m256i_u *) u64, u_i);

          if (rem >= 4u)
            {
              unsigned __int128 part0;

              part0 = (unsigned __int128) u64[0];
              part0 += (unsigned __int128) u64[1] << 32;
              carry += part0;
              rp[limb_idx] = (mp_limb_t) carry;
              carry >>= 64;
            }
        }
    }

  return carry == (unsigned __int128) 0;
}

int
gmp_fft_dp_r22_clean_supported (mp_size_t an, mp_size_t bn)
{
  uint32_t na, nb, n;

  if (an <= 0 || bn <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  nb = 4u * (uint32_t) bn;
  n = gmp_fft_dp_r22_ceil_pow2_u32 (na + nb);
  return n >= 2u && n <= 65536u;
}

int
gmp_fft_dp_r22_clean_mul (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                          mp_srcptr bp, mp_size_t bn)
{
  uint32_t na, nb, n;

  if (!gmp_fft_dp_r22_clean_supported (an, bn))
    return 0;

  na = 4u * (uint32_t) an;
  nb = 4u * (uint32_t) bn;
  n = gmp_fft_dp_r22_ceil_pow2_u32 (na + nb);

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);
  ASSERT (gmp_fft_dp_r22_is_aligned32 (gmp_fft_dp_r22_ws.data));
  ASSERT (gmp_fft_dp_r22_is_aligned32 (gmp_fft_dp_r22_ws.ri_u64));

  gmp_fft_dp_r22_build_interleaved_ri_u64 (gmp_fft_dp_r22_ws.ri_u64, n >> 2,
                                           ap, (size_t) an, bp, (size_t) bn);
  gmp_fft_dp_r22_fwd (gmp_fft_dp_r22_ws.data, gmp_fft_dp_r22_ws.ri_u64, n,
                      &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_pointwise_packed_mul_bitrev (gmp_fft_dp_r22_ws.data, n);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);
  return gmp_fft_dp_r22_recover_limbs_from_fft (rp, gmp_fft_dp_r22_ws.data,
                                                an, bn);
}

int
gmp_fft_dp_r22_clean_mul_balanced (mp_ptr rp, mp_srcptr ap,
                                   mp_srcptr bp, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_mul (rp, ap, n, bp, n);
}

int
gmp_fft_dp_r22_clean_mul_pq (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                             mp_srcptr bp, mp_size_t bn)
{
  uint32_t na, nb, n;

  if (an <= 0 || bn <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  nb = 4u * (uint32_t) bn;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + nb + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_fwd_pq_fusednocopy (gmp_fft_dp_r22_ws.data, ap, (size_t) an, n,
                                     &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_fwd_pq_fusednocopy (gmp_fft_dp_r22_ws.data2, bp, (size_t) bn, n,
                                     &gmp_fft_dp_r22_plan);

  gmp_fft_dp_r22_pointwise_pq_bitrev (gmp_fft_dp_r22_ws.data,
                                      gmp_fft_dp_r22_ws.data2,
                                      n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, bn);
}

int
gmp_fft_dp_r22_clean_mul_pq_balanced (mp_ptr rp, mp_srcptr ap,
                                      mp_srcptr bp, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_mul_pq (rp, ap, n, bp, n);
}

int
gmp_fft_dp_r22_clean_mul_pq_fusedexp (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                                      mp_srcptr bp, mp_size_t bn)
{
  uint32_t na, nb, n;

  if (an <= 0 || bn <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  nb = 4u * (uint32_t) bn;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + nb + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_build_pq_padded_limbs (gmp_fft_dp_r22_ws.ri_u64, n,
                                        ap, (size_t) an);
  gmp_fft_dp_r22_fwd_pq_fusedexp (gmp_fft_dp_r22_ws.data,
                                  gmp_fft_dp_r22_ws.ri_u64, n,
                                  &gmp_fft_dp_r22_plan);

  gmp_fft_dp_r22_build_pq_padded_limbs (gmp_fft_dp_r22_ws.ri_u64, n,
                                        bp, (size_t) bn);
  gmp_fft_dp_r22_fwd_pq_fusedexp (gmp_fft_dp_r22_ws.data2,
                                  gmp_fft_dp_r22_ws.ri_u64, n,
                                  &gmp_fft_dp_r22_plan);

  gmp_fft_dp_r22_pointwise_pq_bitrev (gmp_fft_dp_r22_ws.data,
                                      gmp_fft_dp_r22_ws.data2,
                                      n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, bn);
}

int
gmp_fft_dp_r22_clean_mul_pq_fusedexp_balanced (mp_ptr rp, mp_srcptr ap,
                                               mp_srcptr bp, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_mul_pq_fusedexp (rp, ap, n, bp, n);
}

int
gmp_fft_dp_r22_clean_mul_pq_fusednocopy (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                                         mp_srcptr bp, mp_size_t bn)
{
  uint32_t na, nb, n;

  if (an <= 0 || bn <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  nb = 4u * (uint32_t) bn;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + nb + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_fwd_pq_fusednocopy (gmp_fft_dp_r22_ws.data, ap, (size_t) an, n,
                                     &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_fwd_pq_fusednocopy (gmp_fft_dp_r22_ws.data2, bp, (size_t) bn, n,
                                     &gmp_fft_dp_r22_plan);

  gmp_fft_dp_r22_pointwise_pq_bitrev (gmp_fft_dp_r22_ws.data,
                                      gmp_fft_dp_r22_ws.data2,
                                      n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, bn);
}

int
gmp_fft_dp_r22_clean_mul_pq_fusednocopy_balanced (mp_ptr rp, mp_srcptr ap,
                                                  mp_srcptr bp, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_mul_pq_fusednocopy (rp, ap, n, bp, n);
}

int
gmp_fft_dp_r22_clean_mul_pq_fusedmagic (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                                        mp_srcptr bp, mp_size_t bn)
{
  uint32_t na, nb, n;

  if (an <= 0 || bn <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  nb = 4u * (uint32_t) bn;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + nb + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_fwd_pq_fusedmagic (gmp_fft_dp_r22_ws.data, ap, (size_t) an, n,
                                    &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_fwd_pq_fusedmagic (gmp_fft_dp_r22_ws.data2, bp, (size_t) bn, n,
                                    &gmp_fft_dp_r22_plan);

  gmp_fft_dp_r22_pointwise_pq_bitrev (gmp_fft_dp_r22_ws.data,
                                      gmp_fft_dp_r22_ws.data2,
                                      n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, bn);
}

int
gmp_fft_dp_r22_clean_mul_pq_fusedmagic_balanced (mp_ptr rp, mp_srcptr ap,
                                                 mp_srcptr bp, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_mul_pq_fusedmagic (rp, ap, n, bp, n);
}

int
gmp_fft_dp_r22_clean_sqr (mp_ptr rp, mp_srcptr ap, mp_size_t an)
{
  return gmp_fft_dp_r22_clean_mul (rp, ap, an, ap, an);
}

int
gmp_fft_dp_r22_clean_sqr_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr (rp, ap, n);
}

int
gmp_fft_dp_r22_clean_sqr_pq (mp_ptr rp, mp_srcptr ap, mp_size_t an)
{
  uint32_t na, n;

  if (an <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + na + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_fwd_pq_fusednocopy (gmp_fft_dp_r22_ws.data, ap, (size_t) an, n,
                                     &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_pointwise_pq_sqr_bitrev (gmp_fft_dp_r22_ws.data,
                                          n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, an);
}

int
gmp_fft_dp_r22_clean_sqr_pq_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr_pq (rp, ap, n);
}

int
gmp_fft_dp_r22_clean_sqr_pq_fusedexp (mp_ptr rp, mp_srcptr ap, mp_size_t an)
{
  uint32_t na, n;

  if (an <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + na + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_build_pq_padded_limbs (gmp_fft_dp_r22_ws.ri_u64, n,
                                        ap, (size_t) an);
  gmp_fft_dp_r22_fwd_pq_fusedexp (gmp_fft_dp_r22_ws.data,
                                  gmp_fft_dp_r22_ws.ri_u64, n,
                                  &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_pointwise_pq_sqr_bitrev (gmp_fft_dp_r22_ws.data,
                                          n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, an);
}

int
gmp_fft_dp_r22_clean_sqr_pq_fusedexp_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr_pq_fusedexp (rp, ap, n);
}

int
gmp_fft_dp_r22_clean_sqr_pq_fusednocopy (mp_ptr rp, mp_srcptr ap, mp_size_t an)
{
  uint32_t na, n;

  if (an <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + na + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_fwd_pq_fusednocopy (gmp_fft_dp_r22_ws.data, ap, (size_t) an, n,
                                     &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_pointwise_pq_sqr_bitrev (gmp_fft_dp_r22_ws.data,
                                          n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, an);
}

int
gmp_fft_dp_r22_clean_sqr_pq_fusednocopy_balanced (mp_ptr rp, mp_srcptr ap,
                                                  mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr_pq_fusednocopy (rp, ap, n);
}

int
gmp_fft_dp_r22_clean_sqr_pq_fusedmagic (mp_ptr rp, mp_srcptr ap, mp_size_t an)
{
  uint32_t na, n;

  if (an <= 0)
    return 0;

  na = 4u * (uint32_t) an;
  n = gmp_fft_dp_r22_ceil_pow2_u32 ((na + na + 1u) >> 1);
  if (n > 65536u)
    return 0;

  gmp_fft_dp_r22_plan_ensure (&gmp_fft_dp_r22_plan, n);
  gmp_fft_dp_r22_workspace_ensure (&gmp_fft_dp_r22_ws, n);

  gmp_fft_dp_r22_fwd_pq_fusedmagic (gmp_fft_dp_r22_ws.data, ap, (size_t) an, n,
                                    &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_pointwise_pq_sqr_bitrev (gmp_fft_dp_r22_ws.data,
                                          n, &gmp_fft_dp_r22_plan);
  gmp_fft_dp_r22_inv (gmp_fft_dp_r22_ws.data, n, &gmp_fft_dp_r22_plan);

  return gmp_fft_dp_r22_recover_limbs_from_pq_fft (rp, gmp_fft_dp_r22_ws.data,
                                                   an, an);
}

int
gmp_fft_dp_r22_clean_sqr_pq_fusedmagic_balanced (mp_ptr rp, mp_srcptr ap,
                                                 mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr_pq_fusedmagic (rp, ap, n);
}

int
mpn_fft_dp_r22_clean_mul_balanced (mp_ptr rp, mp_srcptr ap,
                                   mp_srcptr bp, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_mul_balanced (rp, ap, bp, n);
}

int
mpn_fft_dp_r22_clean_sqr_balanced (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr_pq_balanced (rp, ap, n);
}
