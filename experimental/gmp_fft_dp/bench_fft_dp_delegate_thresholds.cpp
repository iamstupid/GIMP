#include "config.h"

#include "gmp-impl.h"
#include "experimental/gmp_fft_dp/gmp_fft_dp_r22_clean.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

struct bench_config
{
  mp_size_t min_limbs;
  mp_size_t max_limbs;
  double factor;
  double seconds;
  unsigned warmups;
  unsigned repeats;
  unsigned long long seed;
  const char *csv_path;
};

enum bench_mode
{
  MODE_BALANCED_MUL = 0,
  MODE_SQR,
  MODE_IMBAL_2,
  MODE_IMBAL_4,
  MODE_IMBAL_8,
  MODE_IMBAL_16,
  MODE_COUNT
};

static const char *const mode_names[MODE_COUNT] =
{
  "balanced_mul",
  "sqr",
  "imbal_2",
  "imbal_4",
  "imbal_8",
  "imbal_16"
};

#if defined(_WIN32)
static double
bench_now (void)
{
  static LARGE_INTEGER freq;
  LARGE_INTEGER counter;

  if (freq.QuadPart == 0)
    QueryPerformanceFrequency (&freq);
  QueryPerformanceCounter (&counter);
  return (double) counter.QuadPart / (double) freq.QuadPart;
}
#else
static double
bench_now (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec + (double) ts.tv_nsec * 1.0e-9;
}
#endif

static unsigned long long
bench_rng_next (unsigned long long *state)
{
  unsigned long long x;

  x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 2685821657736338717ULL;
}

static void
bench_fill_operands (mp_ptr p, mp_size_t n, unsigned long long *state)
{
  mp_size_t i;

  for (i = 0; i < n; ++i)
    p[i] = (mp_limb_t) (bench_rng_next (state) & GMP_NUMB_MASK);

  if (n > 0)
    p[n - 1] |= GMP_NUMB_HIGHBIT;
}

static void
bench_parse_u64 (const char *arg, unsigned long long *dst, const char *name)
{
  char *end;
  unsigned long long value;

  value = strtoull (arg, &end, 0);
  if (arg[0] == '\0' || *end != '\0')
    {
      std::fprintf (stderr, "invalid %s: %s\n", name, arg);
      std::exit (1);
    }
  *dst = value;
}

static void
bench_parse_size (const char *arg, mp_size_t *dst, const char *name)
{
  unsigned long long value;
  bench_parse_u64 (arg, &value, name);
  *dst = (mp_size_t) value;
}

static void
bench_parse_double (const char *arg, double *dst, const char *name)
{
  char *end;
  double value;

  value = std::strtod (arg, &end);
  if (arg[0] == '\0' || *end != '\0')
    {
      std::fprintf (stderr, "invalid %s: %s\n", name, arg);
      std::exit (1);
    }
  *dst = value;
}

static void
bench_usage (const char *argv0)
{
  std::fprintf (stderr,
                "Usage: %s [--min N] [--max N] [--factor F] [--seconds S] [--warmups N] [--repeats N] [--seed N] [--csv PATH]\n",
                argv0);
}

static void
bench_parse_args (bench_config *cfg, int argc, char **argv)
{
  int i;

  for (i = 1; i < argc; ++i)
    {
      if (std::strcmp (argv[i], "--min") == 0)
        {
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          bench_parse_size (argv[i], &cfg->min_limbs, "--min");
        }
      else if (std::strcmp (argv[i], "--max") == 0)
        {
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          bench_parse_size (argv[i], &cfg->max_limbs, "--max");
        }
      else if (std::strcmp (argv[i], "--factor") == 0)
        {
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          bench_parse_double (argv[i], &cfg->factor, "--factor");
        }
      else if (std::strcmp (argv[i], "--seconds") == 0)
        {
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          bench_parse_double (argv[i], &cfg->seconds, "--seconds");
        }
      else if (std::strcmp (argv[i], "--warmups") == 0)
        {
          unsigned long long value;
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          bench_parse_u64 (argv[i], &value, "--warmups");
          cfg->warmups = (unsigned) value;
        }
      else if (std::strcmp (argv[i], "--repeats") == 0)
        {
          unsigned long long value;
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          bench_parse_u64 (argv[i], &value, "--repeats");
          cfg->repeats = (unsigned) value;
        }
      else if (std::strcmp (argv[i], "--seed") == 0)
        {
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          bench_parse_u64 (argv[i], &cfg->seed, "--seed");
        }
      else if (std::strcmp (argv[i], "--csv") == 0)
        {
          if (++i >= argc) { bench_usage (argv[0]); std::exit (1); }
          cfg->csv_path = argv[i];
        }
      else
        {
          bench_usage (argv[0]);
          std::exit (1);
        }
    }
}

static mp_size_t
bench_next_size (mp_size_t cur, double factor)
{
  double nextf = (double) cur * factor;
  mp_size_t next = (mp_size_t) (nextf + 0.5);
  if (next <= cur)
    next = cur + 1;
  return next;
}

static int
bench_direct_mul_ok (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                     mp_srcptr bp, mp_size_t bn)
{
  return gmp_fft_dp_r22_clean_mul_pq_fusednocopy (rp, ap, an, bp, bn);
}

static int
bench_direct_sqr_ok (mp_ptr rp, mp_srcptr ap, mp_size_t n)
{
  return gmp_fft_dp_r22_clean_sqr_pq_fusednocopy_balanced (rp, ap, n);
}

static double
bench_time_mul_dispatch (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                         mp_srcptr bp, mp_size_t bn,
                         double seconds, unsigned warmups)
{
  double t0, t1;
  unsigned long long reps = 0;

  while (warmups-- > 0)
    mpn_mul (rp, ap, an, bp, bn);

  t0 = bench_now ();
  do
    {
      mpn_mul (rp, ap, an, bp, bn);
      ++reps;
      t1 = bench_now ();
    }
  while (t1 - t0 < seconds);

  return (t1 - t0) / (double) reps;
}

static double
bench_time_mul_direct (mp_ptr rp, mp_srcptr ap, mp_size_t an,
                       mp_srcptr bp, mp_size_t bn,
                       double seconds, unsigned warmups)
{
  double t0, t1;
  unsigned long long reps = 0;

  while (warmups-- > 0)
    bench_direct_mul_ok (rp, ap, an, bp, bn);

  t0 = bench_now ();
  do
    {
      bench_direct_mul_ok (rp, ap, an, bp, bn);
      ++reps;
      t1 = bench_now ();
    }
  while (t1 - t0 < seconds);

  return (t1 - t0) / (double) reps;
}

static double
bench_time_sqr_dispatch (mp_ptr rp, mp_srcptr ap, mp_size_t n,
                         double seconds, unsigned warmups)
{
  double t0, t1;
  unsigned long long reps = 0;

  while (warmups-- > 0)
    mpn_sqr (rp, ap, n);

  t0 = bench_now ();
  do
    {
      mpn_sqr (rp, ap, n);
      ++reps;
      t1 = bench_now ();
    }
  while (t1 - t0 < seconds);

  return (t1 - t0) / (double) reps;
}

static double
bench_time_sqr_direct (mp_ptr rp, mp_srcptr ap, mp_size_t n,
                       double seconds, unsigned warmups)
{
  double t0, t1;
  unsigned long long reps = 0;

  while (warmups-- > 0)
    bench_direct_sqr_ok (rp, ap, n);

  t0 = bench_now ();
  do
    {
      bench_direct_sqr_ok (rp, ap, n);
      ++reps;
      t1 = bench_now ();
    }
  while (t1 - t0 < seconds);

  return (t1 - t0) / (double) reps;
}

static void
bench_write_csv_header (FILE *csv)
{
  if (csv == NULL)
    return;
  std::fprintf (csv, "mode,short_limbs,long_limbs,total_limbs,dispatch_ns,direct_ns,speedup\n");
}

static void
bench_write_csv_row (FILE *csv, const char *mode, mp_size_t an, mp_size_t bn,
                     double dispatch_s, double direct_s)
{
  if (csv == NULL)
    return;
  std::fprintf (csv, "%s,%ld,%ld,%ld,%.6f,%.6f,%.6f\n",
                mode,
                (long) std::min (an, bn),
                (long) std::max (an, bn),
                (long) (an + bn),
                dispatch_s * 1.0e9,
                direct_s * 1.0e9,
                dispatch_s / direct_s);
}

int
main (int argc, char **argv)
{
  bench_config cfg;
  unsigned long long state;
  FILE *csv = NULL;
  mp_size_t n;
  mp_size_t ratio_values[4] = { 2, 4, 8, 16 };
  mp_size_t first_win_bal = 0;
  mp_size_t first_win_sqr = 0;
  mp_size_t first_win_imbal[4] = { 0, 0, 0, 0 };

  cfg.min_limbs = 32;
  cfg.max_limbs = 16384;
  cfg.factor = 1.1892071150027211; /* 2^(1/4) */
  cfg.seconds = 0.02;
  cfg.warmups = 1;
  cfg.repeats = 5;
  cfg.seed = 0x1234ULL;
  cfg.csv_path = NULL;

  bench_parse_args (&cfg, argc, argv);
  state = cfg.seed;

  if (cfg.csv_path != NULL)
    {
      csv = std::fopen (cfg.csv_path, "w");
      if (csv == NULL)
        {
          std::perror ("fopen");
          return 1;
        }
      bench_write_csv_header (csv);
    }

  std::printf ("Balanced mul\n");
  for (n = cfg.min_limbs; n <= cfg.max_limbs; n = bench_next_size (n, cfg.factor))
    {
      std::vector<mp_limb_t> a ((size_t) n), b ((size_t) n), ref ((size_t) (2 * n)), got ((size_t) (2 * n));
      double dispatch_best = -1.0;
      double direct_best = -1.0;
      unsigned r;

      bench_fill_operands (a.data (), n, &state);
      bench_fill_operands (b.data (), n, &state);

      mpn_mul_n (ref.data (), a.data (), b.data (), n);
      if (!bench_direct_mul_ok (got.data (), a.data (), n, b.data (), n)
          || mpn_cmp (ref.data (), got.data (), 2 * n) != 0)
        {
          std::fprintf (stderr, "correctness failure balanced mul at %ld limbs\n", (long) n);
          return 2;
        }

      for (r = 0; r < cfg.repeats; ++r)
        {
          double ds = bench_time_mul_dispatch (ref.data (), a.data (), n, b.data (), n,
                                               cfg.seconds, cfg.warmups);
          double fs = bench_time_mul_direct (got.data (), a.data (), n, b.data (), n,
                                             cfg.seconds, cfg.warmups);
          if (dispatch_best < 0.0 || ds < dispatch_best) dispatch_best = ds;
          if (direct_best < 0.0 || fs < direct_best) direct_best = fs;
        }

      if (first_win_bal == 0 && direct_best <= dispatch_best)
        first_win_bal = n;

      std::printf ("  %5ld limbs  dispatch=%8.3f ns  pq=%8.3f ns  speedup=%6.3fx\n",
                   (long) n, dispatch_best * 1e9, direct_best * 1e9, dispatch_best / direct_best);
      bench_write_csv_row (csv, mode_names[MODE_BALANCED_MUL], n, n, dispatch_best, direct_best);
      if (n == cfg.max_limbs)
        break;
    }

  std::printf ("\nSquare\n");
  for (n = cfg.min_limbs; n <= cfg.max_limbs; n = bench_next_size (n, cfg.factor))
    {
      std::vector<mp_limb_t> a ((size_t) n), ref ((size_t) (2 * n)), got ((size_t) (2 * n));
      double dispatch_best = -1.0;
      double direct_best = -1.0;
      unsigned r;

      bench_fill_operands (a.data (), n, &state);

      mpn_sqr (ref.data (), a.data (), n);
      if (!bench_direct_sqr_ok (got.data (), a.data (), n)
          || mpn_cmp (ref.data (), got.data (), 2 * n) != 0)
        {
          std::fprintf (stderr, "correctness failure sqr at %ld limbs\n", (long) n);
          return 3;
        }

      for (r = 0; r < cfg.repeats; ++r)
        {
          double ds = bench_time_sqr_dispatch (ref.data (), a.data (), n,
                                               cfg.seconds, cfg.warmups);
          double fs = bench_time_sqr_direct (got.data (), a.data (), n,
                                             cfg.seconds, cfg.warmups);
          if (dispatch_best < 0.0 || ds < dispatch_best) dispatch_best = ds;
          if (direct_best < 0.0 || fs < direct_best) direct_best = fs;
        }

      if (first_win_sqr == 0 && direct_best <= dispatch_best)
        first_win_sqr = n;

      std::printf ("  %5ld limbs  dispatch=%8.3f ns  pq=%8.3f ns  speedup=%6.3fx\n",
                   (long) n, dispatch_best * 1e9, direct_best * 1e9, dispatch_best / direct_best);
      bench_write_csv_row (csv, mode_names[MODE_SQR], n, n, dispatch_best, direct_best);
      if (n == cfg.max_limbs)
        break;
    }

  for (unsigned ridx = 0; ridx < 4; ++ridx)
    {
      mp_size_t ratio = ratio_values[ridx];
      std::printf ("\nImbalance %ld:1\n", (long) ratio);
      for (n = cfg.min_limbs; n <= cfg.max_limbs; n = bench_next_size (n, cfg.factor))
        {
          mp_size_t sn = n;
          mp_size_t ln = n * ratio;
          std::vector<mp_limb_t> a, b, ref, got;
          double dispatch_best = -1.0;
          double direct_best = -1.0;
          unsigned r;

          if (sn + ln > 32768)
            break;

          a.resize ((size_t) ln);
          b.resize ((size_t) sn);
          ref.resize ((size_t) (ln + sn));
          got.resize ((size_t) (ln + sn));

          bench_fill_operands (a.data (), ln, &state);
          bench_fill_operands (b.data (), sn, &state);

          mpn_mul (ref.data (), a.data (), ln, b.data (), sn);
          if (!bench_direct_mul_ok (got.data (), a.data (), ln, b.data (), sn)
              || mpn_cmp (ref.data (), got.data (), ln + sn) != 0)
            {
              std::fprintf (stderr, "correctness failure imbalance %ld:%ld\n",
                            (long) ln, (long) sn);
              return 4;
            }

          for (r = 0; r < cfg.repeats; ++r)
            {
              double ds = bench_time_mul_dispatch (ref.data (), a.data (), ln, b.data (), sn,
                                                   cfg.seconds, cfg.warmups);
              double fs = bench_time_mul_direct (got.data (), a.data (), ln, b.data (), sn,
                                                 cfg.seconds, cfg.warmups);
              if (dispatch_best < 0.0 || ds < dispatch_best) dispatch_best = ds;
              if (direct_best < 0.0 || fs < direct_best) direct_best = fs;
            }

          if (first_win_imbal[ridx] == 0 && direct_best <= dispatch_best)
            first_win_imbal[ridx] = sn;

          std::printf ("  %5ld x %5ld  dispatch=%8.3f ns  pq=%8.3f ns  speedup=%6.3fx\n",
                       (long) ln, (long) sn, dispatch_best * 1e9, direct_best * 1e9,
                       dispatch_best / direct_best);
          bench_write_csv_row (csv, mode_names[MODE_IMBAL_2 + ridx], sn, ln,
                               dispatch_best, direct_best);
          if (n == cfg.max_limbs)
            break;
        }
    }

  if (csv != NULL)
    std::fclose (csv);

  std::printf ("\nFirst wins:\n");
  std::printf ("  balanced_mul: %ld limbs\n", (long) first_win_bal);
  std::printf ("  sqr:          %ld limbs\n", (long) first_win_sqr);
  std::printf ("  imbal 2:1:    %ld limbs (short side)\n", (long) first_win_imbal[0]);
  std::printf ("  imbal 4:1:    %ld limbs (short side)\n", (long) first_win_imbal[1]);
  std::printf ("  imbal 8:1:    %ld limbs (short side)\n", (long) first_win_imbal[2]);
  std::printf ("  imbal 16:1:   %ld limbs (short side)\n", (long) first_win_imbal[3]);
  return 0;
}
