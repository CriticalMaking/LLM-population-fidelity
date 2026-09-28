
## Program:   fig_pfs_toy.R
## Task:      Stylized Population Fidelity example (Figure 1) and the
##            values reported in the toy-example appendix.
##
## Input:     -
## Output:    figures/stylized_example/fig_pfs_toy.pdf (run from repo root)
##
## Project:   LLM population fidelity
## Author:    Martin Lukk / 2026-09-25 (created)

# 0. Program Setup --------------------------------------------------------
library(tidyverse)
library(patchwork)
library(ggtext)

group_colors <- c(A = "#2a78d6", B = "#eb6834", C = "#1baf7a")
pooled_fill  <- "#dcdbd6"
ink          <- "#2b2b2a"
ink_muted    <- "#6f6e69"

# Opinion scale: 0-10 in steps of 0.1, i.e. K = 101 ordered response options
K <- 101
x <- seq(0, 10, length.out = K)

# nEMD (Equation 1): L1 distance between CDFs, divided by K - 1
nemd <- function(p, q) sum(abs(cumsum(p - q))[-length(p)]) / (length(p) - 1)

make_dist <- function(mu, sd) {
  d <- dnorm(x, mu, sd)
  d / sum(d)
}

# Reassign the respondents in p to three groups with weights
# w_g(x) ∝ exp(-(x - c_g)^2 / (2 tau^2)), normalized over groups at each x,
# with centres (mean - offset, mean, mean + offset). The offset is the
# positive root at which each group receives exactly one third of
# respondents, so the groups average back to p exactly. (offset = 0 is a
# trivial root that reproduces the pooled distribution; the bracket excludes
# it.) Smaller tau = sharper sorting.
sharpen_groups <- function(p, tau) {
  mid <- sum(x * p)
  weights <- function(offset) {
    w <- sapply(mid + c(-offset, 0, offset), \(c_g) exp(-(x - c_g)^2 / (2 * tau^2)))
    w / rowSums(w)
  }
  offset <- uniroot(\(o) sum(p * weights(o)[, 2]) - 1 / 3, c(0.01, 5),
                    tol = .Machine$double.eps)$root
  w <- weights(offset)
  list(offset = offset, dists = map(1:3, \(g) 3 * p * w[, g]))
}

# Pairwise distances in pair order (AB, AC, BC). Pairs that are equal by
# mirror symmetry differ by ~1e-16 in floating point; rounding to 10
# decimals restores those exact ties before ranking (without it, Spearman
# rho for M2 and M3 changes and PFS_M3 becomes positive).
pair_dists <- function(dists) round(combn(dists, 2, \(d) nemd(d[[1]], d[[2]])), 10)

pfs_metrics <- function(s, m) {
  E    <- mean(map2_dbl(s, m, nemd))
  d_s  <- pair_dists(s)
  d_m  <- pair_dists(m)
  A    <- median(d_m) / median(d_s)
  rho  <- if (sd(d_m) == 0) NA_real_ else cor(d_s, d_m, method = "spearman")
  S_acc    <- 1 - E
  S_adapt  <- if (A == 0) 0 else min(A, 1 / A)
  S_struct <- if (is.na(rho)) 0 else max(0, rho)
  tibble(
    E, S_acc, A, S_adapt, rho, S_struct,
    S_center = 1 - nemd(reduce(s, `+`) / length(s), reduce(m, `+`) / length(m)),
    PFS      = (S_acc * S_adapt * S_struct)^(1 / 3)
  )
}

# 1. Create Hypothetical Data ---------------------------------------------
groups <- c("A", "B", "C")
survey <- map(c(A = 4, B = 5, C = 6), make_dist, sd = 1.3)
pooled <- reduce(survey, `+`) / 3

exaggerated <- sharpen_groups(pooled, tau = 0.4)

models <- list(
  "M1: collapsed"   = rep(list(pooled), 3),
  "M2: exaggerated" = exaggerated$dists,
  "M3: swapped"     = survey[c("A", "C", "B")]
) |>
  map(\(m) set_names(m, groups))

# 2. Calculate Metrics ----------------------------------------------------
# The ties the structure scores rely on (see pair_dists)
d_survey <- pair_dists(survey)
d_m2     <- pair_dists(models$`M2: exaggerated`)
d_m3     <- pair_dists(models$`M3: swapped`)
stopifnot(
  d_survey[1] == d_survey[3],  # survey: AB = BC
  d_m2[1] == d_m2[3],          # M2: AB = BC
  d_m3[2] == d_m3[3]           # M3: AC = BC
)

dist_sd <- function(p) sqrt(sum((x - sum(x * p))^2 * p))

metrics <- map(models, \(m) pfs_metrics(survey, m)) |>
  list_rbind(names_to = "panel")

cell_errors <- map(models, \(m) map2_dbl(survey, m, nemd)) |>
  map(\(e) tibble(group = groups, error = e)) |>
  list_rbind(names_to = "panel")

pairwise <- c(list(Survey = survey), models) |>
  map(\(d) tibble(pair = c("AB", "AC", "BC"), dist = pair_dists(d))) |>
  list_rbind(names_to = "panel")

group_summaries <- c(list(Survey = survey), models) |>
  map(\(d) tibble(group = groups, mean = map_dbl(d, \(p) sum(x * p)), sd = map_dbl(d, dist_sd))) |>
  list_rbind(names_to = "panel")

# Values reported in the appendix
cat("M2 sorting: tau = 0.4, c =", round(exaggerated$offset, 4), "\n")
print(metrics, width = Inf)
print(cell_errors |> pivot_wider(names_from = group, values_from = error))
print(pairwise |> pivot_wider(names_from = pair, values_from = dist))
print(group_summaries, n = Inf)

# 3. Plot Hypothetical Data -----------------------------------------------
# Component scores below each model panel, one row per score
score_table <- function(scores) {
  rows <- tibble(
    label = c("Accuracy", "Adaptability", "Structure", "PFS"),
    value = sprintf("%.2f", c(scores$S_acc, scores$S_adapt, scores$S_struct, scores$PFS)),
    y     = c(4, 3, 2, 1),
    face  = c("plain", "plain", "plain", "bold")
  )
  ggplot(rows) +
    geom_text(aes(0.14, y, label = label, fontface = face), hjust = 0,
              size = 6.5 / .pt, color = ink, family = "Helvetica") +
    geom_text(aes(0.86, y, label = value, fontface = face), hjust = 1,
              size = 6.5 / .pt, color = ink, family = "Helvetica") +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.45, 4.55), expand = c(0, 0)) +
    theme_void()
}

# Scale ends, placed in the empty cell below the survey panel
scale_labels <- ggplot() +
  annotate("text", x = c(0.05, 9.95), y = 4.3, label = c("Low", "High"),
           hjust = c(0, 1), vjust = 1,
           size = 6.5 / .pt, color = ink_muted, family = "Helvetica") +
  scale_x_continuous(limits = c(0, 10), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.45, 4.55), expand = c(0, 0)) +
  theme_void() +
  theme(plot.margin = margin(2, 6, 2, 6))

# Each group is drawn at its one-third share of respondents, so the group
# curves add up to the pooled (gray) distribution in every panel
to_curves <- function(dists) {
  imap(dists, \(p, g) tibble(group = g, x = x, y = p / 3)) |>
    list_rbind()
}

# Coincident curves (M1) are drawn as one line of alternating colored dashes
dash_curves <- function(curves, dash = 0.35) {
  curves |>
    mutate(seg = floor(x / dash)) |>
    filter(seg %% 3 == match(group, groups) - 1) |>
    mutate(run = str_c(group, seg))
}

y_top <- max(pooled) * 1.2

plot_panel <- function(dists, title, collapsed = FALSE, y_label = NULL) {
  curves <- to_curves(dists)
  envelope <- tibble(x = x, y = pooled)
  means <- tibble(group = groups, mean = map_dbl(dists, \(p) sum(x * p))) |>
    mutate(height = map2_dbl(group, mean, \(g, m) approx(x, dists[[g]] / 3, m)$y))
  if (collapsed) {
    # Coincident means (M1) share one line whose dashes alternate in color
    step <- means$height[1] / 12
    means <- tibble(group = rep(groups, length.out = 12), mean = means$mean[1],
                    y0 = (0:11) * step, height = y0 + 0.5 * step, linetype = "solid")
  } else {
    means <- means |> mutate(y0 = 0, linetype = "22")
  }

  # Label each group at its mean, just above the highest point of its curve
  # within 0.5 of the mean
  labels <- curves |>
    group_by(group) |>
    summarise(
      xm = sum(x * y) / sum(y),
      ym = max(y[abs(x - xm) <= 0.5]),
      .groups = "drop"
    )
  if (collapsed) {
    labels <- tibble(group = "A = B = C", xm = 5, ym = max(curves$y))
  }

  line_layer <- if (collapsed) {
    geom_line(data = dash_curves(curves), aes(x, y, color = group, group = run),
              linewidth = 0.55, lineend = "butt")
  } else {
    geom_line(data = curves, aes(x, y, color = group), linewidth = 0.45)
  }

  p <- ggplot() +
    geom_area(data = envelope, aes(x, y), fill = pooled_fill) +
    geom_area(data = curves, aes(x, y, fill = group), position = "identity",
              alpha = if (collapsed) 0 else 0.16) +
    line_layer +
    # Group means: faint dashed line from the baseline to each curve
    geom_segment(data = means, aes(x = mean, xend = mean, y = y0, yend = height,
                                   color = group, linetype = linetype),
                 linewidth = 0.35, alpha = 0.7, lineend = "butt") +
    scale_linetype_identity() +
    annotate("segment", x = 0, xend = 10, y = 0, yend = 0,
             color = ink_muted, linewidth = 0.3) +
    geom_text(data = labels, aes(xm, ym, label = group),
              vjust = -0.45, size = 7 / .pt, fontface = "bold", color = ink,
              family = "Helvetica") +
    scale_fill_manual(values = group_colors, guide = "none") +
    scale_color_manual(values = group_colors, guide = "none") +
    scale_x_continuous(limits = c(0, 10), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, y_top), expand = c(0, 0)) +
    labs(title = title) +
    theme_void(base_family = "Helvetica", base_size = 7) +
    theme(
      plot.title   = element_text(face = "bold", size = 7.5, hjust = 0.5, color = ink,
                                  margin = margin(b = 2)),
      plot.margin  = margin(2, 6, 2, 6)
    )

  if (!is.null(y_label)) {
    p <- p +
      annotate("segment", x = 0, xend = 0, y = 0, yend = y_top,
               color = ink_muted, linewidth = 0.3) +
      labs(y = y_label) +
      theme(axis.title.y = element_text(angle = 90, size = 6.5, color = ink_muted,
                                        margin = margin(r = 2)))
  }
  p
}

p_survey <- plot_panel(survey, "Survey (reference)", y_label = "Population share")
p_models <- imap(models, \(m, name) {
  plot_panel(m, name, collapsed = str_detect(name, "collapsed"))
})
p_scores <- map(split(metrics, metrics$panel)[names(models)], score_table)

p_toy <- wrap_plots(c(list(p_survey), p_models, list(scale_labels), p_scores),
                    nrow = 2, heights = c(1, 0.42))

# Base pdf() rather than cairo_pdf: Pango rounds glyph advances at these small
# sizes, which spaces letters unevenly ("swapp ed"). Base pdf() uses the
# Helvetica AFM metrics with kerning; embedFonts() (Ghostscript) then embeds
# the fonts for submission.
out_dir <- file.path("figures", "stylized_example")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
pdf_path <- file.path(out_dir, "fig_pfs_toy.pdf")
ggsave(pdf_path, p_toy, device = pdf, width = 5.5, height = 1.6, units = "in")
embedFonts(pdf_path, options = "-dPDFSETTINGS=/prepress -dEmbedAllFonts=true")
