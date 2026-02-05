#!/usr/bin/env python3
"""
Plot training/validation loss from a TSV file using PyROOT.

Expected input format (whitespace- or tab-separated), with an optional header:
#step   is_val  loss
1       0       0.71015805
2       1       0.695...
...

- is_val == 0  -> training
- is_val == 1  -> validation

Example:
  python plot_loss_root.py loss.tsv -o loss.png
"""

import argparse
import array
import math
import sys


def read_loss_tsv(path: str):
    train = []
    val = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            step = float(parts[0])
            is_val = int(parts[1])
            loss = float(parts[2])

            # log-scale friendly: skip non-positive / non-finite
            if not math.isfinite(loss) or loss <= 0.0:
                continue

            (val if is_val else train).append((step, loss))

    train.sort(key=lambda t: t[0])
    val.sort(key=lambda t: t[0])
    return train, val


def to_graph(points, color, name):
    import ROOT

    xs = array.array("d", [p[0] for p in points])
    ys = array.array("d", [p[1] for p in points])
    gr = ROOT.TGraph(len(points), xs, ys)
    gr.SetName(name)

    gr.SetLineColor(color)
    gr.SetMarkerColor(color)
    gr.SetLineWidth(1)
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(0.45)

    return gr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input TSV (columns: step  is_val  loss)")
    ap.add_argument("-o", "--output", default="loss.png", help="Output image/pdf (default: loss.png)")
    ap.add_argument("--no-logy", action="store_true", help="Disable log-scale on Y")
    ap.add_argument(
        "--xscale",
        type=float,
        default=None,
        help="Scale X by this factor (e.g. 1000). If omitted, auto-scales to 1e3 when max(step)>=1000.",
    )
    ap.add_argument("--xtitle", default="Iteration", help="X axis title (default: Iteration)")
    args = ap.parse_args()

    try:
        import ROOT
    except ImportError:
        sys.stderr.write("ERROR: PyROOT is not available (this script expects `import ROOT` to work).\n")
        return 2

    ROOT.gROOT.SetBatch(True)

    # Style knobs to resemble your example
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetLegendBorderSize(0)

    train, val = read_loss_tsv(args.input)
    if not train and not val:
        sys.stderr.write(f"ERROR: No usable points found in {args.input}\n")
        return 2

    all_steps = [s for s, _ in train] + [s for s, _ in val]
    xmin, xmax = min(all_steps), max(all_steps)

    # Auto-scale X like your plot (0..1.2 with ×10^3) when steps are large
    xscale = args.xscale
    scale_label = ""
    if xscale is None:
        if xmax >= 1000.0:
            xscale = 1000.0
            scale_label = "#times10^{3}"
        else:
            xscale = 1.0
    else:
        if xscale != 1.0:
            # If it's a power of 10, label it nicely
            p = round(math.log10(xscale))
            if abs(xscale - 10**p) / xscale < 1e-9:
                scale_label = f"#times10^{{{int(p)}}}"
            else:
                scale_label = f"/ {xscale:g}"

    train_s = [(s / xscale, l) for s, l in train]
    val_s = [(s / xscale, l) for s, l in val]

    all_x = [s for s, _ in train_s] + [s for s, _ in val_s]
    all_y = [l for _, l in train_s] + [l for _, l in val_s]

    xmin_s, xmax_s = min(all_x), max(all_x)
    ymin_data, ymax_data = min(all_y), max(all_y)

    # Reasonable log-y margins
    ymin = ymin_data * 0.8
    ymax = ymax_data * 1.2
    if ymin <= 0:
        ymin = ymin_data * 0.5
    if ymin <= 0:
        ymin = 1e-6

    c = ROOT.TCanvas("c", "c", 900, 650)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.12)
    c.SetBottomMargin(0.12)
    c.SetTopMargin(0.06)
    if not args.no_logy:
        c.SetLogy()

    gr_train = to_graph(train_s, ROOT.kBlue + 1, "gr_train") if train_s else None
    gr_val = to_graph(val_s, ROOT.kRed + 1, "gr_val") if val_s else None

    base = gr_train if gr_train else gr_val
    base.SetMinimum(ymin)
    base.SetMaximum(ymax)

    # Draw axis + base curve
    base.Draw("ALP")
    base.GetXaxis().SetLimits(xmin_s, xmax_s)
    base.GetXaxis().SetTitle(args.xtitle)
    base.GetYaxis().SetTitle("Loss")
    base.GetYaxis().SetTitleOffset(1.2)
    base.GetXaxis().SetTitleOffset(1.0)

    base.GetXaxis().SetTitleSize(0.05)
    base.GetYaxis().SetTitleSize(0.05)
    base.GetXaxis().SetLabelSize(0.04)
    base.GetYaxis().SetLabelSize(0.04)

    # Overlay the other curve
    if gr_train and gr_train is not base:
        gr_train.Draw("LP same")
    if gr_val and gr_val is not base:
        gr_val.Draw("LP same")

    leg = ROOT.TLegend(0.62, 0.76, 0.88, 0.88)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    if gr_train:
        leg.AddEntry(gr_train, "Training Loss", "lp")
    if gr_val:
        leg.AddEntry(gr_val, "Validation Loss", "lp")
    leg.Draw()

    # Optional “×10^3” style indicator like the screenshot
    if scale_label:
        latex = ROOT.TLatex()
        latex.SetNDC(True)
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.93, 0.04, scale_label)

    c.Modified()
    c.Update()
    c.SaveAs(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
