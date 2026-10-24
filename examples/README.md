# Plotting examples

Scripts in this directory read existing result tables and create figures. They
do not run the nuclear-matter, finite-nucleus, or neutron-star solvers.

Generate an EOS table first, then plot it:

```powershell
python -m crmf --eos-only --output output
python examples/plot_eos.py --input output/EOS.OUT --output output/eos
```

The script writes PNG and PDF files. Paper-result calculations and posterior
contour preparation are kept under `tools/`.
