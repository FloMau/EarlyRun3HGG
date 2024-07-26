import matplotlib.pyplot as plt
import os
import numpy as np
import awkward as ak
import vector
import hist
from copy import deepcopy
import mplhep as hep
import pyarrow.parquet as pq
vector.register_awkward()
plt.style.use([hep.style.CMS])

import time



def plot_stacked_ratio(data_hist, mc_hists, output_filename, density=False, 
                axisLabels=["my x-axis", "Events"], MC_labels=["MC 1", "MC 2"],
                dataLabel="Data", ratioLimits=(0.5, 1.5), lumi=1, text=None):

    data_hist_numpy = data_hist.to_numpy()
    mc_summed = deepcopy(mc_hists[0])
    for mc_hist in mc_hists[1:]:
        mc_summed += mc_hist
    mc_summed_numpy = mc_summed.to_numpy()

    ratio = data_hist_numpy[0] / mc_summed_numpy[0]
    errors_num = (np.sqrt(data_hist_numpy[0])) / mc_summed_numpy[0] 
    errors_den = np.sqrt(mc_summed.variances()) / mc_summed.values()

    ratio = np.nan_to_num(ratio)
    errors_num = np.nan_to_num(errors_num)
    errors_den = np.nan_to_num(errors_den)

    lower_bound = 1 - errors_den
    upper_bound = 1 + errors_den

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    my_colorstack = ["tab:gray","tab:blue", "tab:orange", "tab:red", "tab:green"]
    hep.histplot(
        mc_hists,
        label = MC_labels,
        yerr=True,
        histtype="fill",
        stack=True,
        alpha=0.5,
        color=my_colorstack[:len(mc_hists)],
        density=density,
        linewidth=3,
        ax=ax[0]
    )

    hep.histplot(
        data_hist,
        label=dataLabel,
        yerr=True,
        density=density,
        color="black",
        histtype='errorbar',
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[0]
    )

    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.05*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)

    # line at 1
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=1)#, alpha=0.5)

    # Plot the hatched region
    ax[1].fill_between(data_hist_numpy[1][:-1],
        lower_bound,
        upper_bound,
        hatch='XXX',
        step='post',
        facecolor="none",
        edgecolor="tab:gray", 
        linewidth=0
    )

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="black",
        histtype='errorbar',
        yerr=errors_num,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )

    ax[1].set_xlabel(axisLabels[0], fontsize=26)
    ax[0].set_ylabel(axisLabels[1], fontsize=26)
    ax[0].tick_params(labelsize=24)
    ax[0].set_xlim(data_hist_numpy[1][0], data_hist_numpy[1][-1])
    ax[0].set_ylim(ax[0].get_ylim()[0], 1.15*ax[0].get_ylim()[1])
    ax[1].set_ylabel("Data / MC", fontsize=26)
    ax[1].set_ylim(ratioLimits)

    ax[0].legend(
        loc="upper right", fontsize=24
    )

    if text is not None:
        ax[0].text(
            0.07, 0.75, text, fontsize=22, transform=ax[0].transAxes
        )

    pad = 0 if ax[0].get_ylim()[1] < 1e6 else 0.05
    hep.cms.label(data=True, ax=ax[0], loc=0, label="Work in Progress", com=13.6, lumi=lumi, pad=pad)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.07)
    fig.savefig(output_filename)
    plt.close()


# Constants
inputDir = "/eos/cms/store/group/phys_higgs/cmshgg/earlyRun3Hgg/analysis_Ntuples/prelim_24_01_23/"

subdirs_MC_preEE = ["GG-Box-3Jets_MGG-80_preEE/nominal", "GJet_PT-20to40_DoubleEMEnriched_MGG-80_preEE/nominal", "GJet_PT-40_DoubleEMEnriched_MGG-80_preEE/nominal"]
subdirs_MC_postEE = ["GG-Box-3Jets_MGG-80_postEE/nominal", "GJet_PT-20to40_DoubleEMEnriched_MGG-80_postEE/nominal", "GJet_PT-40_DoubleEMEnriched_MGG-80_postEE/nominal"]

subdirsData = [f"Data{era}_2022/nominal" for era in ["C", "D", "E", "F", "G"]]

subdirs = subdirs_MC_preEE + subdirs_MC_postEE + subdirsData

# Cross section and luminosity values
dict_xSecs = {
    "GG-Box-3Jets": 89.14e3,
    "GJet_PT-20to40": 242.5e3,
    "GJet_PT-40": 919.1e3,
}
lumi_preEE = 8.1  # fb^-1
lumi_postEE = 27.0  # fb^-1

# Dictionary to hold the arrays for each type of modification
allEvents = {}

# Loop through each subdirectory and read events
for subdir in subdirs:
    print("INFO: reading directory", subdir)
    path = os.path.join(inputDir, subdir)
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]# [:20]
    if "Data" in subdir:
        events = ak.from_parquet(files)
        mass_cuts = ((events.mass > 100) & (events.mass < 120)) | ((events.mass > 130) & (events.mass < 180))
        events = events[mass_cuts]
        allEvents[subdir] = events
    else:
        events = ak.from_parquet(files)
        campaign = 'preEE' if 'preEE' in subdir else 'postEE'
        lumi = lumi_preEE if campaign == 'preEE' else lumi_postEE
        process = subdir.split('/')[0].replace("_postEE", "").replace("_preEE", "").replace("_MGG-80", "").replace("_DoubleEMEnriched", "")
        xsec_key = "GG-Box-3Jets" if "GG-Box" in process else process

        # Sum of generated weights before selection
        sum_genw_beforesel = sum(float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in files)

        # Apply weights
        events["weight"] = events["weight"] * (lumi * dict_xSecs[xsec_key] / sum_genw_beforesel)
        ### mass cuts:
        mass_cuts = ((events.mass > 100) & (events.mass < 120)) | ((events.mass > 130) & (events.mass < 180))
        events = events[mass_cuts]

        if "GJet" in subdir:
            print("INFO: applying overlap removal to samples in", subdir)
            events = events[events.lead_genPartFlav + events.sublead_genPartFlav != 2]

        allEvents[subdir] = events

print("INFO: Finished reading all directories.")

dipho = allEvents["GG-Box-3Jets_MGG-80_preEE/nominal"]
data = allEvents["DataC_2022/nominal"]

# print(dipho.fields)
# print("\n\n\n")
# print(data.fields)
# exit()



histsDictBasic = {
        "mvaId": hist.Hist.new.Reg(50, -1, 1).Weight(), 
        "sigmaMoverM": hist.Hist.new.Reg(60, 0.005, 0.035).Weight(), 
        "sigmaMoverM_IDg0": hist.Hist.new.Reg(60, 0.005, 0.035).Weight(), 
    }
hists = {
    subdir: deepcopy(histsDictBasic) for subdir in subdirs
}


# fill all histograms
for subdir in subdirs:

    histograms = hists[subdir]
    events = allEvents[subdir]

    # Fill the histogram
    if "Data" in subdir:
        events["min_mvaID"] = np.min([events.lead_mvaID, events.sublead_mvaID], axis=0)
        # for plots with mvaId > 0 in note:
        # events = events[events.min_mvaID > 0.]
        histograms["mvaId"].fill(events.min_mvaID)
        histograms["sigmaMoverM"].fill(events.sigma_m_over_m_smeared_decorr)
        histograms["sigmaMoverM_IDg0"].fill(events[events.min_mvaID > 0.].sigma_m_over_m_smeared_decorr)
    else:
        events["min_mvaID"] = np.min([events.lead_corr_mvaID_run3, events.sublead_corr_mvaID_run3], axis=0)
        # for plots with mvaId > 0 in note:
        # events = events[events.min_mvaID > 0.]
        histograms["mvaId"].fill(events.min_mvaID, weight=events.weight)
        histograms["sigmaMoverM"].fill(events.sigma_m_over_m_corr_smeared_decorr, weight=events.weight)
        histograms["sigmaMoverM_IDg0"].fill(events[events.min_mvaID > 0.].sigma_m_over_m_corr_smeared_decorr, weight=events[events.min_mvaID > 0.].weight)

# Aggregate data histograms (inclusive, sum from C-G)
histsDataTotal = deepcopy(histsDictBasic)
for era in ["C", "D", "E", "F", "G"]:
    subdir = f"Data{era}_2022/nominal"
    for hist_key in histsDictBasic.keys():
        histsDataTotal[hist_key] += hists[subdir][hist_key]

histsMCSummed = {
    "GG": deepcopy(histsDictBasic),
    "GJet": deepcopy(histsDictBasic),
}
for era in ["preEE", "postEE"]:
    for hist_key in histsDictBasic.keys():
        histsMCSummed["GG"][hist_key] += hists[f"GG-Box-3Jets_MGG-80_{era}/nominal"][hist_key]
        histsMCSummed["GJet"][hist_key] += hists[f"GJet_PT-20to40_DoubleEMEnriched_MGG-80_{era}/nominal"][hist_key]
        histsMCSummed["GJet"][hist_key] += hists[f"GJet_PT-40_DoubleEMEnriched_MGG-80_{era}/nominal"][hist_key]

print("Histograms successfully aggregated.")


### basic plots of minimum MVA ID and sigma:
pathPlots = "./Plots/"
os.makedirs(pathPlots, exist_ok=True)

plot_stacked_ratio(
    data_hist=histsDataTotal["mvaId"],
    mc_hists=[histsMCSummed["GJet"]["mvaId"], histsMCSummed["GG"]["mvaId"]],
    output_filename=pathPlots+"minMvaID.pdf",
    axisLabels=["Diphoton minimum IDMVA", "Events / 0.04"],
    MC_labels=["GJet", "Diphoton"],
    lumi=35.1,
    ratioLimits=(0, 3)
)

plot_stacked_ratio(
    data_hist=histsDataTotal["sigmaMoverM"],
    mc_hists=[histsMCSummed["GJet"]["sigmaMoverM"], histsMCSummed["GG"]["sigmaMoverM"]],
    output_filename=pathPlots+"sigmaMoverM_NF_smeared.pdf",
    axisLabels=[r"$\sigma_m / m$ (decorr., w/ smearing, NF corr.)", "Events / 0.005"],
    MC_labels=["GJet", "Diphoton"],
    lumi=35.1,
    ratioLimits=(0, 3)
)

plot_stacked_ratio(
    data_hist=histsDataTotal["sigmaMoverM_IDg0"],
    mc_hists=[histsMCSummed["GJet"]["sigmaMoverM_IDg0"], histsMCSummed["GG"]["sigmaMoverM_IDg0"]],
    output_filename=pathPlots+"sigmaMoverM_NF_smearedIDg0.pdf",
    axisLabels=[r"$\sigma_m / m$ (decorr., w/ smearing, NF corr.)", "Events / 0.005"],
    MC_labels=["GJet", "Diphoton"],
    lumi=35.1,
    ratioLimits=(0, 3),
    text="Min. IDMVA > 0"
)

plot_stacked_ratio(
    data_hist=histsDataTotal["mvaId"],
    mc_hists=[2.38*histsMCSummed["GJet"]["mvaId"], 1.42*histsMCSummed["GG"]["mvaId"]],
    output_filename=pathPlots+"minMvaID_scaled.pdf",
    axisLabels=["Diphoton minimum IDMVA", "Events / 0.04"],
    MC_labels=["GJet", "Diphoton"],
    lumi=35.1
)


plot_stacked_ratio(
    data_hist=histsDataTotal["sigmaMoverM"],
    mc_hists=[2.38*histsMCSummed["GJet"]["sigmaMoverM"], 1.42*histsMCSummed["GG"]["sigmaMoverM"]],
    output_filename=pathPlots+"sigmaMoverM_NF_smearedScaled.pdf",
    axisLabels=[r"$\sigma_m / m$ (decorr., w/ smearing, NF corr.)", "Events / 0.0005"],
    MC_labels=["GJet", "Diphoton"],
    lumi=35.1
)

plot_stacked_ratio(
    data_hist=histsDataTotal["sigmaMoverM_IDg0"],
    mc_hists=[2.38*histsMCSummed["GJet"]["sigmaMoverM_IDg0"], 1.42*histsMCSummed["GG"]["sigmaMoverM_IDg0"]],
    output_filename=pathPlots+"sigmaMoverM_NF_smearedIDg0Scaled.pdf",
    axisLabels=[r"$\sigma_m / m$ (decorr., w/ smearing, NF corr.)", "Events / 0.0005"],
    MC_labels=["GJet", "Diphoton"],
    lumi=35.1,
    text="Min. IDMVA > 0"
)



get_norm = True
if get_norm:
    ### get scaling factors:
    import numpy as np
    from scipy.optimize import minimize

    # Define the chi-squared function
    def chi_squared(scaling_factors, hist_data, hist_diphoton, hist_GJet, errors):
        scale_diphoton, scale_GJet = scaling_factors
        # Calculate the model by scaling the MC histograms
        model = scale_diphoton * hist_diphoton + scale_GJet * hist_GJet
        # Compute the chi-squared
        chi_sq = np.sum(((hist_data - model) / errors) ** 2)
        return chi_sq

    # Assuming you have the histograms as numpy arrays and the counts for the right half of the spectrum (MVA ID > -0.5)
    # You also need the errors for the data histogram. If you don't have them, you can use the square root of the counts as an approximation
    hist_data=histsDataTotal["mvaId"].to_numpy()[0][25:]
    hist_GJet = histsMCSummed["GJet"]["mvaId"].to_numpy()[0][25:]
    hist_diphoton = histsMCSummed["GG"]["mvaId"].to_numpy()[0][25:]
    errors_data = np.sqrt(hist_data)

    # Initial guess for the scaling factors (you can adjust these as needed)
    initial_guess = [1.0, 1.0]

    # Perform the minimization
    result = minimize(chi_squared, initial_guess, args=(hist_data, hist_diphoton, hist_GJet, errors_data))

    # The result contains the scaling factors for diphoton and GJet
    scale_diphoton, scale_GJet = result.x
    print(f"Scaling factor for diphoton: {scale_diphoton}")
    print(f"Scaling factor for GJet: {scale_GJet}")