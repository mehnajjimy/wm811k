# Wafer map patterns and likely process root causes

This table (item C in RESEARCH.md, sourcing rule d09) links each WM-811K failure pattern class to the process modules most likely to cause it. For each class it gives the first place a fab engineer would look and the evidence that would confirm or rule out the cause.

## How to read the source column

- **Engineering hypothesis** means the row is reasoning from general process knowledge. No verified source was checked for that specific pattern-to-cause link. Treat it as a starting point for a real excursion review, not a fact.
- A source tag such as **[S4]** means the verified reference supports the *general* point named in the row. All source checks were done against bibliographic records and abstracts only (see "Verification limits" below). No full text was read, so no row claims that a paper makes the specific mechanism argument.

Every mechanism in this table is an engineering hypothesis until Mehnaj checks it against the full text of a source or against a process engineer's experience.

## Root cause table

Pattern names are the dataset's exact label strings, as in `wm811k.data.CLASSES`, so the table can be joined to predictions directly.

| Pattern | Likely process modules | Physical mechanism | First fab step to check | Signal that would confirm it | Source |
|---|---|---|---|---|---|
| Center | CMP, deposition, etch | A radial rate profile that is off target at the wafer center. Examples: CMP removal that is center-fast or center-slow from carrier head zone pressure, center-thick CVD from showerhead gas distribution, or a center-heavy etch from plasma density. | Post-CMP or post-deposition film thickness map (radial profile) for the affected lots. | Center thickness or CD is outside the spec window on the same wafers whose center dies fail, and the failing wafers share one tool or chamber in a commonality analysis. | Engineering hypothesis. Ezzat et al. [E] link center patterns to chemical stains or mechanical equipment faults, citing [S5] and [S6]. Systematic patterns pointing to assignable causes: [S1], [S4]. Radial film gradients: Mehnaj's ALD uniformity work [L2] |
| Donut | CMP, deposition, etch, thermal (RTP, anneal) | A rate or temperature that is off target in a ring at mid radius. Examples: a zone boundary in a multi-zone CMP head or heater, or a ring of high plasma density from an inductive coil. | Radial thickness or CD profile, plus zone setpoints of the suspected chamber. | The ring radius matches a hardware zone radius, and the ring disappears when the same product runs in a different chamber. | Engineering hypothesis. Ezzat et al. [E] call donut patterns routine in wafer data and attribute them to possible tooling problems, citing [S8]. [S8] is a published failure analysis of a donut failure, which its title attributes to a photovoltaic electrochemical effect (full text not read). Mid-radius thickness rings: Mehnaj's ALD uniformity work [L2] |
| Edge-Ring | Etch, deposition, CMP, litho | Edge roll-off around the whole circumference. Examples: plasma sheath bending at the edge as a focus ring erodes, deposition thinning at the edge, CMP retaining ring effects, or a spin-coated film left with an edge bead (a thicker rim where surface tension holds liquid at the edge during spin-off) or with edge bead removal set too wide. | Thickness or CD measured within the outer 10 to 20 mm, against focus ring RF hours or retaining ring life. For a coated layer, the edge bead height and the edge bead removal width. | Edge deviation grows with consumable life and resets after preventive maintenance. Edge die yield follows the same trend. For a coated layer, the failing ring width matches the bead or removal width. | Engineering hypothesis. Ezzat et al. [E] trace edge-located zones to uneven polishing or edge-die effects, citing [S7]. Edge bead physics: Mehnaj's spin-coat work [L1] |
| Edge-Loc | Handling, etch, deposition, litho | Edge roll-off confined to one part of the edge. Examples: a wafer placed off center in a chamber, a local hardware feature (clamp finger, gas inlet, lift pin near the edge), a local edge bead problem, or edge chipping during handling. | Angle of the failing zone relative to the notch across many wafers, plus wafer centering or placement data from the suspected tool. | A fixed angle relative to the notch on one tool points to hardware or placement. An angle that varies from wafer to wafer points to handling. | Engineering hypothesis |
| Loc | Deposition, etch, handling, litho | A local cluster of killer defects. Examples: a flake from a chamber wall or shield, a particle burst during pump-down or venting, or a local resist defect. | Inline defect inspection maps from layers before the failing test, overlaid on the bin map. | A defect cluster on an earlier layer sits at the same location as the failing dies (defect-to-bin overlay). The cluster's kill ratio explains how many dies fail. | Engineering hypothesis. Local clustered defects as distinct from global random defects: [S5], [S7] |
| Near-full | Any module; also test | A gross, wafer-level failure. Examples: a skipped or doubled step, the wrong recipe, an implant dose error, a broken film, or a tester or probe card fault that only looks like a process failure. | The wafer's process history (alarms, reworks, holds), and a retest on a different tester or probe card. | A process excursion in that wafer's history, or the wafer passes on retest (which would point to test, not process). | Engineering hypothesis |
| Random | Any module; fab environment | Spatially uniform killer defects with a defect density above the normal baseline. Examples: tool or chemical contamination, or a general rise in airborne particles. It can also be parametric marginality, where the whole wafer sits close to a spec limit. | Defect density trend from inline inspection, and parametric (e-test) distributions for the lot. | Defect density for that period is above baseline, or the e-test means shift toward a limit on the affected wafers. | Engineering hypothesis. Random versus clustered defects in yield modelling: [S5], [S7] |
| Scratch | CMP, handling | A narrow track dragged across the surface. A curved arc fits CMP, where debris or slurry agglomerates are trapped under the rotating pad. A straight line fits handling, such as a robot blade or a wafer sliding in a cassette or on a chuck. | Inline inspection after CMP and after the handling steps that come before the fail. | The scratch shows up on the inspection image of a specific layer. Its curvature matches either the CMP kinematics or a straight handling path. | Engineering hypothesis |
| none | None | Only the normal random defect background. | Check that wafer yield is inside the usual control limits. | Yield in the normal band, with no spatial clustering beyond what a Poisson model predicts. | Engineering hypothesis. Random defect background as the reference case: [S5] |

Notes:

- "Test" is not in the requested module list, but Near-full and some edge patterns can come from the tester or the probe card instead of the process. A real fab rules this out first, because a retest costs much less than a process investigation. This is recorded as an addition in `DECISIONS.md` (d33).
- The boundary between `none` and `Random` is a defect-density threshold, not a different mechanism. The simulator treats them that way too.

## Mehnaj's own lab work

| Tag | Work | Used for |
|---|---|---|
| L1 | Spin coating, including edge beads: the thicker rim of film that forms at the wafer edge as surface tension holds liquid there during spin-off | The Edge-Ring row. The bead is a radially symmetric thickness change confined to the outer edge, the same geometry as an Edge-Ring bin map. |
| L2 | ALD thickness uniformity measurements | The Center and Donut rows. ALD is self-limiting in principle, so any center-to-edge or mid-radius thickness gradient points to precursor dosing, purge or temperature non-uniformity across the wafer. |

These rows cite the physics Mehnaj has seen first hand. No measured values are quoted here until the slides or data are added to the repo (planned under `docs/lab/`).

## Verified sources

### How the citations were checked

The session's network policy blocked doi.org, the Crossref API, IEEE Xplore, arXiv, OpenAlex and Semantic Scholar, so no DOI was resolved directly. Each citation below was checked with web search against publisher-adjacent or institutional repository records (KAIST, SeoulTech and SKKU, NCU and CGU research portals, IAOR, RePEc, library catalogues). The title, authors, venue, volume, issue, pages and DOI were cross-checked across those records. Only abstracts were seen; no full text was read.

Status labels:

- **Record verified**: details agree across at least one repository or catalogue record, but the DOI was not resolved directly.
- **Not found**: no matching record turned up.

| Tag | Citation | Status |
|---|---|---|
| S1 | K. Kyeong and H. Kim, "Classification of mixed-type defect patterns in wafer bin maps using convolutional neural networks," *IEEE Transactions on Semiconductor Manufacturing*, vol. 31, no. 3, pp. 395–402, Aug. 2018. DOI: 10.1109/TSM.2018.2841416. | Record verified. First author's full given name appears as Kiryong in the KAIST record. The DOI came from a library catalogue entry, not the KAIST record. |
| S2 | J. Shim, S. Kang, and S. Cho, "Active learning of convolutional neural network for cost-effective wafer map pattern classification," *IEEE Transactions on Semiconductor Manufacturing*, vol. 33, no. 2, pp. 258–266, 2020. DOI: 10.1109/TSM.2020.2974867. | Record verified |
| S3 | J. Hwang and H. Kim, "Variational deep clustering of wafer map patterns," *IEEE Transactions on Semiconductor Manufacturing*, vol. 33, no. 3, pp. 466–475, 2020. DOI: 10.1109/TSM.2020.3004483. | Record verified. One library catalogue lists a different co-author name; the KAIST record (Jonghyun Hwang, Heeyoung Kim) was used. |
| S4 | K. S.-M. Li, X.-H. Jiang, L. L.-Y. Chen, and S.-J. Wang, "Wafer defect pattern labeling and recognition using semi-supervised learning," *IEEE Transactions on Semiconductor Manufacturing*, vol. 35, no. 2, pp. 291–299, May 2022. DOI: 10.1109/TSM.2022.3159246. | Record verified. Author list taken from the NCU and CGU research portal records. The exact hyphenation of the given names should be checked against IEEE Xplore. |
| S5 | J. Y. Hwang and W. Kuo, "Model-based clustering for integrated circuit yield enhancement," *European Journal of Operational Research*, vol. 178, no. 1, pp. 143–153, Apr. 2007. DOI: 10.1016/j.ejor.2005.11.032. | Record verified |
| S6 | J. Kim, Y. Lee, and H. Kim, "Detection and clustering of mixed-type defect patterns in wafer bin maps," *IISE Transactions*, vol. 50, no. 2, pp. 99–111, 2018. DOI: 10.1080/24725854.2017.1386337. | Record verified. **Venue correction:** this is *IISE Transactions*, not IEEE TSM. |
| S7 | S. P. Cunningham and S. MacKinnon, "Statistical methods for visual defect metrology," *IEEE Transactions on Semiconductor Manufacturing*, vol. 11, no. 1, pp. 48–53, Feb. 1998. DOI: 10.1109/66.661284. | Record verified |
| S8 | S. P. Neo, A. C. T. Quah, G. B. Ang, D. Nagalingam, H. H. Ma, S. L. Ting, C. W. Soo, C. Q. Chen, Z. H. Mai, and J. C. Lam, "Failure analysis methodology on donut pattern failure due to photovoltaic electrochemical effect," *Microelectronics Reliability*, vol. 76–77, pp. 255–260, Sep. 2017. DOI: 10.1016/j.microrel.2017.07.036. | Record verified against Crossref in the main session, after the reference was read off Ezzat et al.'s reference list on ar5iv. Earlier marked not found. |
| E | A. A. Ezzat, S. Liu, D. S. Hochbaum, and Y. Ding, "A graph-theoretic approach for spatial filtering and its impact on mixed-type spatial pattern recognition in wafer bin maps," *IEEE Transactions on Semiconductor Manufacturing*, vol. 34, no. 2, pp. 194–206, 2021. DOI: 10.1109/TSM.2021.3062943. Preprint arXiv:2006.13824. | Record verified through search results. The arXiv page itself could not be opened. |

### What each source is used for (abstract level only)

- **S1, S4:** Their abstracts say that defect pattern type matters for root cause analysis. That supports the general idea behind this table, not any one row's mechanism.
- **S5:** Its abstract separates local defects from assignable causes from global defects from random causes. It models global defects as a spatial non-homogeneous Poisson process and local defects with a bivariate normal distribution or principal curves. This framing supports the Loc, Random and none rows, and the simulator's loc and random generators.
- **S7:** Its abstract says that defect counts and density can mislead as process-control parameters when defects cluster. That supports separating clustered from random defects.
- **S2, S3, S6:** These are ML methods papers (active learning, deep clustering, mixed-type clustering). They are listed for the project's later modelling work and are not used for any root cause claim. Their abstracts describe industrial data: SK hynix for S6, an unnamed global manufacturer for S3, and a semiconductor manufacturer for S2. Whether any of them also used WM-811K was not checked.

## Verification limits

- Ezzat et al.'s citing sentences and reference list were read on ar5iv in the main session. The paper links center patterns to [S5] and [S6], edge zones to [S7], and donut patterns to [S8]. Those statements are Ezzat et al.'s summaries of the cited papers. The cited papers' full texts have not been read.
- A KAIST research webzine article about S6 linked center circular patterns to CMP uniformity problems. It is a second-hand summary and is **not cited** in the table.
- No page numbers or direct quotations are given anywhere in this document, by design.
