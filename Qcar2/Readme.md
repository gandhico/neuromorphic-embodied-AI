# QCar SNNs — Hardware Interface

<!-- One-sentence summary. -->
Hardware-side code for running spiking neural network controllers on the Quanser QCar platform.

> **Status:** Private (pre-publication). Will be made public upon acceptance of the
> associated paper. See [Citation](#citation).
>
> **Scope:** This repository contains the **hardware** portion of the QCar SNN project.
> Simulation / training code lives separately. TODO: link it once public.

## Overview

<!-- 2–4 sentences: what runs on the hardware, how the SNN controller connects to the QCar. -->
TODO: Describe the hardware setup, the QCar interface, and what this code does on-device.

## Repository structure

```
.
├── README.md
├── CITATION.cff
├── requirements.txt        # TODO: add
└── ...                     # TODO: match your hardware/ layout
```

## Requirements

- QCar / Quanser platform  <!-- TODO: model + firmware/SDK version -->
- Python 3.x  <!-- TODO: pin version -->
- TODO: Quanser API and remaining dependencies

Install:

```bash
pip install -r requirements.txt
```

## Hardware setup

TODO: Wiring, calibration, network/IP configuration, and any steps required before
running on the physical QCar. Note any safety precautions for running on hardware.

## Usage

```bash
# TODO: command to deploy / run the controller on the QCar
python run_hardware.py
```

## Data

TODO: State where logs / recorded runs are stored. Keep large logs out of git.

## Citation

If you use this code, please cite the associated paper (see `CITATION.cff`).
TODO: add BibTeX once the paper has a DOI/venue.

## License

TODO: Choose a license (e.g. MIT, Apache-2.0, BSD-3-Clause) and add a `LICENSE` file.
Until a license is added, default copyright applies and others may not reuse the code.

## Contact

TODO: Name / email / lab / ORCID.
