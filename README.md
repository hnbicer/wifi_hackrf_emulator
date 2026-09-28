# Wi-Fi-like HackRF traffic generator

Generate packetized, randomized OFDM activity to test an FPGA channel occupancy
detector. This is a simplified energy source, not an access point or an IEEE
802.11-compliant transmitter. It does not communicate with Wi-Fi clients.

The project lives directly in this folder. Generated IQ lives in RAM; transmission
uses a temporary signed-int8 I/Q file in the operating system's temporary directory
and removes it after the child process stops. There is no waveform cache or IQ
export option. Normal experiments retain only configuration, ground truth, and
summary statistics.

## Requirements and installation

- Python 3.10 or newer, NumPy, PyYAML and Matplotlib. GNU Radio is not required.
- For RF transmission: HackRF One, working USB driver, and `hackrf_info` and
  `hackrf_transfer` from the same versioned host-tools installation, release
  **2024.02.1 or newer**, on `PATH`. Hardware is unnecessary for dry runs and tests.
- A USB connection and host that can sustain 20 million complex samples/second
  (40 MB/s of signed-int8 I/Q).

### Anaconda / Miniconda (recommended for moving to another PC)

Install Anaconda or Miniconda on the other Windows PC and copy this project's
source folder. Exclude `.venv`, `__pycache__`, `.pytest_cache`, and
`wifi_hackrf_emulator.egg-info`; virtual environments should be recreated on each
PC. Copy experiment results separately if you want those records.

Open Anaconda Prompt, change into the copied project folder, and run:

```text
cd C:\wifi_hackrf_emulator
conda env create -f environment.yml
conda activate wifi-hackrf
python scripts/check_hackrf.py
python scripts/random_traffic.py --duration 10 --seed 12345
```

The environment installs Python 3.12, NumPy, Matplotlib, PyYAML, pytest, this project,
and the HackRF host library/command-line tools. The
[conda-forge HackRF package](https://anaconda.org/conda-forge/hackrf) supports Windows
64-bit. No GNU Radio, PothosSDR installation, or Python HackRF bindings are required
for this project when using this environment. The host tools are pinned to
2024.02.1 for use with HackRF One firmware 2024.02.1. For a different
firmware release, check host/firmware compatibility before changing that pin.
The YAML is a portable dependency specification, not a complete version lockfile.

Activate `wifi-hackrf` each time you open a new terminal. Use `python` in this
environment instead of the `.venv\Scripts\python.exe` paths in the examples below.
Run `where.exe python` and `where.exe hackrf_info` if you need to confirm which
executables will be used.

### Windows HackRF USB driver

Conda installs user-space tools and libraries; it does not install a Windows USB
device driver. First connect the HackRF and run `hackrf_info`.
If it reports `Found HackRF`, USB access is working and no driver change is needed.
If an older PothosSDR installation is also present, ensure the Conda tools take
precedence on `PATH`. The project's `check_hackrf.py` also enforces host-tool
compatibility for reliable short bursts.

If the tools are available but the new Windows PC cannot access the device,
[HackRF's Windows instructions](https://github.com/greatscottgadgets/hackrf/blob/main/host/README.md)
recommend WinUSB. Download [Zadig](https://zadig.akeo.ie/), select **Options > List
All Devices**, select the **HackRF** USB device, and install/replace its driver
with **WinUSB**. Verify you selected the HackRF before applying the change.
Reconnect it and repeat the device check. Driver installation may require
administrator access. Driver setup is unnecessary for dry simulations and plots.

### Standard Python virtual environment (alternative)

From this directory on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe scripts/check_hackrf.py
```

Use `.\.venv\Scripts\python.exe` directly; activation is optional. Recreate the
environment if the project moves to another PC or its base Python installation
changes.

On Linux/macOS, use `python3 -m venv .venv`, then `.venv/bin/python` in place of the
Windows interpreter path. After activating an environment, `python` works in the
examples below. Installation also provides `wifi-check-hackrf`,
`wifi-inspect-waveform`, `wifi-transmit-single`, and `wifi-random-traffic` commands.

## Device check and waveform inspection

```powershell
.\.venv\Scripts\python.exe scripts/check_hackrf.py
.\.venv\Scripts\python.exe scripts/inspect_waveform.py --packet-count 5 --seed 123
```

The check verifies both executables, queries device information, and reports a
useful error if no HackRF is detected or the selected host tools cannot be used
reliably for short bursts. Older or unidentified host builds are rejected. It does
not transmit.

Inspection prints requested and actual packet durations, gap durations, sample
count, peak and RMS amplitudes. The plots show the envelope, two-sided PSD from
-10 to +10 MHz relative to the carrier, and a spectrogram with packet/gap timing.
Nothing is saved by default. For a headless plot check use `--no-show`; to explicitly
save a diagnostic image add `--save-plot diagnostic.png`. This saves no IQ samples.

## Dry simulation

```powershell
.\.venv\Scripts\python.exe scripts/random_traffic.py --channels 1-13 --duration 10 --seed 12345
.\.venv\Scripts\python.exe scripts/transmit_single.py --channel 6 --duration-ms 5 --gain 0
```

**Actual transmission requires `--transmit` on that command.** A configuration file
cannot silently enable RF. Dry runs never access the transmitter backend, and their
logs leave HackRF timestamps/results empty. They wait for scheduled idle intervals
and generated burst durations to approximate the experiment's elapsed duration.
`--duration 0` runs until Ctrl+C. A started burst finishes even if it crosses the
experiment deadline, so total runtime may exceed the requested duration.

## RF experiments

RF transmissions must comply with applicable regulations. Prefer a shielded setup,
conducted connections with suitable attenuators, or an RF enclosure. Availability
of channels 12 and 13 and allowed operating conditions vary by location. Use gain
0 initially; the RF amplifier and antenna-port power remain disabled.

```powershell
.\.venv\Scripts\python.exe scripts/transmit_single.py --channel 6 --duration-ms 5 --gain 0 --transmit
.\.venv\Scripts\python.exe scripts/random_traffic.py --channels 1-13 --duration 60 --traffic-mode poisson --lambda-events 10 --packet-count-min 1 --packet-count-max 8 --packet-us-min 100 --packet-us-max 1500 --gap-us-min 30 --gap-us-max 300 --gain 0 --seed 12345 --transmit
```

The single-burst script uses seed 0 unless specified, divides the requested length
into roughly 1.5 ms or shorter packets with 100 us gaps, and rounds each packet up
to an OFDM-symbol boundary. The actual burst can therefore be slightly longer than
`--duration-ms`. It logs both lengths.

Only one center frequency is transmitted at a time. Channels 1–13 map to
`2412000000 + (channel - 1) * 5000000` Hz. Their spectra intentionally overlap;
selection is not restricted to channels 1, 6, and 11. Each burst starts a new
`hackrf_transfer` invocation, which retunes to that event's center frequency.
The runner stops and preserves partial logs on a backend failure.

### Missing short bursts on a spectrum analyzer

A console `success` reports one completed host-tool invocation (one burst), not
RF delivery confirmation or an analyzer packet count. Multiple packets can be
inside that burst. Swept analyzers examine frequencies sequentially, so short
activity can be missed, partially displayed, or combined across trace updates.
Max Hold accumulates a spectrum; it does not count packets.

First check the host tools in the **same activated terminal** as your experiment:

```text
where.exe hackrf_transfer
where.exe hackrf_info
hackrf_info
```

The legacy PothosSDR build `git-7047252` has a concrete short-file problem with
the exact `-n` limit used by this backend: it returns from the callback before
submitting the last filled buffer. At 20 Msps its 262144-byte buffer represents
6.5536 ms of IQ. Bursts at or below that length can lose all intended IQ despite
a successful process exit; longer bursts lose their last buffer. Removing `-n`
alone does not reliably solve its short end-of-file behavior. The 2024.02.1 host
tools submit the final partial buffer and use a TX flush callback. The backend
therefore rejects legacy/unidentified tools before transmission rather than
silently padding or changing the experiment waveform. See the
[legacy callback](https://github.com/greatscottgadgets/hackrf/blob/7047252/host/hackrf-tools/src/hackrf_transfer.c#L423-L455)
and [2024.02.1 callback](https://github.com/greatscottgadgets/hackrf/blob/v2024.02.1/host/hackrf-tools/src/hackrf_transfer.c#L505-L521).

Use the supplied Conda environment for matching 2024.02.1 host tools and the
2024.02.1 firmware. If `where.exe` lists PothosSDR first, the
intended Conda tools are not taking precedence in that terminal. A working USB
driver alone does not establish host-tool compatibility. No driver replacement
or firmware update is required merely to select the correct host executable.

After checking the tools, hold the transmitter on channel 6 and use a 2437 MHz
analyzer center with about 30 MHz span. Start with Peak detection and Max Hold;
300 kHz RBW and VBW at least as wide are useful diagnostic starting settings.
For a longer observable packet train, use the included diagnostic preset:

```text
python scripts/random_traffic.py --config config.analyzer.yaml --transmit --verbose
```

Each generated burst is 54.9 ms: fifty 1 ms packets with forty-nine 100 us gaps.
There is a 100 ms requested idle interval plus generation/process/USB overhead
between invocations. A longer visible burst does not by itself prove all shorter
bursts were transmitted. Packet counting needs time-domain capture or appropriate
triggering; narrow-RBW zero span observes only a spectral slice, not total
wideband channel power. See [Keysight's explanation of swept versus time-record
analysis](https://helpfiles.keysight.com/csg/89600B/Webhelp/Subsystems/concepts/content/concepts_vsa_measadvan.htm).

## Configuration and repeatability

```powershell
.\.venv\Scripts\python.exe scripts/random_traffic.py --config config.example.yaml --duration 20 --seed 42
.\.venv\Scripts\python.exe scripts/random_traffic.py --channels 1,6,11 --channel-mode weighted --channel-weights 1:0.3,6:0.4,11:0.3 --duration 20 --seed 42
```

Command-line values override YAML/JSON. Run `--help` for all parameters. Configuration
validation rejects unknown keys, nonfinite numbers, unsupported PHY settings,
invalid channels/gains, and impossible burst limits. `--burst-ms-min` and
`--burst-ms-max` constrain the actual generated burst including gaps and symbol
rounding. Packet counts, durations and gaps are sampled conditionally to fit those
limits; tightening them changes the marginal distributions.
Selection is uniform over feasible packet counts, followed by feasible total
packet-symbol and gap-sample budgets and bounded random splits. Counts that cannot
fit the burst bounds are excluded without truncating packets.
To limit accidental memory allocations, bursts are limited to 1,000 ms and
10,000 packets; the single-burst CLI accepts up to 500 ms. Large bursts can still
need substantial temporary memory for FFTs and normalization.

`random` mode draws uniform idle intervals. `poisson` mode draws exponential idle
intervals with mean `1 / lambda_events_per_second`. This is a sequential,
Poisson-like arrival model: idle starts after the previous event completes. Process
overhead and burst duration reduce the achieved event rate below the configured
lambda; there is no overlapping RF, queued-arrival catch-up, or strict wall-clock
Poisson process.

Seeded NumPy generators reproduce scheduler parameters. Each event has an independent
logged `waveform_seed`, so waveform generation cannot change subsequent scheduling.
When no experiment seed is supplied, a generated seed is saved and printed. Reuse
the same software version and configuration for reproducibility. Host timing and
the number of events fitting a wall-clock duration can differ between runs.

## Waveform details

- Sample rate: 20 MHz; FFT: 64; cyclic prefix: 16; total symbol: 80 samples / 4 us.
- Occupied signed carrier indices: -26 through -1 and +1 through +26.
- Four pilots at -21, -7, +7, +21; 48 QPSK data carriers; DC and guards unused.
- Signed mathematical carrier `k` is stored at NumPy bin `k % 64` before `ifft`.
  The time waveform needs no `fftshift`; plots shift FFT output and frequency axes
  together. Positive indices generate positive-frequency complex sinusoids.
- Unit-power QPSK mapping: 00 -> (+1+j)/sqrt(2), 01 -> (-1+j)/sqrt(2),
  11 -> (-1-j)/sqrt(2), 10 -> (+1-j)/sqrt(2).
- A repeatable 16 us preamble-like sequence precedes data. It is not STF/LTF/SIGNAL.
  Each packet has at least one data symbol and a 20 us minimum generated length.
- Packet durations round up to 4 us; gap durations round to the nearest sample.
  Gaps contain exact complex zeros. A configurable raised-cosine ramp defaults to
  3 us at both ends of every packet and is capped at half the packet length.
- The outer active carrier centers are at +/-8.125 MHz. The occupied energy is
  approximately 16–18 MHz wide; the exact IEEE spectral mask is not implemented.
  DC is zero in the symbol grid; finite packets and ramps can still cause spectral
  leakage near DC and the guards.
- Each whole burst is peak-normalized then scaled by `--digital-amplitude` (0.6
  by default), rounded and interleaved as signed int8 `I0,Q0,I1,Q1,...`. Peak
  normalization prevents clipping; burst RMS power can vary with its peak factor.

## Ground truth and timing

Each run creates:

```text
results/experiment_YYYYMMDD_HHMMSS_microseconds/
    config.json
    ground_truth.csv
    events.jsonl
    summary.json
```

`config.json` contains the resolved configuration, seed, and program version.
`events.jsonl` durably records each requested event before generation/transmission,
then its terminal state. `ground_truth.csv` contains one terminal row per event.
List fields are JSON-encoded. Failed or interrupted events retain their requested
parameters even if generation never completed. A request without a terminal journal
entry after a hard crash indicates an unresolved event, not successful transmission.

Records include channel/frequency, requested timestamp, host subprocess start/end
UTC timestamps, monotonic elapsed duration and scheduling lateness; requested/actual
packet and gap durations; sample counts; PHY, gain and seed parameters; and process
return code/status. No persistent waveform path is logged.

Timing **within one burst** is deterministic at sample resolution. Timing **between
invocations** includes startup, retuning, USB buffering and host scheduling. The
timestamps do not measure RF start/stop. A zero process return code means the host
tool reported success; it does not prove every sample reached the air. Very short
bursts and installed host-tool/firmware versions should be checked against a receiver
before treating logs as an RF delivery reference.

Summaries report per-channel requested and generated packet airtime, whole waveform
duration including gaps, successful packet airtime, counts and means. Packet duty
cycle excludes gaps and subprocess overhead and is an estimate based on successful
host invocations divided by experiment elapsed time. Dry runs have no successful
RF airtime and expose a separate simulated duty cycle. Totals attribute activity to
the selected center channel; they do not infer occupancy of neighboring channels.

Temporary files are closed before the child opens them on Windows. Normal exits,
backend errors, timeouts, Ctrl+C, SIGTERM and Ctrl+Break unwind cleanup. The child
is stopped/reaped before IQ deletion. No process can guarantee cleanup after power
loss, forced OS termination or an inaccessible filesystem; such failures can leave
an orphan `wifi_hackrf_*` directory in the OS temporary directory. No cache is built
or reused.

## Tests and extension points

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.test-tmp
```

Tests use mocked hardware and cover channel/carrier placement, frequency orientation,
QPSK/CP, packet/burst timing, normalization, reproducibility, weighted selection,
configuration, dry-run isolation, log semantics, and temporary-file cleanup.

`traffic.py` and `models.py` describe hardware-independent events. `packet.py`
provides the replaceable preamble/packet generator; `burst.py` inserts sample-timed
gaps. `hackrf.py` implements the `TransmitterBackend` contract. `runner.py` coordinates
them, and `logger.py` retains experiment records. These boundaries allow future real
Wi-Fi PHYs, alternate modulation, streaming backends, richer traffic models and
multi-radio orchestration without coupling the scheduler to HackRF.

Official references: [HackRF host tools](https://hackrf.readthedocs.io/en/latest/hackrf_tools.html)
and [hackrf_transfer implementation](https://github.com/greatscottgadgets/hackrf/blob/master/host/hackrf-tools/src/hackrf_transfer.c).
