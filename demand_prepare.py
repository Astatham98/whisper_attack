"""
Data preparation for DEMAND dataset using audio from JacobLinCool/VoiceBank-DEMAND-16k
and transcriptions from VCTK.

The DEMAND dataset has IDs in format "p226-001" (speaker-utterance) and contains
noisy audio. This script:
  - Loads VCTK to build a transcription lookup map
  - Loads JacobLinCool/VoiceBank-DEMAND-16k for noisy audio
  - Combines them by parsing DEMAND IDs to extract speaker and utterance

Produces a CSV index compatible with robust_speech.data.dataio.dataio_prepare.
CSV columns
-----------
ID        : unique utterance identifier  (e.g. p226-001)
duration  : duration in seconds          (float as string)
wav       : absolute path to 16 kHz WAV  (written under <data_folder>/audio/)
spk_id    : speaker identifier           (e.g. p226)
wrd       : transcript from VCTK

Usage
-----
Called automatically by run_attack.py via the `dataset_prepare_fct` key in
attack configs. Can also be run standalone:

    python demand_prepare.py \
        --data_folder /path/to/DEMAND \
        --save_folder /path/to/DEMAND/csv \
        --split demand-100
"""

import argparse
import csv
import logging
import os

import torchaudio

logger = logging.getLogger(__name__)

SAMPLERATE = 16000
NUM_SAMPLES = 100
HF_DATASET_ID = "JacobLinCool/VoiceBank-DEMAND-16k"


# ---------------------------------------------------------------------------
# Public entry point (called by run_attack.py / fit_attacker.py)
# ---------------------------------------------------------------------------

def prepare_demand(
    data_folder,
    te_splits,
    save_folder,
    skip_prep=False,
    num_samples=NUM_SAMPLES,
    hf_dataset=HF_DATASET_ID,
    sample_seed=None,
    vctk_csv_path=None,
):
    """Prepare DEMAND data for adversarial attack evaluation.

    Downloads up to `num_samples` audio files from JacobLinCool/VoiceBank-DEMAND-16k
    (using the noisy audio), loads transcriptions from VCTK or a VCTK CSV file,
    combines them by parsing DEMAND IDs (format: speaker-utterance_id), saves 16 kHz
    mono WAV files under ``<data_folder>/audio/``, and writes a CSV index for each
    split name listed in `te_splits`.

    Arguments
    ---------
    data_folder : str
        Root directory where audio files will be stored.
    te_splits : list[str] or str
        Test split names (e.g. ``["demand-100"]``).
    save_folder : str
        Directory where CSV files are written.
    skip_prep : bool
        When *True*, skip preparation if all expected CSV files already exist.
    num_samples : int
        Maximum number of utterances to include (default 100).
    hf_dataset : str
        HuggingFace dataset ID for noisy audio
        (default ``"JacobLinCool/VoiceBank-DEMAND-16k"``).
    sample_seed : int | None
        Optional seed used to randomize the sampled subset.
    vctk_csv_path : str | None
        Optional path to a pre-prepared VCTK CSV file. If provided, uses this instead
        of loading VCTK from HuggingFace. The CSV can contain either:
          - columns: ID,wrd (old format)
          - columns: speaker_id,text_id,text (new vctk_ids.csv format)
    """
    os.makedirs(save_folder, exist_ok=True)

    if isinstance(te_splits, str):
        split_names = [te_splits]
    else:
        split_names = list(te_splits)

    possible_samples = split_names[0].split("-")[-1]
    if possible_samples != "100" and possible_samples.isdigit():
        num_samples = int(possible_samples)
        logger.info(
            "Overriding num_samples to %d based on te_splits argument.",
            num_samples,
        )
    elif possible_samples != "100" and not possible_samples.isdigit() and possible_samples == "all":
        num_samples = None
        logger.info(
            "Overriding num_samples to None (use all samples) based on te_splits argument."
        )
    else:
        logger.info(
            "Using default num_samples=%s since te_splits argument does not specify a sample count.",
            str(num_samples),
        )

    if skip_prep and _all_csvs_exist(split_names, save_folder):
        logger.info("Skipping DEMAND preparation — CSV files already present.")
        return

    audio_dir = os.path.join(data_folder, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    existing_rows = _load_existing_rows(split_names, save_folder, num_samples)
    if existing_rows is not None:
        logger.info(
            "Reusing %d existing DEMAND samples from local CSV/audio cache.",
            len(existing_rows),
        )
        for split in split_names:
            csv_path = os.path.join(save_folder, split + ".csv")
            if not os.path.isfile(csv_path):
                _write_csv(csv_path, existing_rows)
                logger.info(
                    "DEMAND CSV written to %s (%d rows).",
                    csv_path,
                    len(existing_rows),
                )
        return

    rows = _download_demand_samples(
        audio_dir,
        num_samples,
        hf_dataset,
        sample_seed=sample_seed,
        vctk_csv_path=vctk_csv_path,
    )

    for split in split_names:
        csv_path = os.path.join(save_folder, split + ".csv")
        _write_csv(csv_path, rows)
        logger.info(
            "DEMAND CSV written to %s (%d rows).", csv_path, len(rows)
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _all_csvs_exist(splits, save_folder):
    return all(
        os.path.isfile(os.path.join(save_folder, s + ".csv"))
        for s in splits
    )


def _load_existing_rows(splits, save_folder, num_samples):
    for split in splits:
        csv_path = os.path.join(save_folder, split + ".csv")
        rows = _read_csv_rows(csv_path)
        if rows is None:
            continue

        if num_samples is not None:
            rows = rows[:num_samples]
        if num_samples is not None and len(rows) < num_samples:
            logger.info(
                "Existing DEMAND CSV %s has only %d rows; downloading more data.",
                csv_path,
                len(rows),
            )
            continue

        missing_wavs = [row["wav"] for row in rows if not os.path.isfile(row["wav"])]
        if missing_wavs:
            logger.info(
                "Existing DEMAND CSV %s references %d missing WAV files; downloading data again.",
                csv_path,
                len(missing_wavs),
            )
            continue

        return rows

    return None


def _read_csv_rows(csv_path):
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_fields = ["ID", "duration", "wav", "spk_id", "wrd"]
        if reader.fieldnames != expected_fields:
            logger.info(
                "Ignoring cached DEMAND CSV %s because its columns do not match the expected format.",
                csv_path,
            )
            return None
        return list(reader)


def _build_vctk_transcription_map(vctk_csv_path=None):
    """Load VCTK dataset and build a transcription lookup map.

    Returns a nested dict: {speaker_id: {utterance_key: transcription}}
    where utterance_key is formatted as "speaker_id_utterance_id" (e.g., "p226_001").

    Arguments
    ---------
    vctk_csv_path : str | None
        If provided, reads VCTK data from this CSV file instead of downloading.
        The CSV can be either:
          - old format: ID, duration, wav, spk_id, wrd
          - new format: speaker_id, text_id, transcription
    """
    from datasets import load_dataset

    vctk_map = {}

    if vctk_csv_path is not None:
        logger.info("Loading VCTK transcriptions from CSV: %s", vctk_csv_path)
        with open(vctk_csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = [h.strip().lower() for h in reader.fieldnames or []]

            speaker_field = next((f for f in header if f in ["speaker_id", "spk_id", "speaker", "spk"]), None)
            text_id_field = next((f for f in header if f in ["text_id", "utterance_id", "utterance", "file_id", "id", "utt_id", "fileid"]), None)
            transcription_field = next((f for f in header if f in ["transcription", "transcript", "text", "wrd", "sentence", "label"]), None)

            use_legacy_id_wrd = "id" in header and "wrd" in header and transcription_field == "wrd"
            if not use_legacy_id_wrd and (speaker_field is None or text_id_field is None or transcription_field is None):
                raise ValueError(
                    "vctk_csv_path must contain speaker_id/text_id/transcription fields or ID/wrd columns. "
                    f"Found header: {header}"
                )

            for row in reader:
                row_lower = {k.strip().lower(): (v or "") for k, v in row.items() if k is not None}

                if use_legacy_id_wrd:
                    file_id = row_lower.get("id", "").strip()
                    text = row_lower.get("wrd", "").strip()
                    if not file_id or not text:
                        continue
                    parts = file_id.rsplit("_", 1)
                    if len(parts) != 2:
                        logger.debug("Skipping VCTK file with unexpected format: %s", file_id)
                        continue
                    spk_id = parts[0]
                else:
                    speaker = row_lower.get(speaker_field, "").strip()
                    text_id = row_lower.get(text_id_field, "").strip()
                    text = row_lower.get(transcription_field, "").strip()
                    if not speaker or not text_id or not text:
                        continue
                    spk_id = speaker
                    file_id = f"{speaker}_{text_id}"

                if spk_id not in vctk_map:
                    vctk_map[spk_id] = {}
                vctk_map[spk_id][file_id] = text

        logger.info(
            "Loaded VCTK transcriptions from CSV: %d speakers, ~%d total utterances.",
            len(vctk_map),
            sum(len(utts) for utts in vctk_map.values()),
        )
    else:
        logger.info("Loading VCTK transcriptions from HuggingFace...")
        vctk_ds = load_dataset(
            "CSTR-Edinburgh/vctk",
            split="train",
            streaming=False,
            trust_remote_code=True,
        )

        for sample in vctk_ds:
            file_id = sample.get("file_id")
            text = sample.get("text", "").strip()

            if not file_id or not text:
                continue

            parts = file_id.rsplit("_", 1)
            if len(parts) != 2:
                logger.debug("Skipping VCTK file with unexpected format: %s", file_id)
                continue

            spk_id = parts[0]
            if spk_id not in vctk_map:
                vctk_map[spk_id] = {}

            vctk_map[spk_id][file_id] = text

        logger.info(
            "Loaded VCTK transcriptions from HuggingFace: %d speakers, ~%d total utterances.",
            len(vctk_map),
            sum(len(utts) for utts in vctk_map.values()),
        )

    return vctk_map


def _download_demand_samples(audio_dir, num_samples, hf_dataset, sample_seed=None, vctk_csv_path=None):
    """Download DEMAND noisy audio and combine with VCTK transcriptions.

    Loads VCTK (from CSV or HuggingFace) to build a transcription lookup, then
    iterates through JacobLinCool/VoiceBank-DEMAND-16k to get audio from the
    'noisy' column. DEMAND IDs are parsed (e.g., "p226-001") to extract speaker
    and utterance IDs, then transcriptions are looked up from VCTK.

    Returns
    -------
    list[dict]
        One dict per accepted utterance with keys:
        ``ID``, ``duration``, ``wav``, ``spk_id``, ``wrd``.
    """
    import numpy as np
    import torch
    from datasets import load_dataset

    logger.info("Building VCTK transcription lookup...")
    vctk_map = _build_vctk_transcription_map(vctk_csv_path=vctk_csv_path)
    logger.info("Built VCTK map with %d speaker(s).", len(vctk_map))

    if sample_seed is None:
        logger.info(
            "Downloading and randomly sampling %s samples from %s.",
            "all" if num_samples is None else num_samples,
            hf_dataset,
        )
    else:
        logger.info(
            "Downloading and randomly sampling %s samples from %s with seed %s.",
            "all" if num_samples is None else num_samples,
            hf_dataset,
            sample_seed,
        )

    ds = load_dataset(
        hf_dataset,
        split="train",
        streaming=False,
        trust_remote_code=True,
    )
    if num_samples is not None:
        ds = ds.shuffle(seed=sample_seed)

    rows = []
    saved = 0
    skipped_not_in_map = 0
    for sample in ds:
        if num_samples is not None and saved >= num_samples:
            break

        # Parse DEMAND ID (format: "p226-001" -> speaker "p226", utterance "001")
        demand_id = (
            sample.get("id")
            or sample.get("ID")
            or sample.get("file_id")
            or sample.get("name")
            or sample.get("filename")
            or ""
        )
        demand_id = str(demand_id).strip()
        parts = demand_id.rsplit("-", 1)
        if len(parts) != 2:
            parts = demand_id.rsplit("_", 1)
        if len(parts) != 2 or not demand_id:
            logger.warning("Skipping sample with malformed ID: %s", demand_id)
            continue

        spk_id, utt_id = parts
        vctk_utt_key = f"{spk_id}_{utt_id}"  # Convert to VCTK format: "p226_001"

        # Look up transcription from VCTK map
        if spk_id not in vctk_map or vctk_utt_key not in vctk_map[spk_id]:
            logger.debug(
                "VCTK transcription not found for %s (looking for %s in speaker %s).",
                demand_id,
                vctk_utt_key,
                spk_id,
            )
            skipped_not_in_map += 1
            continue

        wrd = vctk_map[spk_id][vctk_utt_key]

        # Get audio from noisy or fallback audio fields
        audio_sample = sample.get("noisy") or sample.get("audio") or sample.get("speech") or sample.get("noisy_audio")
        if audio_sample is None:
            logger.warning("Skipping sample %s because it has no audio field.", demand_id)
            continue

        if isinstance(audio_sample, dict):
            if "array" in audio_sample and "sampling_rate" in audio_sample:
                array = audio_sample["array"]
                src_sr = audio_sample["sampling_rate"]
            elif "path" in audio_sample:
                wav_path = audio_sample["path"]
                if not os.path.isfile(wav_path):
                    logger.warning("Skipping sample %s because audio path does not exist: %s", demand_id, wav_path)
                    continue
                array, src_sr = torchaudio.load(wav_path)
                array = array.squeeze(0).numpy() if array.ndim > 1 else array.numpy()
            else:
                logger.warning("Skipping sample %s because its audio field has no usable content.", demand_id)
                continue
        else:
            logger.warning("Skipping sample %s because audio field is not a dict-like object.", demand_id)
            continue

        # Save 16 kHz WAV
        wav_path = os.path.join(audio_dir, demand_id + ".wav")
        if not os.path.exists(wav_path):
            tensor = torch.from_numpy(np.array(array, dtype="float32")).unsqueeze(0)
            if src_sr != SAMPLERATE:
                tensor = torchaudio.transforms.Resample(src_sr, SAMPLERATE)(tensor)
            torchaudio.save(wav_path, tensor, SAMPLERATE)

        duration = _get_duration(wav_path)
        rows.append(
            {
                "ID": demand_id,
                "duration": f"{duration:.4f}",
                "wav": wav_path,
                "spk_id": str(spk_id),
                "wrd": wrd,
            }
        )
        saved += 1

    if skipped_not_in_map > 0:
        logger.info(
            "Skipped %d DEMAND samples because transcriptions were not found in VCTK map.",
            skipped_not_in_map,
        )

    logger.info("Collected %d DEMAND samples.", len(rows))
    return rows


def _get_duration(wav_path):
    info = torchaudio.info(wav_path)
    return info.num_frames / info.sample_rate


def _write_csv(csv_path, rows):
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ID", "duration", "wav", "spk_id", "wrd"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Prepare a 100-sample DEMAND CSV for whisper_attack."
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        help="Root folder where audio/ sub-directory will be created.",
    )
    parser.add_argument(
        "--save_folder",
        required=True,
        help="Directory where the CSV file(s) are written.",
    )
    parser.add_argument(
        "--split",
        default="demand-100",
        help="Name of the split (determines CSV filename). Default: demand-100.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of utterances to download. Default: 100.",
    )
    parser.add_argument(
        "--hf_dataset",
        default=HF_DATASET_ID,
        help=f"HuggingFace dataset ID. Default: {HF_DATASET_ID}.",
    )
    parser.add_argument(
        "--vctk_csv_path",
        default=None,
        help="Optional path to a VCTK IDs CSV file containing speaker_id,text_id,transcription. If provided, uses this instead of loading VCTK from HuggingFace.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Optional seed for randomized subset selection.",
    )
    parser.add_argument(
        "--skip_prep",
        action="store_true",
        help="Skip preparation if CSV already exists.",
    )
    args = parser.parse_args()

    prepare_demand(
        data_folder=args.data_folder,
        te_splits=[args.split],
        save_folder=args.save_folder,
        skip_prep=args.skip_prep,
        num_samples=args.num_samples,
        hf_dataset=args.hf_dataset,
        sample_seed=args.sample_seed,
        vctk_csv_path=args.vctk_csv_path,
    )
