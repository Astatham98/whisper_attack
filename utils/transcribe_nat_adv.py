import argparse
import whisper
import torch
from jiwer import wer
from whisper.audio import load_audio
from whisper.normalizers import EnglishTextNormalizer
import pandas as pd
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def wers(
    attack_path="whisper-tiny.en-40/420",
    model="tiny.en",
    attack="pgd",
    base_path=None,
    csv_path=None,
):
    base_path = Path(base_path) if base_path is not None else PROJECT_ROOT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vctk_files = [
        "vctk_00{}".format(i) if i > 9 else "vctk_000{}".format(i) for i in range(100)
    ]
    save_dir = base_path / "data" / "attacks" / attack / attack_path / "save"
    saved_audio_files = list(save_dir.iterdir())
    if len(saved_audio_files) < 200:
        vctk_files = sorted(
            f"vctk_{audio_file.stem.split('_')[1]}"
            for audio_file in saved_audio_files
            if audio_file.name.endswith("_adv.wav")
        )

    whisper_model = whisper.load_model(model).to(device)

    if csv_path is None:
        csv_path = base_path / "data" / "data" / "VCTK" / "csv" / "vctk-100.csv"
    else:
        csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    wers = {}
    clean_texts = {}
    adv_texts = {}
    for vctk_file in vctk_files:
        audio_path_adv = save_dir / f"{vctk_file}_adv.wav"
        audio_path_clean = save_dir / f"{vctk_file}_nat.wav"

        transcription_real = df.loc[df["ID"] == vctk_file]["wrd"].values[0]
        transcription_real = EnglishTextNormalizer()(transcription_real)
        inner_wers = {}
        adv_transcription_text = ""

        for label, audio_path in (("clean", audio_path_clean), ("adv", audio_path_adv)):
            audio = load_audio(str(audio_path))
            audio = torch.from_numpy(audio).to(device)
            transcription = whisper_model.transcribe(audio)
            transcription_text = EnglishTextNormalizer()(transcription["text"])
            if label == "clean":
                inner_wers["clean"] = wer(transcription_real, transcription_text)
            else:
                inner_wers["adv"] = wer(transcription_real, transcription_text)
                adv_transcription_text = transcription_text

        clean_texts[f"{vctk_file}_nat.wav"] = transcription_real
        adv_texts[f"{vctk_file}_adv.wav"] = adv_transcription_text
        wers[vctk_file] = inner_wers

    return wers, clean_texts, adv_texts


def run_evaluation(
    model="tiny.en", attack_path=None, attack="pgd", base_path=None, csv_path=None
):
    base_path = Path(base_path) if base_path is not None else PROJECT_ROOT
    if attack_path is None:
        attack_path = f"whisper-{model}-40/420"

    # Recursively evaluate if model is a list of models
    if type(model) == list:
        for m in model:
            run_evaluation(
                model=m,
                attack_path=attack_path,
                attack=attack,
                base_path=base_path,
                csv_path=csv_path,
            )
        return

    wers_dict, clean_texts, adv_texts = wers(
        attack_path=attack_path,
        model=model,
        attack=attack,
        base_path=base_path,
        csv_path=csv_path,
    )
    average_clean_wer = sum(item["clean"] for item in wers_dict.values()) / len(
        wers_dict
    )
    average_adv_wer = sum(item["adv"] for item in wers_dict.values()) / len(wers_dict)

    log_path = Path(f"{attack}_results_{model}.txt")
    with open(log_path, "w") as f:
        f.write(f"Average Clean WER: {average_clean_wer:.4f}\n")
        f.write(f"Average Adversarial WER: {average_adv_wer:.4f}\n")

    save_dir = base_path / "data" / "attacks" / attack / attack_path / "save"

    name_clean = Path(
        save_dir / f"transcriptions_clean_w-{model.replace('.en', '')}.json"
    )
    name_adv = Path(
        save_dir / f"transcriptions_noisy_w-{model.replace('.en', '')}.json"
    )
    with open(name_clean, "w") as f:
        json.dump(clean_texts, f, indent=4, ensure_ascii=False)
    with open(name_adv, "w") as f:
        json.dump(adv_texts, f, indent=4, ensure_ascii=False)

    return average_clean_wer, average_adv_wer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument(
        "--attack-path", "--attack_path", dest="attack_path", default=None
    )
    parser.add_argument("--attack", default="pgd")
    parser.add_argument("--base-path", "--base_path", dest="base_path", default=None)
    parser.add_argument(
        "--csv-path",
        "--csv_path",
        dest="csv_path",
        default=None,
        help="Path to the CSV file containing the VCTK transcriptions. If not provided, it will default to data/data/VCTK/csv/vctk-100.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    average_clean_wer, average_adv_wer = run_evaluation(
        model=args.model,
        attack_path=args.attack_path,
        attack=args.attack,
        base_path=args.base_path,
        csv_path=args.csv_path,
    )
    print(f"Average Clean WER: {average_clean_wer:.4f}")
    print(f"Average Adversarial WER: {average_adv_wer:.4f}")
