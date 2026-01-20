import json
import random
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    source_dir = Path(sys.argv[1])
    target_dir = Path(sys.argv[2])
    num_samples = int(sys.argv[3])

    source_audio_dir = source_dir / "source_audio"
    source_transcript_dir = source_dir / "transcripts"

    target_audio_dir = target_dir / "source_audio"
    target_transcript_dir = target_dir / "transcripts"

    files = [x.stem for x in source_audio_dir.glob("*.wav")]

    for _ in tqdm(range(num_samples), total=num_samples):
        basename = random.choice(files)
        shutil.copy(
            source_audio_dir / f"{basename}.wav", target_audio_dir / f"{basename}.wav"
        )

        with (
            open(
                source_transcript_dir / f"{basename}.json", "r", encoding="utf-8"
            ) as src_f,
            open(
                target_transcript_dir / f"{basename}.json", "w", encoding="utf-8"
            ) as target_f,
        ):
            data = json.load(src_f)
            json.dump(data, target_f, indent=4, ensure_ascii=False)
