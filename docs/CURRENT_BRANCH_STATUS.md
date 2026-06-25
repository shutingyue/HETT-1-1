# Current Branch Status

This branch is a backup of the old research code before clean migration to official HETT baseline.

Known status:
- Branch contains RegionPrompt / StopContrast / Conversion-Aware StopContrast experiments.
- This branch is NOT official-baseline equivalent.
- Official progress-based stop was disabled/commented in this branch.
- rollout state update order differs from upstream HETT.
- train.sh defaults differ from upstream HETT.
- This branch should be used as a code backup and migration source only, not as clean baseline.

Planned next step:
- Re-clone https://github.com/crotonyl/HETT.git into a clean directory.
- Reproduce official baseline.
- Migrate RegionPrompt, region diversity, StopContrast / Conversion-Aware StopContrast into the clean baseline with all new flags disabled by default.
