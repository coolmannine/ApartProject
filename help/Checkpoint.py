# Cell 4 — Find & filter checkpoints on HF
STEP_RX = re.compile(rf"{CKPT_DIR}/checkpoint_(\d+)\.pth$")
all_files = list_repo_files(REPO_ID, repo_type="model")
pairs = [(int(m.group(1)),f) for f in all_files if (m:=STEP_RX.match(f))]
ckpts = sorted(
    (s,f) for s,f in pairs
    if s%KEEP_EVERY==0 and RANGE_START<=s<=RANGE_END
)
print("Using steps:",[s for s,_ in ckpts])
