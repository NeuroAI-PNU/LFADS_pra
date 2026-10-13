from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# LFADS 결과 파일
run_dir = Path(
    "/home/changhyun/LFADS_pra/lfads-torch/runs/"
    "lfads-torch-example/ibl_m1/261012_162141_ibl_m1"
)

h5_files = list(run_dir.glob("lfads_output_*.h5"))
if len(h5_files) != 1:
    raise RuntimeError(f"LFADS 출력 파일을 확인하세요: {h5_files}")

with h5py.File(h5_files[0], "r") as f:
    print("Available keys:", list(f.keys()))
    factors = f["valid_factors"][:]
    edges = f["bin_edges_s"][:]

print("Factors shape:", factors.shape)

# Trial × Time × Factors
n_trials, n_bins, n_factors = factors.shape

if n_factors < 3:
    raise ValueError("3D PCA에는 최소 3개의 factor가 필요합니다.")

# 모든 Validation trial과 시간점으로 PCA 학습
pca = PCA(n_components=3)

Z = pca.fit_transform(
    factors.reshape(-1, n_factors)
).reshape(n_trials, n_bins, 3)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Trial 평균 trajectory
mean_trajectory = Z.mean(axis=0)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# 개별 Trial 5개
for i in range(min(5, n_trials)):
    ax.plot(
        Z[i, :, 0],
        Z[i, :, 1],
        Z[i, :, 2],
        alpha=0.35,
        linewidth=1,
    )

# Trial 평균 trajectory
ax.plot(
    mean_trajectory[:, 0],
    mean_trajectory[:, 1],
    mean_trajectory[:, 2],
    color="black",
    linewidth=3,
    label="Mean trajectory",
)

# First movement onset (t=0)
time = (edges[:-1] + edges[1:]) / 2
movement_idx = np.argmin(np.abs(time))

ax.scatter(
    *mean_trajectory[movement_idx],
    color="red",
    s=70,
    label="Movement onset",
)

# 평균 trajectory의 시작점: t = -0.5 s
ax.scatter(
    *mean_trajectory[0],
    color="blue",
    s=90,
    label="Start (-0.5 s)"
)

# 평균 trajectory의 끝점: t = +1.0 s 부근
ax.scatter(
    *mean_trajectory[-1],
    color="green",
    s=90,
    label="End (+1.0 s)"
)

ax.legend()

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("IBL M1 LFADS Latent Trajectory")
ax.legend()

plt.tight_layout()

out_path = run_dir / "m1_lfads_pca_3d.png"
plt.savefig(out_path, dpi=200)
plt.show()

print("Saved:", out_path)


from scipy.ndimage import gaussian_filter1d

# =============================
# 1. State-space speed 계산
# =============================

dt = np.mean(np.diff(time))

# Trial별 상태 이동 속도
trial_speed = np.linalg.norm(
    np.diff(Z, axis=1), axis=2
) / dt

# 개별 Trial 속도의 평균
mean_trial_speed = trial_speed.mean(axis=0)

# 평균 trajectory 자체의 이동 속도
mean_speed = np.linalg.norm(
    np.diff(mean_trajectory, axis=0), axis=1
) / dt

speed_time = (time[:-1] + time[1:]) / 2

# =============================
# 2. 실제 Spike 기반 Firing rate
# =============================

input_h5 = (
    "/home/changhyun/LFADS_pra/lfads_data/"
    "m1_0ffa052a-fe3f-46c2-999d-43bd56b7826e.h5"
)

with h5py.File(input_h5, "r") as f:
    spike_counts = f["valid_encod_data"][:]

# Trial별 Spike count를 Hz로 변환
firing_rates = spike_counts / dt

# Trial 평균: Time × Neuron
mean_firing = firing_rates.mean(axis=0)

# 시각화용 평활화 (Gaussian, sigma=30 ms)
smooth_firing = gaussian_filter1d(
    mean_firing, sigma=3, axis=0
)

# =============================
# 3. 결과 시각화
# =============================

fig, axes = plt.subplots(
    2, 1, figsize=(11, 8), sharex=True
)

# A. 잠재 상태 이동 속도
axes[0].plot(
    speed_time,
    mean_trial_speed,
    label="Mean trial state speed",
    alpha=0.65
)

axes[0].plot(
    speed_time,
    mean_speed,
    color="black",
    linewidth=2,
    label="Mean trajectory speed"
)

axes[0].axvline(
    0, color="red", linestyle="--"
)

axes[0].set_ylabel("State-space speed (PC units/s)")
axes[0].set_title("LFADS State-space Dynamics")
axes[0].legend()

# B. 뉴런별 firing rate
for neuron in range(smooth_firing.shape[1]):
    axes[1].plot(
        time,
        smooth_firing[:, neuron],
        label=f"Neuron {neuron+1}"
    )

axes[1].axvline(
    0, color="red", linestyle="--"
)

axes[1].set_ylabel("Firing rate (Hz)")
axes[1].set_xlabel("Time from movement onset (s)")
axes[1].set_title("Observed M1 Firing Rates")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    run_dir / "m1_speed_vs_firing.png",
    dpi=200
)

plt.show()



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    mean_trajectory[:, 0],
    mean_trajectory[:, 1],
    mean_trajectory[:, 2],
    color="gray",
    alpha=0.5
)

points = ax.scatter(
    mean_trajectory[:, 0],
    mean_trajectory[:, 1],
    mean_trajectory[:, 2],
    c=time,
    cmap="coolwarm",
    s=18
)

plt.colorbar(
    points,
    ax=ax,
    label="Time from movement onset (s)",
    shrink=0.65
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("LFADS Trajectory Colored by Time")

plt.savefig(
    run_dir / "m1_trajectory_time_color.png",
    dpi=200
)

plt.show()
