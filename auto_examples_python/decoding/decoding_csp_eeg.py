"""
.. _ex-decoding-csp-eeg:

===========================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
===========================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP. A
classifier is then applied to features extracted on CSP-filtered signals.

See https://en.wikipedia.org/wiki/Common_spatial_pattern and
:footcite:`Koles1991`. The EEGBCI dataset is documented in
:footcite:`SchalkEtAl2004` and is available at PhysioNet
:footcite:`GoldbergerEtAl2000`.
"""
# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1.0, 4.0
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
print("=================================================")
print("raw_fnames: ") # 
print(raw_fnames)
print("raw_fnames info: ")
print(raw.info)

# data
data = raw.get_data()
print("data: ")
print(data.shape)
print(data)

# Montages은 기존 EEG / MEG 데이터에 할당할 수 있는 3D 센서 위치 (미터의 x, y, z)를 포함한다. 
# 뇌에 상대적인 센서의 위치를 ​​지정함으로써 Montages 순방향 솔루션과 역 추정값을 계산하는 데 중요한 역할을 한다.
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage("standard_1005") 
raw.set_montage(montage)
raw.annotations.rename(dict(T1="hands", T2="feet"))

raw.plot()

# eeg 신호는 전극간의 전위 차이를 측정하는데, 이 함수를 통해 기준을 명시할 수 있음
# projection operator: 데이터를 바꾸지 않고, 특정 변환을 나타내는 형렬을 미리 계산해 두는 방식. 
# 이 연산자는 필요 시 데이터를 변경하는 데 사용되지만, 원시 데이터 자체는 변하지 않으므로 데이터의 원본 상태를 유지할 수 있음
raw.set_eeg_reference(projection=True) 

# Apply band-pass filter
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

# 필요한 유형의 채널을 선택하는 데 사용되는 함수
# return: 선택된 채널들의 인덱스 목록 반환
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads") 

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs( # Epochs 객체 생성
    raw,
    event_id=["hands", "feet"],
    tmin=tmin,      # 각 이벤트의 시간 범위 설정 (-1초 ~ 4초)
    tmax=tmax,
    proj=True,      # 프로젝현을 적용할지 여부 지정
    picks=picks,    # 데이터에서 선택할 채널의 인덱스
    baseline=None,  # 베이스 라인 보정: None은 보정하지 않겠다는 의미
    preload=True,   # 데이터를 미리 메모리에 로드할지 여부 결정 -> 빨라짐
)
# epocs 객체의 복사본 생성 
# -1초 ~ 2초 구간: 학습용 데이터를 선택할 때는 특정 시간 구간의 신호에 집중 -> 모터 이미지 관련 신호는 이벤트 1초~2초 사이에 주로 발생
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0) 
labels = epochs.events[:, -1] - 2
print(epochs.events)
print(labels)

# %%
# Classification with linear discrimant analysis

# Define a monte-carlo cross-validation generator (reduce variance): 
scores = []
epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)

# ShuffleSplit: 데이터를 무작위로 섞은 후, 지정된 비율만큼 데이터를 학습용과 테스트 용으로 분할하는 방법
# n_splits=10 : 데이터를 10분할. 10개의 교차 검증 수행
# test_size=0.2: 테스트 데이터의 비율. 0.2는 20%를 테스트 데이터로 사용, 나머지 80%는 학습 데이터
# random_state=42: 난수 생성에 사용되는 시드. 이 값을 지정하면 항상 동일한 방식으로 이루어져서, 실험 재현 가능
cv = ShuffleSplit(10, test_size=0.2, random_state=42) 
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
# n_components=4: CSP 필터로 추출할 성분의 수를 4로 설정
# reg=None: 정규화 X
# log=True: 로그 스케일로 변환 -> 분류가 더 잘되도록 함
# norm_trace=False: 공분산 행렬의 추적을 정규화하지 않음
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None) # n_jobs: 병렬처리

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}") # chance level은 무작위로 추측했을 때의 정확도를 의미
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels) # CSP 모델을 학습(fit)하고, 동시에 학습된 CSP 모델을 사용하여 데이터 변환(transform)을 수행

csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

# %%
# Look at performance over time (추가 학습 X, 각 시간대별 분류 성능 평가)

sfreq = raw.info["sfreq"]
w_length = int(sfreq * 0.5)  # running classifier: window length (0.5초 간격, sfreq에 0.5를 곱해 윈도우의 길이를 초 단위로 변환한 값을 설정)
w_step = int(sfreq * 0.1)  # running classifier: window step size (슬라이딩 윈도우가 얼마나 이동할지를 설정. 0.1초씩 이동)
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step) # 슬라이딩 윈도우의 시작 위치 설정. 

scores_windows = []

for train_idx, test_idx in cv_split: # 슬라이딩 윈도우를 이용한 교차검증
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = [] 
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
plt.axvline(0, linestyle="--", color="k", label="Onset")
plt.axhline(0.5, linestyle="-", color="k", label="Chance")
plt.xlabel("time (s)")
plt.ylabel("classification accuracy")
plt.title("Classification score over time")
plt.legend(loc="lower right")
plt.show()

##############################################################################
# References
# ----------
# .. footbibliography::
