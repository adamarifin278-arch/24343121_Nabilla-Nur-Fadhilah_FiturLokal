import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# =====================================================
# DATASET
# =====================================================

files = [

    ("buku1.JPG", "buku"),
    ("buku2.JPG", "buku"),
    ("buku3.JPG", "buku"),

    ("mug1.JPG", "mug"),
    ("mug2.JPG", "mug"),
    ("mug3.JPG", "mug"),

    ("botol1.JPG", "botol"),
    ("botol2.JPG", "botol"),
    ("botol3.JPG", "botol"),

    ("mainan1.JPG", "mainan"),
    ("mainan2.JPG", "mainan"),
    ("mainan3.JPG", "mainan"),

    ("remote1.JPG", "remote"),
    ("remote2.JPG", "remote"),
    ("remote3.JPG", "remote")

]

# =====================================================
# FEATURE DETECTOR
# =====================================================

sift = cv2.SIFT_create()

orb = cv2.ORB_create()

# SURF (jika tersedia)
try:

    surf = cv2.xfeatures2d.SURF_create()

    surf_available = True

except:

    surf_available = False

# =====================================================
# STORAGE
# =====================================================

all_descriptors = []

data = []

label = []

# =====================================================
# PROCESS IMAGE
# =====================================================

for file, kelas in files:

    img = cv2.imread(file)

    if img is None:

        print("Gambar tidak ditemukan:", file)

        continue

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # =================================================
    # SIFT
    # =================================================

    start = time.time()

    kp_sift, des_sift = sift.detectAndCompute(
        gray,
        None
    )

    sift_time = time.time() - start

    print("\n==========================")
    print("FILE :", file)
    print("==========================")

    print("Jumlah Keypoints SIFT :",
          len(kp_sift))

    print("Dimensi Descriptor :",
          des_sift.shape)

    print("Waktu Ekstraksi :",
          sift_time)

    # =================================================
    # VISUALISASI KEYPOINTS
    # =================================================

    sift_img = cv2.drawKeypoints(
        img,
        kp_sift,
        None
    )

    plt.figure(figsize=(5,5))

    plt.imshow(
        cv2.cvtColor(
            sift_img,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("SIFT Keypoints")

    plt.axis("off")

    plt.show()

    # =================================================
    # ORB
    # =================================================

    kp_orb, des_orb = orb.detectAndCompute(
        gray,
        None
    )

    print("Jumlah Keypoints ORB :",
          len(kp_orb))

    # =================================================
    # SURF
    # =================================================

    if surf_available:

        kp_surf, des_surf = surf.detectAndCompute(
            gray,
            None
        )

        print("Jumlah Keypoints SURF :",
              len(kp_surf))

    # =================================================
    # SIMPAN DESCRIPTOR
    # =================================================

    if des_sift is not None:

        all_descriptors.extend(des_sift)

        feature = np.mean(
            des_sift,
            axis=0
        )

        data.append(feature)

        label.append(kelas)

# =====================================================
# FEATURE MATCHING
# =====================================================

img1 = cv2.imread("buku1.jpg")
img2 = cv2.imread("buku2.jpg")

gray1 = cv2.cvtColor(
    img1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    img2,
    cv2.COLOR_BGR2GRAY
)

kp1, des1 = sift.detectAndCompute(
    gray1,
    None
)

kp2, des2 = sift.detectAndCompute(
    gray2,
    None
)

# =====================================================
# BRUTE FORCE MATCHING
# =====================================================

bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)

matches = bf.knnMatch(
    des1,
    des2,
    k=2
)

# =====================================================
# LOWE RATIO TEST
# =====================================================

good = []

for m,n in matches:

    if m.distance < 0.75 * n.distance:

        good.append(m)

print("\nJumlah Good Match :",
      len(good))

# =====================================================
# VISUALISASI MATCHING
# =====================================================

match_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good,
    None,
    flags=2
)

plt.figure(figsize=(12,6))

plt.imshow(
    cv2.cvtColor(
        match_img,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Feature Matching")

plt.axis("off")

plt.show()

# =====================================================
# RANSAC HOMOGRAPHY
# =====================================================

if len(good) > 10:

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1,1,2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1,1,2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    print("\nHomography Matrix")
    print(H)

# =====================================================
# FLANN MATCHING
# =====================================================

index_params = dict(
    algorithm=1,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)

matches_flann = flann.knnMatch(
    des1,
    des2,
    k=2
)

good_flann = []

for m,n in matches_flann:

    if m.distance < 0.75 * n.distance:

        good_flann.append(m)

print("\nGood Match FLANN :",
      len(good_flann))

# =====================================================
# BAG OF VISUAL WORDS
# =====================================================

all_descriptors = np.array(
    all_descriptors
)

k = 20

kmeans = KMeans(
    n_clusters=k,
    random_state=42
)

kmeans.fit(all_descriptors)

# =====================================================
# HISTOGRAM VISUAL WORDS
# =====================================================

histograms = []

for file, kelas in files:

    img = cv2.imread(file)

    if img is None:
        continue

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    kp, des = sift.detectAndCompute(
        gray,
        None
    )

    if des is None:
        continue

    words = kmeans.predict(des)

    hist, _ = np.histogram(
        words,
        bins=np.arange(k+1)
    )

    histograms.append(hist)

# =====================================================
# PCA
# =====================================================

X = np.array(histograms)

y = np.array(label)

pca = PCA(
    n_components=16
)

X_pca = pca.fit_transform(X)

print("\nUkuran Setelah PCA :",
      X_pca.shape)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.3,
    random_state=42
)

# =====================================================
# KNN CLASSIFIER
# =====================================================

knn = KNeighborsClassifier(
    n_neighbors=3
)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAkurasi KNN :",
      acc * 100,"%")

# =====================================================
# SVM CLASSIFIER
# =====================================================

svm = SVC()

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(
    y_test,
    y_pred_svm
)

print("Akurasi SVM :",
      acc_svm * 100,"%")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred_svm
)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# GRAFIK PCA COMPONENTS
# =====================================================

components = [16,32,64,128]

accuracy_list = []

for c in components:

    pca = PCA(
        n_components=min(c, X.shape[1])
    )

    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.3,
        random_state=42
    )

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    accuracy_list.append(acc)

plt.figure(figsize=(6,4))

plt.plot(
    components,
    accuracy_list,
    marker='o'
)

plt.title("Pengaruh PCA Components")

plt.xlabel("Jumlah Components")

plt.ylabel("Accuracy")

plt.show()

print("\nPROGRAM SELESAI")